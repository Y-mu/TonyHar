# RAG 文档处理管线设计

本文档描述如何使用 `PipelineContext` 将文档任务从调度器贯通到向量存储。当前方案已经落地到 `rag/` 目录。

实现状态：已完成基础管线（Parser、Splitter、Context、Scheduler、DocumentService、VectorStore、ChromaStoreImp）。

## 1. 总体流程

```text
Scheduler
    ↓
PipelineContext
    ↓
Parser
    ↓
Splitter
    ↓
DocumentService
    ↓
VectorStore
```

各层职责如下：

| 组件 | 职责 |
| --- | --- |
| `Scheduler` | 编排一次文档处理任务的执行顺序 |
| `PipelineContext` | 保存本次任务的数据、状态、进度和错误 |
| `Parser` | 将文件内容解析为 `RawDocument` |
| `Splitter` | 将 `RawDocument` 切分为 `Chunk` |
| `DocumentService` | 处理业务 metadata，并调用向量存储 |
| `VectorStore` | 定义统一的向量存储接口 |
| `ChromaStoreImp` | 实现 Chroma 的连接、写入、删除和查询 |

## 2. 推荐目录结构

```text
rag/
├── pipeline_context.py       # 任务上下文
├── scheduler.py              # 处理流程编排
├── document_service.py       # 文档入库服务
├── retriever.py              # 检索服务
├── splitter.py               # 文档切分
├── vector_store_base.py      # VectorStore 抽象接口
├── chroma_store_imp.py       # Chroma 具体实现
└── PIPELINE_CONTEXT_DESIGN.md
```

## 3. PipelineContext

`PipelineContext` 是一次文档处理任务的上下文对象。它应该保存任务数据和处理状态，但不应该负责业务逻辑。

```python
from dataclasses import dataclass, field
from typing import Any

from core.parser import RawDocument
from .vector_store_base import Chunk


@dataclass
class PipelineContext:
    document_id: str
    filename: str

    binary: bytes | None = None
    raw_document: RawDocument | None = None
    chunks: list[Chunk] = field(default_factory=list)

    metadata: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    progress: float = 0.0

    def add_error(self, message: str) -> None:
        self.errors.append(message)

    def update_progress(self, value: float) -> None:
        self.progress = max(0.0, min(1.0, value))
```

## 4. DocumentService

`DocumentService` 负责文档入库相关的业务规则，例如补充 metadata、校验切片和调用 `VectorStore`。

```python
from .pipeline_context import PipelineContext
from .vector_store_base import VectorStore


class DocumentService:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def index(self, ctx: PipelineContext) -> int:
        if not ctx.chunks:
            return 0

        prepared_chunks = [
            self.prepare_chunk(chunk)
            for chunk in ctx.chunks
        ]

        ctx.chunks = prepared_chunks
        count = self.vector_store.upsert(prepared_chunks)
        ctx.update_progress(1.0)
        return count

    @staticmethod
    def prepare_chunk(chunk):
        metadata = dict(chunk.metadata)
        metadata.update({
            "document_id": chunk.document_id,
            "filename": chunk.filename,
            "chunk_index": chunk.chunk_index,
        })

        if chunk.title:
            metadata.setdefault("title", chunk.title)
        if chunk.page is not None:
            metadata.setdefault("page", chunk.page)
        if chunk.position is not None:
            metadata.setdefault("position", str(chunk.position))

        chunk.metadata = metadata
        return chunk
```

`DocumentService` 依赖的是 `VectorStore` 接口，而不是 `ChromaStoreImp`。因此未来替换为 Milvus、FAISS 或其他向量数据库时，不需要修改文档服务。

## 5. Scheduler

调度器只负责编排，不负责解析细节、切片细节或数据库细节。

```python
class DocumentScheduler:
    def __init__(self, parser, splitter, document_service):
        self.parser = parser
        self.splitter = splitter
        self.document_service = document_service

    def run(self, ctx: PipelineContext) -> PipelineContext:
        try:
            ctx.update_progress(0.1)

            if ctx.binary is None:
                raise ValueError("ctx.binary 不能为空")

            # 文件内容 → RawDocument
            ctx.raw_document = self.parser.parse(
                filename=ctx.filename,
                binary=ctx.binary,
            )
            ctx.update_progress(0.35)

            # RawDocument → Chunk 列表
            ctx.chunks = self.splitter.split(ctx.raw_document)
            ctx.update_progress(0.7)

            # metadata 处理并写入 VectorStore
            self.document_service.index(ctx)
            ctx.update_progress(1.0)

        except Exception as exc:
            ctx.add_error(str(exc))
            raise

        return ctx
```

## 6. VectorStore 的边界

建议保持如下接口：

```python
class VectorStore(ABC):
    @abstractmethod
    def upsert(self, chunks: Sequence[Chunk]) -> int:
        ...

    @abstractmethod
    def delete_document(self, document_id: str) -> int:
        ...

    @abstractmethod
    def search(self, query: str, top_k: int) -> Sequence[Chunk]:
        ...
```

`VectorStore` 接收 `Chunk`，而不是 `PipelineContext`：

```python
vector_store.upsert(ctx.chunks)
```

原因是向量存储不应该知道：

- 文件二进制内容
- Parser
- Splitter
- 处理进度
- 调度错误
- 用户请求状态

`ChromaStoreImp` 只负责把 `Chunk` 转换成 Chroma 所需要的 `ids`、`documents` 和 `metadatas`，并执行数据库操作。

## 7. 使用示例

```python
from rag.chroma_store_imp import ChromaStoreImp
from rag.document_service import DocumentService
from rag.pipeline_context import PipelineContext

store = ChromaStoreImp(
    database_name="data/chroma",
    collection_name="documents",
)

document_service = DocumentService(store)

scheduler = DocumentScheduler(
    parser=TxtParser(),
    splitter=HybridSplitter(),
    document_service=document_service,
)

ctx = PipelineContext(
    document_id="doc-001",
    filename="test.txt",
    binary=file_bytes,
)

scheduler.run(ctx)

print(ctx.progress)
print(len(ctx.chunks))
print(ctx.errors)
```

## 8. 已完成实现

- `core/parser.py` 的 Parser 接口已统一为 `parse(filename, binary, callback)`。
- `TxtParser` 已支持 UTF-8、UTF-8 BOM 和 GB18030 解码。
- `rag/splitter.py` 已实现带 overlap 的文本切分，并生成稳定的 Chunk ID。
- `rag/pipeline_context.py` 已提供任务级上下文、进度和错误收集。
- `rag/scheduler.py` 已串联 Parser、Splitter 和 DocumentService。
- `DocumentService` 负责业务 metadata 处理。
- `ChromaStoreImp` 负责 Chroma 连接及数据库格式适配。

## 9. 核心原则

```text
ctx 是应用层的任务状态
Chunk 是领域层的数据对象
VectorStore 是基础设施抽象
ChromaStoreImp 是具体数据库适配器
```

因此推荐的依赖方向是：

```text
Scheduler → DocumentService → VectorStore ← ChromaStoreImp
```

而不是：

```text
Scheduler → ChromaStoreImp → Parser / Splitter
```

## 10. 下一步

1. 为 `TxtParser`、`HybridSplitter` 和 `DocumentScheduler` 增加单元测试。
2. 为 `HybridSplitter` 增加更多可插拔的文档块策略。
3. 增加 PDF、Markdown 等 Parser，并统一 Parser 注册机制。
4. 为 `DocumentScheduler` 增加可选的进度回调和可恢复错误处理。
5. 增加 Retriever 的 metadata 过滤、距离分数和重排能力。
6. 为 `ChromaStoreImp` 增加批量写入、连接配置和健康检查。

## 11. 当前测试入口

项目根目录的 `main.py` 保留聊天 Agent，并在 Agent 前增加文件规则层。

```bash
env3.13/bin/python main.py
```

执行顺序为：

```text
用户输入
  → 文件规则层（检测本地路径）
  → 读取文件
  → TxtParser
  → HybridSplitter
  → DocumentService
  → ChromaStoreImp.upsert
  → 将入库结果附加到 Agent 输入
  → Agent 正常对话/调用工具
```

普通对话仍按原有 Agent 流程执行：

```bash
你: 3+3 等于多少？
```

消息中包含现有的 `.txt` 或 `.md` 文件路径时，规则层会尝试解析并入库：

```text
你: 帮我处理 ./manual.txt
```

Chroma 和 embedding 模型采用延迟初始化；只有检测到文件时才加载。首次文件入库需要模型已经存在于项目 `.hf-cache`，或者运行环境允许 Hugging Face 下载模型。
