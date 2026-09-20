# TonyHar

本项目是一个使用 Chroma 和本地中文嵌入模型的工程化 RAG 学习项目。

## 目录

- `RAG/Chroma/Parser/`：文件解析器与 Parser 注册表
- `RAG/Chroma/Chunker/`：基于语义边界和 token 上限的文本切块
- `RAG/Chroma/model/Chunk.py`：贯穿入库流水线的统一 Chunk 模型
- `RAG/Chroma/chunk_service.py`：编排解析和切块，并构造 Chunk
- `RAG/Chroma/DocumentIndexer.py`：校验并批量写入 ChromaDB
- `RAG/Chroma/search.py`：查询已建立的向量索引
- `RAG/Chroma/main.py`：运行完整的建索引与查询示例
- `RAG/Chroma/chroma_config.py`：统一管理跨平台设备、模型和 Chroma 客户端配置
- `requirements.txt`：Python 依赖

## 安装

Windows PowerShell：

```powershell
python -m venv env
.\env\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS/Linux：

```bash
python3 -m venv env
source env/bin/activate
python -m pip install -r requirements.txt
```

Windows 上的 NVIDIA 显卡可安装 CUDA 版 PyTorch：

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
```

Python 代码会自动选择 CUDA（Windows/NVIDIA）、MPS（Apple Silicon）或 CPU，不需要维护不同平台的代码。

## 运行

首次运行需要下载 `BAAI/bge-small-zh-v1.5`。网络较慢时可使用 Hugging Face 镜像：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HOME = "$PWD\.hf-cache"
python "RAG\Chroma\main.py"
```

macOS/Linux：

```bash
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="$PWD/.hf-cache"
python RAG/Chroma/main.py
```

模型缓存以及生成的 `my_vector_db/` 和 `chroma_db/` 已加入 `.gitignore`，不会提交到仓库。

## 当前架构

项目已经从“切分字符串后直接写入 Chroma”的验证脚本，演进为一条最小工程化 RAG 入库流水线：

```text
main.py
  → TaskContext
  → Parser Registry
  → Parser.parse()
  → TextChunker.split()
  → ChunkService
  → Chunk
  → DocumentIndexer
  → Chroma
```

目前具备：

- 分层与明确的模块职责
- 统一的 `Chunk` 流水线模型
- 根据文件类型和解析策略选择 Parser
- Parser、Chunker 和 Collection 的依赖注入
- 对 RAGFlow `one.chunk()` 旧接口的适配
- 基于段落、句子边界和 token 上限的语义切块
- Chunk 批量索引
- 原文字符位置记录
- 根据 `document_id` 删除文档 Chunk

纵向模块负责提供可替换的能力，横向流水线负责让数据依次经过解析、切块和索引：

```text
纵向结构：Parser / Chunker / ChunkService / DocumentIndexer
横向流程：bytes → parsed document → segment → Chunk → Chroma
```

## 当前边界

这个项目已经不是单文件 Demo，但还没有达到生产级。当前最明显的耦合是：

```text
DocumentIndexer
  → Chroma Collection
```

`DocumentIndexer` 仍然直接调用：

```python
collection.upsert(...)
collection.delete(...)
```

下一步应增加统一的 `VectorStore` 接口，并提供 `ChromaVectorStore` 实现：

```text
DocumentIndexer
  → VectorStore
  → ChromaVectorStore
  → Chroma
```

这样 `DocumentIndexer` 只负责索引流程，不需要了解 Chroma 的具体 API，也能在以后替换为 Qdrant、Milvus 或 Elasticsearch。

## 后续演进路线

距离生产级 RAG 系统仍缺少：

- 独立的 `EmbeddingService`
- 独立的 `VectorStore` 接口
- Chroma 与 `DocumentIndexer` 解耦
- 文档更新事务及旧 Chunk 自动清理
- 失败重试
- 部分写入后的恢复或回滚
- 任务状态与进度管理
- 结构化日志
- 正式单元测试
- 配置管理
- 搜索结果模型
- Reranker
- 引用构建
- 异步任务队列

推荐按以下顺序继续演进：

```text
VectorStore 抽象
  → EmbeddingService
  → IngestionPipeline
  → 文档更新与失败恢复
  → Retrieval / Reranker / Citation
  → 异步任务与可观测性
```
