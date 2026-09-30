# RAG 文本入库管线设计

当前管线把“从内容来源取得字符串”和“字符串切块入库”分成两个正式边界，使本地文件、
网页抓取及后续其他来源复用同一套索引逻辑。

## 总体流程

```text
本地文件 → FileTextReader ─┐
网页正文 → Web 抓取/清洗 ──┤
其他来源 → str ────────────┤
                            ↓
                 TextIngestionService
                   → RawDocument
                   → HybridSplitter
                   → DocumentService
                   → VectorStore（dense + BM25 混合检索）
                   → ChromaStoreImp
```

## 组件职责

| 组件 | 职责 |
| --- | --- |
| `DocumentParser` | 将一种文件编码或格式解析为正文 |
| `FileTextReader` | 读取单个本地文件，调用 Parser 并返回 `str` |
| `TextIngestionService` | 接收正文与文档身份，切块、规范 metadata 并写库 |
| `PipelineContext` | 保存本次文本入库的正文、切片、进度和错误 |
| `HybridSplitter` | 将 `RawDocument` 切分为稳定的 `Chunk` |
| `DocumentService` | 规范 Chunk metadata，并通过存储端口写入 |
| `VectorStore` | 定义 dense 与 keyword 检索端口 |
| `ChromaStoreImp` | 实现 Chroma 存储适配 |

`TextIngestionService` 的正式入口为：

```python
context = text_ingestion.ingest(
    text,
    document_id="稳定的来源标识",
    filename="article.md",
    metadata={
        "source_type": "web",
        "source_url": "https://example.com/article",
    },
)
```

它不接收文件路径或二进制内容。来源适配器必须先完成读取、格式解析与必要的正文清洗。

## 本地文件入库

```text
FileIngestionTool
  → FileIngestionService 发现用户消息中的文件
  → FileTextReader.read(path)
  → TextIngestionService.ingest(text, ...)
```

`FileIngestionTool` 是完整的用户用例入口，但内部读取和入库可以分别测试、分别复用。
支持的文件类型和目录扫描限制由 `FileIngestionService` 管理。

## Web 接入

Web 工具完成安全 URL 校验、HTTP 获取、Trafilatura 提取和 Markdown 清洗后，可直接调用：

```python
context = text_ingestion.ingest(
    cleaned_markdown,
    document_id=url_hash,
    filename=page_filename,
    metadata={"source_type": "web", "source_url": final_url},
)
```

网页正文不需要先写入临时文件，也不应该通过 LLM 从一个 Tool 转发给另一个 Tool。

## 依赖方向

```text
tools → FileTextReader / TextIngestionService
TextIngestionService → HybridSplitter / DocumentService
DocumentService → VectorStore ← ChromaStoreImp
```

RAG 模块不依赖 Tool、Agent 或 Web 层。组合根负责创建实例并注入依赖。

## PipelineContext

`PipelineContext` 只描述一次已经取得正文后的入库任务：

- `document_id`、`filename`：文档稳定身份；
- `raw_document`：正文及来源 metadata；
- `chunks`：实际写入向量库的规范化切片；
- `progress`、`errors`：本次任务状态。

旧的 `binary` 字段和 `DocumentScheduler` 已删除，不保留双协议。

## 测试要求

- 文件读取覆盖 UTF-8、UTF-8 BOM 和 GB18030 解码；
- 文本入库测试不依赖文件系统，并覆盖 Web 来源 metadata；
- 文件入库组合测试覆盖路径发现、读取、切块和写库；
- VectorStore 使用内存实现测试，不加载 embedding 模型。
