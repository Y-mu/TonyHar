"""列出知识库中文档的工具。"""

import asyncio

from tonyhar.tooling import BaseTool, ToolPolicy, tool

from tonyhar.rag.document_service import DocumentService


def _inline_code(value: object) -> str:
    normalized = str(value).replace("\r", " ").replace("\n", " ")
    return normalized.replace("`", "ˋ")


def _markdown_text(value: object) -> str:
    normalized = str(value).replace("\r", " ").replace("\n", " ")
    for character in ("\\", "*", "_", "[", "]", "<", ">"):
        normalized = normalized.replace(character, f"\\{character}")
    return normalized


def format_knowledge_list(data: object) -> str:
    """将知识库目录转换为适合 CLI 和 Web 展示的 Markdown。"""
    if not isinstance(data, dict):
        return "## 知识库文档\n\n暂时无法读取文档目录。"

    raw_documents = data.get("documents", [])
    documents = raw_documents if isinstance(raw_documents, list) else []
    if not documents:
        return "## 知识库文档\n\n当前还没有已入库的文档。"

    lines = [
        "## 知识库文档",
        "",
        f"当前共有 **{len(documents)}** 篇已入库文档：",
        "",
    ]
    for index, document in enumerate(documents, start=1):
        if not isinstance(document, dict):
            lines.append(f"{index}. `{_inline_code(document)}`")
            continue

        filename = document.get("filename") or "未命名文档"
        document_id = document.get("document_id") or ""
        lines.append(f"{index}. **{_markdown_text(filename)}**")
        if document_id:
            lines.append(f"   - 文档 ID：`{_inline_code(document_id)}`")

    return "\n".join(lines)


def create_knowledge_list_tool(
    document_service: DocumentService,
) -> BaseTool:
    """使用已装配的文档服务创建知识库目录工具。"""

    @tool(
        policy=ToolPolicy(
            max_attempts=2,
            idempotent=True,
            parallel_safe=True,
        ),
        result_formatter=format_knowledge_list,
    )
    async def knowledge_list() -> dict:
        """列出知识库中已经入库的文档，不检索文档正文。"""
        documents = await asyncio.to_thread(document_service.list_documents)
        return {
            "count": len(documents),
            "documents": [
                {
                    "document_id": document.document_id,
                    "filename": document.filename,
                }
                for document in documents
            ],
        }

    return knowledge_list
