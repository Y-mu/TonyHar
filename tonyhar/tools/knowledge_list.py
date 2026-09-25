"""列出知识库中文档的工具。"""

import asyncio

from tonyhar.tooling import BaseTool, ToolPolicy, tool

from tonyhar.rag.document_service import DocumentService


def create_knowledge_list_tool(
    document_service: DocumentService,
) -> BaseTool:
    """使用已装配的文档服务创建知识库目录工具。"""

    @tool(policy=ToolPolicy(
        max_attempts=2,
        idempotent=True,
        parallel_safe=True,
    ))
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
