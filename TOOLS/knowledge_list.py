"""列出 Chroma 知识库中文档的工具。"""

from pathlib import Path

import chromadb

from core.tool import BaseTool


class KnowledgeListTool(BaseTool):
    name = "knowledge_list"
    description = "列出知识库中已经入库的文档，不检索文档正文"
    parameters = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    def __init__(
        self,
        database_path: str | Path = "data/chroma",
        collection_name: str = "documents",
    ):
        path = Path(database_path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[1] / path
        self.client = chromadb.PersistentClient(path=str(path))
        self.collection_name = collection_name

    def run(self) -> dict:
        try:
            collection = self.client.get_collection(self.collection_name)
        except Exception:
            return {"count": 0, "documents": []}

        result = collection.get(include=["metadatas"])
        documents: dict[str, dict] = {}
        for metadata in result.get("metadatas") or []:
            metadata = metadata or {}
            document_id = str(metadata.get("document_id", ""))
            filename = str(metadata.get("filename", ""))
            key = document_id or filename
            if not key:
                continue
            documents.setdefault(key, {
                "document_id": document_id,
                "filename": filename,
            })

        items = sorted(documents.values(), key=lambda item: item["filename"])
        return {"count": len(items), "documents": items}
