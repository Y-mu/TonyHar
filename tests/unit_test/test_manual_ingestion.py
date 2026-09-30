import hashlib
import tempfile
import unittest
from pathlib import Path
from typing import Sequence

from tonyhar.rag.document_service import DocumentService
from tonyhar.rag.file_reader import FileTextReader
from tonyhar.rag.parser import TxtParser
from tonyhar.rag.splitter import HybridSplitter, SplitterConfig
from tonyhar.rag.text_ingestion import TextIngestionService
from tonyhar.rag.vector_store_base import Chunk, DocumentSummary, VectorStore
from tonyhar.tools.file_ingestion import FileIngestionService


class MemoryVectorStore(VectorStore):
    """测试用内存存储，不加载 embedding 模型。"""

    def __init__(self):
        self.chunks: dict[str, Chunk] = {}

    def upsert(self, chunks: Sequence[Chunk]) -> int:
        for chunk in chunks:
            self.chunks[chunk.id] = chunk
        return len(chunks)

    def delete_document(self, document_id: str) -> int:
        ids = [
            chunk_id for chunk_id, chunk in self.chunks.items()
            if chunk.document_id == document_id
        ]
        for chunk_id in ids:
            del self.chunks[chunk_id]
        return len(ids)

    def search(self, query: str, top_k: int) -> Sequence[Chunk]:
        return list(self.chunks.values())[:top_k]

    def keyword_search(self, query: str, top_k: int) -> Sequence[Chunk]:
        return list(self.chunks.values())[:top_k]

    def list_documents(self) -> Sequence[DocumentSummary]:
        documents = {
            chunk.document_id: DocumentSummary(
                document_id=chunk.document_id,
                filename=chunk.filename,
            )
            for chunk in self.chunks.values()
        }
        return list(documents.values())


class ManualIngestionTest(unittest.TestCase):
    @staticmethod
    def text_ingestion(store: MemoryVectorStore) -> TextIngestionService:
        return TextIngestionService(
            splitter=HybridSplitter(
                config=SplitterConfig(chunk_size=128, chunk_overlap=16)
            ),
            document_service=DocumentService(store),
        )

    def test_normalize_metadata_removes_empty_values(self):
        result = DocumentService.normalize_metadata({
            "none": None,
            "empty_string": "",
            "empty_list": [],
            "empty_dict": {},
            "zero": 0,
            "false": False,
            "value": ["heading"],
        })

        self.assertEqual(result, {
            "zero": 0,
            "false": False,
            "value": ["heading"],
        })

    def test_file_reader_parses_file_and_returns_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manual_path = Path(temp_dir) / "manual.txt"
            manual_path.write_bytes("发动机检查步骤。".encode("gb18030"))

            text = FileTextReader(TxtParser()).read(manual_path)

            self.assertEqual(text, "发动机检查步骤。")

    def test_text_is_chunked_and_stored_without_file_dependency(self):
        store = MemoryVectorStore()
        service = self.text_ingestion(store)

        context = service.ingest(
            "发动机检查步骤。" * 300,
            document_id="web-document-id",
            filename="article.md",
            metadata={
                "source_type": "web",
                "source_url": "https://example.com/article",
            },
        )

        self.assertEqual(context.progress, 1.0)
        self.assertEqual(context.errors, [])
        self.assertGreater(len(context.chunks), 1)
        self.assertEqual(len(context.chunks), len(store.chunks))
        for chunk in context.chunks:
            self.assertEqual(chunk.document_id, "web-document-id")
            self.assertEqual(chunk.filename, "article.md")
            self.assertEqual(chunk.metadata["source_type"], "web")
            self.assertEqual(
                chunk.metadata["source_url"],
                "https://example.com/article",
            )

    def test_file_ingestion_composes_reader_and_text_ingestion(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manual_path = Path(temp_dir) / "manual.txt"
            manual_path.write_text(
                "发动机检查步骤。" * 300,
                encoding="utf-8",
            )

            store = MemoryVectorStore()
            service = FileIngestionService(
                reader=FileTextReader(TxtParser()),
                text_ingestion=self.text_ingestion(store),
            )

            document_id = hashlib.sha1(
                str(manual_path.resolve()).encode("utf-8")
            ).hexdigest()
            result = service.ingest(f"请导入 {manual_path}")

            self.assertTrue(result["success"])
            self.assertEqual(result["files"][0]["document_id"], document_id)
            self.assertGreater(result["files"][0]["chunk_count"], 1)
            for chunk in store.chunks.values():
                self.assertEqual(chunk.document_id, document_id)
                self.assertEqual(chunk.filename, "manual.txt")
                self.assertTrue(chunk.text.strip())
                self.assertEqual(chunk.metadata["document_id"], document_id)
                self.assertEqual(chunk.metadata["filename"], "manual.txt")
                self.assertEqual(chunk.metadata["source_type"], "file")


if __name__ == "__main__":
    unittest.main()
