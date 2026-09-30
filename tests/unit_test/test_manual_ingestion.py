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
from tonyhar.tools.document_ingestion import DocumentIngestionTool
from tonyhar.tools.file_read import FileReadTool


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


class ManualIngestionTest(unittest.IsolatedAsyncioTestCase):
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

    async def test_file_read_tool_parses_file_and_returns_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manual_path = Path(temp_dir) / "manual.txt"
            manual_path.write_bytes("发动机检查步骤。".encode("gb18030"))

            tool = FileReadTool(FileTextReader(TxtParser()))
            text = await tool.execute(path=str(manual_path))

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

    async def test_file_read_output_can_be_ingested_as_document(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manual_path = Path(temp_dir) / "manual.txt"
            manual_path.write_text(
                "发动机检查步骤。" * 300,
                encoding="utf-8",
            )

            store = MemoryVectorStore()
            file_read = FileReadTool(FileTextReader(TxtParser()))
            document_ingestion = DocumentIngestionTool(
                self.text_ingestion(store)
            )

            text = await file_read.execute(path=str(manual_path))
            result = await document_ingestion.execute(
                text=text,
                filename=manual_path.name,
                source_type="file",
                source_uri=str(manual_path.resolve()),
            )

            self.assertTrue(result["success"])
            self.assertGreater(result["chunk_count"], 1)
            for chunk in store.chunks.values():
                self.assertEqual(chunk.document_id, result["document_id"])
                self.assertEqual(chunk.filename, "manual.txt")
                self.assertTrue(chunk.text.strip())
                self.assertEqual(
                    chunk.metadata["document_id"],
                    result["document_id"],
                )
                self.assertEqual(chunk.metadata["filename"], "manual.txt")
                self.assertEqual(chunk.metadata["source_type"], "file")
                self.assertEqual(
                    chunk.metadata["source_uri"],
                    str(manual_path.resolve()),
                )

    async def test_file_read_tool_rejects_directories_and_unsupported_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tool = FileReadTool(FileTextReader(TxtParser()))

            with self.assertRaisesRegex(ValueError, "只支持"):
                await tool.execute(path=str(Path(temp_dir) / "manual.pdf"))

            with self.assertRaisesRegex(ValueError, "单个文件"):
                await tool.execute(path=temp_dir)


if __name__ == "__main__":
    unittest.main()
