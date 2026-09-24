import hashlib
import tempfile
import unittest
from pathlib import Path
from typing import Sequence

from core.parser import TxtParser
from rag.document_service import DocumentService
from rag.pipeline_context import PipelineContext
from rag.scheduler import DocumentScheduler
from rag.splitter import HybridSplitter, SplitterConfig
from rag.vector_store_base import Chunk, DocumentSummary, VectorStore


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

    def test_manual_txt_is_parsed_and_stored(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manual_path = Path(temp_dir) / "manual.txt"
            manual_path.write_text(
                "发动机检查步骤。" * 300,
                encoding="utf-8",
            )

            store = MemoryVectorStore()
            scheduler = DocumentScheduler(
                parser=TxtParser(),
                splitter=HybridSplitter(
                    config=SplitterConfig(chunk_size=128, chunk_overlap=16)
                ),
                document_service=DocumentService(store),
            )

            document_id = hashlib.sha1(
                str(manual_path.resolve()).encode("utf-8")
            ).hexdigest()
            context = PipelineContext(
                document_id=document_id,
                filename=manual_path.name,
                binary=manual_path.read_bytes(),
            )

            result = scheduler.run(context)

            self.assertIs(result, context)
            self.assertEqual(context.progress, 1.0)
            self.assertEqual(context.errors, [])
            self.assertGreater(len(context.chunks), 1)
            self.assertEqual(len(context.chunks), len(store.chunks))

            for chunk in context.chunks:
                self.assertEqual(chunk.document_id, document_id)
                self.assertEqual(chunk.filename, "manual.txt")
                self.assertTrue(chunk.text.strip())
                self.assertEqual(chunk.metadata["document_id"], document_id)
                self.assertEqual(chunk.metadata["filename"], "manual.txt")


if __name__ == "__main__":
    unittest.main()
