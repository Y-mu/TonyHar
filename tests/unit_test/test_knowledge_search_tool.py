import unittest

from tonyhar.rag.vector_store_base import Chunk
from tonyhar.tools.knowledge_search import KnowledgeSearchTool


class FakeRetriever:
    def retrieve_hybrid(self, query: str, top_k: int):
        return [
            Chunk(
                id="chunk-1",
                document_id="doc-1",
                filename="manual.md",
                text="允许 ECU 调校，但必须符合排放要求。",
            )
        ][:top_k]


class HybridRetriever:
    def retrieve_hybrid(self, query: str, top_k: int):
        return FakeRetriever().retrieve_hybrid(query, top_k)


class KnowledgeSearchToolTest(unittest.IsolatedAsyncioTestCase):
    QUERY = '有资料称“允许ECU调校提升发动机性能，但需符合国六排放标准'

    def setUp(self):
        self.tool = KnowledgeSearchTool(HybridRetriever())

    async def test_knowledge_search_tool_parameters(self):
        result = await self.tool.execute(query=self.QUERY, top_k=5)

        self.assertEqual(result["count"], 1)
        self.assertEqual(result["matches"][0]["chunk_id"], "chunk-1")


if __name__ == "__main__":
    unittest.main()
