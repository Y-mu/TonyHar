import unittest

from rag.vector_store_base import Chunk
from tools.knowledge_search import create_knowledge_search_tool


class FakeRetriever:
    def retrieve(self, query: str, top_k: int):
        return [
            Chunk(
                id="chunk-1",
                document_id="doc-1",
                filename="manual.md",
                text="允许 ECU 调校，但必须符合排放要求。",
            )
        ][:top_k]


class KnowledgeSearchToolTest(unittest.TestCase):
    QUERY = '有资料称“允许ECU调校提升发动机性能，但需符合国六排放标准'

    def setUp(self):
        self.tool = create_knowledge_search_tool(FakeRetriever())

    def test_knowledge_search_tool_parameters(self):
        result = self.tool.run(query=self.QUERY, top_k=5)

        self.assertEqual(result["count"], 1)
        self.assertEqual(result["matches"][0]["chunk_id"], "chunk-1")


if __name__ == "__main__":
    unittest.main()
