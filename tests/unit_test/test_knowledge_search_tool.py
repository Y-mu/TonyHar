import unittest
from tools.knowledge_search import KnowledgeSearchTool


class KnowledgeSearchToolTest(unittest.TestCase):
    QUERY = (
        '有资料称“允许ECU调校提升发动机性能，但需符合国六排放标准'
    )

    @classmethod
    def setUpClass(cls):
        cls.tool = KnowledgeSearchTool()

    def test_knowledge_search_tool_parameters(self):
        result = self.tool.run(self.QUERY, top_k=5)
        self.assertIsNotNone(result)
        self.assertGreater(len(result), 0)



if __name__ == "__main__":
    unittest.main()