import unittest

from tonyhar.tools.knowledge_list import format_knowledge_list


class KnowledgeListToolTest(unittest.TestCase):
    def test_formats_documents_as_markdown(self):
        result = format_knowledge_list({
            "count": 2,
            "documents": [
                {
                    "document_id": "manual-1",
                    "filename": "维修手册.md",
                },
                {
                    "document_id": "guide-2",
                    "filename": "使用指南.txt",
                },
            ],
        })

        self.assertEqual(
            result,
            "## 知识库文档\n\n"
            "当前共有 **2** 篇已入库文档：\n\n"
            "1. **维修手册.md**\n"
            "   - 文档 ID：`manual-1`\n"
            "2. **使用指南.txt**\n"
            "   - 文档 ID：`guide-2`",
        )

    def test_formats_empty_knowledge_base(self):
        self.assertEqual(
            format_knowledge_list({"count": 0, "documents": []}),
            "## 知识库文档\n\n当前还没有已入库的文档。",
        )


if __name__ == "__main__":
    unittest.main()
