import unittest
from unittest.mock import Mock, patch

from tonyhar.tools.spider_url import GetUrlTool, clean_extracted_markdown


class CleanExtractedMarkdownTest(unittest.TestCase):
    def test_normalizes_web_whitespace_and_soft_line_breaks(self):
        dirty = (
            "\ufeff到每个部位\u00a0【电压值】\r\n"
            "变化，驱动功能。\u200b\r\n\r\n\r\n"
            "open\u202floop\t开路模式。\x00"
        )

        cleaned = clean_extracted_markdown(dirty)

        self.assertEqual(
            cleaned,
            "到每个部位 【电压值】变化，驱动功能。\n\nopen loop 开路模式。",
        )

    def test_preserves_markdown_blocks_and_fenced_code(self):
        dirty = (
            "# 标题\n"
            "正文第一行\n"
            "继续正文。\n\n"
            "- 项目一\n"
            "- 项目二\n\n"
            "| 名称 | 值 |\n"
            "| --- | --- |\n"
            "| ECU | 12 V |\n\n"
            "```python\n"
            "value  =  1\n\n"
            "print(value)\n"
            "```\n\n"
            "    indented  =  True"
        )

        cleaned = clean_extracted_markdown(dirty)

        self.assertIn("# 标题\n正文第一行继续正文。", cleaned)
        self.assertIn("- 项目一\n- 项目二", cleaned)
        self.assertIn("| 名称 | 值 |\n| --- | --- |", cleaned)
        self.assertIn("```python\nvalue  =  1\n\nprint(value)\n```", cleaned)
        self.assertTrue(cleaned.endswith("    indented  =  True"))

    def test_decodes_html_entities_and_is_idempotent(self):
        dirty = "ECU&nbsp;&amp;&nbsp;OBD II。\n\n参数：１２Ｖ。"

        cleaned = clean_extracted_markdown(dirty)

        self.assertEqual(cleaned, "ECU & OBD II。\n\n参数：12V。")
        self.assertEqual(clean_extracted_markdown(cleaned), cleaned)


class GetUrlToolTest(unittest.IsolatedAsyncioTestCase):
    @patch("tonyhar.tools.spider_url.trafilatura.extract")
    @patch("tonyhar.tools.spider_url.requests.get")
    async def test_execute_returns_cleaned_markdown(self, get, extract):
        response = Mock(text="<html>正文</html>")
        get.return_value = response
        extract.return_value = ("引擎管理系统\u00a0ECU。\n" * 12) + "\u200b"

        result = await GetUrlTool().execute("https://example.com/article")

        response.raise_for_status.assert_called_once_with()
        self.assertNotIn("\u00a0", result)
        self.assertNotIn("\u200b", result)
        self.assertIn("引擎管理系统 ECU。引擎管理系统 ECU。", result)

    @patch("tonyhar.tools.spider_url.trafilatura.extract")
    @patch("tonyhar.tools.spider_url.requests.get")
    async def test_execute_validates_content_after_cleaning(self, get, extract):
        get.return_value = Mock(text="<html></html>")
        extract.return_value = "\u200b" * 200

        with self.assertRaisesRegex(ValueError, "没有提取到有效正文"):
            await GetUrlTool().execute("https://example.com/empty")


if __name__ == "__main__":
    unittest.main()
