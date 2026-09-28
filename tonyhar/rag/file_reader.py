"""本地文本文件读取器。"""

from pathlib import Path

from .parser import DocumentParser


class FileTextReader:
    """读取并解析单个文本文件，只返回后续流程需要的正文字符串。"""

    def __init__(self, parser: DocumentParser):
        self.parser = parser

    def read(self, path: Path) -> str:
        if not path.is_file():
            raise ValueError(f"文件不存在: {path}")
        return self.parser.parse(path.name, path.read_bytes()).text
