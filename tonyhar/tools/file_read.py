"""读取并解析单个本地文本文件。"""

import asyncio
from pathlib import Path

from tonyhar.rag.file_reader import FileTextReader
from tonyhar.tooling import BaseTool, ToolPolicy, tool


@tool(
    policy=ToolPolicy(
        timeout_seconds=30.0,
        parallel_safe=True,
        max_model_output_chars=12_000,
    )
)
class FileReadTool(BaseTool):
    """读取单个本地文本文件，解析后返回正文字符串。"""

    name = "file_read"
    description = (
        "读取单个本地 .txt 或 .md 文件，完成文本解码和解析后返回正文字符串。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "需要读取的本地文件路径。",
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    }
    supported_suffixes = frozenset({".txt", ".md"})

    def __init__(self, reader: FileTextReader):
        self.reader = reader

    async def execute(self, path: str) -> str:
        if not isinstance(path, str) or not path.strip():
            raise ValueError("文件路径不能为空")

        resolved_path = Path(path).expanduser()
        if not resolved_path.is_absolute():
            resolved_path = Path.cwd() / resolved_path
        resolved_path = resolved_path.resolve()

        if resolved_path.is_dir():
            raise ValueError("file_read 只支持读取单个文件")
        if resolved_path.suffix.lower() not in self.supported_suffixes:
            raise ValueError("只支持读取 .txt 和 .md 文件")

        return await asyncio.to_thread(self.reader.read, resolved_path)
