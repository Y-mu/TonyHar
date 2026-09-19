"""Parser + ChunkService + Chroma 的最小可执行示例。

运行：
    python -m RAG.Chroma.main ./manual.pdf --document-id manual-v1
"""

import argparse
import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from RAG.Chroma.DocumentIndexer import DocumentIndexer
from RAG.Chroma.chunk_service import ChunkService


@dataclass
class RecordingContext:
    records: Dict[str, Any] = field(default_factory=dict)

    def record(self, key: str, value: Any) -> None:
        self.records[key] = value


@dataclass
class DemoTaskContext:
    filename: str
    size: int
    document_id: str
    parser_id: str = "auto"
    language: str = "Chinese"
    parser_config: Dict[str, Any] = field(default_factory=dict)
    tenant_id: Optional[str] = None
    from_page: int = 0
    to_page: Optional[int] = None
    recording_context: RecordingContext = field(default_factory=RecordingContext)

    def progress_cb(self, prog: Any = None, msg: str = "") -> None:
        if msg:
            print(f"[Parser] {msg}")


async def index_file(path: Path, document_id: str, parser_id: str = "auto") -> int:
    binary = path.read_bytes()
    context = DemoTaskContext(
        filename=path.name,
        size=len(binary),
        document_id=document_id,
        parser_id=parser_id,
    )

    chunk_service = ChunkService(context)
    chunks = await chunk_service.build_chunks(binary)
    print(f"解析完成：{len(chunks)} 个 Chunk")

    indexer = DocumentIndexer()
    count = indexer.index_chunks(chunks)
    print(f"索引完成：写入 {count} 个 Chunk")
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse, chunk, and index a file into Chroma")
    parser.add_argument("file", type=Path, help="待索引文件：pdf/docx/xlsx/txt/html 等")
    parser.add_argument("--document-id", help="文档 ID；默认使用文件名（不含扩展名）")
    parser.add_argument("--parser-id", default="auto", choices=["auto", "txt", "one"], help="解析策略")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.file.is_file():
        raise SystemExit(f"文件不存在：{args.file}")

    document_id = args.document_id or args.file.stem
    try:
        asyncio.run(index_file(args.file, document_id, args.parser_id))
    except Exception as error:
        raise SystemExit(f"索引失败：{error}") from error


if __name__ == "__main__":
    main()
