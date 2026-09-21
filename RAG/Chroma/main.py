"""Parser + ChunkService + Chroma 的最小可执行示例。

运行：
    python -m RAG.Chroma.main ./manual.pdf --document-id manual-v1
"""

import argparse
import asyncio
from pathlib import Path

from .svr.DocumentIndexer import DocumentIndexer
from .svr.TaskHandler import TaskHandler
from .svr.chunk_service import ChunkService
from .store.chroma_vector_store import ChromaVectorStore
from .model import IngestionConfig, RAGContext




async def index_file(path: Path, document_id: str, parser_id: str = "auto") -> int:
    binary = path.read_bytes()
    context = RAGContext(
        filename=path.name,
        size=len(binary),
        document_id=document_id,
        ingestion=IngestionConfig(parser_id=parser_id),
    )

    vector_store = ChromaVectorStore()
    indexer = DocumentIndexer(vector_store=vector_store)
    chunk_service = ChunkService(context)
    handler = TaskHandler(context, chunk_service, indexer)
    result = await handler.handle_task(
        binary,
        replace_existing=context.ingestion.replace_existing,
    )
    print(
        f"处理完成：生成 {result.chunk_count} 个 Chunk，"
        f"写入 {result.indexed_count} 个，耗时 {result.elapsed_seconds:.2f}s"
    )
    return result.indexed_count


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
