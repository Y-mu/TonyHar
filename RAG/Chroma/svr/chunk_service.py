from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from RAG.Chroma.model.Chunk import Chunk
from RAG.Chroma.Chunker import TextChunker, TextChunkerConfig

DOC_MAXIMUM_SIZE = 10 * 1024 * 1024


def get_parser(filename: str, parser_id: str):
    import sys

    # Parser.one 沿用了 RAGFlow 的顶层导入（deepdoc、rag、api 等）。
    # 将本地 RAGFlow 源码根目录加入搜索路径后，这些导入才能解析。
    ragflow_root = Path(__file__).resolve().parents[2] / "ragflow"
    if ragflow_root.exists() and str(ragflow_root) not in sys.path:
        sys.path.insert(0, str(ragflow_root))
    from RAG.Chroma.Parser.registry import get_parser as resolve_parser
    return resolve_parser(filename, parser_id)


class ChunkService:
    """使用``Parser.one`` 解析文件并构建流水线记录。"""

    def __init__(self, task_context: Any, parser=None, text_chunker=None, maximum_size: int = DOC_MAXIMUM_SIZE):
        self._task_context = task_context
        self._parser = parser or get_parser(
            str(getattr(task_context, "filename", None) or getattr(task_context, "name", "")),
            getattr(task_context, "parser_id", "auto"),
        )
        parser_config = dict(getattr(task_context, "parser_config", {}) or {})
        self._text_chunker = text_chunker or TextChunker(TextChunkerConfig(
            chunk_size=parser_config.get("chunk_token_num", 256),
            chunk_overlap=parser_config.get("overlap_token_num", 30),
        ))
        self._maximum_size = maximum_size

    async def build_chunks(
        self,
        storage_binary: bytes,
        on_chunking_start: Optional[Callable[..., Any]] = None,
    ) -> List[Chunk]:
        ctx = self._task_context
        recorded_size = getattr(ctx, "size", len(storage_binary))
        if recorded_size > self._maximum_size or len(storage_binary) > self._maximum_size:
            self._record("file_size_exceeded", True)
            self._progress(-1, "File size exceeds (<= %dMb)" % (self._maximum_size // 1024 // 1024))
            return []

        self._record("file_size_exceeded", False)
        self._record("parser_id", getattr(ctx, "parser_id", "one"))
        filename = str(getattr(ctx, "filename", None) or getattr(ctx, "name", ""))
        if not filename:
            raise ValueError("TaskContext must provide filename or name")

        parser_kwargs = {
            "filename": filename,
            "binary": storage_binary,
            "lang": getattr(ctx, "language", "Chinese"),
            "callback": on_chunking_start or self._progress,
            "parser_config": dict(getattr(ctx, "parser_config", {}) or {}),
            "tenant_id": getattr(ctx, "tenant_id", None),
        }
        if getattr(ctx, "from_page", None) is not None:
            parser_kwargs["from_page"] = ctx.from_page
        if getattr(ctx, "to_page", None) is not None:
            parser_kwargs["to_page"] = ctx.to_page

        parsed_documents = await run_parser(self._parser, parser_kwargs)
        self._record("parsed_documents", parsed_documents)
        document_id = str(getattr(ctx, "document_id", None) or getattr(ctx, "doc_id", None) or Path(filename).stem)
        records = []
        for parsed_document in parsed_documents:
            text = str(
                parsed_document.get("content_with_weight")
                or parsed_document.get("content")
                or parsed_document.get("text")
                or ""
            )
            for segment in self._text_chunker.split(text):
                records.append(
                    self._to_record(
                        parsed_document,
                        document_id,
                        filename,
                        len(records),
                        segment,
                    )
                )
        self._record("chunks", records)
        return records

    @staticmethod
    def _to_record(raw: Dict[str, Any], document_id: str, filename: str, index: int, segment) -> Chunk:
        if not isinstance(raw, dict):
            raise TypeError("one.py must return dictionaries")
        text = segment.text
        metadata = {"document_id": document_id, "filename": filename, "chunk_index": index}
        metadata.update(raw.get("metadata") or {})
        return Chunk(
            id=str(raw.get("id") or f"{document_id}_chunk_{index}"),
            document_id=document_id,
            filename=filename,
            text=text,
            chunk_index=index,
            content_with_weight=text,
            title=str(raw.get("title") or raw.get("docnm_kwd") or ""),
            page=raw.get("page") or raw.get("page_num"),
            position={"start": segment.start, "end": segment.end},
            metadata=metadata,
            raw=dict(raw),
        )

    def _progress(self, prog: Any = None, msg: str = "") -> None:
        callback = getattr(self._task_context, "progress_cb", None)
        if callback:
            callback(prog=prog, msg=msg)

    def _record(self, key: str, value: Any) -> None:
        recording = getattr(self._task_context, "recording_context", None)
        if recording and hasattr(recording, "record"):
            recording.record(key, value)


async def run_parser(parser, kwargs: Dict[str, Any]) -> List[Dict[str, Any]]:
    """在线程中运行同步 Parser，避免阻塞事件循环。"""
    import asyncio

    if not hasattr(parser, "parse"):
        raise TypeError("parser must implement parse()")
    return await asyncio.to_thread(parser.parse, **kwargs)
