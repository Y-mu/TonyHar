from dataclasses import dataclass
from time import perf_counter
from typing import Any

from .DocumentIndexer import DocumentIndexer
from .chunk_service import ChunkService


@dataclass(frozen=True)
class TaskResult:
    """一次文档入库任务的执行结果。"""

    document_id: str
    chunk_count: int
    indexed_count: int
    elapsed_seconds: float


class TaskHandler:
    """编排一次文档解析、切块和索引任务。"""

    def __init__(
        self,
        task_context: Any,
        chunk_service: ChunkService,
        document_indexer: DocumentIndexer,
    ) -> None:
        self._task_context = task_context
        self._chunk_service = chunk_service
        self._document_indexer = document_indexer

    async def handle_task(
        self,
        storage_binary: bytes,
        replace_existing: bool = False,
    ) -> TaskResult:
        """执行完整入库任务，并保留原始异常堆栈。"""
        started_at = perf_counter()
        document_id = self._document_id()

        try:
            chunks = await self._chunk_service.build_chunks(storage_binary)
            deleted_count = 0
            if replace_existing:
                deleted_count = self._document_indexer.delete_document(document_id)
            self._record("deleted_count", deleted_count)

            indexed_count = self._document_indexer.index_chunks(chunks)
            result = TaskResult(
                document_id=document_id,
                chunk_count=len(chunks),
                indexed_count=indexed_count,
                elapsed_seconds=perf_counter() - started_at,
            )
            self._record("task_result", result)
            self._progress(1.0, f"Task completed: indexed {indexed_count} chunks")
            return result
        except Exception as error:
            self._record("task_error", str(error))
            self._progress(-1.0, f"Task failed: {error}")
            raise

    def _document_id(self) -> str:
        value = getattr(self._task_context, "document_id", None) or getattr(
            self._task_context, "doc_id", None
        )
        if not value or not str(value).strip():
            raise ValueError("TaskContext must provide document_id or doc_id")
        return str(value)

    def _progress(self, prog: float, msg: str) -> None:
        callback = getattr(self._task_context, "progress_cb", None)
        if callback:
            callback(prog=prog, msg=msg)

    def _record(self, key: str, value: Any) -> None:
        recording = getattr(self._task_context, "recording_context", None)
        if recording and hasattr(recording, "record"):
            recording.record(key, value)
