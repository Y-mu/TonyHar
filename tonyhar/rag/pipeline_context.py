"""文档处理任务上下文。"""
from dataclasses import dataclass, field
from typing import Any

from .parser import RawDocument
from .vector_store_base import Chunk


@dataclass
class PipelineContext:
    document_id: str
    filename: str
    binary: bytes | None = None
    raw_document: RawDocument | None = None
    chunks: list[Chunk] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    progress: float = 0.0

    def update_progress(self, value: float) -> None:
        self.progress = max(0.0, min(1.0, value))

    def add_error(self, message: str) -> None:
        self.errors.append(message)
