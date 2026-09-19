from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Chunk:
    id: str
    document_id: str
    filename: str
    text: str
    chunk_index: int = 0
    content_with_weight: str = ""
    title: str = ""
    page: Optional[int] = None
    position: Optional[Any] = None
    embedding: Optional[list[float]] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
