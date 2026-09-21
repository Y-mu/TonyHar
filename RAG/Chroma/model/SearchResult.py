from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    document_id: str
    text: str
    distance: float
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)