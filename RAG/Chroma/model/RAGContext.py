from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional
from uuid import uuid4


ProgressCallback = Callable[[Any, str], None]


@dataclass
class RecordingContext:
    """记录一条 RAG 流水线各阶段的中间结果。"""

    records: Dict[str, Any] = field(default_factory=dict)

    def record(self, key: str, value: Any) -> None:
        self.records[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.records.get(key, default)


@dataclass
class IngestionConfig:
    parser_id: str = "auto"
    parser_config: Dict[str, Any] = field(default_factory=dict)
    from_page: int = 0
    to_page: Optional[int] = None
    replace_existing: bool = False


@dataclass
class RetrievalConfig:
    top_k: int = 5
    score_threshold: Optional[float] = None
    filters: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if self.score_threshold is not None and not 0 <= self.score_threshold <= 1:
            raise ValueError("score_threshold must be between zero and one")


@dataclass
class GenerationConfig:
    model: Optional[str] = None
    system_prompt: str = "请仅根据提供的上下文回答问题。"
    temperature: float = 0.2
    max_tokens: int = 1024

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must not be negative")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")


@dataclass
class RAGContext:
    """贯穿入库、召回和生成阶段的一次 RAG 任务上下文。"""

    document_id: Optional[str] = None
    filename: Optional[str] = None
    size: int = 0
    language: str = "Chinese"
    tenant_id: Optional[str] = None
    knowledge_base_id: Optional[str] = None
    task_id: str = field(default_factory=lambda: uuid4().hex)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    recording_context: RecordingContext = field(default_factory=RecordingContext)
    progress_callback: Optional[ProgressCallback] = None

    # 以下属性为现有 ChunkService 提供稳定接口；配置源仍归 ingestion 管理。
    @property
    def parser_id(self) -> str:
        return self.ingestion.parser_id

    @property
    def parser_config(self) -> Dict[str, Any]:
        return self.ingestion.parser_config

    @property
    def from_page(self) -> int:
        return self.ingestion.from_page

    @property
    def to_page(self) -> Optional[int]:
        return self.ingestion.to_page

    def progress_cb(self, prog: Any = None, msg: str = "") -> None:
        if self.progress_callback:
            self.progress_callback(prog, msg)
        elif msg:
            print(f"[RAG:{self.task_id[:8]}] {msg}")
