"""可插拔、可组合的文档切块策略。"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import hashlib
import re
from typing import Any, Sequence
from .parser import RawDocument
from .vector_store_base import Chunk

@dataclass(frozen=True)
class SplitterConfig:
    chunk_size: int = 512
    chunk_overlap: int = 64

@dataclass(frozen=True)
class ChunkCandidate:
    text: str
    chunk_type: str = "text"
    heading_path: tuple[str, ...] = ()
    char_start: int | None = None
    char_end: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

class TextSplitter(ABC):
    @abstractmethod
    def split(self, doc: RawDocument) -> list[Chunk]: ...

class ChunkingStrategy(ABC):
    name = "strategy"
    @abstractmethod
    def supports(self, doc: RawDocument) -> bool: ...
    @abstractmethod
    def split_candidates(self, doc: RawDocument) -> Sequence[ChunkCandidate]: ...

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\u4e00-\u9fff]|[A-Za-z]+(?:['-][A-Za-z]+)*|\d+(?:\.\d+)?|[^\w\s]", text)


def _join_tokens(tokens: list[str]) -> str:
    result = ""
    for token in tokens:
        if result and re.match(r"^[A-Za-z0-9]", token) and re.search(r"[A-Za-z0-9]$", result): result += " "
        result += token
    return result

class MarkdownStrategy(ChunkingStrategy):
    name = "markdown"
    heading = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
    def supports(self, doc: RawDocument) -> bool: return doc.filename.lower().endswith((".md", ".markdown"))
    def split_candidates(self, doc: RawDocument) -> Sequence[ChunkCandidate]:
        path, blocks, buffer, start, offset, in_code = [], [], [], 0, 0, False
        def flush(end: int, kind: str = "text"):
            text = "".join(buffer).strip()
            if text: blocks.append(ChunkCandidate(text, kind, tuple(path), start, end))
            buffer.clear()
        for line in doc.text.splitlines(True):
            stripped = line.strip()
            if stripped.startswith("```"):
                if in_code: buffer.append(line); offset += len(line); flush(offset, "code"); in_code = False
                else: flush(offset); in_code = True; start = offset; buffer.append(line); offset += len(line)
                continue
            if in_code: buffer.append(line); offset += len(line); continue
            match = self.heading.match(stripped)
            if match:
                flush(offset); level = len(match.group(1)); del path[level - 1:]; path.append(match.group(2).strip()); offset += len(line); start = offset; continue
            if not stripped and buffer: flush(offset); offset += len(line); start = offset
            else:
                if not buffer: start = offset
                buffer.append(line); offset += len(line)
        flush(offset)
        return blocks

class PlainTextStrategy(ChunkingStrategy):
    name = "plain_text"
    def __init__(self, config: SplitterConfig): self.config = config
    def supports(self, doc: RawDocument) -> bool: return True
    def split_candidates(self, doc: RawDocument) -> Sequence[ChunkCandidate]:
        tokens = _tokenize(doc.text)
        result, start = [], 0
        while start < len(tokens):
            end = min(start + self.config.chunk_size, len(tokens))
            result.append(ChunkCandidate(_join_tokens(tokens[start:end])))
            if end >= len(tokens): break
            start = end - self.config.chunk_overlap
        return result

class HybridSplitter(TextSplitter):
    """策略选择 + 候选块统一装配；Scheduler 只依赖 TextSplitter。"""
    def __init__(self, strategies: Sequence[ChunkingStrategy] | None = None, config: SplitterConfig | None = None):
        self.config = config or SplitterConfig(); self.strategies = list(strategies or [MarkdownStrategy(), PlainTextStrategy(self.config)])
    def split(self, doc: RawDocument) -> list[Chunk]:
        if not doc or not doc.text.strip(): return []
        strategy = next(s for s in self.strategies if s.supports(doc)); result, index = [], 0
        for candidate in strategy.split_candidates(doc):
            tokens, start = _tokenize(candidate.text), 0
            while start < len(tokens):
                end = min(start + self.config.chunk_size, len(tokens)); text = _join_tokens(tokens[start:end]).strip()
                if text:
                    prefix = " > ".join(candidate.heading_path); content = f"[{prefix}]\n{text}" if prefix else text
                    metadata = {
                        **doc.metadata,
                        **candidate.metadata,
                        "chunk_type": candidate.chunk_type,
                        "strategy": strategy.name,
                    }
                    if candidate.heading_path:
                        metadata["heading_path"] = list(candidate.heading_path)
                    raw = f"{doc.document_id}:{index}:{content}"
                    result.append(Chunk(hashlib.sha1(raw.encode()).hexdigest(), doc.document_id, doc.filename, content, index, metadata=metadata, raw={"char_start": candidate.char_start, "char_end": candidate.char_end})); index += 1
                if end >= len(tokens): break
                start = end - self.config.chunk_overlap
        return result
