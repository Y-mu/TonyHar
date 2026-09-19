import re
from dataclasses import dataclass
from typing import List

import tiktoken


@dataclass(frozen=True)
class TextChunkerConfig:
    chunk_size: int = 256
    chunk_overlap: int = 30
    encoding_name: str = "cl100k_base"

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must be between zero and chunk_size")


@dataclass(frozen=True)
class TextSegment:
    text: str
    start: int
    end: int
    token_count: int


class TextChunker:
    """优先按段落/句子边界切分，再按 token 上限合并。"""

    _BOUNDARY = re.compile(r".+?(?:\n\s*\n|(?<=[。！？!?；;])(?:\s+|$)|$)", re.S)

    def __init__(self, config: TextChunkerConfig = None) -> None:
        self.config = config or TextChunkerConfig()
        self._encoding = tiktoken.get_encoding(self.config.encoding_name)

    def split(self, text: str) -> List[TextSegment]:
        if not text or not text.strip():
            return []
        atoms = self._semantic_atoms(text)
        segments: List[TextSegment] = []
        current = []
        current_tokens = 0

        for atom in atoms:
            if current and current_tokens + atom.token_count > self.config.chunk_size:
                segments.append(self._merge(current))
                current = self._overlap_atoms(current)
                current_tokens = sum(item.token_count for item in current)
                while current and current_tokens + atom.token_count > self.config.chunk_size:
                    current_tokens -= current.pop(0).token_count
            current.append(atom)
            current_tokens += atom.token_count

        if current:
            merged = self._merge(current)
            if not segments or merged.start != segments[-1].start or merged.end != segments[-1].end:
                segments.append(merged)
        return segments

    def _semantic_atoms(self, text: str) -> List[TextSegment]:
        atoms = []
        for match in self._BOUNDARY.finditer(text):
            start, end = match.span()
            value = match.group()
            if not value.strip():
                continue
            tokens = self._encoding.encode(value)
            if len(tokens) <= self.config.chunk_size:
                atoms.append(TextSegment(value, start, end, len(tokens)))
            else:
                atoms.extend(self._split_oversized(value, start, tokens))
        return atoms

    def _split_oversized(self, value: str, offset: int, tokens: List[int]) -> List[TextSegment]:
        result = []
        cursor = 0
        for start in range(0, len(tokens), self.config.chunk_size):
            piece_tokens = tokens[start : start + self.config.chunk_size]
            piece = self._encoding.decode(piece_tokens)
            local_start = value.find(piece, cursor)
            if local_start < 0:
                local_start = cursor
            local_end = min(len(value), local_start + len(piece))
            result.append(TextSegment(value[local_start:local_end], offset + local_start, offset + local_end, len(piece_tokens)))
            cursor = local_end
        return result

    def _overlap_atoms(self, atoms: List[TextSegment]) -> List[TextSegment]:
        overlap = []
        count = 0
        for atom in reversed(atoms):
            if count and count + atom.token_count > self.config.chunk_overlap:
                break
            if atom.token_count > self.config.chunk_overlap:
                break
            overlap.insert(0, atom)
            count += atom.token_count
        return overlap

    def _merge(self, atoms: List[TextSegment]) -> TextSegment:
        start, end = atoms[0].start, atoms[-1].end
        return TextSegment("".join(item.text for item in atoms), start, end, sum(item.token_count for item in atoms))
