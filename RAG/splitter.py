"""文档切分器。"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib
import re

from core.parser import RawDocument
from .vector_store_base import Chunk


@dataclass(frozen=True)
class SplitterConfig:
    chunk_size: int = 512
    chunk_overlap: int = 64


class TextSplitter(ABC):
    @abstractmethod
    def split(self, doc: RawDocument) -> list[Chunk]:
        raise NotImplementedError


class TokenSplitter(TextSplitter):
    """按中英文 token 近似切分文本，并保留窗口重叠。"""

    def __init__(self, config: SplitterConfig | None = None):
        self.config = config or SplitterConfig()
        if self.config.chunk_size <= 0:
            raise ValueError("chunk_size 必须大于 0")
        if not 0 <= self.config.chunk_overlap < self.config.chunk_size:
            raise ValueError("chunk_overlap 必须在 [0, chunk_size) 范围内")

    def split(self, doc: RawDocument) -> list[Chunk]:
        if not doc or not doc.text.strip():
            return []

        tokens = self._tokenize(doc.text)
        if not tokens:
            return []

        chunks: list[Chunk] = []
        start = 0
        index = 0

        while start < len(tokens):
            end = min(start + self.config.chunk_size, len(tokens))
            part = self._join_tokens(tokens[start:end]).strip()
            if part:
                raw_id = f"{doc.document_id}:{index}:{part}"
                chunk_id = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()
                chunks.append(Chunk(
                    id=chunk_id,
                    document_id=doc.document_id,
                    filename=doc.filename,
                    text=part,
                    chunk_index=index,
                    metadata=dict(doc.metadata),
                    raw={"token_start": start, "token_end": end},
                ))
                index += 1

            if end >= len(tokens):
                break

            # 下一块从当前块尾部向前回退 overlap 个 token。
            start = end - self.config.chunk_overlap

        return chunks

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """中文按字、英文按词切分；连续空白只作为分隔符。"""
        pattern = r"[\u4e00-\u9fff]|[A-Za-z]+(?:['-][A-Za-z]+)*|\d+(?:\.\d+)?|[^\w\s]"
        return re.findall(pattern, text)

    @staticmethod
    def _join_tokens(tokens: list[str]) -> str:
        """拼接 token，避免英文词之间和中文字符之间出现异常空格。"""
        result = ""
        for token in tokens:
            if not result:
                result = token
            elif re.match(r"^[A-Za-z0-9]", token) and re.search(r"[A-Za-z0-9]$", result):
                result += " " + token
            else:
                result += token
        return result
