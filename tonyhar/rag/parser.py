"""文档解析端口与文本解析器。"""
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, Optional

@dataclass
class RawDocument:
    """解析结果：纯文本 + 文档级元数据。"""
    text: str
    document_id: str
    filename: str
    metadata: dict
    
class DocumentParser(ABC):
    @abstractmethod
    def parse(self, filename: str, binary: bytes, callback=None) -> "RawDocument": ...


class TxtParser(DocumentParser):
    def parse(
        self,
        filename: str,
        binary: bytes,
        callback: Optional[Callable[..., Any]] = None,
    ) -> RawDocument:
        callback = callback or (lambda **kwargs: None)
        callback(prog=0.1, msg="Start to parse text file.")
        text = self._decode(binary)
        if not text.strip():
            return RawDocument(text="", document_id=filename, filename=filename, metadata={})

        callback(prog=0.8, msg="Finish parsing text file.")
        return RawDocument(
            text=text,
            document_id=str(hash(filename)),
            filename=filename,
            metadata={"file_type": Path(filename).suffix.lower() or ".txt"}
        )

    @staticmethod
    def _decode(binary: bytes) -> str:
        for encoding in ("utf-8", "utf-8-sig", "gb18030"):
            try:
                return binary.decode(encoding)
            except UnicodeDecodeError:
                continue
        return binary.decode("utf-8", errors="replace")
