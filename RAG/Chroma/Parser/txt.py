from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


class TxtParser:
    """轻量 TXT/Markdown 解析器，不依赖 RAGFlow。"""

    def parse(
        self,
        filename: str,
        binary: bytes,
        callback: Optional[Callable[..., Any]] = None,
        **_: Any,
    ) -> List[Dict[str, Any]]:
        callback = callback or (lambda **kwargs: None)
        callback(prog=0.1, msg="Start to parse text file.")
        text = self._decode(binary)
        if not text.strip():
            return []

        callback(prog=0.8, msg="Finish parsing text file.")
        return [
            {
                "text": text,
                "content_with_weight": text,
                "docnm_kwd": filename,
                "metadata": {"file_type": Path(filename).suffix.lower() or ".txt"},
            }
        ]

    @staticmethod
    def _decode(binary: bytes) -> str:
        for encoding in ("utf-8", "utf-8-sig", "gb18030"):
            try:
                return binary.decode(encoding)
            except UnicodeDecodeError:
                continue
        return binary.decode("utf-8", errors="replace")
