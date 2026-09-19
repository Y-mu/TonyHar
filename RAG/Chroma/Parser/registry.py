from pathlib import Path
from typing import Any

from .txt import TxtParser


def get_parser(filename: str, parser_id: str = "auto") -> Any:
    """根据文件后缀和解析策略返回 Parser。"""
    suffix = Path(filename).suffix.lower()
    parser_id = (parser_id or "auto").lower()

    if parser_id in {"auto", "txt"} and suffix in {".txt", ".md", ".markdown", ".mdx"}:
        return TxtParser()
    if parser_id == "one":
        from .one_adapter import OneParser
        return OneParser()
    raise ValueError(f"unsupported parser: parser_id={parser_id}, suffix={suffix}")
