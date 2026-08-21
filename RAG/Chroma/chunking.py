from typing import List

import tiktoken


def chunk_text(
    text: str,
    chunk_size: int = 256,
    chunk_overlap: int = 30,
    encoding_name: str = "cl100k_base",
) -> List[str]:
    """Split text into overlapping token-based chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be between zero and chunk_size")

    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    chunks = []
    step = chunk_size - chunk_overlap

    for start in range(0, len(tokens), step):
        chunks.append(encoding.decode(tokens[start : start + chunk_size]))
        if start + chunk_size >= len(tokens):
            break

    return chunks
