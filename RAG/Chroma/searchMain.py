"""向量召回 Demo。

运行：python -m RAG.Chroma.searchMain "宝马发动机有什么特点"
"""

import argparse

from .model import RAGContext, RetrievalConfig
from .store.chroma_vector_store import ChromaVectorStore
from .svr.retriever_service import Retriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve chunks from Chroma")
    parser.add_argument("query", help="检索问题")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--score-threshold", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context = RAGContext(
        retrieval=RetrievalConfig(
            top_k=args.top_k,
            score_threshold=args.score_threshold,
        )
    )
    retriever = Retriever(ChromaVectorStore(), context)
    results = retriever.retrieve(args.query)

    for index, result in enumerate(results, start=1):
        print(f"{index}. score={result.score:.4f} chunk={result.chunk_id}")
        print(result.text)
        print()


if __name__ == "__main__":
    main()
