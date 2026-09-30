import unittest

from tonyhar.rag.retriever import Retriever
from tonyhar.rag.vector_store_base import Chunk, DocumentSummary, VectorStore


class FakeStore(VectorStore):
    def __init__(self):
        self.dense = [Chunk(id="dense", document_id="d", filename="d", text="dense")]
        self.sparse = [Chunk(id="sparse", document_id="s", filename="s", text="sparse")]

    def upsert(self, chunks):
        return len(chunks)

    def delete_document(self, document_id):
        return 0

    def search(self, query, top_k):
        return self.dense[:top_k]

    def keyword_search(self, query, top_k):
        return self.sparse[:top_k]

    def list_documents(self):
        return [DocumentSummary(document_id="d", filename="d")]


class RetrieverTest(unittest.TestCase):
    def test_hybrid_fuses_both_ranked_lists(self):
        results = Retriever(FakeStore()).retrieve_hybrid("query", top_k=2)
        self.assertEqual([chunk.id for chunk in results], ["dense", "sparse"])

    def test_hybrid_deduplicates_same_chunk(self):
        store = FakeStore()
        store.sparse = store.dense
        results = Retriever(store).retrieve_hybrid("query", top_k=2)
        self.assertEqual([chunk.id for chunk in results], ["dense"])


if __name__ == "__main__":
    unittest.main()
