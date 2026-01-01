"""
Naive RAG Baseline: Standard chunking + vector search.

This is the primary baseline we compare against.
Implementation uses ChromaDB for fair comparison.
"""

# TODO: Implement naive RAG baseline
# - Document chunking (512 tokens, 50 token overlap)
# - ChromaDB for vector storage
# - Top-K retrieval
# - LLM answer generation


class NaiveRAG:
    """
    Standard vector RAG baseline.

    Pipeline:
    1. Chunk document into fixed-size segments
    2. Embed and store in vector DB
    3. Query: embed question, retrieve top-K chunks
    4. Feed chunks to LLM for answer
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # TODO: Initialize ChromaDB

    def ingest(self, document: str, doc_id: str) -> None:
        """Chunk and ingest a document."""
        raise NotImplementedError

    def query(self, question: str, top_k: int = 5) -> dict:
        """
        Query the RAG system.

        Returns:
            Dict with 'answer', 'retrieved_chunks', 'scores'
        """
        raise NotImplementedError
