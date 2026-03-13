"""
Naive RAG Baseline: Fixed-size chunking + vector search.

Chunks documents into fixed-size segments (no structure awareness),
embeds them, and retrieves top-K by cosine similarity. Uses the same
DuckDB + embedder infrastructure as EngramDB for a fair comparison.
"""

import math
from dataclasses import dataclass

from engramdb.core.engram import Engram, EngramType
from engramdb.embeddings.embedder import Embedder, create_embedder
from engramdb.storage.duckdb import DuckDBStorage
from engramdb.retrieval.hybrid import RetrievalResult


@dataclass
class ChunkInfo:
    """A fixed-size text chunk."""
    text: str
    start_char: int
    end_char: int
    chunk_index: int


class NaiveRAG:
    """
    Standard vector RAG baseline with fixed-size chunking.

    Pipeline:
    1. Chunk document into fixed-size segments (word-boundary aware)
    2. Embed and store in DuckDB (same backend as EngramDB)
    3. Query: embed question, retrieve top-K chunks by cosine similarity
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        embedding_backend: str = "mock",
        embedder: Embedder | None = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.storage = DuckDBStorage(None)
        self.storage.connect()
        self.embedder = embedder or create_embedder(backend=embedding_backend)

    def ingest(self, document: str, doc_id: str) -> int:
        """
        Chunk and ingest a document.

        Returns the number of chunks created.
        """
        chunks = self._chunk_text(document)
        engrams = []
        for chunk in chunks:
            engram = Engram(
                content=chunk.text,
                engram_type=EngramType.SECTION,
                metadata={
                    "document_id": doc_id,
                    "chunk_index": chunk.chunk_index,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                },
            )
            engrams.append(engram)

        # Embed
        texts = [e.content for e in engrams]
        embeddings = self.embedder.embed_batch(texts)
        for engram, embedding in zip(engrams, embeddings):
            engram.embedding = embedding

        self.storage.insert_engrams_batch(engrams)
        return len(engrams)

    def query(self, question: str, top_k: int = 10) -> RetrievalResult:
        """Retrieve top-K chunks by cosine similarity."""
        query_embedding = self.embedder.embed(question)
        results = self.storage.search_similar(query_embedding, top_k=top_k)

        engrams = [engram for engram, _ in results]
        scores = {engram.id: score for engram, score in results}

        return RetrievalResult(
            engrams=engrams,
            anchor_ids=[e.id for e in engrams],
            traversed_ids=[],
            scores=scores,
        )

    def _chunk_text(self, text: str) -> list[ChunkInfo]:
        """Split text into fixed-size chunks with overlap (word-boundary aware)."""
        words = text.split()
        if not words:
            return []

        chunks = []
        word_idx = 0
        chunk_num = 0

        while word_idx < len(words):
            chunk_words = words[word_idx: word_idx + self.chunk_size]
            chunk_text = " ".join(chunk_words)

            # Approximate character positions
            prefix = " ".join(words[:word_idx])
            start_char = len(prefix) + (1 if prefix else 0)
            end_char = start_char + len(chunk_text)

            chunks.append(ChunkInfo(
                text=chunk_text,
                start_char=start_char,
                end_char=end_char,
                chunk_index=chunk_num,
            ))
            chunk_num += 1

            step = max(1, self.chunk_size - self.chunk_overlap)
            word_idx += step

        return chunks

    def close(self):
        self.storage.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
