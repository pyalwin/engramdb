"""
Hybrid Retriever: Combines vector search and graph traversal.

This is the core innovation of EngramDB - using vector search to find
anchor points, then graph traversal to gather connected context that
enables multi-hop reasoning.
"""

from dataclasses import dataclass, field
from typing import Optional

from ..core.engram import Engram, EngramType
from ..storage.duckdb import DuckDBStorage
from ..embeddings.embedder import Embedder


@dataclass
class RetrievalResult:
    """Result from hybrid retrieval."""
    engrams: list[Engram]
    anchor_ids: list[str]  # IDs of vector search anchors
    traversed_ids: list[str]  # IDs discovered via graph traversal
    scores: dict[str, float]  # engram_id -> relevance score


@dataclass
class RetrievalTrace:
    """Detailed trace of retrieval for debugging/evaluation."""
    query: str
    query_embedding: list[float]
    anchors: list[tuple[Engram, float]]  # (engram, similarity)
    traversal_paths: dict[str, list[str]]  # anchor_id -> [traversed_ids]
    final_context: list[Engram]
    reasoning_chain: list[str] = field(default_factory=list)


class HybridRetriever:
    """
    Hybrid vector + graph retrieval.

    The retrieval pipeline:
    1. Anchor: Vector search finds top-K semantically similar engrams
    2. Traverse: Graph traversal expands N hops from anchors
    3. Aggregate: Combine and deduplicate retrieved engrams
    4. Rank: Order by relevance for context assembly
    """

    def __init__(
        self,
        storage: DuckDBStorage,
        embedder: Embedder
    ):
        """
        Initialize hybrid retriever.

        Args:
            storage: DuckDBStorage instance (must be connected)
            embedder: Embedder instance for query embedding
        """
        self.storage = storage
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        top_k_anchors: int = 3,
        max_hops: int = 2,
        max_context_items: int = 10,
        engram_types: Optional[list[EngramType]] = None
    ) -> RetrievalResult:
        """
        Retrieve relevant engrams using hybrid approach.

        Args:
            query: Natural language query
            top_k_anchors: Number of vector search anchors
            max_hops: Graph traversal depth
            max_context_items: Maximum engrams to return
            engram_types: Filter anchors by type (optional)

        Returns:
            RetrievalResult with engrams and metadata
        """
        # Step 1: Embed query
        query_embedding = self.embedder.embed(query)

        # Step 2: Vector search for anchors
        anchor_results = self.storage.search_similar(
            query_embedding,
            top_k=top_k_anchors,
            engram_types=engram_types
        )

        if not anchor_results:
            return RetrievalResult(
                engrams=[],
                anchor_ids=[],
                traversed_ids=[],
                scores={}
            )

        anchor_ids = [engram.id for engram, _ in anchor_results]
        scores = {engram.id: score for engram, score in anchor_results}

        # Step 3: Graph traversal from anchors
        all_connected_ids = set(anchor_ids)
        traversed_ids = []

        for anchor_id in anchor_ids:
            connected = self.storage.get_connected(
                anchor_id,
                hops=max_hops,
                direction="both"
            )
            for cid in connected:
                if cid not in all_connected_ids:
                    traversed_ids.append(cid)
                    all_connected_ids.add(cid)

        # Step 4: Fetch all engrams
        all_ids = list(all_connected_ids)
        engrams = self.storage.get_engrams(all_ids)

        # Step 5: Score traversed nodes (lower score than anchors)
        for engram in engrams:
            if engram.id not in scores:
                # Assign a base score for traversed nodes
                # Could be improved with re-ranking
                scores[engram.id] = 0.5

        # Step 6: Sort by score and limit
        engrams.sort(key=lambda e: scores.get(e.id, 0), reverse=True)
        engrams = engrams[:max_context_items]

        return RetrievalResult(
            engrams=engrams,
            anchor_ids=anchor_ids,
            traversed_ids=traversed_ids,
            scores=scores
        )

    def retrieve_with_trace(
        self,
        query: str,
        top_k_anchors: int = 3,
        max_hops: int = 2,
        max_context_items: int = 10
    ) -> RetrievalTrace:
        """
        Retrieve with full trace for debugging/evaluation.

        Returns detailed information about each retrieval step.
        """
        # Step 1: Embed query
        query_embedding = self.embedder.embed(query)

        # Step 2: Vector search for anchors
        anchor_results = self.storage.search_similar(
            query_embedding,
            top_k=top_k_anchors
        )

        if not anchor_results:
            return RetrievalTrace(
                query=query,
                query_embedding=query_embedding,
                anchors=[],
                traversal_paths={},
                final_context=[]
            )

        # Step 3: Graph traversal with path tracking
        traversal_paths = {}
        all_connected_ids = set()

        for anchor_engram, _ in anchor_results:
            connected = self.storage.get_connected(
                anchor_engram.id,
                hops=max_hops,
                direction="both"
            )
            traversal_paths[anchor_engram.id] = [
                cid for cid in connected if cid != anchor_engram.id
            ]
            all_connected_ids.update(connected)

        # Step 4: Fetch all engrams
        all_ids = list(all_connected_ids)
        engrams = self.storage.get_engrams(all_ids)

        # Step 5: Build scores
        scores = {engram.id: score for engram, score in anchor_results}
        for engram in engrams:
            if engram.id not in scores:
                scores[engram.id] = 0.5

        # Step 6: Sort and limit
        engrams.sort(key=lambda e: scores.get(e.id, 0), reverse=True)
        final_context = engrams[:max_context_items]

        return RetrievalTrace(
            query=query,
            query_embedding=query_embedding,
            anchors=anchor_results,
            traversal_paths=traversal_paths,
            final_context=final_context
        )

    def retrieve_vector_only(
        self,
        query: str,
        top_k: int = 10,
        engram_types: Optional[list[EngramType]] = None
    ) -> RetrievalResult:
        """
        Vector-only retrieval (baseline for comparison).

        This is equivalent to standard RAG without graph traversal.
        """
        query_embedding = self.embedder.embed(query)

        results = self.storage.search_similar(
            query_embedding,
            top_k=top_k,
            engram_types=engram_types
        )

        engrams = [engram for engram, _ in results]
        scores = {engram.id: score for engram, score in results}

        return RetrievalResult(
            engrams=engrams,
            anchor_ids=[e.id for e in engrams],
            traversed_ids=[],
            scores=scores
        )

    def format_context(
        self,
        result: RetrievalResult,
        include_metadata: bool = False
    ) -> str:
        """
        Format retrieved engrams as context string for LLM.

        Args:
            result: RetrievalResult from retrieve()
            include_metadata: Include section numbers, types, etc.

        Returns:
            Formatted context string
        """
        if not result.engrams:
            return ""

        parts = []
        for i, engram in enumerate(result.engrams, 1):
            if include_metadata:
                meta_parts = []
                if engram.metadata.get("section_number"):
                    meta_parts.append(f"Section {engram.metadata['section_number']}")
                if engram.metadata.get("title"):
                    meta_parts.append(engram.metadata["title"])
                meta_str = " - ".join(meta_parts)
                if meta_str:
                    parts.append(f"[{i}] {meta_str}:\n{engram.content}")
                else:
                    parts.append(f"[{i}] ({engram.engram_type.value}):\n{engram.content}")
            else:
                parts.append(f"[{i}] {engram.content}")

        return "\n\n".join(parts)
