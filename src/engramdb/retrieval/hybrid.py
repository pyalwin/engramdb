"""
Hybrid Retriever: Combines vector search and graph traversal.

This is the core innovation of EngramDB - using vector search to find
anchor points, then graph traversal to gather connected context that
enables multi-hop reasoning.
"""

from dataclasses import dataclass, field
import math
import re
from typing import Optional

from ..core.engram import Engram, EngramType
from ..core.synapse import SynapseType
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
        self.edge_type_weights = {
            SynapseType.REFERENCES: 1.0,
            SynapseType.DEFINES: 0.9,
            SynapseType.RELATED_TO: 0.75,
            SynapseType.SUPERSEDES: 0.75,
            SynapseType.PARENT_OF: 0.55,
            SynapseType.CHILD_OF: 0.55,
        }
        self.default_edge_weight = 0.6
        self.hop_decay = 0.75
        self.semantic_weight = 0.5

    def retrieve(
        self,
        query: str,
        top_k_anchors: int = 3,
        max_hops: int = 2,
        max_context_items: int = 10,
        min_traversed_items: int = 0,
        engram_types: Optional[list[EngramType]] = None
    ) -> RetrievalResult:
        """
        Retrieve relevant engrams using hybrid approach.

        Args:
            query: Natural language query
            top_k_anchors: Number of vector search anchors
            max_hops: Graph traversal depth
            max_context_items: Maximum engrams to return
            min_traversed_items: Minimum non-anchor traversed items to keep in final context
            engram_types: Filter anchors by type (optional)

        Returns:
            RetrievalResult with engrams and metadata
        """
        data = self._retrieve_core(
            query=query,
            top_k_anchors=top_k_anchors,
            max_hops=max_hops,
            max_context_items=max_context_items,
            min_traversed_items=min_traversed_items,
            engram_types=engram_types,
        )
        return data["result"]

    def retrieve_with_trace(
        self,
        query: str,
        top_k_anchors: int = 3,
        max_hops: int = 2,
        max_context_items: int = 10,
        min_traversed_items: int = 0
    ) -> RetrievalTrace:
        """
        Retrieve with full trace for debugging/evaluation.

        Returns detailed information about each retrieval step.
        """
        data = self._retrieve_core(
            query=query,
            top_k_anchors=top_k_anchors,
            max_hops=max_hops,
            max_context_items=max_context_items,
            min_traversed_items=min_traversed_items,
            engram_types=None,
        )

        return RetrievalTrace(
            query=query,
            query_embedding=data["query_embedding"],
            anchors=data["anchors"],
            traversal_paths=data["traversal_paths"],
            final_context=data["result"].engrams
        )

    def _retrieve_core(
        self,
        query: str,
        top_k_anchors: int,
        max_hops: int,
        max_context_items: int,
        min_traversed_items: int,
        engram_types: Optional[list[EngramType]]
    ) -> dict:
        """Run hybrid retrieval and return full intermediate data."""
        query_embedding = self.embedder.embed(query)
        anchor_results = self.storage.search_similar(
            query_embedding,
            top_k=top_k_anchors,
            engram_types=engram_types
        )

        if not anchor_results:
            return {
                "query_embedding": query_embedding,
                "anchors": [],
                "traversal_paths": {},
                "result": RetrievalResult(
                    engrams=[],
                    anchor_ids=[],
                    traversed_ids=[],
                    scores={}
                ),
            }

        # Inject section-number anchors for queries mentioning "Section N"
        section_matches = re.findall(r'[Ss]ection\s+(\d+(?:\.\d+)*)', query)
        if section_matches:
            existing_anchor_ids = {e.id for e, _ in anchor_results}
            for sec_num in section_matches:
                sec_engrams = self.storage.find_by_section_number(sec_num)
                for engram in sec_engrams:
                    if engram.id not in existing_anchor_ids and engram.embedding:
                        sim = self._cosine_similarity(query_embedding, engram.embedding)
                        anchor_results.append((engram, sim))
                        existing_anchor_ids.add(engram.id)

        anchor_ids = [engram.id for engram, _ in anchor_results]
        anchor_id_set = set(anchor_ids)

        traversal_details = {}
        traversed_info: dict[str, tuple[int, float]] = {}
        for anchor_id in anchor_ids:
            discovered = self._traverse_from_anchor(
                anchor_id=anchor_id,
                max_hops=max_hops,
                direction="both",
            )
            traversal_details[anchor_id] = discovered
            for node_id, (hop, edge_weight) in discovered.items():
                prior = traversed_info.get(node_id)
                if prior is None or hop < prior[0] or (hop == prior[0] and edge_weight > prior[1]):
                    traversed_info[node_id] = (hop, edge_weight)

        all_ids = set(anchor_ids) | set(traversed_info.keys())

        # Backfill with additional semantic results so hybrid does not collapse to tiny sets
        # when graph expansion is sparse.
        if len(all_ids) < max_context_items:
            backfill_top_k = max(max_context_items * 3, top_k_anchors * 2)
            backfill_results = self.storage.search_similar(
                query_embedding,
                top_k=backfill_top_k,
                engram_types=engram_types
            )
            for engram, _score in backfill_results:
                if len(all_ids) >= max_context_items:
                    break
                all_ids.add(engram.id)

        engrams = self.storage.get_engrams(list(all_ids))

        # Uniform scoring: every node gets the same blended formula.
        # Anchors: hop=0, structural=1.0 (best possible graph signal).
        # Traversed: structural decays by hop distance and edge type.
        # Backfill / unknown: structural=0 (pure semantic).
        scores: dict[str, float] = {}
        for engram in engrams:
            semantic_score = max(self._cosine_similarity(query_embedding, engram.embedding), 0.0)

            if engram.id in anchor_id_set:
                traversal_score = 1.0
            elif engram.id in traversed_info:
                hop, edge_weight = traversed_info[engram.id]
                traversal_score = edge_weight * (self.hop_decay ** max(hop - 1, 0))
            else:
                traversal_score = 0.0

            scores[engram.id] = (
                self.semantic_weight * semantic_score
                + (1 - self.semantic_weight) * traversal_score
            )

        engrams.sort(key=lambda e: scores.get(e.id, 0.0), reverse=True)
        engrams = self._apply_traversal_reservation(
            ranked_engrams=engrams,
            anchor_ids=anchor_id_set,
            traversed_ids=set(traversed_info.keys()),
            max_context_items=max_context_items,
            min_traversed_items=min_traversed_items,
            traversed_info=traversed_info,
            scores=scores,
            query_embedding=query_embedding,
        )

        traversed_ids = [
            engram_id
            for engram_id, _ in sorted(
                traversed_info.items(),
                key=lambda item: (item[1][0], -item[1][1], -scores.get(item[0], 0.0))
            )
        ]
        traversal_paths = {
            anchor_id: [
                node_id
                for node_id, _ in sorted(
                    discovered.items(),
                    key=lambda item: (item[1][0], -item[1][1], -scores.get(item[0], 0.0))
                )
            ]
            for anchor_id, discovered in traversal_details.items()
        }

        return {
            "query_embedding": query_embedding,
            "anchors": anchor_results,
            "traversal_paths": traversal_paths,
            "result": RetrievalResult(
                engrams=engrams,
                anchor_ids=anchor_ids,
                traversed_ids=traversed_ids,
                scores=scores
            ),
        }

    def _apply_traversal_reservation(
        self,
        ranked_engrams: list[Engram],
        anchor_ids: set[str],
        traversed_ids: set[str],
        max_context_items: int,
        min_traversed_items: int,
        traversed_info: Optional[dict[str, tuple[int, float]]] = None,
        scores: Optional[dict[str, float]] = None,
        query_embedding: Optional[list[float]] = None,
    ) -> list[Engram]:
        """
        Ensure traversed context can contribute to final ranking.

        Without this, large anchor budgets can saturate context and make hybrid
        outputs identical to vector-only even when traversal discovers useful nodes.

        When traversed_info is provided, reserved slots prioritize nodes reached
        via high-value edge types (REFERENCES/DEFINES with weight ≥0.9) over
        lower-value edges like PARENT_OF. Within each tier, candidates are ranked
        by semantic similarity rather than blended score — this prevents structural
        score differences within the same tier from crowding out semantically
        relevant nodes.
        """
        if max_context_items <= 0:
            return []

        if min_traversed_items <= 0:
            return ranked_engrams[:max_context_items]

        traversed_candidates = [
            e
            for e in ranked_engrams
            if e.id in traversed_ids and e.id not in anchor_ids
        ]
        reserved = min(min_traversed_items, len(traversed_candidates), max_context_items)
        if reserved <= 0:
            return ranked_engrams[:max_context_items]

        # Sort reserved candidates: high-value edge types first, then by
        # semantic similarity within each tier. Using semantic similarity
        # (not blended score) as tiebreaker prevents structural score
        # differences within a tier from biasing slot allocation.
        if traversed_info is not None and query_embedding is not None:
            def _reservation_key(e: Engram) -> tuple[int, float]:
                _hop, edge_weight = traversed_info.get(e.id, (99, 0.0))
                # Tier 0 = high-value edges (REFERENCES/DEFINES, weight ≥ 0.9)
                # Tier 1 = everything else
                tier = 0 if edge_weight >= 0.9 else 1
                sem = self._cosine_similarity(query_embedding, e.embedding)
                return (tier, -sem)

            traversed_candidates.sort(key=_reservation_key)

        selected: list[Engram] = traversed_candidates[:reserved]
        selected_ids = {e.id for e in selected}

        for engram in ranked_engrams:
            if len(selected) >= max_context_items:
                break
            if engram.id in selected_ids:
                continue
            selected.append(engram)
            selected_ids.add(engram.id)

        return selected

    def _traverse_from_anchor(
        self,
        anchor_id: str,
        max_hops: int,
        direction: str = "both"
    ) -> dict[str, tuple[int, float]]:
        """
        Traverse from anchor and keep best hop/edge-weight signal for each discovered node.

        Returns:
            node_id -> (hop_distance, edge_type_weight)
        """
        visited_hops = {anchor_id: 0}
        best_edge_weight: dict[str, float] = {}
        frontier = {anchor_id}
        discovered: dict[str, tuple[int, float]] = {}

        for hop in range(1, max_hops + 1):
            next_frontier = set()

            for node_id in frontier:
                neighbors = []
                if direction in ("outgoing", "both"):
                    neighbors.extend(
                        (syn.target_id, syn.synapse_type)
                        for syn in self.storage.get_synapses_from(node_id)
                    )
                if direction in ("incoming", "both"):
                    neighbors.extend(
                        (syn.source_id, syn.synapse_type)
                        for syn in self.storage.get_synapses_to(node_id)
                    )

                for neighbor_id, synapse_type in neighbors:
                    edge_weight = self.edge_type_weights.get(
                        synapse_type,
                        self.default_edge_weight
                    )
                    prior_hop = visited_hops.get(neighbor_id)
                    prior_weight = best_edge_weight.get(neighbor_id, 0.0)
                    is_better = (
                        prior_hop is None
                        or hop < prior_hop
                        or (hop == prior_hop and edge_weight > prior_weight)
                    )
                    if not is_better:
                        continue

                    visited_hops[neighbor_id] = hop
                    best_edge_weight[neighbor_id] = edge_weight
                    if neighbor_id != anchor_id:
                        discovered[neighbor_id] = (hop, edge_weight)
                    next_frontier.add(neighbor_id)

            frontier = next_frontier
            if not frontier:
                break

        return discovered

    def _cosine_similarity(
        self,
        query_embedding: list[float],
        candidate_embedding: Optional[list[float]]
    ) -> float:
        """Cosine similarity with zero-norm safety."""
        if not candidate_embedding:
            return 0.0

        if len(query_embedding) != len(candidate_embedding):
            return 0.0

        dot = 0.0
        query_norm_sq = 0.0
        candidate_norm_sq = 0.0
        for q, c in zip(query_embedding, candidate_embedding):
            dot += q * c
            query_norm_sq += q * q
            candidate_norm_sq += c * c

        if query_norm_sq <= 0 or candidate_norm_sq <= 0:
            return 0.0

        return dot / math.sqrt(query_norm_sq * candidate_norm_sq)

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
