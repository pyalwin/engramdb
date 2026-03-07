"""
Benchmark Evaluation for EngramDB

Compares hybrid retrieval (vector + graph) vs vector-only retrieval
on multi-hop QA tasks from CUAD contracts.

Metrics:
- Retrieval Recall: % of required sections retrieved
- Hop Coverage: % of reasoning chain covered
- Context Relevance: How much of retrieved context is useful
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from engramdb import EngramDB
from engramdb.embeddings.embedder import MockEmbedder, create_embedder
from engramdb.core.synapse import SynapseType


@dataclass
class RetrievalMetrics:
    """Metrics for a single retrieval."""
    question_id: str
    question_type: str
    hop_count: int
    required_sections: list[str]

    # Hybrid retrieval results
    hybrid_retrieved: list[str]
    hybrid_recall: float
    hybrid_hop_coverage: float
    hybrid_time_ms: float
    anchors_count: int
    traversed_discovered: int
    traversed_in_final: int
    anchor_only_recall: float
    hybrid_gain_over_anchor_only: float

    # Vector-only results
    vector_retrieved: list[str]
    vector_recall: float
    vector_hop_coverage: float
    vector_time_ms: float

    # Graph-only results
    graph_retrieved: list[str] = None
    graph_recall: float = 0.0
    graph_hop_coverage: float = 0.0
    graph_time_ms: float = 0.0

    # Comparison
    hybrid_advantage: float = 0.0  # hybrid_recall - vector_recall
    hybrid_vs_graph: float = 0.0  # hybrid_recall - graph_recall


@dataclass
class BenchmarkResults:
    """Aggregate benchmark results."""
    total_questions: int
    total_contracts: int

    # Aggregate metrics
    avg_hybrid_recall: float
    avg_vector_recall: float
    avg_graph_recall: float
    avg_hybrid_hop_coverage: float
    avg_vector_hop_coverage: float
    avg_graph_hop_coverage: float
    avg_anchors_count: float
    avg_traversed_discovered: float
    avg_traversed_in_final: float
    avg_anchor_only_recall: float
    avg_hybrid_gain_over_anchor_only: float

    # By question type
    metrics_by_type: dict

    # By hop count
    metrics_by_hops: dict

    # Timing
    avg_hybrid_time_ms: float
    avg_vector_time_ms: float
    avg_graph_time_ms: float

    # Individual results
    per_question_metrics: list[RetrievalMetrics]

    def to_dict(self) -> dict:
        return {
            "summary": {
                "total_questions": self.total_questions,
                "total_contracts": self.total_contracts,
                "avg_hybrid_recall": self.avg_hybrid_recall,
                "avg_vector_recall": self.avg_vector_recall,
                "avg_graph_recall": self.avg_graph_recall,
                "avg_hybrid_hop_coverage": self.avg_hybrid_hop_coverage,
                "avg_vector_hop_coverage": self.avg_vector_hop_coverage,
                "avg_graph_hop_coverage": self.avg_graph_hop_coverage,
                "avg_hybrid_time_ms": self.avg_hybrid_time_ms,
                "avg_vector_time_ms": self.avg_vector_time_ms,
                "avg_graph_time_ms": self.avg_graph_time_ms,
                "avg_anchors_count": self.avg_anchors_count,
                "avg_traversed_discovered": self.avg_traversed_discovered,
                "avg_traversed_in_final": self.avg_traversed_in_final,
                "avg_anchor_only_recall": self.avg_anchor_only_recall,
                "avg_hybrid_gain_over_anchor_only": self.avg_hybrid_gain_over_anchor_only,
                "hybrid_vs_vector": self.avg_hybrid_recall - self.avg_vector_recall,
                "hybrid_vs_graph": self.avg_hybrid_recall - self.avg_graph_recall,
            },
            "by_question_type": self.metrics_by_type,
            "by_hop_count": self.metrics_by_hops,
            "per_question": [asdict(m) for m in self.per_question_metrics],
        }

    def save(self, filepath: Path):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Saved benchmark results to {filepath}")


class Benchmark:
    """
    Benchmark runner for EngramDB vs vector-only retrieval.
    """

    def __init__(
        self,
        embedding_backend: str = "mock",
        top_k_anchors: int = 3,
        max_hops: int = 2,
        max_context_items: int = 10,
        min_traversed_items: int = 0,
        semantic_weight: Optional[float] = None,
        hop_decay: Optional[float] = None,
        default_edge_weight: Optional[float] = None,
        edge_type_weights: Optional[dict[str, float]] = None,
    ):
        """
        Initialize benchmark.

        Args:
            embedding_backend: "mock", "openai", or "local"
            top_k_anchors: Number of vector search anchors for hybrid
            max_hops: Graph traversal depth
            max_context_items: Max items to retrieve
            min_traversed_items: Minimum non-anchor traversed items retained in final context
            semantic_weight: Override hybrid semantic score weight (optional)
            hop_decay: Override graph hop decay factor (optional)
            default_edge_weight: Override fallback edge weight (optional)
            edge_type_weights: Override per-edge-type weights, keyed by synapse value/name
        """
        self.embedding_backend = embedding_backend
        self.top_k_anchors = top_k_anchors
        self.max_hops = max_hops
        self.max_context_items = max_context_items
        self.min_traversed_items = min_traversed_items
        self.semantic_weight = semantic_weight
        self.hop_decay = hop_decay
        self.default_edge_weight = default_edge_weight
        self.edge_type_weights = edge_type_weights or {}

    def load_dataset(self, filepath: Path) -> dict:
        """Load multi-hop QA dataset."""
        with open(filepath) as f:
            return json.load(f)

    def run(self, dataset_path: Path) -> BenchmarkResults:
        """
        Run the full benchmark.

        Args:
            dataset_path: Path to multi-hop QA dataset JSON

        Returns:
            BenchmarkResults with all metrics
        """
        print("=" * 60)
        print("EngramDB Benchmark")
        print("=" * 60)
        print(f"Embedding backend: {self.embedding_backend}")
        print(f"Top-K anchors: {self.top_k_anchors}")
        print(f"Max hops: {self.max_hops}")
        print(f"Min traversed items: {self.min_traversed_items}")
        if self.semantic_weight is not None:
            print(f"Semantic weight: {self.semantic_weight}")
        if self.hop_decay is not None:
            print(f"Hop decay: {self.hop_decay}")
        print()

        # Load dataset
        dataset = self.load_dataset(dataset_path)
        contracts = dataset["contracts"]
        questions = dataset["questions"]

        print(f"Loaded {len(contracts)} contracts, {len(questions)} questions")

        # Group questions by contract
        questions_by_contract = {}
        for q in questions:
            cid = q["contract_id"]
            if cid not in questions_by_contract:
                questions_by_contract[cid] = []
            questions_by_contract[cid].append(q)

        all_metrics = []

        # Process each contract
        for i, contract in enumerate(contracts):
            contract_id = contract["id"]
            contract_questions = questions_by_contract.get(contract_id, [])

            if not contract_questions:
                continue

            print(f"\n[{i+1}/{len(contracts)}] {contract_id[:50]}...")
            print(f"  Questions: {len(contract_questions)}")

            # Create EngramDB instance for this contract
            with EngramDB(embedding_backend=self.embedding_backend) as db:
                self._configure_retriever(db)
                # Ingest contract
                print(f"  Ingesting...")
                db.ingest(contract["text"], doc_id=contract_id)

                # Run queries for each question
                for q in contract_questions:
                    metrics = self._evaluate_question(db, q)
                    all_metrics.append(metrics)

                    # Print progress
                    print(f"    Q[{q['question_type']}]: hybrid={metrics.hybrid_recall:.0%} vector={metrics.vector_recall:.0%} graph={metrics.graph_recall:.0%}")

        # Aggregate results
        results = self._aggregate_results(all_metrics, len(contracts))

        return results

    def _evaluate_question(self, db: EngramDB, question: dict) -> RetrievalMetrics:
        """Evaluate a single question with both retrieval methods."""
        query = question["question"]
        required_sections = question["reasoning_chain"]

        # Hybrid retrieval
        start = time.time()
        hybrid_result = db.query(
            query,
            top_k_anchors=self.top_k_anchors,
            max_hops=self.max_hops,
            max_context_items=self.max_context_items,
            min_traversed_items=self.min_traversed_items
        )
        hybrid_time = (time.time() - start) * 1000

        # Extract section IDs from retrieved engrams
        hybrid_sections = self._extract_sections(hybrid_result.engrams)

        # Vector-only retrieval
        start = time.time()
        vector_result = db.query_vector_only(
            query,
            top_k=self.max_context_items
        )
        vector_time = (time.time() - start) * 1000

        vector_sections = self._extract_sections(vector_result.engrams)

        # Graph-only retrieval (ablation)
        start = time.time()
        graph_result = db.query_graph_only(
            query,
            top_k_anchors=self.top_k_anchors,
            max_hops=self.max_hops,
            max_context_items=self.max_context_items
        )
        graph_time = (time.time() - start) * 1000

        graph_sections = self._extract_sections(graph_result.engrams)

        # Calculate metrics
        hybrid_recall = self._calculate_recall(required_sections, hybrid_sections)
        vector_recall = self._calculate_recall(required_sections, vector_sections)
        graph_recall = self._calculate_recall(required_sections, graph_sections)

        hybrid_hop_coverage = self._calculate_hop_coverage(required_sections, hybrid_sections)
        vector_hop_coverage = self._calculate_hop_coverage(required_sections, vector_sections)
        graph_hop_coverage = self._calculate_hop_coverage(required_sections, graph_sections)

        anchor_engrams = db.storage.get_engrams(hybrid_result.anchor_ids)
        anchor_sections = self._extract_sections(anchor_engrams)
        anchor_only_recall = self._calculate_recall(required_sections, anchor_sections)

        anchor_ids = set(hybrid_result.anchor_ids)
        traversed_ids = set(hybrid_result.traversed_ids)
        traversed_in_final = sum(
            1
            for e in hybrid_result.engrams
            if e.id in traversed_ids and e.id not in anchor_ids
        )

        return RetrievalMetrics(
            question_id=question["id"],
            question_type=question["question_type"],
            hop_count=question["hop_count"],
            required_sections=required_sections,
            hybrid_retrieved=hybrid_sections,
            hybrid_recall=hybrid_recall,
            hybrid_hop_coverage=hybrid_hop_coverage,
            hybrid_time_ms=hybrid_time,
            anchors_count=len(anchor_ids),
            traversed_discovered=len(traversed_ids),
            traversed_in_final=traversed_in_final,
            anchor_only_recall=anchor_only_recall,
            hybrid_gain_over_anchor_only=hybrid_recall - anchor_only_recall,
            vector_retrieved=vector_sections,
            vector_recall=vector_recall,
            vector_hop_coverage=vector_hop_coverage,
            vector_time_ms=vector_time,
            graph_retrieved=graph_sections,
            graph_recall=graph_recall,
            graph_hop_coverage=graph_hop_coverage,
            graph_time_ms=graph_time,
            hybrid_advantage=hybrid_recall - vector_recall,
            hybrid_vs_graph=hybrid_recall - graph_recall,
        )

    def _configure_retriever(self, db: EngramDB) -> None:
        """Apply optional retrieval scoring overrides for tuning experiments."""
        retriever = db.retriever

        if self.semantic_weight is not None:
            retriever.semantic_weight = self.semantic_weight
        if self.hop_decay is not None:
            retriever.hop_decay = self.hop_decay
        if self.default_edge_weight is not None:
            retriever.default_edge_weight = self.default_edge_weight

        if not self.edge_type_weights:
            return

        for key, value in self.edge_type_weights.items():
            synapse_type = None
            for member in SynapseType:
                if key == member.value or key == member.name:
                    synapse_type = member
                    break
            if synapse_type is None:
                raise ValueError(f"Unknown synapse type override: {key}")
            retriever.edge_type_weights[synapse_type] = value

    def _extract_sections(self, engrams) -> list[str]:
        """Extract section identifiers from engrams."""
        sections = []
        for e in engrams:
            # Try to get section number from metadata
            if e.metadata.get("section_number"):
                sections.append(e.metadata["section_number"])
            elif e.metadata.get("title"):
                sections.append(e.metadata["title"])
            # Also check content for section references
            content = e.content[:200].lower()
            # Extract any section numbers mentioned
            import re
            for match in re.finditer(r'section\s+(\d+(?:\.\d+)*)', content):
                sections.append(match.group(1))
        return list(set(sections))

    def _calculate_recall(self, required: list[str], retrieved: list[str]) -> float:
        """Calculate retrieval recall."""
        if not required:
            return 1.0

        # Normalize for comparison
        required_norm = set(str(s).lower() for s in required)
        retrieved_norm = set(str(s).lower() for s in retrieved)

        # Count matches
        matches = len(required_norm & retrieved_norm)

        return matches / len(required_norm)

    def _calculate_hop_coverage(self, chain: list[str], retrieved: list[str]) -> float:
        """Calculate what fraction of the reasoning chain is covered."""
        if not chain:
            return 1.0

        chain_norm = [str(s).lower() for s in chain]
        retrieved_norm = set(str(s).lower() for s in retrieved)

        # Check consecutive coverage
        covered = 0
        for section in chain_norm:
            if section in retrieved_norm:
                covered += 1

        return covered / len(chain_norm)

    def _aggregate_results(
        self,
        metrics: list[RetrievalMetrics],
        num_contracts: int
    ) -> BenchmarkResults:
        """Aggregate individual metrics into summary."""
        if not metrics:
            return BenchmarkResults(
                total_questions=0,
                total_contracts=num_contracts,
                avg_hybrid_recall=0,
                avg_vector_recall=0,
                avg_graph_recall=0,
                avg_hybrid_hop_coverage=0,
                avg_vector_hop_coverage=0,
                avg_graph_hop_coverage=0,
                avg_anchors_count=0,
                avg_traversed_discovered=0,
                avg_traversed_in_final=0,
                avg_anchor_only_recall=0,
                avg_hybrid_gain_over_anchor_only=0,
                metrics_by_type={},
                metrics_by_hops={},
                avg_hybrid_time_ms=0,
                avg_vector_time_ms=0,
                avg_graph_time_ms=0,
                per_question_metrics=[],
            )

        n = len(metrics)

        # Overall averages
        avg_hybrid_recall = sum(m.hybrid_recall for m in metrics) / n
        avg_vector_recall = sum(m.vector_recall for m in metrics) / n
        avg_graph_recall = sum(m.graph_recall for m in metrics) / n
        avg_hybrid_hop = sum(m.hybrid_hop_coverage for m in metrics) / n
        avg_vector_hop = sum(m.vector_hop_coverage for m in metrics) / n
        avg_graph_hop = sum(m.graph_hop_coverage for m in metrics) / n
        avg_hybrid_time = sum(m.hybrid_time_ms for m in metrics) / n
        avg_vector_time = sum(m.vector_time_ms for m in metrics) / n
        avg_graph_time = sum(m.graph_time_ms for m in metrics) / n
        avg_anchors_count = sum(m.anchors_count for m in metrics) / n
        avg_traversed_discovered = sum(m.traversed_discovered for m in metrics) / n
        avg_traversed_in_final = sum(m.traversed_in_final for m in metrics) / n
        avg_anchor_only_recall = sum(m.anchor_only_recall for m in metrics) / n
        avg_hybrid_gain_over_anchor_only = sum(m.hybrid_gain_over_anchor_only for m in metrics) / n

        # By question type
        by_type = {}
        for m in metrics:
            qtype = m.question_type
            if qtype not in by_type:
                by_type[qtype] = {"hybrid_recall": [], "vector_recall": [], "graph_recall": [], "count": 0}
            by_type[qtype]["hybrid_recall"].append(m.hybrid_recall)
            by_type[qtype]["vector_recall"].append(m.vector_recall)
            by_type[qtype]["graph_recall"].append(m.graph_recall)
            by_type[qtype]["count"] += 1

        for qtype in by_type:
            cnt = by_type[qtype]["count"]
            by_type[qtype]["avg_hybrid_recall"] = sum(by_type[qtype]["hybrid_recall"]) / cnt
            by_type[qtype]["avg_vector_recall"] = sum(by_type[qtype]["vector_recall"]) / cnt
            by_type[qtype]["avg_graph_recall"] = sum(by_type[qtype]["graph_recall"]) / cnt
            by_type[qtype]["hybrid_vs_vector"] = by_type[qtype]["avg_hybrid_recall"] - by_type[qtype]["avg_vector_recall"]
            by_type[qtype]["hybrid_vs_graph"] = by_type[qtype]["avg_hybrid_recall"] - by_type[qtype]["avg_graph_recall"]
            del by_type[qtype]["hybrid_recall"]
            del by_type[qtype]["vector_recall"]
            del by_type[qtype]["graph_recall"]

        # By hop count
        by_hops = {}
        for m in metrics:
            hops = m.hop_count
            if hops not in by_hops:
                by_hops[hops] = {"hybrid_recall": [], "vector_recall": [], "graph_recall": [], "count": 0}
            by_hops[hops]["hybrid_recall"].append(m.hybrid_recall)
            by_hops[hops]["vector_recall"].append(m.vector_recall)
            by_hops[hops]["graph_recall"].append(m.graph_recall)
            by_hops[hops]["count"] += 1

        for hops in by_hops:
            cnt = by_hops[hops]["count"]
            by_hops[hops]["avg_hybrid_recall"] = sum(by_hops[hops]["hybrid_recall"]) / cnt
            by_hops[hops]["avg_vector_recall"] = sum(by_hops[hops]["vector_recall"]) / cnt
            by_hops[hops]["avg_graph_recall"] = sum(by_hops[hops]["graph_recall"]) / cnt
            by_hops[hops]["hybrid_vs_vector"] = by_hops[hops]["avg_hybrid_recall"] - by_hops[hops]["avg_vector_recall"]
            by_hops[hops]["hybrid_vs_graph"] = by_hops[hops]["avg_hybrid_recall"] - by_hops[hops]["avg_graph_recall"]
            del by_hops[hops]["hybrid_recall"]
            del by_hops[hops]["vector_recall"]
            del by_hops[hops]["graph_recall"]

        return BenchmarkResults(
            total_questions=n,
            total_contracts=num_contracts,
            avg_hybrid_recall=avg_hybrid_recall,
            avg_vector_recall=avg_vector_recall,
            avg_graph_recall=avg_graph_recall,
            avg_hybrid_hop_coverage=avg_hybrid_hop,
            avg_vector_hop_coverage=avg_vector_hop,
            avg_graph_hop_coverage=avg_graph_hop,
            avg_anchors_count=avg_anchors_count,
            avg_traversed_discovered=avg_traversed_discovered,
            avg_traversed_in_final=avg_traversed_in_final,
            avg_anchor_only_recall=avg_anchor_only_recall,
            avg_hybrid_gain_over_anchor_only=avg_hybrid_gain_over_anchor_only,
            metrics_by_type=by_type,
            metrics_by_hops={str(k): v for k, v in sorted(by_hops.items())},
            avg_hybrid_time_ms=avg_hybrid_time,
            avg_vector_time_ms=avg_vector_time,
            avg_graph_time_ms=avg_graph_time,
            per_question_metrics=metrics,
        )


def print_results(results: BenchmarkResults):
    """Print benchmark results in a nice format."""
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    print(f"\nContracts: {results.total_contracts}")
    print(f"Questions: {results.total_questions}")

    print("\n--- Overall Retrieval Recall ---")
    print(f"  Hybrid (vector + graph): {results.avg_hybrid_recall:.1%}")
    print(f"  Vector-only:             {results.avg_vector_recall:.1%}")
    print(f"  Graph-only:              {results.avg_graph_recall:.1%}")
    print(f"  Hybrid vs Vector:        {results.avg_hybrid_recall - results.avg_vector_recall:+.1%}")
    print(f"  Hybrid vs Graph:         {results.avg_hybrid_recall - results.avg_graph_recall:+.1%}")

    print("\n--- Hop Coverage ---")
    print(f"  Hybrid: {results.avg_hybrid_hop_coverage:.1%}")
    print(f"  Vector: {results.avg_vector_hop_coverage:.1%}")
    print(f"  Graph:  {results.avg_graph_hop_coverage:.1%}")

    print("\n--- By Question Type ---")
    for qtype, data in results.metrics_by_type.items():
        print(f"  {qtype}:")
        print(f"    Hybrid: {data['avg_hybrid_recall']:.1%}  Vector: {data['avg_vector_recall']:.1%}  Graph: {data['avg_graph_recall']:.1%}")

    print("\n--- By Hop Count ---")
    for hops, data in results.metrics_by_hops.items():
        print(f"  {hops} hops (n={data['count']}):")
        print(f"    Hybrid: {data['avg_hybrid_recall']:.1%}  Vector: {data['avg_vector_recall']:.1%}  Graph: {data['avg_graph_recall']:.1%}")

    print("\n--- Timing ---")
    print(f"  Hybrid avg: {results.avg_hybrid_time_ms:.1f}ms")
    print(f"  Vector avg: {results.avg_vector_time_ms:.1f}ms")
    print(f"  Graph avg:  {results.avg_graph_time_ms:.1f}ms")
    print("\n--- Hybrid Diagnostics ---")
    print(f"  Avg anchors/query:            {results.avg_anchors_count:.2f}")
    print(f"  Avg traversed discovered:     {results.avg_traversed_discovered:.2f}")
    print(f"  Avg traversed in final:       {results.avg_traversed_in_final:.2f}")
    print(f"  Avg anchor-only recall:       {results.avg_anchor_only_recall:.1%}")
    print(f"  Avg gain vs anchor-only:      {results.avg_hybrid_gain_over_anchor_only:+.1%}")


def main():
    """Run the benchmark."""
    dataset_path = Path("data/cuad/multihop_qa_dataset.json")

    if not dataset_path.exists():
        print(f"Dataset not found at {dataset_path}")
        print("Run multihop_generator.py first to create the dataset.")
        return

    # Run benchmark with separate semantic and graph slots so traversal can contribute.
    benchmark = Benchmark(
        embedding_backend="openai",  # Use OpenAI embeddings for real evaluation
        top_k_anchors=8,
        max_hops=2,
        max_context_items=15,  # Larger context window
        min_traversed_items=4
    )

    results = benchmark.run(dataset_path)

    # Print results
    print_results(results)

    # Save results
    output_path = Path("data/cuad/benchmark_results.json")
    results.save(output_path)


if __name__ == "__main__":
    main()
