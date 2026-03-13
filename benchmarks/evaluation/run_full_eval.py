"""
Full Evaluation Orchestrator for EngramDB.

Runs all retrieval systems on the same multi-hop QA dataset and
produces a unified results report with statistical significance tests.

Systems evaluated:
  1. EngramDB Hybrid (vector + graph)
  2. Vector-only (EngramDB without graph traversal)
  3. Graph-only (EngramDB without semantic blending)
  4. NaiveRAG (fixed-size chunking + vector search)
  5. Parent-Document RAG (section chunking + parent expansion)
"""

import json
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from engramdb import EngramDB
from engramdb.embeddings.embedder import create_embedder
from engramdb.retrieval.hybrid import RetrievalResult

from benchmarks.baselines.naive_rag import NaiveRAG
from benchmarks.baselines.parent_doc_rag import ParentDocumentRAG
from benchmarks.evaluation.statistics import (
    paired_t_test,
    wilcoxon_signed_rank,
    bootstrap_ci,
    format_significance_table,
)
from benchmarks.evaluation.metrics import Evaluator, _calculate_recall, _calculate_hop_coverage


SYSTEM_NAMES = ["hybrid", "vector_only", "graph_only", "naive_rag", "parent_doc_rag"]


@dataclass
class SystemResult:
    """Per-question result for a single system."""
    question_id: str
    question_type: str
    hop_count: int
    required_sections: list[str]
    retrieved_sections: list[str]
    recall: float
    hop_coverage: float
    time_ms: float


@dataclass
class FullEvalResults:
    """Complete evaluation results across all systems."""
    total_questions: int
    total_contracts: int
    systems: dict[str, list[SystemResult]]
    summary: dict  # system_name -> {avg_recall, avg_hop_coverage, ...}
    by_type: dict   # question_type -> {system_name -> avg_recall}
    by_hops: dict   # hop_count -> {system_name -> avg_recall}
    significance_tests: list[dict]

    def to_dict(self) -> dict:
        per_question = {}
        for sys_name, results in self.systems.items():
            per_question[sys_name] = [asdict(r) for r in results]

        return {
            "total_questions": self.total_questions,
            "total_contracts": self.total_contracts,
            "summary": self.summary,
            "by_question_type": self.by_type,
            "by_hop_count": self.by_hops,
            "significance_tests": self.significance_tests,
            "per_question": per_question,
        }

    def save(self, filepath: Path):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Saved full evaluation to {filepath}")


def extract_sections(engrams) -> list[str]:
    """Extract section identifiers from retrieved engrams."""
    sections = []
    for e in engrams:
        if e.metadata.get("section_number"):
            sections.append(e.metadata["section_number"])
        elif e.metadata.get("title"):
            sections.append(e.metadata["title"])
        content = e.content[:200].lower()
        for match in re.finditer(r'section\s+(\d+(?:\.\d+)*)', content):
            sections.append(match.group(1))
    return list(set(sections))


class FullEvaluator:
    """Orchestrates evaluation of all retrieval systems."""

    def __init__(
        self,
        embedding_backend: str = "mock",
        top_k_anchors: int = 3,
        max_hops: int = 2,
        max_context_items: int = 10,
        min_traversed_items: int = 4,
        naive_chunk_size: int = 512,
        naive_chunk_overlap: int = 50,
    ):
        self.embedding_backend = embedding_backend
        self.top_k_anchors = top_k_anchors
        self.max_hops = max_hops
        self.max_context_items = max_context_items
        self.min_traversed_items = min_traversed_items
        self.naive_chunk_size = naive_chunk_size
        self.naive_chunk_overlap = naive_chunk_overlap

    def run(self, dataset_path: Path) -> FullEvalResults:
        """Run full evaluation on all systems."""
        print("=" * 70)
        print("EngramDB Full Evaluation")
        print("=" * 70)

        with open(dataset_path) as f:
            dataset = json.load(f)

        contracts = dataset["contracts"]
        questions = dataset["questions"]
        print(f"Loaded {len(contracts)} contracts, {len(questions)} questions\n")

        # Group questions by contract
        questions_by_contract: dict[str, list[dict]] = {}
        for q in questions:
            cid = q["contract_id"]
            if cid not in questions_by_contract:
                questions_by_contract[cid] = []
            questions_by_contract[cid].append(q)

        all_results: dict[str, list[SystemResult]] = {name: [] for name in SYSTEM_NAMES}

        for i, contract in enumerate(contracts):
            contract_id = contract["id"]
            contract_questions = questions_by_contract.get(contract_id, [])
            if not contract_questions:
                continue

            print(f"[{i+1}/{len(contracts)}] {contract_id[:50]}... ({len(contract_questions)} questions)")

            self._evaluate_contract(
                contract["text"], contract_id, contract_questions, all_results
            )

        # Aggregate
        summary = self._compute_summary(all_results)
        by_type = self._compute_by_type(all_results)
        by_hops = self._compute_by_hops(all_results)
        sig_tests = self._compute_significance(all_results)

        results = FullEvalResults(
            total_questions=len(questions),
            total_contracts=len(contracts),
            systems=all_results,
            summary=summary,
            by_type=by_type,
            by_hops=by_hops,
            significance_tests=sig_tests,
        )

        self._print_results(results)
        return results

    def _evaluate_contract(
        self,
        text: str,
        contract_id: str,
        questions: list[dict],
        all_results: dict[str, list[SystemResult]],
    ):
        """Evaluate all systems on a single contract's questions."""
        embedder = create_embedder(backend=self.embedding_backend)

        # --- System 1-3: EngramDB (hybrid, vector-only, graph-only) ---
        with EngramDB(embedding_backend=self.embedding_backend, embedder=embedder) as db:
            db.ingest(text, doc_id=contract_id)

            for q in questions:
                qid = q["id"]
                query = q["question"]
                required = q["reasoning_chain"]
                qtype = q["question_type"]
                hops = q["hop_count"]

                # Hybrid
                t0 = time.time()
                hr = db.query(
                    query,
                    top_k_anchors=self.top_k_anchors,
                    max_hops=self.max_hops,
                    max_context_items=self.max_context_items,
                    min_traversed_items=self.min_traversed_items,
                )
                ht = (time.time() - t0) * 1000
                hs = extract_sections(hr.engrams)
                all_results["hybrid"].append(SystemResult(
                    qid, qtype, hops, required, hs,
                    _calculate_recall(required, hs),
                    _calculate_hop_coverage(required, hs), ht,
                ))

                # Vector-only
                t0 = time.time()
                vr = db.query_vector_only(query, top_k=self.max_context_items)
                vt = (time.time() - t0) * 1000
                vs = extract_sections(vr.engrams)
                all_results["vector_only"].append(SystemResult(
                    qid, qtype, hops, required, vs,
                    _calculate_recall(required, vs),
                    _calculate_hop_coverage(required, vs), vt,
                ))

                # Graph-only
                t0 = time.time()
                gr = db.query_graph_only(
                    query,
                    top_k_anchors=self.top_k_anchors,
                    max_hops=self.max_hops,
                    max_context_items=self.max_context_items,
                )
                gt = (time.time() - t0) * 1000
                gs = extract_sections(gr.engrams)
                all_results["graph_only"].append(SystemResult(
                    qid, qtype, hops, required, gs,
                    _calculate_recall(required, gs),
                    _calculate_hop_coverage(required, gs), gt,
                ))

        # --- System 4: NaiveRAG ---
        with NaiveRAG(
            chunk_size=self.naive_chunk_size,
            chunk_overlap=self.naive_chunk_overlap,
            embedder=embedder,
        ) as naive:
            naive.ingest(text, doc_id=contract_id)
            for q in questions:
                qid = q["id"]
                query = q["question"]
                required = q["reasoning_chain"]
                qtype = q["question_type"]
                hops = q["hop_count"]

                t0 = time.time()
                nr = naive.query(query, top_k=self.max_context_items)
                nt = (time.time() - t0) * 1000
                ns = extract_sections(nr.engrams)
                all_results["naive_rag"].append(SystemResult(
                    qid, qtype, hops, required, ns,
                    _calculate_recall(required, ns),
                    _calculate_hop_coverage(required, ns), nt,
                ))

        # --- System 5: Parent-Document RAG ---
        with ParentDocumentRAG(embedder=embedder) as pdrag:
            pdrag.ingest(text, doc_id=contract_id)
            for q in questions:
                qid = q["id"]
                query = q["question"]
                required = q["reasoning_chain"]
                qtype = q["question_type"]
                hops = q["hop_count"]

                t0 = time.time()
                pr = pdrag.query(
                    query,
                    top_k=self.top_k_anchors,
                    max_context_items=self.max_context_items,
                )
                pt = (time.time() - t0) * 1000
                ps = extract_sections(pr.engrams)
                all_results["parent_doc_rag"].append(SystemResult(
                    qid, qtype, hops, required, ps,
                    _calculate_recall(required, ps),
                    _calculate_hop_coverage(required, ps), pt,
                ))

    def _compute_summary(self, all_results: dict[str, list[SystemResult]]) -> dict:
        """Compute per-system summary metrics."""
        summary = {}
        for sys_name, results in all_results.items():
            if not results:
                continue
            n = len(results)
            summary[sys_name] = {
                "n": n,
                "avg_recall": round(sum(r.recall for r in results) / n, 4),
                "avg_hop_coverage": round(sum(r.hop_coverage for r in results) / n, 4),
                "avg_time_ms": round(sum(r.time_ms for r in results) / n, 2),
                "perfect_recall_pct": round(sum(1 for r in results if r.recall >= 1.0) / n, 4),
            }
        return summary

    def _compute_by_type(self, all_results: dict[str, list[SystemResult]]) -> dict:
        """Break down recall by question type for each system."""
        # Collect all question types
        qtypes: set[str] = set()
        for results in all_results.values():
            for r in results:
                qtypes.add(r.question_type)

        by_type = {}
        for qtype in sorted(qtypes):
            by_type[qtype] = {}
            for sys_name, results in all_results.items():
                typed = [r for r in results if r.question_type == qtype]
                if typed:
                    by_type[qtype][sys_name] = {
                        "count": len(typed),
                        "avg_recall": round(sum(r.recall for r in typed) / len(typed), 4),
                    }
        return by_type

    def _compute_by_hops(self, all_results: dict[str, list[SystemResult]]) -> dict:
        """Break down recall by hop count for each system."""
        hop_counts: set[int] = set()
        for results in all_results.values():
            for r in results:
                hop_counts.add(r.hop_count)

        by_hops = {}
        for hops in sorted(hop_counts):
            by_hops[str(hops)] = {}
            for sys_name, results in all_results.items():
                hopped = [r for r in results if r.hop_count == hops]
                if hopped:
                    by_hops[str(hops)][sys_name] = {
                        "count": len(hopped),
                        "avg_recall": round(sum(r.recall for r in hopped) / len(hopped), 4),
                    }
        return by_hops

    def _compute_significance(self, all_results: dict[str, list[SystemResult]]) -> list[dict]:
        """Run paired t-tests: hybrid vs each other system."""
        hybrid = all_results.get("hybrid", [])
        if not hybrid:
            return []

        hybrid_recalls = [r.recall for r in hybrid]
        tests = []

        for sys_name in SYSTEM_NAMES:
            if sys_name == "hybrid":
                continue
            other = all_results.get(sys_name, [])
            if len(other) != len(hybrid):
                continue

            other_recalls = [r.recall for r in other]
            try:
                t_result = paired_t_test(
                    hybrid_recalls, other_recalls,
                    system_a="hybrid", system_b=sys_name,
                    metric="recall",
                )
                tests.append(t_result.to_dict())

                w_result = wilcoxon_signed_rank(
                    hybrid_recalls, other_recalls,
                    system_a="hybrid", system_b=sys_name,
                    metric="recall",
                )
                tests.append({**w_result.to_dict(), "test_type": "wilcoxon"})
            except ValueError:
                pass

        return tests

    def _print_results(self, results: FullEvalResults):
        """Print formatted results table."""
        print("\n" + "=" * 70)
        print("FULL EVALUATION RESULTS")
        print("=" * 70)

        print(f"\nQuestions: {results.total_questions}  |  Contracts: {results.total_contracts}\n")

        # Summary table
        print(f"{'System':<20} {'Recall':>8} {'Hop Cov':>8} {'Perfect%':>9} {'Time(ms)':>10}")
        print("-" * 60)
        for sys_name in SYSTEM_NAMES:
            s = results.summary.get(sys_name)
            if s:
                print(
                    f"{sys_name:<20} {s['avg_recall']:>7.1%} "
                    f"{s['avg_hop_coverage']:>7.1%} "
                    f"{s['perfect_recall_pct']:>8.1%} "
                    f"{s['avg_time_ms']:>9.1f}"
                )

        # By hop count
        print("\n--- By Hop Count ---")
        for hops, data in sorted(results.by_hops.items()):
            print(f"\n  {hops} hops:")
            for sys_name in SYSTEM_NAMES:
                if sys_name in data:
                    d = data[sys_name]
                    print(f"    {sys_name:<20} {d['avg_recall']:>7.1%}  (n={d['count']})")

        # By question type
        print("\n--- By Question Type ---")
        for qtype, data in sorted(results.by_type.items()):
            print(f"\n  {qtype}:")
            for sys_name in SYSTEM_NAMES:
                if sys_name in data:
                    d = data[sys_name]
                    print(f"    {sys_name:<20} {d['avg_recall']:>7.1%}  (n={d['count']})")

        # Significance
        if results.significance_tests:
            print("\n--- Statistical Significance (hybrid vs others) ---")
            for t in results.significance_tests:
                test_type = t.get("test_type", "t-test")
                sig = "***" if t["p_value"] < 0.001 else (
                    "**" if t["p_value"] < 0.01 else (
                    "*" if t["p_value"] < 0.05 else "ns"))
                print(
                    f"  [{test_type}] hybrid vs {t['system_b']}: "
                    f"Δ={t['mean_diff']:+.4f}  p={t['p_value']:.6f}  "
                    f"d={t['cohens_d']:.3f}  {sig}"
                )


def main():
    """Run the full evaluation."""
    dataset_path = Path("data/cuad/multihop_qa_dataset.json")

    if not dataset_path.exists():
        print(f"Dataset not found at {dataset_path}")
        print("Run multihop_generator.py first to create the dataset.")
        return

    evaluator = FullEvaluator(
        embedding_backend="openai",
        top_k_anchors=8,
        max_hops=2,
        max_context_items=15,
        min_traversed_items=4,
    )

    results = evaluator.run(dataset_path)

    output_path = Path("data/cuad/full_eval_results.json")
    results.save(output_path)


if __name__ == "__main__":
    main()
