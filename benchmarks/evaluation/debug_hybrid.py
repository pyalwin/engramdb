"""
Hybrid Retrieval Debugger

Diagnose whether graph traversal is actually contributing to final retrieval
or getting ranked out by semantic anchors.

Usage:
  uv run python benchmarks/evaluation/debug_hybrid.py
  uv run python benchmarks/evaluation/debug_hybrid.py --embedding-backend openai --top-k-anchors 5,10,15 --max-questions 40
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from engramdb import EngramDB


def extract_sections(engrams) -> list[str]:
    """Extract section-like labels from retrieved engrams."""
    sections = []
    for engram in engrams:
        if engram.metadata.get("section_number"):
            sections.append(engram.metadata["section_number"])
        elif engram.metadata.get("title"):
            sections.append(engram.metadata["title"])
        content = engram.content[:200].lower()
        for match in re.finditer(r"section\s+(\d+(?:\.\d+)*)", content):
            sections.append(match.group(1))
    return sorted(set(str(s).lower() for s in sections))


def calculate_recall(required: list[str], retrieved: list[str]) -> float:
    if not required:
        return 1.0
    required_norm = set(str(s).lower() for s in required)
    retrieved_norm = set(str(s).lower() for s in retrieved)
    return len(required_norm & retrieved_norm) / len(required_norm)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug hybrid retrieval contribution.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/cuad/multihop_qa_dataset.json"),
        help="Path to multi-hop dataset JSON.",
    )
    parser.add_argument(
        "--embedding-backend",
        choices=["openai", "local", "mock"],
        default="openai",
        help="Embedding backend for ingestion/query.",
    )
    parser.add_argument(
        "--top-k-anchors",
        type=str,
        default="5,10,15",
        help="Comma-separated anchor budgets to evaluate (e.g., 3,5,8,15).",
    )
    parser.add_argument(
        "--max-hops",
        type=int,
        default=2,
        help="Graph traversal depth.",
    )
    parser.add_argument(
        "--max-context-items",
        type=int,
        default=15,
        help="Final context window size for both hybrid and vector baseline.",
    )
    parser.add_argument(
        "--min-traversed-items",
        type=int,
        default=0,
        help="Minimum non-anchor traversed nodes retained in final hybrid context.",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=40,
        help="Limit number of questions for fast diagnostics. Use 0 for all.",
    )
    parser.add_argument(
        "--question-types",
        type=str,
        default="",
        help="Optional comma-separated question_type filter.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    data = json.loads(args.dataset.read_text())
    contracts = {c["id"]: c for c in data["contracts"]}
    questions = data["questions"]

    if args.question_types.strip():
        allowed = {q.strip() for q in args.question_types.split(",") if q.strip()}
        questions = [q for q in questions if q["question_type"] in allowed]

    if args.max_questions and args.max_questions > 0:
        questions = questions[: args.max_questions]

    if not questions:
        print("No questions selected. Check filters.")
        return

    anchor_budgets = [int(x.strip()) for x in args.top_k_anchors.split(",") if x.strip()]
    questions_by_contract = defaultdict(list)
    for q in questions:
        questions_by_contract[q["contract_id"]].append(q)

    print("=" * 72)
    print("Hybrid Debug Report")
    print("=" * 72)
    print(f"Questions: {len(questions)}")
    print(f"Contracts: {len(questions_by_contract)}")
    print(f"Backend: {args.embedding_backend}")
    print(f"Max hops: {args.max_hops}")
    print(f"Context window: {args.max_context_items}")
    print(f"Min traversed items: {args.min_traversed_items}")
    print()

    for top_k_anchors in anchor_budgets:
        hybrid_recall_sum = 0.0
        vector_recall_sum = 0.0
        same_order_count = 0
        same_set_count = 0
        traversed_discovered_sum = 0
        non_anchor_in_final_sum = 0
        queries_with_traversal = 0
        queries_with_traversal_contribution = 0
        total = 0

        for contract_id, contract_questions in questions_by_contract.items():
            contract = contracts[contract_id]
            with EngramDB(embedding_backend=args.embedding_backend) as db:
                db.ingest(contract["text"], doc_id=contract_id)

                for q in contract_questions:
                    total += 1
                    query = q["question"]
                    required_sections = q["reasoning_chain"]

                    hybrid = db.query(
                        query,
                        top_k_anchors=top_k_anchors,
                        max_hops=args.max_hops,
                        max_context_items=args.max_context_items,
                        min_traversed_items=args.min_traversed_items,
                    )
                    vector = db.query_vector_only(
                        query,
                        top_k=args.max_context_items,
                    )

                    hybrid_sections = extract_sections(hybrid.engrams)
                    vector_sections = extract_sections(vector.engrams)
                    hybrid_recall_sum += calculate_recall(required_sections, hybrid_sections)
                    vector_recall_sum += calculate_recall(required_sections, vector_sections)

                    hybrid_ids = [e.id for e in hybrid.engrams]
                    vector_ids = [e.id for e in vector.engrams]
                    if hybrid_ids == vector_ids:
                        same_order_count += 1
                    if set(hybrid_ids) == set(vector_ids):
                        same_set_count += 1

                    trace = db.retriever.retrieve_with_trace(
                        query=query,
                        top_k_anchors=top_k_anchors,
                        max_hops=args.max_hops,
                        max_context_items=args.max_context_items,
                        min_traversed_items=args.min_traversed_items,
                    )
                    anchor_ids = {engram.id for engram, _ in trace.anchors}
                    traversed_ids = set()
                    for ids in trace.traversal_paths.values():
                        traversed_ids.update(ids)

                    traversed_discovered_sum += len(traversed_ids)
                    if traversed_ids:
                        queries_with_traversal += 1

                    non_anchor_in_final = sum(
                        1
                        for e in trace.final_context
                        if e.id in traversed_ids and e.id not in anchor_ids
                    )
                    non_anchor_in_final_sum += non_anchor_in_final
                    if non_anchor_in_final > 0:
                        queries_with_traversal_contribution += 1

        print(f"[top_k_anchors={top_k_anchors}]")
        print(f"  Avg hybrid recall:            {hybrid_recall_sum / total:.4f}")
        print(f"  Avg vector recall:            {vector_recall_sum / total:.4f}")
        print(f"  Recall delta (hybrid-vector): {(hybrid_recall_sum - vector_recall_sum) / total:+.4f}")
        print(f"  Identical final IDs (order):  {same_order_count}/{total} ({same_order_count / total:.1%})")
        print(f"  Identical final IDs (set):    {same_set_count}/{total} ({same_set_count / total:.1%})")
        print(f"  Avg traversed discovered:     {traversed_discovered_sum / total:.2f}")
        print(f"  Avg non-anchor in final:      {non_anchor_in_final_sum / total:.2f}")
        print(f"  Traversal discovered >0:      {queries_with_traversal}/{total} ({queries_with_traversal / total:.1%})")
        print(f"  Traversal contributes >0:     {queries_with_traversal_contribution}/{total} ({queries_with_traversal_contribution / total:.1%})")
        print()


if __name__ == "__main__":
    main()
