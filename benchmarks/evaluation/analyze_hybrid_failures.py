"""
Analyze Hybrid Retrieval Failures

For each question where hybrid recall is not perfect, replay retrieval with trace,
inspect graph hops to required sections, and classify likely failure reasons.

Usage:
  uv run python benchmarks/evaluation/analyze_hybrid_failures.py
  uv run python benchmarks/evaluation/analyze_hybrid_failures.py --limit-failures 20
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import json
from pathlib import Path
import re
import sys
from typing import Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from engramdb import EngramDB


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze failed hybrid retrieval cases.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/cuad/multihop_qa_dataset.json"),
        help="Path to multihop dataset.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("data/cuad/benchmark_results.json"),
        help="Path to benchmark results JSON to identify failed questions.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/cuad/hybrid_failure_analysis.json"),
        help="Output analysis report path.",
    )
    parser.add_argument(
        "--embedding-backend",
        choices=["openai", "local", "mock"],
        default="openai",
        help="Embedding backend to replay retrieval.",
    )
    parser.add_argument(
        "--top-k-anchors",
        type=int,
        default=8,
        help="Hybrid anchor budget used for replay.",
    )
    parser.add_argument(
        "--max-hops",
        type=int,
        default=2,
        help="Hybrid traversal depth used for replay.",
    )
    parser.add_argument(
        "--max-context-items",
        type=int,
        default=15,
        help="Final context size used for replay.",
    )
    parser.add_argument(
        "--min-traversed-items",
        type=int,
        default=4,
        help="Traversal reservation used for replay.",
    )
    parser.add_argument(
        "--counterfactual-anchors-multiplier",
        type=float,
        default=1.5,
        help="Multiplier for top_k_anchors counterfactual test.",
    )
    parser.add_argument(
        "--counterfactual-extra-hops",
        type=int,
        default=1,
        help="Additional hops for max_hops counterfactual test.",
    )
    parser.add_argument(
        "--limit-failures",
        type=int,
        default=0,
        help="Analyze only first N failed questions (0 = all).",
    )
    return parser.parse_args()


def extract_sections(engrams) -> list[str]:
    """Extract section-like labels from engrams using benchmark-compatible logic."""
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


def build_graph(db: EngramDB) -> dict[str, set[str]]:
    """Build undirected adjacency map from synapses."""
    rows = db.storage._connection.execute(
        "SELECT source_id, target_id FROM synapses"
    ).fetchall()
    adjacency: dict[str, set[str]] = defaultdict(set)
    for source_id, target_id in rows:
        adjacency[source_id].add(target_id)
        adjacency[target_id].add(source_id)
    return adjacency


def shortest_path_to_targets(
    adjacency: dict[str, set[str]],
    source_ids: set[str],
    target_ids: set[str],
    max_depth: int,
) -> Optional[list[str]]:
    """Shortest path from any source to any target within max_depth hops."""
    if not source_ids or not target_ids:
        return None

    overlap = source_ids & target_ids
    if overlap:
        source = next(iter(overlap))
        return [source]

    queue = deque()
    parent: dict[str, Optional[str]] = {}
    depth_map: dict[str, int] = {}

    for source in source_ids:
        queue.append(source)
        parent[source] = None
        depth_map[source] = 0

    visited = set(source_ids)

    while queue:
        node = queue.popleft()
        depth = depth_map[node]
        if depth >= max_depth:
            continue

        for neighbor in adjacency.get(node, set()):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            parent[neighbor] = node
            depth_map[neighbor] = depth + 1

            if neighbor in target_ids:
                path = [neighbor]
                cursor = neighbor
                while parent[cursor] is not None:
                    cursor = parent[cursor]
                    path.append(cursor)
                path.reverse()
                return path

            queue.append(neighbor)

    return None


def choose_primary_reason(
    required_item_analysis: list[dict],
    cf_anchor_improved: bool,
    cf_hops_improved: bool,
    cf_both_improved: bool,
) -> str:
    statuses = [item["status"] for item in required_item_analysis if not item["captured_in_final"]]
    if not statuses:
        return "fully_captured"
    if any(status == "missing_engram" for status in statuses):
        return "missing_engram"
    if any(status == "traversed_ranked_out" for status in statuses):
        return "traversed_ranked_out"
    if any(status == "anchor_ranked_out" for status in statuses):
        return "anchor_ranked_out"
    if cf_hops_improved:
        return "insufficient_hops"
    if cf_anchor_improved:
        return "insufficient_anchors"
    if cf_both_improved:
        return "needs_joint_tuning"
    return "not_reachable_from_anchors"


def main() -> None:
    args = parse_args()
    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")
    if not args.results.exists():
        raise FileNotFoundError(f"Results not found: {args.results}")

    dataset = json.loads(args.dataset.read_text())
    results = json.loads(args.results.read_text())

    questions_by_id = {q["id"]: q for q in dataset["questions"]}
    contracts_by_id = {c["id"]: c for c in dataset["contracts"]}

    failed_question_ids = [
        row["question_id"]
        for row in results["per_question"]
        if row["hybrid_recall"] < 1.0
    ]
    if args.limit_failures and args.limit_failures > 0:
        failed_question_ids = failed_question_ids[: args.limit_failures]

    if not failed_question_ids:
        print("No failed hybrid questions found (hybrid_recall < 1.0).")
        return

    failed_questions = [questions_by_id[qid] for qid in failed_question_ids]
    questions_by_contract: dict[str, list[dict]] = defaultdict(list)
    for question in failed_questions:
        questions_by_contract[question["contract_id"]].append(question)

    print("=" * 72)
    print("Hybrid Failure Analysis")
    print("=" * 72)
    print(f"Failed questions selected: {len(failed_questions)}")
    print(f"Contracts touched: {len(questions_by_contract)}")
    print(f"Replay params: anchors={args.top_k_anchors}, hops={args.max_hops}, context={args.max_context_items}, min_traversed={args.min_traversed_items}")
    print()

    failures = []
    reason_counts = Counter()
    status_counts = Counter()

    cf_anchor_k = max(args.top_k_anchors + 1, int(round(args.top_k_anchors * args.counterfactual_anchors_multiplier)))
    cf_hops = args.max_hops + args.counterfactual_extra_hops

    for idx, (contract_id, contract_questions) in enumerate(questions_by_contract.items(), start=1):
        print(f"[{idx}/{len(questions_by_contract)}] {contract_id[:70]}... ({len(contract_questions)} failed questions)")
        contract = contracts_by_id[contract_id]

        with EngramDB(embedding_backend=args.embedding_backend) as db:
            db.ingest(contract["text"], doc_id=contract_id)
            adjacency = build_graph(db)
            all_engrams = db.storage.get_all_engrams()
            label_to_ids: dict[str, set[str]] = defaultdict(set)
            id_to_label: dict[str, str] = {}
            for engram in all_engrams:
                label = engram.metadata.get("section_number") or engram.metadata.get("title") or engram.metadata.get("term") or engram.id
                id_to_label[engram.id] = str(label)
                if engram.metadata.get("section_number"):
                    label_to_ids[str(engram.metadata["section_number"]).lower()].add(engram.id)
                if engram.metadata.get("title"):
                    label_to_ids[str(engram.metadata["title"]).lower()].add(engram.id)

            for question in contract_questions:
                query = question["question"]
                required = question["reasoning_chain"]
                required_norm = sorted(set(str(s).lower() for s in required))

                trace = db.retriever.retrieve_with_trace(
                    query=query,
                    top_k_anchors=args.top_k_anchors,
                    max_hops=args.max_hops,
                    max_context_items=args.max_context_items,
                    min_traversed_items=args.min_traversed_items,
                )

                base_result = db.query(
                    query=query,
                    top_k_anchors=args.top_k_anchors,
                    max_hops=args.max_hops,
                    max_context_items=args.max_context_items,
                    min_traversed_items=args.min_traversed_items,
                )
                base_sections = extract_sections(base_result.engrams)
                base_recall = calculate_recall(required, base_sections)

                anchor_result = db.query_vector_only(query=query, top_k=args.top_k_anchors)
                anchor_sections = extract_sections(anchor_result.engrams)
                anchor_only_recall = calculate_recall(required, anchor_sections)

                cf_anchor_result = db.query(
                    query=query,
                    top_k_anchors=cf_anchor_k,
                    max_hops=args.max_hops,
                    max_context_items=args.max_context_items,
                    min_traversed_items=args.min_traversed_items,
                )
                cf_anchor_recall = calculate_recall(required, extract_sections(cf_anchor_result.engrams))

                cf_hops_result = db.query(
                    query=query,
                    top_k_anchors=args.top_k_anchors,
                    max_hops=cf_hops,
                    max_context_items=args.max_context_items,
                    min_traversed_items=args.min_traversed_items,
                )
                cf_hops_recall = calculate_recall(required, extract_sections(cf_hops_result.engrams))

                cf_both_result = db.query(
                    query=query,
                    top_k_anchors=cf_anchor_k,
                    max_hops=cf_hops,
                    max_context_items=args.max_context_items,
                    min_traversed_items=args.min_traversed_items,
                )
                cf_both_recall = calculate_recall(required, extract_sections(cf_both_result.engrams))

                cf_anchor_improved = cf_anchor_recall > base_recall
                cf_hops_improved = cf_hops_recall > base_recall
                cf_both_improved = cf_both_recall > base_recall

                anchor_ids = {engram.id for engram, _ in trace.anchors}
                traversed_ids = set()
                for ids in trace.traversal_paths.values():
                    traversed_ids.update(ids)
                final_ids = {engram.id for engram in trace.final_context}

                required_item_analysis = []
                for section_label in required_norm:
                    candidate_ids = label_to_ids.get(section_label, set())
                    captured = bool(candidate_ids & final_ids)

                    if not candidate_ids:
                        status = "missing_engram"
                        path_labels = []
                        shortest_hop = None
                    elif captured:
                        status = "captured"
                        path_labels = []
                        shortest_hop = 0
                    elif candidate_ids & traversed_ids:
                        status = "traversed_ranked_out"
                        shortest_hop = None
                        path = shortest_path_to_targets(
                            adjacency=adjacency,
                            source_ids=anchor_ids,
                            target_ids=candidate_ids,
                            max_depth=cf_hops,
                        )
                        if path is None:
                            path_labels = []
                        else:
                            shortest_hop = len(path) - 1
                            path_labels = [id_to_label.get(node_id, node_id) for node_id in path]
                    elif candidate_ids & anchor_ids:
                        status = "anchor_ranked_out"
                        shortest_hop = 0
                        path_labels = []
                    else:
                        status = "not_reached"
                        path = shortest_path_to_targets(
                            adjacency=adjacency,
                            source_ids=anchor_ids,
                            target_ids=candidate_ids,
                            max_depth=cf_hops,
                        )
                        if path is None:
                            shortest_hop = None
                            path_labels = []
                        else:
                            shortest_hop = len(path) - 1
                            path_labels = [id_to_label.get(node_id, node_id) for node_id in path]

                    status_counts[status] += 1
                    required_item_analysis.append(
                        {
                            "required_section": section_label,
                            "captured_in_final": captured,
                            "status": status,
                            "candidate_ids_count": len(candidate_ids),
                            "shortest_hop_from_anchors_within_counterfactual": shortest_hop,
                            "shortest_path_labels": path_labels,
                        }
                    )

                primary_reason = choose_primary_reason(
                    required_item_analysis=required_item_analysis,
                    cf_anchor_improved=cf_anchor_improved,
                    cf_hops_improved=cf_hops_improved,
                    cf_both_improved=cf_both_improved,
                )
                reason_counts[primary_reason] += 1

                failures.append(
                    {
                        "question_id": question["id"],
                        "contract_id": question["contract_id"],
                        "question_type": question["question_type"],
                        "hop_count": question["hop_count"],
                        "question": query,
                        "required_sections": required,
                        "base": {
                            "hybrid_recall": base_recall,
                            "anchor_only_recall": anchor_only_recall,
                            "anchors_count": len(anchor_ids),
                            "traversed_discovered": len(traversed_ids),
                            "traversed_in_final": sum(
                                1
                                for engram in trace.final_context
                                if engram.id in traversed_ids and engram.id not in anchor_ids
                            ),
                            "anchor_labels": [
                                engram.metadata.get("section_number")
                                or engram.metadata.get("title")
                                or engram.metadata.get("term")
                                or engram.id
                                for engram, _ in trace.anchors
                            ],
                            "final_labels": [
                                engram.metadata.get("section_number")
                                or engram.metadata.get("title")
                                or engram.metadata.get("term")
                                or engram.id
                                for engram in trace.final_context
                            ],
                        },
                        "counterfactuals": {
                            "anchor_multiplier": args.counterfactual_anchors_multiplier,
                            "extra_hops": args.counterfactual_extra_hops,
                            "more_anchors_recall": cf_anchor_recall,
                            "more_hops_recall": cf_hops_recall,
                            "more_anchors_and_hops_recall": cf_both_recall,
                            "more_anchors_improved": cf_anchor_improved,
                            "more_hops_improved": cf_hops_improved,
                            "more_anchors_and_hops_improved": cf_both_improved,
                            "counterfactual_top_k_anchors": cf_anchor_k,
                            "counterfactual_max_hops": cf_hops,
                        },
                        "required_item_analysis": required_item_analysis,
                        "primary_reason": primary_reason,
                    }
                )

    report = {
        "replay_config": {
            "embedding_backend": args.embedding_backend,
            "top_k_anchors": args.top_k_anchors,
            "max_hops": args.max_hops,
            "max_context_items": args.max_context_items,
            "min_traversed_items": args.min_traversed_items,
            "counterfactual_anchors_multiplier": args.counterfactual_anchors_multiplier,
            "counterfactual_extra_hops": args.counterfactual_extra_hops,
        },
        "summary": {
            "failed_questions_analyzed": len(failures),
            "reason_counts": dict(reason_counts),
            "required_item_status_counts": dict(status_counts),
        },
        "failures": failures,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))

    print()
    print("=" * 72)
    print("Failure Analysis Complete")
    print("=" * 72)
    print(f"Analyzed failures: {len(failures)}")
    print(f"Reason counts: {dict(reason_counts)}")
    print(f"Required-item status counts: {dict(status_counts)}")
    print(f"Saved report: {args.output}")


if __name__ == "__main__":
    main()
