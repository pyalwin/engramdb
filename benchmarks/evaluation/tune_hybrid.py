"""
Hybrid Retriever Parameter Sweep

Runs a grid search over hybrid retrieval/scoring parameters, ranks configurations
by objective, and exports failure buckets for targeted debugging.

Usage:
  uv run python benchmarks/evaluation/tune_hybrid.py
  uv run python benchmarks/evaluation/tune_hybrid.py --max-questions 80 --top-k-anchors 5,8,10 --min-traversed-items 2,4,6
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from datetime import datetime, UTC
import io
import itertools
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable

from benchmark import Benchmark, RetrievalMetrics


def parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_float_list(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep hybrid retrieval parameters.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/cuad/multihop_qa_dataset.json"),
        help="Path to multihop dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/cuad/hybrid_tuning_report.json"),
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--embedding-backend",
        choices=["openai", "local", "mock"],
        default="openai",
        help="Embedding backend.",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=0,
        help="Use only first N questions for faster sweep (0 = all).",
    )
    parser.add_argument(
        "--question-types",
        type=str,
        default="",
        help="Optional comma-separated filter (e.g. cross_reference,definition_usage).",
    )
    parser.add_argument(
        "--max-context-items",
        type=int,
        default=15,
        help="Final context size.",
    )
    parser.add_argument(
        "--max-hops",
        type=int,
        default=2,
        help="Graph traversal depth.",
    )
    parser.add_argument(
        "--top-k-anchors",
        type=str,
        default="5,8,10",
        help="Comma-separated anchor budgets.",
    )
    parser.add_argument(
        "--min-traversed-items",
        type=str,
        default="0,2,4",
        help="Comma-separated traversal reservation values.",
    )
    parser.add_argument(
        "--semantic-weight",
        type=str,
        default="0.65,0.7,0.75",
        help="Comma-separated semantic score weights.",
    )
    parser.add_argument(
        "--hop-decay",
        type=str,
        default="0.65,0.75,0.85",
        help="Comma-separated hop decay values.",
    )
    parser.add_argument(
        "--defines-weight",
        type=str,
        default="0.8,0.9,1.0",
        help="Comma-separated DEFINES edge weights.",
    )
    parser.add_argument(
        "--parent-weight",
        type=str,
        default="0.45,0.55,0.65",
        help="Comma-separated hierarchy edge weights (PARENT_OF/CHILD_OF).",
    )
    parser.add_argument(
        "--references-weight",
        type=str,
        default="1.0",
        help="Comma-separated REFERENCES edge weights.",
    )
    parser.add_argument(
        "--default-edge-weight",
        type=str,
        default="0.6",
        help="Comma-separated default fallback edge weights.",
    )
    parser.add_argument(
        "--objective",
        choices=["improvement", "hybrid_recall", "gain_over_anchor"],
        default="improvement",
        help="Primary optimization objective.",
    )
    parser.add_argument(
        "--latency-penalty",
        type=float,
        default=0.0,
        help="Subtract penalty * max(hybrid_ms - vector_ms, 0) from objective.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help="Cap number of configurations evaluated (0 = all).",
    )
    parser.add_argument(
        "--failure-examples",
        type=int,
        default=20,
        help="Number of worst examples to keep per failure bucket.",
    )
    return parser.parse_args()


def make_subset_dataset(
    dataset: dict,
    max_questions: int,
    question_types: set[str],
) -> dict:
    questions = dataset["questions"]
    if question_types:
        questions = [q for q in questions if q["question_type"] in question_types]
    if max_questions > 0:
        questions = questions[:max_questions]

    needed_contracts = {q["contract_id"] for q in questions}
    contracts = [c for c in dataset["contracts"] if c["id"] in needed_contracts]

    return {"contracts": contracts, "questions": questions}


def score_result(summary: dict, objective: str, latency_penalty: float) -> float:
    if objective == "improvement":
        base = summary["avg_hybrid_recall"] - summary["avg_vector_recall"]
    elif objective == "hybrid_recall":
        base = summary["avg_hybrid_recall"]
    else:
        base = summary.get("avg_hybrid_gain_over_anchor_only", 0.0)

    latency_delta = max(summary["avg_hybrid_time_ms"] - summary["avg_vector_time_ms"], 0.0)
    return base - latency_penalty * (latency_delta / 1000.0)


def simplify_metric(m: RetrievalMetrics) -> dict:
    return {
        "question_id": m.question_id,
        "question_type": m.question_type,
        "hop_count": m.hop_count,
        "hybrid_advantage": m.hybrid_advantage,
        "hybrid_gain_over_anchor_only": m.hybrid_gain_over_anchor_only,
        "hybrid_recall": m.hybrid_recall,
        "vector_recall": m.vector_recall,
        "anchor_only_recall": m.anchor_only_recall,
        "traversed_discovered": m.traversed_discovered,
        "traversed_in_final": m.traversed_in_final,
        "required_sections": m.required_sections,
        "hybrid_retrieved": m.hybrid_retrieved,
        "vector_retrieved": m.vector_retrieved,
    }


def build_failure_buckets(metrics: list[RetrievalMetrics], max_examples: int) -> dict:
    negative_adv = sorted(
        [m for m in metrics if m.hybrid_advantage < 0],
        key=lambda m: m.hybrid_advantage,
    )
    traversal_hurts = sorted(
        [m for m in metrics if m.hybrid_advantage < 0 and m.traversed_in_final > 0],
        key=lambda m: m.hybrid_advantage,
    )
    traversal_discovered_but_ranked_out = sorted(
        [m for m in metrics if m.traversed_discovered > 0 and m.traversed_in_final == 0],
        key=lambda m: m.hybrid_advantage,
    )
    no_gain_over_anchor = sorted(
        [m for m in metrics if m.hybrid_gain_over_anchor_only <= 0],
        key=lambda m: m.hybrid_gain_over_anchor_only,
    )

    by_type: dict[str, list[RetrievalMetrics]] = {}
    for m in metrics:
        by_type.setdefault(m.question_type, []).append(m)

    type_diagnostics = {}
    for qtype, items in by_type.items():
        type_diagnostics[qtype] = {
            "count": len(items),
            "avg_hybrid_advantage": sum(m.hybrid_advantage for m in items) / len(items),
            "avg_gain_over_anchor_only": sum(m.hybrid_gain_over_anchor_only for m in items) / len(items),
            "negative_advantage_count": sum(1 for m in items if m.hybrid_advantage < 0),
            "no_gain_over_anchor_count": sum(1 for m in items if m.hybrid_gain_over_anchor_only <= 0),
        }

    def examples(rows: Iterable[RetrievalMetrics]) -> list[dict]:
        return [simplify_metric(m) for m in list(rows)[:max_examples]]

    return {
        "counts": {
            "total_questions": len(metrics),
            "negative_advantage": len(negative_adv),
            "traversal_hurts": len(traversal_hurts),
            "traversal_discovered_but_ranked_out": len(traversal_discovered_but_ranked_out),
            "no_gain_over_anchor_only": len(no_gain_over_anchor),
        },
        "by_question_type": type_diagnostics,
        "examples": {
            "negative_advantage": examples(negative_adv),
            "traversal_hurts": examples(traversal_hurts),
            "traversal_discovered_but_ranked_out": examples(traversal_discovered_but_ranked_out),
            "no_gain_over_anchor_only": examples(no_gain_over_anchor),
        },
    }


def main() -> None:
    args = parse_args()
    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    top_k_anchors_space = parse_int_list(args.top_k_anchors)
    min_traversed_space = parse_int_list(args.min_traversed_items)
    semantic_space = parse_float_list(args.semantic_weight)
    hop_decay_space = parse_float_list(args.hop_decay)
    defines_space = parse_float_list(args.defines_weight)
    parent_space = parse_float_list(args.parent_weight)
    references_space = parse_float_list(args.references_weight)
    default_edge_space = parse_float_list(args.default_edge_weight)

    question_types = {q.strip() for q in args.question_types.split(",") if q.strip()}

    dataset = json.loads(args.dataset.read_text())
    subset = make_subset_dataset(dataset, args.max_questions, question_types)
    if not subset["questions"]:
        raise ValueError("No questions selected for tuning. Check filters.")

    with NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        json.dump(subset, tmp)

    grid = list(
        itertools.product(
            top_k_anchors_space,
            min_traversed_space,
            semantic_space,
            hop_decay_space,
            defines_space,
            parent_space,
            references_space,
            default_edge_space,
        )
    )
    if args.max_runs > 0:
        grid = grid[: args.max_runs]

    print("=" * 72)
    print("Hybrid Tuning Sweep")
    print("=" * 72)
    print(f"Questions: {len(subset['questions'])}")
    print(f"Contracts: {len(subset['contracts'])}")
    print(f"Runs: {len(grid)}")
    print(f"Objective: {args.objective} (latency penalty={args.latency_penalty})")
    print()

    all_runs = []
    best = None
    best_result = None

    for i, combo in enumerate(grid, start=1):
        (
            top_k_anchors,
            min_traversed_items,
            semantic_weight,
            hop_decay,
            defines_weight,
            parent_weight,
            references_weight,
            default_edge_weight,
        ) = combo

        config = {
            "top_k_anchors": top_k_anchors,
            "min_traversed_items": min_traversed_items,
            "max_hops": args.max_hops,
            "max_context_items": args.max_context_items,
            "semantic_weight": semantic_weight,
            "hop_decay": hop_decay,
            "default_edge_weight": default_edge_weight,
            "edge_type_weights": {
                "REFERENCES": references_weight,
                "DEFINES": defines_weight,
                "PARENT_OF": parent_weight,
                "CHILD_OF": parent_weight,
            },
        }

        bench = Benchmark(
            embedding_backend=args.embedding_backend,
            top_k_anchors=top_k_anchors,
            max_hops=args.max_hops,
            max_context_items=args.max_context_items,
            min_traversed_items=min_traversed_items,
            semantic_weight=semantic_weight,
            hop_decay=hop_decay,
            default_edge_weight=default_edge_weight,
            edge_type_weights=config["edge_type_weights"],
        )

        # Suppress verbose benchmark progress during sweep.
        with redirect_stdout(io.StringIO()):
            result = bench.run(tmp_path)

        summary = result.to_dict()["summary"]
        objective_score = score_result(summary, args.objective, args.latency_penalty)
        run_payload = {
            "config": config,
            "summary": summary,
            "objective_score": objective_score,
        }
        all_runs.append(run_payload)

        if best is None or objective_score > best["objective_score"]:
            best = run_payload
            best_result = result

        print(
            f"[{i}/{len(grid)}] score={objective_score:+.4f} "
            f"delta={summary['hybrid_improvement']:+.4f} "
            f"hybrid={summary['avg_hybrid_recall']:.4f} "
            f"vector={summary['avg_vector_recall']:.4f} "
            f"cfg=anchors:{top_k_anchors} traversed:{min_traversed_items} "
            f"sem:{semantic_weight} hop:{hop_decay} defines:{defines_weight} parent:{parent_weight}"
        )

    ranked_runs = sorted(all_runs, key=lambda r: r["objective_score"], reverse=True)
    failure_buckets = build_failure_buckets(best_result.per_question_metrics, args.failure_examples)

    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "objective": args.objective,
        "latency_penalty": args.latency_penalty,
        "embedding_backend": args.embedding_backend,
        "dataset": str(args.dataset),
        "subset": {
            "question_count": len(subset["questions"]),
            "contract_count": len(subset["contracts"]),
            "question_types_filter": sorted(question_types),
            "max_questions": args.max_questions,
        },
        "search_space": {
            "top_k_anchors": top_k_anchors_space,
            "min_traversed_items": min_traversed_space,
            "semantic_weight": semantic_space,
            "hop_decay": hop_decay_space,
            "defines_weight": defines_space,
            "parent_weight": parent_space,
            "references_weight": references_space,
            "default_edge_weight": default_edge_space,
        },
        "best_run": best,
        "top_runs": ranked_runs[:10],
        "failure_buckets_best_run": failure_buckets,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))

    print()
    print("=" * 72)
    print("Sweep Complete")
    print("=" * 72)
    print(f"Best objective score: {best['objective_score']:+.4f}")
    print(f"Best config: {json.dumps(best['config'])}")
    print(f"Best summary: {json.dumps(best['summary'])}")
    print(f"Report saved to: {args.output}")


if __name__ == "__main__":
    main()
