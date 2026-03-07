"""
Evaluation Metrics for EngramDB benchmarks.

Provides retrieval recall, answer accuracy (exact match + normalized
string matching), and aggregate metrics broken down by question type
and hop count.
"""

import re
import string
from dataclasses import dataclass, field


@dataclass
class EvaluationResult:
    """Results from evaluating a single question."""
    question_id: str
    question_type: str
    hop_count: int
    required_sections: list[str]
    retrieved_sections: list[str]
    retrieval_recall: float
    hop_coverage: float
    predicted_answer: str = ""
    ground_truth: str = ""
    answer_exact_match: bool = False
    answer_normalized_match: bool = False


@dataclass
class AggregateMetrics:
    """Aggregate evaluation metrics."""
    total_questions: int
    avg_retrieval_recall: float
    avg_hop_coverage: float
    exact_match_accuracy: float
    normalized_match_accuracy: float

    by_question_type: dict = field(default_factory=dict)
    by_hop_count: dict = field(default_factory=dict)

    per_question: list[EvaluationResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_questions": self.total_questions,
            "avg_retrieval_recall": round(self.avg_retrieval_recall, 4),
            "avg_hop_coverage": round(self.avg_hop_coverage, 4),
            "exact_match_accuracy": round(self.exact_match_accuracy, 4),
            "normalized_match_accuracy": round(self.normalized_match_accuracy, 4),
            "by_question_type": self.by_question_type,
            "by_hop_count": {str(k): v for k, v in self.by_hop_count.items()},
        }


class Evaluator:
    """Evaluates retrieval and answer quality against ground truth."""

    def evaluate_retrieval(
        self,
        required_sections: list[str],
        retrieved_sections: list[str],
    ) -> tuple[float, float]:
        """
        Compute retrieval recall and hop coverage.

        Returns (recall, hop_coverage).
        """
        recall = _calculate_recall(required_sections, retrieved_sections)
        hop_coverage = _calculate_hop_coverage(required_sections, retrieved_sections)
        return recall, hop_coverage

    def evaluate_answer(
        self,
        predicted: str,
        ground_truth: str,
    ) -> tuple[bool, bool]:
        """
        Check if predicted answer matches ground truth.

        Returns (exact_match, normalized_match).
        """
        exact = predicted.strip() == ground_truth.strip()
        normalized = _normalize_text(predicted) == _normalize_text(ground_truth)
        return exact, normalized

    def evaluate_question(
        self,
        question_id: str,
        question_type: str,
        hop_count: int,
        required_sections: list[str],
        retrieved_sections: list[str],
        predicted_answer: str = "",
        ground_truth: str = "",
    ) -> EvaluationResult:
        """Evaluate a single question across all metrics."""
        recall, hop_cov = self.evaluate_retrieval(required_sections, retrieved_sections)
        exact, normalized = self.evaluate_answer(predicted_answer, ground_truth)

        return EvaluationResult(
            question_id=question_id,
            question_type=question_type,
            hop_count=hop_count,
            required_sections=required_sections,
            retrieved_sections=retrieved_sections,
            retrieval_recall=recall,
            hop_coverage=hop_cov,
            predicted_answer=predicted_answer,
            ground_truth=ground_truth,
            answer_exact_match=exact,
            answer_normalized_match=normalized,
        )

    def evaluate_dataset(
        self,
        results: list[EvaluationResult],
    ) -> AggregateMetrics:
        """Aggregate per-question results into summary metrics."""
        if not results:
            return AggregateMetrics(
                total_questions=0,
                avg_retrieval_recall=0.0,
                avg_hop_coverage=0.0,
                exact_match_accuracy=0.0,
                normalized_match_accuracy=0.0,
            )

        n = len(results)
        avg_recall = sum(r.retrieval_recall for r in results) / n
        avg_hop = sum(r.hop_coverage for r in results) / n
        exact_acc = sum(1 for r in results if r.answer_exact_match) / n
        norm_acc = sum(1 for r in results if r.answer_normalized_match) / n

        # Group by question type
        by_type: dict[str, dict] = {}
        for r in results:
            if r.question_type not in by_type:
                by_type[r.question_type] = {"recalls": [], "hop_covs": [], "count": 0}
            by_type[r.question_type]["recalls"].append(r.retrieval_recall)
            by_type[r.question_type]["hop_covs"].append(r.hop_coverage)
            by_type[r.question_type]["count"] += 1

        by_type_summary = {}
        for qtype, data in by_type.items():
            by_type_summary[qtype] = {
                "count": data["count"],
                "avg_recall": round(sum(data["recalls"]) / data["count"], 4),
                "avg_hop_coverage": round(sum(data["hop_covs"]) / data["count"], 4),
            }

        # Group by hop count
        by_hops: dict[int, dict] = {}
        for r in results:
            if r.hop_count not in by_hops:
                by_hops[r.hop_count] = {"recalls": [], "count": 0}
            by_hops[r.hop_count]["recalls"].append(r.retrieval_recall)
            by_hops[r.hop_count]["count"] += 1

        by_hops_summary = {}
        for hops, data in by_hops.items():
            by_hops_summary[hops] = {
                "count": data["count"],
                "avg_recall": round(sum(data["recalls"]) / data["count"], 4),
            }

        return AggregateMetrics(
            total_questions=n,
            avg_retrieval_recall=avg_recall,
            avg_hop_coverage=avg_hop,
            exact_match_accuracy=exact_acc,
            normalized_match_accuracy=norm_acc,
            by_question_type=by_type_summary,
            by_hop_count=by_hops_summary,
            per_question=results,
        )


def _calculate_recall(required: list[str], retrieved: list[str]) -> float:
    """Calculate retrieval recall (fraction of required sections retrieved)."""
    if not required:
        return 1.0
    required_norm = set(str(s).lower().strip() for s in required)
    retrieved_norm = set(str(s).lower().strip() for s in retrieved)
    matches = len(required_norm & retrieved_norm)
    return matches / len(required_norm)


def _calculate_hop_coverage(chain: list[str], retrieved: list[str]) -> float:
    """Calculate fraction of reasoning chain covered by retrieved sections."""
    if not chain:
        return 1.0
    chain_norm = [str(s).lower().strip() for s in chain]
    retrieved_norm = set(str(s).lower().strip() for s in retrieved)
    covered = sum(1 for s in chain_norm if s in retrieved_norm)
    return covered / len(chain_norm)


def _normalize_text(text: str) -> str:
    """Normalize text for fuzzy answer matching."""
    text = text.lower().strip()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text
