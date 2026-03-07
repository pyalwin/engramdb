"""
Statistical significance testing for EngramDB benchmark results.

Provides paired t-tests, Wilcoxon signed-rank tests, bootstrap
confidence intervals, and effect size (Cohen's d) for comparing
retrieval systems.
"""

import math
import random
from dataclasses import dataclass


@dataclass
class PairedTestResult:
    """Result of a paired statistical test between two systems."""
    system_a: str
    system_b: str
    metric: str
    n: int
    mean_a: float
    mean_b: float
    mean_diff: float
    std_diff: float
    t_statistic: float
    p_value: float
    cohens_d: float
    ci_lower: float
    ci_upper: float
    significant: bool  # p < 0.05

    def to_dict(self) -> dict:
        return {
            "system_a": self.system_a,
            "system_b": self.system_b,
            "metric": self.metric,
            "n": self.n,
            "mean_a": round(self.mean_a, 4),
            "mean_b": round(self.mean_b, 4),
            "mean_diff": round(self.mean_diff, 4),
            "t_statistic": round(self.t_statistic, 4),
            "p_value": round(self.p_value, 6),
            "cohens_d": round(self.cohens_d, 4),
            "ci_95_lower": round(self.ci_lower, 4),
            "ci_95_upper": round(self.ci_upper, 4),
            "significant_p05": self.significant,
        }


@dataclass
class BootstrapCI:
    """Bootstrap confidence interval."""
    mean: float
    ci_lower: float
    ci_upper: float
    n_bootstrap: int


def paired_t_test(
    scores_a: list[float],
    scores_b: list[float],
    system_a: str = "system_a",
    system_b: str = "system_b",
    metric: str = "recall",
) -> PairedTestResult:
    """
    Paired t-test comparing two systems on matched samples.

    Uses the standard paired t-test formula without scipy dependency.
    """
    n = len(scores_a)
    if n != len(scores_b):
        raise ValueError(f"Score lists must have equal length: {n} vs {len(scores_b)}")
    if n < 2:
        raise ValueError(f"Need at least 2 paired samples, got {n}")

    diffs = [a - b for a, b in zip(scores_a, scores_b)]
    mean_diff = sum(diffs) / n
    var_diff = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
    std_diff = math.sqrt(var_diff) if var_diff > 0 else 0.0

    if std_diff == 0:
        t_stat = float("inf") if mean_diff != 0 else 0.0
        p_value = 0.0 if mean_diff != 0 else 1.0
    else:
        se = std_diff / math.sqrt(n)
        t_stat = mean_diff / se
        # Two-tailed p-value approximation using t-distribution
        p_value = _t_distribution_p_value(abs(t_stat), n - 1)

    # Cohen's d for paired samples
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0.0

    # 95% CI for mean difference
    t_crit = _t_critical_95(n - 1)
    se = std_diff / math.sqrt(n) if std_diff > 0 else 0.0
    ci_lower = mean_diff - t_crit * se
    ci_upper = mean_diff + t_crit * se

    mean_a = sum(scores_a) / n
    mean_b = sum(scores_b) / n

    return PairedTestResult(
        system_a=system_a,
        system_b=system_b,
        metric=metric,
        n=n,
        mean_a=mean_a,
        mean_b=mean_b,
        mean_diff=mean_diff,
        std_diff=std_diff,
        t_statistic=t_stat,
        p_value=p_value,
        cohens_d=cohens_d,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        significant=p_value < 0.05,
    )


def wilcoxon_signed_rank(
    scores_a: list[float],
    scores_b: list[float],
    system_a: str = "system_a",
    system_b: str = "system_b",
    metric: str = "recall",
) -> PairedTestResult:
    """
    Wilcoxon signed-rank test (non-parametric alternative to paired t-test).

    Implements the test without scipy dependency.
    """
    n = len(scores_a)
    if n != len(scores_b):
        raise ValueError("Score lists must have equal length")

    diffs = [a - b for a, b in zip(scores_a, scores_b)]

    # Remove zero differences
    nonzero = [(abs(d), d) for d in diffs if d != 0]
    nr = len(nonzero)

    if nr == 0:
        mean_a = sum(scores_a) / n
        mean_b = sum(scores_b) / n
        return PairedTestResult(
            system_a=system_a, system_b=system_b, metric=metric,
            n=n, mean_a=mean_a, mean_b=mean_b, mean_diff=0.0,
            std_diff=0.0, t_statistic=0.0, p_value=1.0,
            cohens_d=0.0, ci_lower=0.0, ci_upper=0.0, significant=False,
        )

    # Rank by absolute value
    nonzero.sort(key=lambda x: x[0])

    # Assign ranks (handle ties with average rank)
    ranks: list[tuple[float, float]] = []  # (rank, signed_diff)
    i = 0
    while i < nr:
        j = i
        while j < nr and nonzero[j][0] == nonzero[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks.append((avg_rank, nonzero[k][1]))
        i = j

    w_plus = sum(r for r, d in ranks if d > 0)
    w_minus = sum(r for r, d in ranks if d < 0)
    w = min(w_plus, w_minus)

    # Normal approximation for p-value (valid for nr >= 10)
    mean_w = nr * (nr + 1) / 4.0
    var_w = nr * (nr + 1) * (2 * nr + 1) / 24.0
    std_w = math.sqrt(var_w) if var_w > 0 else 1.0
    z = (w - mean_w) / std_w if std_w > 0 else 0.0
    p_value = 2.0 * _normal_cdf(-abs(z))

    mean_a = sum(scores_a) / n
    mean_b = sum(scores_b) / n
    mean_diff = mean_a - mean_b
    var_diff = sum((d - mean_diff) ** 2 for d in diffs) / max(n - 1, 1)
    std_diff = math.sqrt(var_diff)
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0.0

    # Bootstrap CI for mean difference
    bci = bootstrap_ci([a - b for a, b in zip(scores_a, scores_b)])

    return PairedTestResult(
        system_a=system_a, system_b=system_b, metric=metric,
        n=n, mean_a=mean_a, mean_b=mean_b, mean_diff=mean_diff,
        std_diff=std_diff, t_statistic=z, p_value=p_value,
        cohens_d=cohens_d, ci_lower=bci.ci_lower, ci_upper=bci.ci_upper,
        significant=p_value < 0.05,
    )


def bootstrap_ci(
    diffs: list[float],
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrapCI:
    """Compute bootstrap confidence interval for the mean of differences."""
    n = len(diffs)
    if n == 0:
        return BootstrapCI(mean=0.0, ci_lower=0.0, ci_upper=0.0, n_bootstrap=n_bootstrap)

    rng = random.Random(seed)
    means = []
    for _ in range(n_bootstrap):
        sample = [rng.choice(diffs) for _ in range(n)]
        means.append(sum(sample) / n)

    means.sort()
    alpha = 1 - confidence
    lo_idx = int(math.floor(alpha / 2 * n_bootstrap))
    hi_idx = int(math.ceil((1 - alpha / 2) * n_bootstrap)) - 1
    lo_idx = max(0, min(lo_idx, n_bootstrap - 1))
    hi_idx = max(0, min(hi_idx, n_bootstrap - 1))

    return BootstrapCI(
        mean=sum(diffs) / n,
        ci_lower=means[lo_idx],
        ci_upper=means[hi_idx],
        n_bootstrap=n_bootstrap,
    )


def compute_all_tests(
    results_by_system: dict[str, list[float]],
    reference_system: str = "hybrid",
    metric: str = "recall",
) -> list[PairedTestResult]:
    """
    Run paired t-test and Wilcoxon test for each system vs the reference.

    Args:
        results_by_system: system_name -> per-question scores
        reference_system: the system to compare against
        metric: name of the metric being compared

    Returns:
        List of PairedTestResult (one per comparison)
    """
    ref_scores = results_by_system.get(reference_system)
    if ref_scores is None:
        raise ValueError(f"Reference system '{reference_system}' not found")

    tests = []
    for system_name, scores in results_by_system.items():
        if system_name == reference_system:
            continue
        tests.append(paired_t_test(
            ref_scores, scores,
            system_a=reference_system,
            system_b=system_name,
            metric=metric,
        ))
    return tests


def format_significance_table(tests: list[PairedTestResult]) -> str:
    """Format test results as a readable table."""
    lines = []
    lines.append(f"{'Comparison':<35} {'Δ':>7} {'t/z':>8} {'p':>10} {'d':>7} {'95% CI':>20} {'Sig?':>5}")
    lines.append("-" * 95)
    for t in tests:
        comp = f"{t.system_a} vs {t.system_b}"
        ci = f"[{t.ci_lower:+.3f}, {t.ci_upper:+.3f}]"
        sig = "***" if t.p_value < 0.001 else ("**" if t.p_value < 0.01 else ("*" if t.p_value < 0.05 else "ns"))
        lines.append(f"{comp:<35} {t.mean_diff:>+.4f} {t.t_statistic:>8.3f} {t.p_value:>10.6f} {t.cohens_d:>7.3f} {ci:>20} {sig:>5}")
    return "\n".join(lines)


# --- Internal math utilities (no scipy dependency) ---

def _normal_cdf(x: float) -> float:
    """Approximation of the standard normal CDF (Abramowitz & Stegun)."""
    if x < -8:
        return 0.0
    if x > 8:
        return 1.0
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    d = 0.3989422804014327  # 1/sqrt(2*pi)
    p = d * math.exp(-x * x / 2.0) * (
        t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    )
    return 1.0 - p if x > 0 else p


def _t_distribution_p_value(t_abs: float, df: int) -> float:
    """
    Approximate two-tailed p-value for t-distribution.

    Uses the normal approximation for df >= 30, and a rough
    incomplete beta approximation for smaller df.
    """
    if df >= 30:
        return 2.0 * (1.0 - _normal_cdf(t_abs))

    # For smaller df, use the relationship: p = I_x(df/2, 1/2)
    # where x = df / (df + t^2)
    x = df / (df + t_abs * t_abs)
    a = df / 2.0
    b = 0.5
    # Regularized incomplete beta via continued fraction (rough)
    p = _regularized_incomplete_beta(x, a, b)
    return p


def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """Rough approximation of regularized incomplete beta I_x(a, b)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0

    # Use the Lentz continued fraction method (limited iterations)
    ln_beta = _ln_beta(a, b)
    front = math.exp(a * math.log(x) + b * math.log(1.0 - x) - ln_beta) / a

    # Simple continued fraction
    f = 1.0
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    f = d

    for m in range(1, 100):
        # Even step
        numerator = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        d = 1.0 + numerator * d
        if abs(d) < 1e-30:
            d = 1e-30
        d = 1.0 / d
        c = 1.0 + numerator / c
        if abs(c) < 1e-30:
            c = 1e-30
        f *= c * d

        # Odd step
        numerator = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + numerator * d
        if abs(d) < 1e-30:
            d = 1e-30
        d = 1.0 / d
        c = 1.0 + numerator / c
        if abs(c) < 1e-30:
            c = 1e-30
        delta = c * d
        f *= delta

        if abs(delta - 1.0) < 1e-8:
            break

    return front * f


def _ln_beta(a: float, b: float) -> float:
    """Log of the beta function using lgamma."""
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _t_critical_95(df: int) -> float:
    """Approximate t critical value for 95% CI (two-tailed)."""
    # Common values for small df
    table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042,
        40: 2.021, 60: 2.000, 120: 1.980,
    }
    if df in table:
        return table[df]
    # Interpolate or use normal approximation
    if df > 120:
        return 1.96
    # Find bounding entries
    keys = sorted(table.keys())
    for i in range(len(keys) - 1):
        if keys[i] <= df <= keys[i + 1]:
            lo, hi = keys[i], keys[i + 1]
            frac = (df - lo) / (hi - lo)
            return table[lo] + frac * (table[hi] - table[lo])
    return 1.96
