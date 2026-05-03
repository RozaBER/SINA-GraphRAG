from __future__ import annotations

from collections import defaultdict
from statistics import mean, stdev


SUMMARY_BY_RUN_FIELDS = [
    "run_id",
    "seed",
    "test_ratio",
    "negative_strategy",
    "negatives_per_source",
    "k",
    "algorithm",
    "sources_evaluated",
    "precision_at_k",
    "hit_rate_at_k",
    "mrr",
    "positive_candidates",
    "negative_candidates",
    "runtime_seconds",
]

RUN_FIELDS = [
    "run_id",
    "seed",
    "test_ratio",
    "negative_strategy",
    "negatives_per_source",
    "sources_evaluated",
    "positive_candidates",
    "negative_candidates",
    "candidate_build_seconds",
    "total_runtime_seconds",
]

AGGREGATE_FIELDS = [
    "algorithm",
    "k",
    "negative_strategy",
    "negatives_per_source",
    "test_ratio",
    "runs",
    "mean_precision_at_k",
    "std_precision_at_k",
    "mean_hit_rate_at_k",
    "std_hit_rate_at_k",
    "mean_mrr",
    "std_mrr",
    "mean_runtime_seconds",
    "std_runtime_seconds",
]


def aggregate_summary_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["algorithm"]),
            str(row["k"]),
            str(row["negative_strategy"]),
            str(row["negatives_per_source"]),
            str(row["test_ratio"]),
        )
        grouped[key].append(row)

    aggregated: list[dict[str, object]] = []
    for key, group_rows in sorted(grouped.items()):
        algorithm, k, negative_strategy, negatives_per_source, test_ratio = key
        precisions = [float(row["precision_at_k"]) for row in group_rows]
        hit_rates = [float(row["hit_rate_at_k"]) for row in group_rows]
        mrr_values = [float(row["mrr"]) for row in group_rows]
        runtimes = [float(row["runtime_seconds"]) for row in group_rows]

        aggregated.append(
            {
                "algorithm": algorithm,
                "k": int(k),
                "negative_strategy": negative_strategy,
                "negatives_per_source": int(negatives_per_source),
                "test_ratio": float(test_ratio),
                "runs": len(group_rows),
                "mean_precision_at_k": _round(mean(precisions)),
                "std_precision_at_k": _round(_std(precisions)),
                "mean_hit_rate_at_k": _round(mean(hit_rates)),
                "std_hit_rate_at_k": _round(_std(hit_rates)),
                "mean_mrr": _round(mean(mrr_values)),
                "std_mrr": _round(_std(mrr_values)),
                "mean_runtime_seconds": _round(mean(runtimes)),
                "std_runtime_seconds": _round(_std(runtimes)),
            }
        )

    return aggregated


def _std(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def _round(value: float) -> float:
    return round(value, 6)
