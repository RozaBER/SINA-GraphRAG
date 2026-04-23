from __future__ import annotations

from pathlib import Path

from .data import write_rows_csv
from .scorers import BaseScorer


SUMMARY_FIELDS = [
    "algorithm",
    "k",
    "sources_evaluated",
    "precision_at_k",
    "hit_rate_at_k",
    "mrr",
    "positive_candidates",
    "negative_candidates",
]

RANKING_FIELDS = [
    "source_node",
    "target_node",
    "label",
    "score",
    "rank",
    "algorithm",
]


def evaluate_candidate_sets(
    *,
    candidate_sets: dict[int, list[tuple[int, int]]],
    scorers: list[BaseScorer],
    k: int,
) -> tuple[list[dict[str, object]], dict[str, list[dict[str, object]]]]:
    summary_rows: list[dict[str, object]] = []
    ranking_rows_by_algorithm: dict[str, list[dict[str, object]]] = {}

    for scorer in scorers:
        algorithm_rows: list[dict[str, object]] = []
        precision_values: list[float] = []
        hit_rate_values: list[float] = []
        mrr_values: list[float] = []
        positive_candidates = 0
        negative_candidates = 0

        for source in sorted(candidate_sets):
            labeled_candidates = candidate_sets[source]
            candidate_nodes = [target for target, _ in labeled_candidates]
            labels_by_target = {target: label for target, label in labeled_candidates}
            positive_candidates += sum(labels_by_target.values())
            negative_candidates += len(labels_by_target) - sum(labels_by_target.values())

            scores = scorer.score(source, candidate_nodes)
            ranked_targets = sorted(
                candidate_nodes,
                key=lambda target: (-scores.get(target, 0.0), target),
            )

            for rank, target in enumerate(ranked_targets, start=1):
                algorithm_rows.append(
                    {
                        "source_node": source,
                        "target_node": target,
                        "label": labels_by_target[target],
                        "score": round(scores.get(target, 0.0), 8),
                        "rank": rank,
                        "algorithm": scorer.name,
                    }
                )

            precision, hit_rate, reciprocal_rank = compute_metrics_for_source(
                ranked_targets=ranked_targets,
                labels_by_target=labels_by_target,
                k=k,
            )
            precision_values.append(precision)
            hit_rate_values.append(hit_rate)
            mrr_values.append(reciprocal_rank)

        source_count = len(candidate_sets)
        summary_rows.append(
            {
                "algorithm": scorer.name,
                "k": k,
                "sources_evaluated": source_count,
                "precision_at_k": round(sum(precision_values) / source_count, 6) if source_count else 0.0,
                "hit_rate_at_k": round(sum(hit_rate_values) / source_count, 6) if source_count else 0.0,
                "mrr": round(sum(mrr_values) / source_count, 6) if source_count else 0.0,
                "positive_candidates": positive_candidates,
                "negative_candidates": negative_candidates,
            }
        )
        ranking_rows_by_algorithm[scorer.name] = algorithm_rows

    return summary_rows, ranking_rows_by_algorithm


def compute_metrics_for_source(
    *,
    ranked_targets: list[int],
    labels_by_target: dict[int, int],
    k: int,
) -> tuple[float, float, float]:
    top_k_targets = ranked_targets[:k]
    hits_at_k = sum(labels_by_target[target] for target in top_k_targets)
    precision_at_k = hits_at_k / k if k else 0.0
    hit_rate_at_k = 1.0 if hits_at_k > 0 else 0.0

    reciprocal_rank = 0.0
    for rank, target in enumerate(ranked_targets, start=1):
        if labels_by_target[target] == 1:
            reciprocal_rank = 1.0 / rank
            break

    return precision_at_k, hit_rate_at_k, reciprocal_rank


def write_evaluation_outputs(
    *,
    output_dir: str | Path,
    summary_rows: list[dict[str, object]],
    ranking_rows_by_algorithm: dict[str, list[dict[str, object]]],
) -> None:
    base_dir = Path(output_dir)
    write_rows_csv(base_dir / "metrics" / "summary.csv", SUMMARY_FIELDS, summary_rows)

    ranking_dir = base_dir / "rankings"
    for algorithm, rows in ranking_rows_by_algorithm.items():
        write_rows_csv(ranking_dir / f"{algorithm}.csv", RANKING_FIELDS, rows)


def plot_algorithm_comparison(summary_rows: list[dict[str, object]], output_path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    palette = ["#1c2b7c", "#42559a", "#6e7eb7", "#9da8d4", "#cfd3f1"]
    algorithms = [str(row["algorithm"]) for row in summary_rows]
    precision_values = [float(row["precision_at_k"]) for row in summary_rows]
    hit_rate_values = [float(row["hit_rate_at_k"]) for row in summary_rows]
    mrr_values = [float(row["mrr"]) for row in summary_rows]

    positions = list(range(len(algorithms)))
    bar_width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar([pos - bar_width for pos in positions], precision_values, width=bar_width, label="Precision@K", color=palette[0])
    plt.bar(positions, hit_rate_values, width=bar_width, label="HitRate@K", color=palette[1])
    plt.bar([pos + bar_width for pos in positions], mrr_values, width=bar_width, label="MRR", color=palette[2])

    plt.xticks(positions, algorithms, rotation=20)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("GraphRAG Retrieval Algorithm Comparison")
    plt.legend()
    plt.tight_layout()

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=180)
    plt.close()
