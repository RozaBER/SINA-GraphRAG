from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PALETTE = ["#1c2b7c", "#42559a", "#6e7eb7", "#9da8d4", "#cfd3f1"]


def generate_horizontal_figures(output_dir: str | Path) -> list[Path]:
    base_dir = Path(output_dir)
    rows = _read_csv_rows(base_dir / "summary_mean_std.csv")
    if not rows:
        return []

    figure_dir = base_dir / "figures"
    generated = [
        _plot_algorithm_mean_std(rows, figure_dir),
        _plot_precision_at_k(rows, figure_dir),
        _plot_negative_strategy_compare(rows, figure_dir),
        _plot_runtime_quality(rows, figure_dir),
    ]
    return [path for path in generated if path is not None]


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _save(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def _preferred_rows(rows: list[dict[str, str]], *, k: int | None = None, strategy: str | None = None) -> list[dict[str, str]]:
    selected = rows
    if k is not None:
        k_rows = [row for row in selected if int(row["k"]) == k]
        selected = k_rows or selected
    if strategy is not None:
        strategy_rows = [row for row in selected if row["negative_strategy"] == strategy]
        selected = strategy_rows or selected
    return selected


def _plot_algorithm_mean_std(rows: list[dict[str, str]], figure_dir: Path) -> Path | None:
    selected = _preferred_rows(rows, k=10, strategy="random")
    if not selected:
        return None

    algorithms = [row["algorithm"] for row in selected]
    values = [float(row["mean_precision_at_k"]) for row in selected]
    errors = [float(row["std_precision_at_k"]) for row in selected]

    plt.figure(figsize=(12, 6))
    plt.bar(algorithms, values, yerr=errors, color=PALETTE[0], capsize=4)
    plt.xticks(rotation=25, ha="right")
    plt.ylim(0, max(values + errors) * 1.18 if values else 1)
    plt.ylabel("Mean Precision@K")
    plt.title("Algorithm Comparison with Seed Variance")
    return _save(figure_dir / "algorithm_mean_std.png")


def _plot_precision_at_k(rows: list[dict[str, str]], figure_dir: Path) -> Path | None:
    selected = _preferred_rows(rows, strategy="random")
    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in selected:
        grouped[row["algorithm"]].append((int(row["k"]), float(row["mean_precision_at_k"])))
    if not grouped:
        return None

    plt.figure(figsize=(11, 6))
    for index, (algorithm, points) in enumerate(sorted(grouped.items())):
        ordered = sorted(points)
        plt.plot(
            [point[0] for point in ordered],
            [point[1] for point in ordered],
            marker="o",
            linewidth=2,
            label=algorithm,
            color=PALETTE[index % len(PALETTE)],
        )

    plt.xlabel("K")
    plt.ylabel("Mean Precision@K")
    plt.title("Precision@K by Algorithm")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    return _save(figure_dir / "precision_at_k_by_algorithm.png")


def _plot_negative_strategy_compare(rows: list[dict[str, str]], figure_dir: Path) -> Path | None:
    selected = _preferred_rows(rows, k=10)
    algorithms = sorted({row["algorithm"] for row in selected})
    strategies = sorted({row["negative_strategy"] for row in selected})
    if not algorithms or not strategies:
        return None

    lookup = {
        (row["algorithm"], row["negative_strategy"]): float(row["mean_precision_at_k"])
        for row in selected
    }
    positions = list(range(len(algorithms)))
    width = min(0.8 / len(strategies), 0.25)

    plt.figure(figsize=(13, 6))
    for index, strategy in enumerate(strategies):
        shifted = [pos + (index - (len(strategies) - 1) / 2) * width for pos in positions]
        values = [lookup.get((algorithm, strategy), 0.0) for algorithm in algorithms]
        plt.bar(shifted, values, width=width, label=strategy, color=PALETTE[index % len(PALETTE)])

    plt.xticks(positions, algorithms, rotation=25, ha="right")
    plt.ylabel("Mean Precision@K")
    plt.title("Negative Sampling Strategy Comparison")
    plt.legend()
    return _save(figure_dir / "negative_strategy_compare.png")


def _plot_runtime_quality(rows: list[dict[str, str]], figure_dir: Path) -> Path | None:
    selected = _preferred_rows(rows, k=10, strategy="random")
    if not selected:
        return None

    plt.figure(figsize=(9, 6))
    for index, row in enumerate(selected):
        runtime = float(row["mean_runtime_seconds"])
        precision = float(row["mean_precision_at_k"])
        plt.scatter(runtime, precision, s=90, color=PALETTE[index % len(PALETTE)])
        plt.text(runtime, precision, row["algorithm"], fontsize=8, ha="left", va="bottom")

    plt.xlabel("Mean Runtime Seconds")
    plt.ylabel("Mean Precision@K")
    plt.title("Runtime and Retrieval Quality Tradeoff")
    plt.grid(alpha=0.25)
    return _save(figure_dir / "runtime_quality_tradeoff.png")
