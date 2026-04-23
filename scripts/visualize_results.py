from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx


METRIC_LABELS = {
    "precision_at_k": "Precision@K",
    "hit_rate_at_k": "HitRate@K",
    "mrr": "MRR",
}

PALETTE = ["#1c2b7c", "#42559a", "#6e7eb7", "#9da8d4", "#cfd3f1"]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_edges(path: Path) -> list[tuple[int, int]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [(int(row["source_node"]), int(row["target_node"])) for row in reader]


def save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_metrics_summary(summary_path: Path, figure_dir: Path) -> Path:
    rows = read_csv_rows(summary_path)
    algorithms = [row["algorithm"] for row in rows]
    metrics = ["precision_at_k", "hit_rate_at_k", "mrr"]
    positions = list(range(len(algorithms)))
    bar_width = 0.24

    plt.figure(figsize=(11, 6))
    for offset, metric in enumerate(metrics):
        values = [float(row[metric]) for row in rows]
        shifted = [pos + (offset - 1) * bar_width for pos in positions]
        plt.bar(shifted, values, width=bar_width, label=METRIC_LABELS[metric], color=PALETTE[offset])

    plt.xticks(positions, algorithms, rotation=18)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("GraphRAG Retrieval Metrics by Algorithm")
    plt.legend()

    output_path = figure_dir / "metrics_summary.png"
    save_figure(output_path)
    return output_path


def plot_split_summary(meta_path: Path, figure_dir: Path) -> Path:
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    labels = ["Train edges", "Test edges"]
    values = [metadata["train_edge_count"], metadata["test_edge_count"]]
    colors = [PALETTE[0], PALETTE[3]]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=colors)
    plt.ylabel("Edges")
    plt.title("Train/Test Edge Split")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:,}", ha="center", va="bottom")

    output_path = figure_dir / "train_test_split.png"
    save_figure(output_path)
    return output_path


def compute_precision_curve(ranking_path: Path, max_k: int) -> tuple[str, list[float]]:
    totals = [0.0 for _ in range(max_k)]
    source_count = 0
    current_source: int | None = None
    current_labels: list[int] = []
    algorithm = ranking_path.stem

    def finalize_source() -> None:
        nonlocal source_count
        if not current_labels:
            return
        source_count += 1
        hits = 0
        for index in range(max_k):
            if index < len(current_labels):
                hits += current_labels[index]
            totals[index] += hits / (index + 1)

    with ranking_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source = int(row["source_node"])
            rank = int(row["rank"])
            if current_source is None:
                current_source = source
            elif source != current_source:
                finalize_source()
                current_source = source
                current_labels = []

            if rank <= max_k:
                current_labels.append(int(row["label"]))

    finalize_source()
    if source_count == 0:
        return algorithm, [0.0 for _ in range(max_k)]
    return algorithm, [value / source_count for value in totals]


def plot_precision_curves(ranking_dir: Path, figure_dir: Path, max_k: int) -> Path:
    plt.figure(figsize=(10, 6))
    for index, ranking_path in enumerate(sorted(ranking_dir.glob("*.csv"))):
        algorithm, values = compute_precision_curve(ranking_path, max_k=max_k)
        plt.plot(
            range(1, max_k + 1),
            values,
            marker="o",
            linewidth=2,
            label=algorithm,
            color=PALETTE[index % len(PALETTE)],
        )

    plt.xlabel("K")
    plt.ylabel("Macro Precision@K")
    plt.title("Precision@K Curve")
    plt.ylim(bottom=0)
    plt.grid(alpha=0.25)
    plt.legend()

    output_path = figure_dir / "precision_at_k_curve.png"
    save_figure(output_path)
    return output_path


def compute_hit_rate_curve(ranking_path: Path, max_k: int) -> tuple[str, list[float]]:
    totals = [0.0 for _ in range(max_k)]
    source_count = 0
    current_source: int | None = None
    current_labels: list[int] = []
    algorithm = ranking_path.stem

    def finalize_source() -> None:
        nonlocal source_count
        if not current_labels:
            return
        source_count += 1
        hits = 0
        for index in range(max_k):
            if index < len(current_labels):
                hits += current_labels[index]
            totals[index] += 1.0 if hits > 0 else 0.0

    with ranking_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source = int(row["source_node"])
            rank = int(row["rank"])
            if current_source is None:
                current_source = source
            elif source != current_source:
                finalize_source()
                current_source = source
                current_labels = []

            if rank <= max_k:
                current_labels.append(int(row["label"]))

    finalize_source()
    if source_count == 0:
        return algorithm, [0.0 for _ in range(max_k)]
    return algorithm, [value / source_count for value in totals]


def plot_hit_rate_curves(ranking_dir: Path, figure_dir: Path, max_k: int) -> Path:
    plt.figure(figsize=(10, 6))
    for index, ranking_path in enumerate(sorted(ranking_dir.glob("*.csv"))):
        algorithm, values = compute_hit_rate_curve(ranking_path, max_k=max_k)
        plt.plot(
            range(1, max_k + 1),
            values,
            marker="o",
            linewidth=2,
            label=algorithm,
            color=PALETTE[index % len(PALETTE)],
        )

    plt.xlabel("K")
    plt.ylabel("HitRate@K")
    plt.title("HitRate@K Curve")
    plt.ylim(0, 1)
    plt.grid(alpha=0.25)
    plt.legend()

    output_path = figure_dir / "hit_rate_at_k_curve.png"
    save_figure(output_path)
    return output_path


def plot_precision_heatmap(ranking_dir: Path, figure_dir: Path, max_k: int) -> Path:
    algorithms: list[str] = []
    curves: list[list[float]] = []
    for ranking_path in sorted(ranking_dir.glob("*.csv")):
        algorithm, values = compute_precision_curve(ranking_path, max_k=max_k)
        algorithms.append(algorithm)
        curves.append(values)

    plt.figure(figsize=(12, 5))
    plt.imshow(curves, aspect="auto", cmap=build_palette_colormap(), vmin=0)
    plt.colorbar(label="Macro Precision@K")
    plt.xticks(range(max_k), range(1, max_k + 1))
    plt.yticks(range(len(algorithms)), algorithms)
    plt.xlabel("K")
    plt.title("Precision@K Heatmap")

    output_path = figure_dir / "precision_at_k_heatmap.png"
    save_figure(output_path)
    return output_path


def compute_first_hit_cdf(ranking_path: Path, cutoffs: list[int]) -> tuple[str, list[float]]:
    first_ranks: list[int] = []
    current_source: int | None = None
    first_positive_rank: int | None = None
    algorithm = ranking_path.stem

    def finalize_source() -> None:
        if first_positive_rank is not None:
            first_ranks.append(first_positive_rank)

    with ranking_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source = int(row["source_node"])
            if current_source is None:
                current_source = source
            elif source != current_source:
                finalize_source()
                current_source = source
                first_positive_rank = None

            if first_positive_rank is None and int(row["label"]) == 1:
                first_positive_rank = int(row["rank"])

    finalize_source()
    if not first_ranks:
        return algorithm, [0.0 for _ in cutoffs]

    return algorithm, [
        sum(1 for rank in first_ranks if rank <= cutoff) / len(first_ranks)
        for cutoff in cutoffs
    ]


def plot_first_hit_cdf(ranking_dir: Path, figure_dir: Path) -> Path:
    cutoffs = [1, 3, 5, 10, 20, 50, 100]

    plt.figure(figsize=(10, 6))
    for index, ranking_path in enumerate(sorted(ranking_dir.glob("*.csv"))):
        algorithm, values = compute_first_hit_cdf(ranking_path, cutoffs)
        plt.plot(
            cutoffs,
            values,
            marker="o",
            linewidth=2,
            label=algorithm,
            color=PALETTE[index % len(PALETTE)],
        )

    plt.xlabel("Rank cutoff")
    plt.ylabel("Fraction of sources with a hit")
    plt.title("First Relevant Target Rank CDF")
    plt.ylim(0, 1)
    plt.grid(alpha=0.25)
    plt.legend()

    output_path = figure_dir / "first_hit_rank_cdf.png"
    save_figure(output_path)
    return output_path


def plot_score_distributions(ranking_dir: Path, figure_dir: Path) -> Path:
    ranking_paths = sorted(ranking_dir.glob("*.csv"))
    columns = 2
    rows = (len(ranking_paths) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(13, 4.5 * rows))
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for index, ranking_path in enumerate(ranking_paths):
        positives: list[float] = []
        negatives: list[float] = []
        with ranking_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                score = float(row["score"])
                if int(row["label"]) == 1:
                    positives.append(score)
                else:
                    negatives.append(score)

        axis = axes_list[index]
        axis.hist(negatives, bins=45, density=True, alpha=0.72, color=PALETTE[3], label="negative candidates")
        axis.hist(positives, bins=45, density=True, alpha=0.78, color=PALETTE[0], label="true test edges")
        axis.set_title(f"Score Distribution: {ranking_path.stem}")
        axis.set_xlabel("Score")
        axis.set_ylabel("Density")
        axis.legend()

    for axis in axes_list[len(ranking_paths):]:
        axis.axis("off")

    output_path = figure_dir / "score_distribution_positive_negative.png"
    save_figure(output_path)
    return output_path


def plot_test_positive_distribution(test_edges_path: Path, figure_dir: Path) -> Path:
    counts: dict[int, int] = {}
    for source, _ in read_edges(test_edges_path):
        counts[source] = counts.get(source, 0) + 1

    plt.figure(figsize=(10, 6))
    plt.hist(list(counts.values()), bins=40, color=PALETTE[1], edgecolor=PALETTE[0])
    plt.xlabel("Hidden positive edges per source node")
    plt.ylabel("Number of source nodes")
    plt.title("Test Positive Edge Count Distribution")
    plt.grid(axis="y", alpha=0.25)

    output_path = figure_dir / "test_positives_per_source_distribution.png"
    save_figure(output_path)
    return output_path


def plot_train_degree_distribution(train_edges_path: Path, figure_dir: Path) -> Path:
    in_degree: dict[int, int] = {}
    out_degree: dict[int, int] = {}
    nodes: set[int] = set()

    for source, target in read_edges(train_edges_path):
        nodes.add(source)
        nodes.add(target)
        out_degree[source] = out_degree.get(source, 0) + 1
        in_degree[target] = in_degree.get(target, 0) + 1

    in_values = sorted([in_degree.get(node, 0) for node in nodes], reverse=True)
    out_values = sorted([out_degree.get(node, 0) for node in nodes], reverse=True)

    plt.figure(figsize=(10, 6))
    plt.loglog(range(1, len(out_values) + 1), out_values, color=PALETTE[0], linewidth=2, label="out-degree")
    plt.loglog(range(1, len(in_values) + 1), in_values, color=PALETTE[2], linewidth=2, label="in-degree")
    plt.xlabel("Node rank")
    plt.ylabel("Degree")
    plt.title("Training Graph Degree Distribution")
    plt.grid(alpha=0.25)
    plt.legend()

    output_path = figure_dir / "train_degree_distribution.png"
    save_figure(output_path)
    return output_path


def build_palette_colormap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("net_rag_palette", PALETTE)


def plot_retrieval_network(output_dir: Path, figure_dir: Path, user_id: int, algo: str) -> Path | None:
    retrieval_path = output_dir / "retrievals" / f"user_{user_id}_{algo}.csv"
    train_path = output_dir / "splits" / "train_edges.csv"
    if not retrieval_path.exists() or not train_path.exists():
        return None

    retrieval_rows = read_csv_rows(retrieval_path)
    train_edges = read_edges(train_path)
    train_graph = nx.DiGraph()
    train_graph.add_edges_from(train_edges)
    undirected_graph = train_graph.to_undirected(as_view=False)

    targets = [int(row["target_node"]) for row in retrieval_rows]
    subgraph_nodes = {user_id, *targets}
    edge_colors: list[str] = []
    subgraph_edges: list[tuple[int, int]] = []

    for target in targets:
        try:
            path = nx.shortest_path(undirected_graph, user_id, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

        for node in path[:4]:
            subgraph_nodes.add(node)
        for left, right in zip(path[:3], path[1:4]):
            edge = (left, right)
            if edge not in subgraph_edges and (right, left) not in subgraph_edges:
                subgraph_edges.append(edge)
                edge_colors.append(PALETTE[4])

    display_graph = nx.Graph()
    display_graph.add_nodes_from(subgraph_nodes)
    display_graph.add_edges_from(subgraph_edges)

    node_colors = []
    node_sizes = []
    for node in display_graph.nodes:
        if node == user_id:
            node_colors.append(PALETTE[0])
            node_sizes.append(900)
        elif node in targets:
            node_colors.append(PALETTE[1])
            node_sizes.append(650)
        else:
            node_colors.append(PALETTE[3])
            node_sizes.append(480)

    plt.figure(figsize=(11, 8))
    positions = nx.spring_layout(display_graph, seed=42, k=0.9)
    nx.draw_networkx_edges(display_graph, positions, edge_color=edge_colors or PALETTE[4], width=1.5, alpha=0.85)
    nx.draw_networkx_nodes(display_graph, positions, node_color=node_colors, node_size=node_sizes, linewidths=1.5, edgecolors=PALETTE[2])
    nx.draw_networkx_labels(display_graph, positions, font_size=8, font_color=PALETTE[0])

    plt.title(f"Top-{len(targets)} Retrieval Network for User {user_id} ({algo})")
    plt.axis("off")

    output_path = figure_dir / f"user_{user_id}_{algo}_retrieval_network.png"
    save_figure(output_path)
    return output_path


def visualize(output_dir: Path, max_k: int, user_id: int, algo: str) -> list[Path]:
    figure_dir = output_dir / "figures"
    generated: list[Path] = []

    generated.append(plot_metrics_summary(output_dir / "metrics" / "summary.csv", figure_dir))
    generated.append(plot_split_summary(output_dir / "splits" / "split_meta.json", figure_dir))
    generated.append(plot_precision_curves(output_dir / "rankings", figure_dir, max_k=max_k))
    generated.append(plot_hit_rate_curves(output_dir / "rankings", figure_dir, max_k=max_k))
    generated.append(plot_precision_heatmap(output_dir / "rankings", figure_dir, max_k=max_k))
    generated.append(plot_first_hit_cdf(output_dir / "rankings", figure_dir))
    generated.append(plot_score_distributions(output_dir / "rankings", figure_dir))
    generated.append(plot_test_positive_distribution(output_dir / "splits" / "test_edges.csv", figure_dir))
    generated.append(plot_train_degree_distribution(output_dir / "splits" / "train_edges.csv", figure_dir))

    retrieval_path = plot_retrieval_network(output_dir, figure_dir, user_id=user_id, algo=algo)
    if retrieval_path is not None:
        generated.append(retrieval_path)

    return generated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate visualizations for GraphRAG retrieval results.")
    parser.add_argument("--output-dir", default="outputs", help="Directory containing evaluation outputs.")
    parser.add_argument("--max-k", type=int, default=20, help="Maximum K shown in the Precision@K curve.")
    parser.add_argument("--user-id", type=int, default=30, help="User ID used for the retrieval network figure.")
    parser.add_argument("--algo", default="ppr", help="Retrieval result suffix used for the network figure.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    generated = visualize(Path(args.output_dir), max_k=args.max_k, user_id=args.user_id, algo=args.algo)
    for path in generated:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
