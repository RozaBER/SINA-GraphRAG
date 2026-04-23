from __future__ import annotations

from pathlib import Path

import networkx as nx

from .candidates import build_retrieval_candidates
from .data import Edge, write_rows_csv
from .scorers import BaseScorer


RETRIEVAL_FIELDS = [
    "source_node",
    "target_node",
    "score",
    "rank",
    "algorithm",
    "evidence",
]


def retrieve_related_nodes(
    *,
    graph: nx.DiGraph,
    all_nodes: list[int],
    train_edge_set: set[Edge],
    source: int,
    scorer: BaseScorer,
    k: int,
) -> list[dict[str, object]]:
    candidate_nodes = build_retrieval_candidates(
        source=source,
        all_nodes=all_nodes,
        train_edge_set=train_edge_set,
    )
    scores = scorer.score(source, candidate_nodes)
    ranked_targets = sorted(
        candidate_nodes,
        key=lambda target: (-scores.get(target, 0.0), target),
    )[:k]

    explanations = build_explanations(graph=graph, source=source, targets=ranked_targets)

    return [
        {
            "source_node": source,
            "target_node": target,
            "score": round(scores.get(target, 0.0), 8),
            "rank": index,
            "algorithm": scorer.name,
            "evidence": explanations.get(target, "No short structural explanation available."),
        }
        for index, target in enumerate(ranked_targets, start=1)
    ]


def build_explanations(
    *,
    graph: nx.DiGraph,
    source: int,
    targets: list[int],
) -> dict[int, str]:
    explanations: dict[int, str] = {}
    undirected_graph = graph.to_undirected(as_view=False)

    try:
        short_paths = nx.single_source_shortest_path(undirected_graph, source, cutoff=3)
    except nx.NodeNotFound:
        short_paths = {}

    source_neighbors = set(undirected_graph.neighbors(source)) if undirected_graph.has_node(source) else set()

    for target in targets:
        path = short_paths.get(target)
        if path and len(path) == 3:
            explanations[target] = f"2-hop path via node {path[1]}."
            continue
        if path and len(path) > 3:
            path_text = " -> ".join(str(node) for node in path)
            explanations[target] = f"Shortest path in training graph: {path_text}."
            continue

        target_neighbors = set(undirected_graph.neighbors(target)) if undirected_graph.has_node(target) else set()
        common_neighbors = sorted(source_neighbors & target_neighbors)
        if common_neighbors:
            preview = ", ".join(str(node) for node in common_neighbors[:3])
            explanations[target] = f"Shared neighbors: {preview}."
        else:
            explanations[target] = "No short path found; relevance comes from broader graph structure."

    return explanations


def write_retrieval_output(output_dir: str | Path, user_id: int, algo_name: str, rows: list[dict[str, object]]) -> Path:
    output_file = Path(output_dir) / "retrievals" / f"user_{user_id}_{algo_name}.csv"
    write_rows_csv(output_file, RETRIEVAL_FIELDS, rows)
    return output_file
