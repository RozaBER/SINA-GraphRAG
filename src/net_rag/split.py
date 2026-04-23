from __future__ import annotations

import random
from collections import defaultdict

from .data import Edge


def stratified_random_split(
    edges: list[Edge],
    test_ratio: float,
    seed: int,
) -> tuple[list[Edge], list[Edge]]:
    outgoing: dict[int, list[int]] = defaultdict(list)
    for source, target in edges:
        outgoing[source].append(target)

    rng = random.Random(seed)
    test_edge_set: set[Edge] = set()

    for source in sorted(outgoing):
        targets = sorted(outgoing[source])
        if len(targets) < 2:
            continue

        test_count = max(1, int(round(len(targets) * test_ratio)))
        test_count = min(test_count, len(targets) - 1)
        sampled_targets = rng.sample(targets, test_count)
        for target in sampled_targets:
            test_edge_set.add((source, target))

    train_edges = [edge for edge in edges if edge not in test_edge_set]
    test_edges = [edge for edge in edges if edge in test_edge_set]
    return train_edges, test_edges


def build_split_metadata(
    *,
    total_edges: int,
    total_nodes: int,
    train_edges: list[Edge],
    test_edges: list[Edge],
    seed: int,
    test_ratio: float,
) -> dict[str, object]:
    train_count = len(train_edges)
    test_count = len(test_edges)
    actual_ratio = test_count / total_edges if total_edges else 0.0
    sources_with_test_edges = len({source for source, _ in test_edges})

    return {
        "seed": seed,
        "requested_test_ratio": test_ratio,
        "actual_test_ratio": round(actual_ratio, 6),
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "train_edge_count": train_count,
        "test_edge_count": test_count,
        "sources_with_test_edges": sources_with_test_edges,
    }
