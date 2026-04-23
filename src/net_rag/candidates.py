from __future__ import annotations

import random
from collections import defaultdict

from .data import Edge


CandidateLabel = tuple[int, int]


def group_targets_by_source(edges: list[Edge]) -> dict[int, set[int]]:
    grouped: dict[int, set[int]] = defaultdict(set)
    for source, target in edges:
        grouped[source].add(target)
    return grouped


def build_candidate_sets(
    *,
    test_edges: list[Edge],
    all_nodes: list[int],
    full_edge_set: set[Edge],
    negatives_per_source: int,
    seed: int,
) -> dict[int, list[CandidateLabel]]:
    positives_by_source = group_targets_by_source(test_edges)
    candidate_sets: dict[int, list[CandidateLabel]] = {}

    for source in sorted(positives_by_source):
        positive_targets = positives_by_source[source]
        negative_pool = [
            node
            for node in all_nodes
            if node != source and (source, node) not in full_edge_set
        ]

        local_rng = random.Random((seed * 1000003) ^ source)
        sample_size = min(negatives_per_source, len(negative_pool))
        negative_targets = local_rng.sample(negative_pool, sample_size)

        candidates: list[CandidateLabel] = []
        candidates.extend((target, 1) for target in sorted(positive_targets))
        candidates.extend((target, 0) for target in sorted(negative_targets))
        candidate_sets[source] = candidates

    return candidate_sets


def build_retrieval_candidates(
    *,
    source: int,
    all_nodes: list[int],
    train_edge_set: set[Edge],
) -> list[int]:
    return [
        node
        for node in all_nodes
        if node != source and (source, node) not in train_edge_set
    ]
