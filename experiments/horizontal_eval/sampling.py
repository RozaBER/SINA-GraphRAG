from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterable

import networkx as nx

from net_rag.candidates import CandidateLabel, group_targets_by_source
from net_rag.data import Edge


SUPPORTED_NEGATIVE_STRATEGIES = ["random", "degree_matched", "hard_2hop"]


def build_candidate_sets_with_strategy(
    *,
    test_edges: list[Edge],
    all_nodes: list[int],
    full_edge_set: set[Edge],
    train_graph: nx.DiGraph,
    negatives_per_source: int,
    seed: int,
    strategy: str,
) -> dict[int, list[CandidateLabel]]:
    if strategy not in SUPPORTED_NEGATIVE_STRATEGIES:
        supported = ", ".join(SUPPORTED_NEGATIVE_STRATEGIES)
        raise ValueError(f"Unsupported negative strategy '{strategy}'. Supported: {supported}")

    positives_by_source = group_targets_by_source(test_edges)
    undirected_graph = train_graph.to_undirected(as_view=False)
    candidate_sets: dict[int, list[CandidateLabel]] = {}

    for source in sorted(positives_by_source):
        positive_targets = sorted(positives_by_source[source])
        negative_pool = [
            node
            for node in all_nodes
            if node != source and (source, node) not in full_edge_set
        ]
        local_rng = random.Random((seed * 1000003) ^ (source * 9176) ^ _stable_strategy_salt(strategy))

        if strategy == "random":
            negative_targets = _sample_random(negative_pool, negatives_per_source, local_rng)
        elif strategy == "degree_matched":
            negative_targets = _sample_degree_matched(
                positive_targets=positive_targets,
                negative_pool=negative_pool,
                train_graph=train_graph,
                sample_size=negatives_per_source,
                rng=local_rng,
            )
        else:
            negative_targets = _sample_hard_2hop(
                source=source,
                negative_pool=negative_pool,
                full_edge_set=full_edge_set,
                undirected_graph=undirected_graph,
                sample_size=negatives_per_source,
                rng=local_rng,
            )

        candidates: list[CandidateLabel] = []
        candidates.extend((target, 1) for target in positive_targets)
        candidates.extend((target, 0) for target in sorted(negative_targets))
        candidate_sets[source] = candidates

    return candidate_sets


def _sample_random(pool: list[int], sample_size: int, rng: random.Random) -> list[int]:
    return rng.sample(pool, min(sample_size, len(pool)))


def _stable_strategy_salt(strategy: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(strategy))


def _sample_degree_matched(
    *,
    positive_targets: list[int],
    negative_pool: list[int],
    train_graph: nx.DiGraph,
    sample_size: int,
    rng: random.Random,
) -> list[int]:
    if not negative_pool or sample_size <= 0:
        return []

    target_degrees = dict(train_graph.in_degree())
    desired_degrees = [target_degrees.get(target, 0) for target in positive_targets] or [0]
    buckets: dict[int, list[int]] = defaultdict(list)
    for node in negative_pool:
        buckets[target_degrees.get(node, 0)].append(node)

    for nodes in buckets.values():
        nodes.sort()

    selected: list[int] = []
    max_degree = max(buckets) if buckets else 0
    target_size = min(sample_size, len(negative_pool))

    # 反复围绕正样本入度找最近的负样本，避免 popularity baseline 占便宜。
    while len(selected) < target_size and buckets:
        desired = desired_degrees[len(selected) % len(desired_degrees)]
        chosen = _pop_nearest_degree_bucket(buckets, desired, max_degree, rng)
        if chosen is None:
            break
        selected.append(chosen)

    if len(selected) < target_size:
        remaining = [node for nodes in buckets.values() for node in nodes]
        selected.extend(_sample_random(remaining, target_size - len(selected), rng))

    return selected


def _pop_nearest_degree_bucket(
    buckets: dict[int, list[int]],
    desired_degree: int,
    max_degree: int,
    rng: random.Random,
) -> int | None:
    search_limit = max(max_degree, desired_degree)
    for distance in range(search_limit + 1):
        for degree in _candidate_degrees(desired_degree, distance):
            nodes = buckets.get(degree)
            if not nodes:
                continue
            index = rng.randrange(len(nodes))
            chosen = nodes.pop(index)
            if not nodes:
                buckets.pop(degree, None)
            return chosen
    return None


def _candidate_degrees(center: int, distance: int) -> Iterable[int]:
    if distance == 0:
        yield center
        return
    if center - distance >= 0:
        yield center - distance
    yield center + distance


def _sample_hard_2hop(
    *,
    source: int,
    negative_pool: list[int],
    full_edge_set: set[Edge],
    undirected_graph: nx.Graph,
    sample_size: int,
    rng: random.Random,
) -> list[int]:
    if not negative_pool or sample_size <= 0:
        return []

    pool_set = set(negative_pool)
    try:
        lengths = nx.single_source_shortest_path_length(undirected_graph, source, cutoff=3)
    except nx.NodeNotFound:
        lengths = {}

    hard_pool = [
        node
        for node, distance in lengths.items()
        if 2 <= distance <= 3 and node in pool_set and (source, node) not in full_edge_set
    ]
    selected = _sample_random(sorted(hard_pool), sample_size, rng)

    if len(selected) < min(sample_size, len(negative_pool)):
        selected_set = set(selected)
        fallback_pool = [node for node in negative_pool if node not in selected_set]
        selected.extend(_sample_random(fallback_pool, sample_size - len(selected), rng))

    return selected
