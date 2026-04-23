from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from collections import Counter
from typing import Iterable

import networkx as nx


class BaseScorer(ABC):
    name: str

    @abstractmethod
    def score(self, source: int, candidates: Iterable[int]) -> dict[int, float]:
        raise NotImplementedError


class LocalUndirectedScorer(BaseScorer):
    def __init__(self, graph: nx.DiGraph) -> None:
        self.graph = graph.to_undirected(as_view=False)
        self.neighbors = {
            node: set(self.graph.neighbors(node))
            for node in self.graph.nodes
        }

    def _get_neighbors(self, node: int) -> set[int]:
        return self.neighbors.get(node, set())


class CommonNeighborsScorer(LocalUndirectedScorer):
    name = "common_neighbors"

    def score(self, source: int, candidates: Iterable[int]) -> dict[int, float]:
        source_neighbors = self._get_neighbors(source)
        return {
            candidate: float(len(source_neighbors & self._get_neighbors(candidate)))
            for candidate in candidates
        }


class JaccardScorer(LocalUndirectedScorer):
    name = "jaccard"

    def score(self, source: int, candidates: Iterable[int]) -> dict[int, float]:
        source_neighbors = self._get_neighbors(source)
        scores: dict[int, float] = {}
        for candidate in candidates:
            candidate_neighbors = self._get_neighbors(candidate)
            union = source_neighbors | candidate_neighbors
            if not union:
                scores[candidate] = 0.0
                continue
            intersection = source_neighbors & candidate_neighbors
            scores[candidate] = len(intersection) / len(union)
        return scores


class AdamicAdarScorer(LocalUndirectedScorer):
    name = "adamic_adar"

    def score(self, source: int, candidates: Iterable[int]) -> dict[int, float]:
        source_neighbors = self._get_neighbors(source)
        scores: dict[int, float] = {}

        for candidate in candidates:
            candidate_neighbors = self._get_neighbors(candidate)
            common_neighbors = source_neighbors & candidate_neighbors
            value = 0.0
            for neighbor in common_neighbors:
                degree = len(self._get_neighbors(neighbor))
                if degree > 1:
                    value += 1.0 / math.log(degree)
            scores[candidate] = value

        return scores


class PersonalizedPageRankScorer(BaseScorer):
    name = "personalized_pagerank"

    def __init__(
        self,
        graph: nx.DiGraph,
        *,
        restart_probability: float = 0.15,
        num_walks: int = 400,
        walk_length: int = 20,
        seed: int = 42,
    ) -> None:
        self.graph = graph
        self.restart_probability = restart_probability
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.seed = seed
        self.out_neighbors = {
            node: tuple(sorted(graph.successors(node)))
            for node in graph.nodes
        }
        self._cache: dict[int, dict[int, float]] = {}

    def _distribution_for_source(self, source: int) -> dict[int, float]:
        if source in self._cache:
            return self._cache[source]

        rng = random.Random((self.seed << 16) ^ (source * 2654435761))
        visits: Counter[int] = Counter()
        total_visits = 0

        for _ in range(self.num_walks):
            current = source
            for _ in range(self.walk_length):
                visits[current] += 1
                total_visits += 1
                neighbors = self.out_neighbors.get(current, ())
                if not neighbors or rng.random() < self.restart_probability:
                    current = source
                else:
                    current = neighbors[rng.randrange(len(neighbors))]

        if total_visits == 0:
            distribution = {}
        else:
            distribution = {
                node: count / total_visits
                for node, count in visits.items()
            }

        self._cache[source] = distribution
        return distribution

    def score(self, source: int, candidates: Iterable[int]) -> dict[int, float]:
        distribution = self._distribution_for_source(source)
        return {
            candidate: distribution.get(candidate, 0.0)
            for candidate in candidates
        }


def build_default_scorers(graph: nx.DiGraph, *, seed: int) -> list[BaseScorer]:
    return [
        CommonNeighborsScorer(graph),
        JaccardScorer(graph),
        AdamicAdarScorer(graph),
        PersonalizedPageRankScorer(graph, seed=seed),
    ]
