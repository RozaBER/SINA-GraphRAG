from __future__ import annotations

import math
from typing import Iterable

import networkx as nx

from net_rag.scorers import BaseScorer, build_default_scorers


class _UndirectedNeighborhoodMixin:
    def __init__(self, graph: nx.DiGraph) -> None:
        self.graph = graph.to_undirected(as_view=False)
        self.neighbors = {
            node: set(self.graph.neighbors(node))
            for node in self.graph.nodes
        }

    def _get_neighbors(self, node: int) -> set[int]:
        return self.neighbors.get(node, set())


class ResourceAllocationScorer(_UndirectedNeighborhoodMixin, BaseScorer):
    name = "resource_allocation"

    def score(self, source: int, candidates: Iterable[int]) -> dict[int, float]:
        source_neighbors = self._get_neighbors(source)
        scores: dict[int, float] = {}

        for candidate in candidates:
            common_neighbors = source_neighbors & self._get_neighbors(candidate)
            value = 0.0
            for neighbor in common_neighbors:
                degree = len(self._get_neighbors(neighbor))
                if degree:
                    value += 1.0 / degree
            scores[candidate] = value

        return scores


class PreferentialAttachmentScorer(_UndirectedNeighborhoodMixin, BaseScorer):
    name = "preferential_attachment"

    def score(self, source: int, candidates: Iterable[int]) -> dict[int, float]:
        source_degree = len(self._get_neighbors(source))
        return {
            candidate: float(source_degree * len(self._get_neighbors(candidate)))
            for candidate in candidates
        }


class CosineSimilarityScorer(_UndirectedNeighborhoodMixin, BaseScorer):
    name = "cosine_similarity"

    def score(self, source: int, candidates: Iterable[int]) -> dict[int, float]:
        source_neighbors = self._get_neighbors(source)
        source_degree = len(source_neighbors)
        scores: dict[int, float] = {}

        for candidate in candidates:
            candidate_neighbors = self._get_neighbors(candidate)
            denominator = math.sqrt(source_degree * len(candidate_neighbors))
            if denominator == 0:
                scores[candidate] = 0.0
                continue
            scores[candidate] = len(source_neighbors & candidate_neighbors) / denominator

        return scores


class SorensenScorer(_UndirectedNeighborhoodMixin, BaseScorer):
    name = "sorensen"

    def score(self, source: int, candidates: Iterable[int]) -> dict[int, float]:
        source_neighbors = self._get_neighbors(source)
        source_degree = len(source_neighbors)
        scores: dict[int, float] = {}

        for candidate in candidates:
            candidate_neighbors = self._get_neighbors(candidate)
            denominator = source_degree + len(candidate_neighbors)
            if denominator == 0:
                scores[candidate] = 0.0
                continue
            scores[candidate] = 2.0 * len(source_neighbors & candidate_neighbors) / denominator

        return scores


class TargetInDegreeScorer(BaseScorer):
    name = "target_indegree"

    def __init__(self, graph: nx.DiGraph) -> None:
        # 这个基线只看目标节点流行度，用来检查复杂算法是否只是推荐热门节点。
        self.in_degrees = dict(graph.in_degree())

    def score(self, source: int, candidates: Iterable[int]) -> dict[int, float]:
        return {
            candidate: float(self.in_degrees.get(candidate, 0))
            for candidate in candidates
        }


def build_horizontal_scorers(graph: nx.DiGraph, *, seed: int) -> list[BaseScorer]:
    return [
        *build_default_scorers(graph, seed=seed),
        ResourceAllocationScorer(graph),
        PreferentialAttachmentScorer(graph),
        CosineSimilarityScorer(graph),
        SorensenScorer(graph),
        TargetInDegreeScorer(graph),
    ]


def filter_scorers(scorers: list[BaseScorer], requested: list[str]) -> list[BaseScorer]:
    if requested == ["all"]:
        return scorers

    scorers_by_name = {scorer.name: scorer for scorer in scorers}
    missing = [name for name in requested if name not in scorers_by_name]
    if missing:
        supported = ", ".join(sorted(scorers_by_name))
        raise ValueError(f"Unsupported algorithms: {', '.join(missing)}. Supported: {supported}")

    return [scorers_by_name[name] for name in requested]
