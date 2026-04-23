import math

from net_rag.candidates import build_candidate_sets
from net_rag.data import build_digraph
from net_rag.scorers import build_default_scorers


def test_negative_sampling_excludes_existing_edges():
    train_edges = [(1, 2), (1, 3), (2, 4)]
    test_edges = [(1, 4)]
    full_edge_set = set(train_edges) | set(test_edges)
    all_nodes = [1, 2, 3, 4, 5]

    candidate_sets = build_candidate_sets(
        test_edges=test_edges,
        all_nodes=all_nodes,
        full_edge_set=full_edge_set,
        negatives_per_source=3,
        seed=11,
    )

    targets = {target for target, _ in candidate_sets[1]}
    assert 2 not in targets
    assert 3 not in targets
    assert 4 in targets


def test_all_scorers_return_finite_scores():
    train_graph = build_digraph(
        [
            (1, 2),
            (1, 3),
            (2, 3),
            (2, 4),
            (3, 4),
            (4, 5),
        ]
    )

    scorers = build_default_scorers(train_graph, seed=13)
    candidates = [4, 5, 6]

    for scorer in scorers:
        scores = scorer.score(1, candidates)
        assert set(scores) == set(candidates)
        assert all(math.isfinite(value) for value in scores.values())
