import csv
import math

from experiments.horizontal_eval.extra_scorers import build_horizontal_scorers, filter_scorers
from experiments.horizontal_eval.run import main
from experiments.horizontal_eval.sampling import SUPPORTED_NEGATIVE_STRATEGIES, build_candidate_sets_with_strategy
from net_rag.data import build_digraph


def test_horizontal_scorers_return_finite_scores():
    graph = build_digraph(
        [
            (1, 2),
            (1, 3),
            (2, 3),
            (2, 4),
            (3, 4),
            (4, 5),
        ]
    )

    scorers = build_horizontal_scorers(graph, seed=5)
    names = {scorer.name for scorer in scorers}
    assert {
        "resource_allocation",
        "preferential_attachment",
        "cosine_similarity",
        "sorensen",
        "target_indegree",
    } <= names

    for scorer in scorers:
        scores = scorer.score(1, [4, 5, 6])
        assert set(scores) == {4, 5, 6}
        assert all(math.isfinite(value) for value in scores.values())

    selected = filter_scorers(scorers, ["target_indegree", "resource_allocation"])
    assert [scorer.name for scorer in selected] == ["target_indegree", "resource_allocation"]


def test_negative_sampling_strategies_exclude_existing_edges():
    train_edges = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 6), (5, 6)]
    test_edges = [(1, 4), (2, 6)]
    full_edge_set = set(train_edges) | set(test_edges)
    all_nodes = [1, 2, 3, 4, 5, 6, 7]
    train_graph = build_digraph(train_edges)

    for strategy in SUPPORTED_NEGATIVE_STRATEGIES:
        candidate_sets = build_candidate_sets_with_strategy(
            test_edges=test_edges,
            all_nodes=all_nodes,
            full_edge_set=full_edge_set,
            train_graph=train_graph,
            negatives_per_source=2,
            seed=11,
            strategy=strategy,
        )

        assert set(candidate_sets) == {1, 2}
        for source, candidates in candidate_sets.items():
            labels = {target: label for target, label in candidates}
            assert any(label == 1 for label in labels.values())
            for target, label in candidates:
                if label == 0:
                    assert (source, target) not in full_edge_set


def test_horizontal_eval_cli_on_small_graph(tmp_path):
    raw_file = tmp_path / "wiki-small.txt"
    raw_file.write_text(
        "# header\n"
        "1 2\n"
        "1 3\n"
        "1 4\n"
        "2 3\n"
        "2 4\n"
        "2 5\n"
        "3 4\n"
        "3 5\n"
        "4 5\n"
        "4 6\n"
        "5 1\n"
        "5 6\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "horizontal"

    assert main(
        [
            "--input",
            str(raw_file),
            "--output-dir",
            str(output_dir),
            "--seeds",
            "3",
            "--k-values",
            "1,2",
            "--negative-strategies",
            "random,hard_2hop",
            "--negatives-per-source",
            "2",
            "--algorithms",
            "common_neighbors,target_indegree,resource_allocation",
            "--max-sources",
            "3",
        ]
    ) == 0

    assert (output_dir / "runs.csv").exists()
    assert (output_dir / "summary_by_run.csv").exists()
    assert (output_dir / "summary_mean_std.csv").exists()
    assert (output_dir / "configs" / "default_config.json").exists()
    assert (output_dir / "figures" / "algorithm_mean_std.png").exists()

    with (output_dir / "summary_mean_std.csv").open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    algorithms = {row["algorithm"] for row in rows}
    assert algorithms == {"common_neighbors", "resource_allocation", "target_indegree"}
    assert all(row["mean_precision_at_k"] != "" for row in rows)
