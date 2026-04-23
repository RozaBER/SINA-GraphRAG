from net_rag.data import load_raw_edges
from net_rag.split import stratified_random_split


def test_load_raw_edges_skips_comments_and_duplicates(tmp_path):
    raw_file = tmp_path / "sample.txt"
    raw_file.write_text(
        "# comment\n"
        "1\t2\n"
        "1 2\n"
        "2 3\n",
        encoding="utf-8",
    )

    edges = load_raw_edges(raw_file)
    assert edges == [(1, 2), (2, 3)]


def test_stratified_random_split_is_disjoint_and_deterministic():
    edges = [
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 3),
        (2, 4),
        (3, 4),
    ]

    train_a, test_a = stratified_random_split(edges, test_ratio=0.2, seed=7)
    train_b, test_b = stratified_random_split(edges, test_ratio=0.2, seed=7)

    assert train_a == train_b
    assert test_a == test_b
    assert set(train_a).isdisjoint(set(test_a))
    assert set(train_a) | set(test_a) == set(edges)
