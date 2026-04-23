from net_rag.cli import main


def test_cli_end_to_end_on_small_graph(tmp_path, capsys):
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
        "5 1\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "outputs"

    assert main(["prepare", "--input", str(raw_file), "--output-dir", str(output_dir), "--seed", "7"]) == 0
    assert main(
        [
            "evaluate",
            "--output-dir",
            str(output_dir),
            "--k",
            "2",
            "--negatives-per-source",
            "2",
            "--seed",
            "7",
        ]
    ) == 0
    assert main(
        [
            "retrieve",
            "--output-dir",
            str(output_dir),
            "--user-id",
            "1",
            "--algo",
            "ppr",
            "--k",
            "3",
            "--seed",
            "7",
        ]
    ) == 0

    captured = capsys.readouterr().out
    assert "Prepared split" in captured
    assert "personalized_pagerank" in captured
    assert (output_dir / "splits" / "train_edges.csv").exists()
    assert (output_dir / "splits" / "test_edges.csv").exists()
    assert (output_dir / "splits" / "split_meta.json").exists()
    assert (output_dir / "metrics" / "summary.csv").exists()
    assert (output_dir / "rankings" / "common_neighbors.csv").exists()
    assert (output_dir / "rankings" / "personalized_pagerank.csv").exists()
    assert (output_dir / "figures" / "algorithm_compare.png").exists()
    assert (output_dir / "retrievals" / "user_1_ppr.csv").exists()
