from __future__ import annotations

import argparse
from pathlib import Path

from .candidates import build_candidate_sets
from .constants import DEFAULT_K, DEFAULT_NEGATIVES_PER_SOURCE, DEFAULT_OUTPUT_DIR, DEFAULT_SEED, DEFAULT_TEST_RATIO
from .data import build_digraph, collect_nodes, load_raw_edges, read_edges_csv, read_json, write_edges_csv, write_json
from .evaluation import plot_algorithm_comparison, evaluate_candidate_sets, write_evaluation_outputs
from .retrieval import retrieve_related_nodes, write_retrieval_output
from .scorers import build_default_scorers
from .split import build_split_metadata, stratified_random_split


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simplified GraphRAG retrieval module.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Parse raw data and create train/test splits.")
    prepare_parser.add_argument("--input", default="Wiki-Vote.txt", help="Path to the raw Wiki-Vote dataset.")
    prepare_parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for generated artifacts.")
    prepare_parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducible splits.")
    prepare_parser.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO, help="Approximate test edge ratio.")

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate scoring algorithms on the prepared split.")
    evaluate_parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory containing split artifacts.")
    evaluate_parser.add_argument("--k", type=int, default=DEFAULT_K, help="Top-K cutoff for evaluation metrics.")
    evaluate_parser.add_argument(
        "--negatives-per-source",
        type=int,
        default=DEFAULT_NEGATIVES_PER_SOURCE,
        help="Number of sampled negative targets for each test source node.",
    )
    evaluate_parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for candidate sampling and PPR.")

    retrieve_parser = subparsers.add_parser("retrieve", help="Retrieve top related nodes for a given user.")
    retrieve_parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory containing split artifacts.")
    retrieve_parser.add_argument("--user-id", type=int, required=True, help="User ID to retrieve context for.")
    retrieve_parser.add_argument("--algo", default="ppr", help="Algorithm to use: cn, jaccard, aa, ppr.")
    retrieve_parser.add_argument("--k", type=int, default=DEFAULT_K, help="Number of related nodes to return.")
    retrieve_parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for scorer construction.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "prepare":
        run_prepare(args)
        return 0
    if args.command == "evaluate":
        run_evaluate(args)
        return 0
    if args.command == "retrieve":
        run_retrieve(args)
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 1


def run_prepare(args: argparse.Namespace) -> None:
    raw_edges = load_raw_edges(args.input)
    train_edges, test_edges = stratified_random_split(
        raw_edges,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    all_nodes = collect_nodes(raw_edges)
    split_dir = Path(args.output_dir) / "splits"
    write_edges_csv(split_dir / "train_edges.csv", train_edges)
    write_edges_csv(split_dir / "test_edges.csv", test_edges)

    metadata = build_split_metadata(
        total_edges=len(raw_edges),
        total_nodes=len(all_nodes),
        train_edges=train_edges,
        test_edges=test_edges,
        seed=args.seed,
        test_ratio=args.test_ratio,
    )
    metadata["input_path"] = str(args.input)
    write_json(split_dir / "split_meta.json", metadata)

    print(
        "Prepared split:",
        f"nodes={metadata['total_nodes']},",
        f"train_edges={metadata['train_edge_count']},",
        f"test_edges={metadata['test_edge_count']},",
        f"actual_test_ratio={metadata['actual_test_ratio']}",
    )


def run_evaluate(args: argparse.Namespace) -> None:
    train_edges, test_edges, metadata = load_split_artifacts(args.output_dir)
    all_nodes = collect_nodes(train_edges + test_edges)
    full_edge_set = set(train_edges) | set(test_edges)

    candidate_sets = build_candidate_sets(
        test_edges=test_edges,
        all_nodes=all_nodes,
        full_edge_set=full_edge_set,
        negatives_per_source=args.negatives_per_source,
        seed=args.seed,
    )

    train_graph = build_digraph(train_edges)
    scorers = build_default_scorers(train_graph, seed=args.seed)
    summary_rows, ranking_rows_by_algorithm = evaluate_candidate_sets(
        candidate_sets=candidate_sets,
        scorers=scorers,
        k=args.k,
    )

    output_dir = Path(args.output_dir)
    write_evaluation_outputs(
        output_dir=output_dir,
        summary_rows=summary_rows,
        ranking_rows_by_algorithm=ranking_rows_by_algorithm,
    )
    plot_algorithm_comparison(summary_rows, output_dir / "figures" / "algorithm_compare.png")

    print(f"Loaded split created from: {metadata.get('input_path', 'unknown input')}")
    for row in summary_rows:
        print(
            f"{row['algorithm']}: "
            f"Precision@{row['k']}={row['precision_at_k']}, "
            f"HitRate@{row['k']}={row['hit_rate_at_k']}, "
            f"MRR={row['mrr']}"
        )


def run_retrieve(args: argparse.Namespace) -> None:
    train_edges, test_edges, _ = load_split_artifacts(args.output_dir)
    all_nodes = collect_nodes(train_edges + test_edges)
    if args.user_id not in all_nodes:
        raise SystemExit(f"User {args.user_id} does not exist in the prepared dataset.")

    train_graph = build_digraph(train_edges)
    scorers = {scorer.name: scorer for scorer in build_default_scorers(train_graph, seed=args.seed)}
    alias_map = {
        "cn": "common_neighbors",
        "common_neighbors": "common_neighbors",
        "jaccard": "jaccard",
        "aa": "adamic_adar",
        "adamic_adar": "adamic_adar",
        "ppr": "personalized_pagerank",
        "personalized_pagerank": "personalized_pagerank",
    }

    algorithm_name = alias_map.get(args.algo.lower())
    if algorithm_name is None:
        supported = ", ".join(sorted(alias_map))
        raise SystemExit(f"Unsupported algorithm '{args.algo}'. Available choices: {supported}.")

    rows = retrieve_related_nodes(
        graph=train_graph,
        all_nodes=all_nodes,
        train_edge_set=set(train_edges),
        source=args.user_id,
        scorer=scorers[algorithm_name],
        k=args.k,
    )
    output_path = write_retrieval_output(args.output_dir, args.user_id, args.algo.lower(), rows)

    print(f"Top {args.k} results for user {args.user_id} using {algorithm_name}:")
    for row in rows:
        print(
            f"#{row['rank']}: target={row['target_node']}, "
            f"score={row['score']}, evidence={row['evidence']}"
        )
    print(f"Saved retrieval results to {output_path}")


def load_split_artifacts(output_dir: str | Path) -> tuple[list[tuple[int, int]], list[tuple[int, int]], dict[str, object]]:
    split_dir = Path(output_dir) / "splits"
    train_path = split_dir / "train_edges.csv"
    test_path = split_dir / "test_edges.csv"
    meta_path = split_dir / "split_meta.json"

    missing_files = [path for path in (train_path, test_path, meta_path) if not path.exists()]
    if missing_files:
        missing = ", ".join(str(path) for path in missing_files)
        raise SystemExit(f"Missing split artifacts: {missing}. Run 'prepare' first.")

    train_edges = read_edges_csv(train_path)
    test_edges = read_edges_csv(test_path)
    metadata = read_json(meta_path)
    return train_edges, test_edges, metadata


if __name__ == "__main__":
    raise SystemExit(main())
