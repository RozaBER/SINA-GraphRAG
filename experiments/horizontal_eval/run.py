from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from net_rag.data import build_digraph, collect_nodes, load_raw_edges, write_json, write_rows_csv
from net_rag.evaluation import compute_metrics_for_source
from net_rag.scorers import BaseScorer
from net_rag.split import stratified_random_split

from .aggregate import AGGREGATE_FIELDS, RUN_FIELDS, SUMMARY_BY_RUN_FIELDS, aggregate_summary_rows
from .extra_scorers import build_horizontal_scorers, filter_scorers
from .plot import generate_horizontal_figures
from .sampling import SUPPORTED_NEGATIVE_STRATEGIES, build_candidate_sets_with_strategy


DEFAULT_K_VALUES = [1, 3, 5, 10, 20]
DEFAULT_SEEDS = [7, 13, 21, 42, 100]
DEFAULT_NEGATIVE_STRATEGIES = ["random", "degree_matched", "hard_2hop"]


@dataclass(frozen=True)
class HorizontalEvalConfig:
    input_path: str = "Wiki-Vote.txt"
    output_dir: str = "outputs/experiments/horizontal"
    seeds: tuple[int, ...] = tuple(DEFAULT_SEEDS)
    k_values: tuple[int, ...] = tuple(DEFAULT_K_VALUES)
    negative_strategies: tuple[str, ...] = tuple(DEFAULT_NEGATIVE_STRATEGIES)
    negatives_per_source: int = 100
    test_ratio: float = 0.2
    algorithms: tuple[str, ...] = ("all",)
    max_sources: int | None = None


def run_horizontal_evaluation(config: HorizontalEvalConfig) -> dict[str, list[dict[str, object]]]:
    output_dir = Path(config.output_dir)
    raw_edges = load_raw_edges(config.input_path)
    run_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for seed in config.seeds:
        train_edges, test_edges = stratified_random_split(raw_edges, test_ratio=config.test_ratio, seed=seed)
        if config.max_sources is not None:
            test_edges = _limit_test_edges_by_source(test_edges, max_sources=config.max_sources)

        all_nodes = collect_nodes(train_edges + test_edges)
        full_edge_set = set(train_edges) | set(test_edges)
        train_graph = build_digraph(train_edges)

        for strategy in config.negative_strategies:
            run_id = f"seed{seed}_{strategy}"
            run_start = time.perf_counter()
            candidate_start = time.perf_counter()
            candidate_sets = build_candidate_sets_with_strategy(
                test_edges=test_edges,
                all_nodes=all_nodes,
                full_edge_set=full_edge_set,
                train_graph=train_graph,
                negatives_per_source=config.negatives_per_source,
                seed=seed,
                strategy=strategy,
            )
            candidate_seconds = time.perf_counter() - candidate_start
            positive_candidates, negative_candidates = _count_candidate_labels(candidate_sets)

            # 每个策略单独构建 scorer，避免 PPR 缓存让后续策略的耗时偏低。
            scorers = filter_scorers(
                build_horizontal_scorers(train_graph, seed=seed),
                list(config.algorithms),
            )
            for scorer in scorers:
                scorer_start = time.perf_counter()
                ranked_sets = _rank_candidate_sets(candidate_sets, scorer)
                scorer_seconds = time.perf_counter() - scorer_start

                for k in config.k_values:
                    summary_rows.append(
                        _build_summary_row(
                            run_id=run_id,
                            seed=seed,
                            test_ratio=config.test_ratio,
                            strategy=strategy,
                            negatives_per_source=config.negatives_per_source,
                            k=k,
                            scorer=scorer,
                            ranked_sets=ranked_sets,
                            positive_candidates=positive_candidates,
                            negative_candidates=negative_candidates,
                            runtime_seconds=scorer_seconds,
                        )
                    )

            run_rows.append(
                {
                    "run_id": run_id,
                    "seed": seed,
                    "test_ratio": config.test_ratio,
                    "negative_strategy": strategy,
                    "negatives_per_source": config.negatives_per_source,
                    "sources_evaluated": len(candidate_sets),
                    "positive_candidates": positive_candidates,
                    "negative_candidates": negative_candidates,
                    "candidate_build_seconds": round(candidate_seconds, 6),
                    "total_runtime_seconds": round(time.perf_counter() - run_start, 6),
                }
            )

    aggregate_rows = aggregate_summary_rows(summary_rows)
    _write_outputs(output_dir, config, run_rows, summary_rows, aggregate_rows)
    generated_figures = generate_horizontal_figures(output_dir)

    print(f"Saved horizontal evaluation outputs to {output_dir}")
    for figure in generated_figures:
        print(figure)

    return {
        "runs": run_rows,
        "summary_by_run": summary_rows,
        "summary_mean_std": aggregate_rows,
    }


def _limit_test_edges_by_source(test_edges: list[tuple[int, int]], *, max_sources: int) -> list[tuple[int, int]]:
    selected_sources = set()
    limited_edges: list[tuple[int, int]] = []
    for source, target in test_edges:
        if source not in selected_sources and len(selected_sources) >= max_sources:
            continue
        selected_sources.add(source)
        limited_edges.append((source, target))
    return limited_edges


def _count_candidate_labels(candidate_sets: dict[int, list[tuple[int, int]]]) -> tuple[int, int]:
    positive = 0
    negative = 0
    for candidates in candidate_sets.values():
        for _, label in candidates:
            if label == 1:
                positive += 1
            else:
                negative += 1
    return positive, negative


def _rank_candidate_sets(
    candidate_sets: dict[int, list[tuple[int, int]]],
    scorer: BaseScorer,
) -> list[tuple[int, list[int], dict[int, int]]]:
    ranked_sets: list[tuple[int, list[int], dict[int, int]]] = []
    for source in sorted(candidate_sets):
        labeled_candidates = candidate_sets[source]
        candidate_nodes = [target for target, _ in labeled_candidates]
        labels_by_target = {target: label for target, label in labeled_candidates}
        scores = scorer.score(source, candidate_nodes)
        ranked_targets = sorted(candidate_nodes, key=lambda target: (-scores.get(target, 0.0), target))
        ranked_sets.append((source, ranked_targets, labels_by_target))
    return ranked_sets


def _build_summary_row(
    *,
    run_id: str,
    seed: int,
    test_ratio: float,
    strategy: str,
    negatives_per_source: int,
    k: int,
    scorer: BaseScorer,
    ranked_sets: list[tuple[int, list[int], dict[int, int]]],
    positive_candidates: int,
    negative_candidates: int,
    runtime_seconds: float,
) -> dict[str, object]:
    precision_values: list[float] = []
    hit_rate_values: list[float] = []
    mrr_values: list[float] = []

    for _, ranked_targets, labels_by_target in ranked_sets:
        precision, hit_rate, reciprocal_rank = compute_metrics_for_source(
            ranked_targets=ranked_targets,
            labels_by_target=labels_by_target,
            k=k,
        )
        precision_values.append(precision)
        hit_rate_values.append(hit_rate)
        mrr_values.append(reciprocal_rank)

    source_count = len(ranked_sets)
    return {
        "run_id": run_id,
        "seed": seed,
        "test_ratio": test_ratio,
        "negative_strategy": strategy,
        "negatives_per_source": negatives_per_source,
        "k": k,
        "algorithm": scorer.name,
        "sources_evaluated": source_count,
        "precision_at_k": _mean_or_zero(precision_values),
        "hit_rate_at_k": _mean_or_zero(hit_rate_values),
        "mrr": _mean_or_zero(mrr_values),
        "positive_candidates": positive_candidates,
        "negative_candidates": negative_candidates,
        "runtime_seconds": round(runtime_seconds, 6),
    }


def _mean_or_zero(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def _write_outputs(
    output_dir: Path,
    config: HorizontalEvalConfig,
    run_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    aggregate_rows: list[dict[str, object]],
) -> None:
    write_rows_csv(output_dir / "runs.csv", RUN_FIELDS, run_rows)
    write_rows_csv(output_dir / "summary_by_run.csv", SUMMARY_BY_RUN_FIELDS, summary_rows)
    write_rows_csv(output_dir / "summary_mean_std.csv", AGGREGATE_FIELDS, aggregate_rows)
    write_json(output_dir / "configs" / "default_config.json", asdict(config))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run horizontal evaluation for Net RAG graph retrieval algorithms.")
    parser.add_argument("--input", default="Wiki-Vote.txt", help="Path to the raw Wiki-Vote dataset.")
    parser.add_argument("--output-dir", default="outputs/experiments/horizontal", help="Directory for experiment outputs.")
    parser.add_argument("--seeds", default=_join(DEFAULT_SEEDS), help="Comma-separated random seeds.")
    parser.add_argument("--k-values", default=_join(DEFAULT_K_VALUES), help="Comma-separated K values.")
    parser.add_argument(
        "--negative-strategies",
        default=",".join(DEFAULT_NEGATIVE_STRATEGIES),
        help=f"Comma-separated strategies: {', '.join(SUPPORTED_NEGATIVE_STRATEGIES)}.",
    )
    parser.add_argument("--negatives-per-source", type=int, default=100, help="Negative candidates sampled per source.")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Held-out edge ratio for each seed.")
    parser.add_argument("--algorithms", default="all", help="Comma-separated algorithm names, or 'all'.")
    parser.add_argument("--max-sources", type=int, default=None, help="Optional cap for quick smoke runs.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = HorizontalEvalConfig(
        input_path=args.input,
        output_dir=args.output_dir,
        seeds=tuple(_parse_int_list(args.seeds)),
        k_values=tuple(_parse_int_list(args.k_values)),
        negative_strategies=tuple(_parse_str_list(args.negative_strategies)),
        negatives_per_source=args.negatives_per_source,
        test_ratio=args.test_ratio,
        algorithms=tuple(_parse_str_list(args.algorithms)),
        max_sources=args.max_sources,
    )
    run_horizontal_evaluation(config)
    return 0


def _parse_int_list(value: str) -> list[int]:
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("Expected at least one integer value.")
    return parsed


def _parse_str_list(value: str) -> list[str]:
    parsed = [item.strip() for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("Expected at least one string value.")
    return parsed


def _join(values: list[int]) -> str:
    return ",".join(str(value) for value in values)


if __name__ == "__main__":
    raise SystemExit(main())
