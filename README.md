# Net RAG: Simplified GraphRAG Retrieval on Wiki-Vote

English | [简体中文](README.zh-CN.md)

This project implements a simplified GraphRAG-style retrieval module for the Wikipedia Vote Network. It does not build a full LLM question-answering system. Instead, it treats link prediction as context retrieval: if the system predicts that user `A` is likely to connect to user `B`, then user `B` is considered relevant graph context for user `A`.

## What This Project Does

- Parses the raw SNAP `Wiki-Vote.txt` dataset.
- Builds a reproducible stratified random train/test split.
- Samples negative candidate edges for efficient link prediction evaluation.
- Scores candidate links with four graph-based algorithms.
- Evaluates Top-K retrieval quality with `Precision@K`, `HitRate@K`, and `MRR`.
- Produces rich result visualizations from multiple analysis angles.
- Provides a simple `retrieve` command for user-level context retrieval with structural explanations.

## Project Layout

```text
.
├── net_rag/                         # Lightweight package bridge for python -m net_rag.cli
├── scripts/
│   └── visualize_results.py         # Visualization script
├── src/net_rag/
│   ├── candidates.py                # Positive/negative candidate construction
│   ├── cli.py                       # prepare / evaluate / retrieve commands
│   ├── constants.py                 # Default parameters
│   ├── data.py                      # Dataset and artifact I/O
│   ├── evaluation.py                # Metrics and built-in comparison plot
│   ├── retrieval.py                 # User-level retrieval and explanations
│   ├── scorers.py                   # Graph scoring algorithms
│   └── split.py                     # Stratified random split
├── tests/                           # Unit and smoke tests
├── GraphRAG_Context_Retrieval_Topic2.md
├── Wiki-Vote.txt
├── requirements.txt
└── sitecustomize.py
```

Generated files are written under `outputs/`.

## Setup

Create a local virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

The repository includes a lightweight package bridge, so commands can be run from the repository root:

```powershell
.\.venv\Scripts\python -m net_rag.cli --help
```

## Workflow

### 1. Prepare Train/Test Splits

```powershell
.\.venv\Scripts\python -m net_rag.cli prepare --input Wiki-Vote.txt --seed 42
```

Outputs:

- `outputs/splits/train_edges.csv`
- `outputs/splits/test_edges.csv`
- `outputs/splits/split_meta.json`

The split is stratified by source node. For each source node with at least two outgoing edges, about 20% of its edges are hidden for testing while at least one training edge is kept.

### 2. Evaluate Retrieval Algorithms

```powershell
.\.venv\Scripts\python -m net_rag.cli evaluate --k 10
```

Outputs:

- `outputs/metrics/summary.csv`
- `outputs/rankings/common_neighbors.csv`
- `outputs/rankings/jaccard.csv`
- `outputs/rankings/adamic_adar.csv`
- `outputs/rankings/personalized_pagerank.csv`
- `outputs/figures/algorithm_compare.png`

### 3. Retrieve Context for One User

```powershell
.\.venv\Scripts\python -m net_rag.cli retrieve --user-id 30 --algo ppr --k 10
```

Output:

- `outputs/retrievals/user_30_ppr.csv`

The retrieval output includes the recommended target node, score, rank, algorithm name, and a short graph-based explanation such as a 2-hop path.

### 4. Generate Visualizations

```powershell
.\.venv\Scripts\python scripts\visualize_results.py --output-dir outputs --max-k 20 --user-id 30 --algo ppr
```

Outputs:

- `outputs/figures/metrics_summary.png`
- `outputs/figures/train_test_split.png`
- `outputs/figures/precision_at_k_curve.png`
- `outputs/figures/hit_rate_at_k_curve.png`
- `outputs/figures/precision_at_k_heatmap.png`
- `outputs/figures/first_hit_rank_cdf.png`
- `outputs/figures/score_distribution_positive_negative.png`
- `outputs/figures/test_positives_per_source_distribution.png`
- `outputs/figures/train_degree_distribution.png`
- `outputs/figures/user_30_ppr_retrieval_network.png`

All visualizations use this palette:

```text
#1c2b7c
#42559a
#6e7eb7
#9da8d4
#cfd3f1
```

## Algorithms

The project compares four scoring algorithms:

- `common_neighbors`
- `jaccard`
- `adamic_adar`
- `personalized_pagerank`

The first three local similarity methods are computed on an undirected projection of the training graph. This reduces the number of zero-score pairs in a sparse directed graph.

`personalized_pagerank` uses an approximate random-walk implementation instead of running exact PageRank once per source node. This keeps the full Wiki-Vote evaluation practical while preserving the core idea of source-centered graph relevance propagation.

## Evaluation Protocol

For each source node that appears in the test set:

1. Use all hidden test edges from that source as positive candidates.
2. Sample random non-existent edges as negative candidates.
3. Score the candidate targets with each algorithm.
4. Rank candidates by score.
5. Compute Top-K retrieval metrics.

The main metric is macro `Precision@10`. The project also reports `HitRate@10` and `MRR`.

## Current Full-Dataset Result

Using `Wiki-Vote.txt`, `seed=42`, `K=10`, and `100` negative samples per source:

| Algorithm | Precision@10 | HitRate@10 | MRR |
| --- | ---: | ---: | ---: |
| `common_neighbors` | 0.229641 | 0.711373 | 0.493400 |
| `jaccard` | 0.219742 | 0.698766 | 0.420111 |
| `adamic_adar` | 0.230472 | 0.714056 | 0.502139 |
| `personalized_pagerank` | 0.239297 | 0.791845 | 0.551049 |

The prepared split contains:

- `82,822` training edges
- `20,867` test edges
- `103,689` total unique edges

## Tests

Run the test suite:

```powershell
.\.venv\Scripts\python -m pytest
```

The tests cover:

- raw edge parsing
- deterministic train/test splitting
- candidate sampling
- scoring behavior
- small end-to-end CLI workflow

## Notes

- `outputs/` is ignored by Git because it contains generated experiment artifacts.
- The repository uses relative paths in code and documentation unless a file is outside the working directory.
- The project is intentionally script-oriented and lightweight so it can be inspected, rerun, and extended easily.
