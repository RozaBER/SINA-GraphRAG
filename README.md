# Net RAG: GraphRAG Context Retrieval on Wiki-Vote

English | [简体中文](README.zh-CN.md)

![Python 3.12](https://img.shields.io/badge/Python-3.12-1c2b7c?style=flat-square)
![NetworkX](https://img.shields.io/badge/NetworkX-Graph%20Algorithms-42559a?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-Wiki--Vote-6e7eb7?style=flat-square)
![Task](https://img.shields.io/badge/Task-Link%20Prediction-9da8d4?style=flat-square)
![Metrics](https://img.shields.io/badge/Metrics-Precision%40K%20%7C%20MRR-cfd3f1?style=flat-square)

This repository is a student coursework implementation for **Selected Topic 2: GraphRAG and Context Retrieval**. The project builds a simplified retrieval module for a GraphRAG system by turning context retrieval into a link prediction task on the Wikipedia Vote Network.

The main idea is simple: in a graph, relevant context may not appear through direct keyword matching. If the model can predict that user `A` is strongly related to user `B`, then user `B` can be treated as useful context for user `A`. This project implements that idea with graph algorithms, evaluates the retrieval quality, and visualizes the results.

## Project Snapshot

| Item | Description |
| --- | --- |
| Coursework topic | GraphRAG and Context Retrieval |
| Dataset | Wikipedia Vote Network, stored as `Wiki-Vote.txt` |
| Task formulation | Link prediction as context retrieval |
| Train/test split | Source-node stratified random split, about 80/20 |
| Algorithms | Common Neighbors, Jaccard, Adamic/Adar, Personalized PageRank |
| Main metric | Macro Precision@10 |
| Extra outputs | Ranking CSV files, retrieval examples, and visualization figures |

## Assignment Background

Large Language Models can hallucinate when they do not have enough reliable context. Standard RAG reduces this problem by retrieving related documents before generation, but simple keyword-based retrieval may miss indirect relationships.

GraphRAG adds graph structure to the retrieval process. Instead of only asking whether two items share similar text, it also asks whether they are connected through graph relationships. In this assignment, the graph is the Wikipedia Vote Network, and the retrieval task is framed as predicting missing links.

## Dataset

The project uses the SNAP Wikipedia Vote Network:

- **Nodes**: Wikipedia users or editors.
- **Directed edges**: an edge from user `i` to user `j` means user `i` voted for user `j` to become a Wikipedia administrator.
- **Local file used in this project**: `Wiki-Vote.txt`.

The raw file contains comments and tab-separated edge rows. The implementation parses the raw file directly, skips comment lines, removes duplicate edges, and builds a directed graph.

## Data Processing

Because the dataset does not provide timestamps, this project uses a reproducible random split instead of a time split.

The split strategy is source-node stratified:

- For each source node with at least two outgoing edges, about 20% of its outgoing edges are hidden as test edges.
- At least one outgoing edge is kept in the training graph so the source node still has graph context.
- Nodes with only one outgoing edge are kept in the training graph.

The prepared split is saved to:

- `outputs/splits/train_edges.csv`
- `outputs/splits/test_edges.csv`
- `outputs/splits/split_meta.json`

With the current default setting, the split contains:

| Split | Edge Count |
| --- | ---: |
| Training edges | 82,822 |
| Testing edges | 20,867 |
| Total unique edges | 103,689 |

## Retrieval Algorithms

The project compares four graph-based scoring algorithms:

| Algorithm | Role in this project |
| --- | --- |
| `common_neighbors` | Scores two nodes by how many neighbors they share. |
| `jaccard` | Normalizes shared neighbors by the size of the neighbor union. |
| `adamic_adar` | Gives more weight to rare shared neighbors. |
| `personalized_pagerank` | Uses source-centered random walks to estimate graph relevance. |

The first three algorithms run on an undirected projection of the training graph. This makes the local similarity scores less sparse. `personalized_pagerank` runs on the directed training graph with an approximate random-walk implementation, which keeps the full dataset evaluation practical.

## Evaluation Design

The evaluation follows a standard link prediction setup:

1. The training graph is built from the visible 80% of edges.
2. The hidden 20% of edges are treated as true missing links.
3. For each source node in the test set, the system ranks its true hidden targets together with randomly sampled non-existent targets.
4. The Top-K predictions are compared with the hidden test edges.

The main metric is **Macro Precision@10**. The project also reports:

- `HitRate@10`
- `MRR`

Current full-dataset result using `seed=42`, `K=10`, and `100` negative samples per source:

| Algorithm | Precision@10 | HitRate@10 | MRR |
| --- | ---: | ---: | ---: |
| `common_neighbors` | 0.229641 | 0.711373 | 0.493400 |
| `jaccard` | 0.219742 | 0.698766 | 0.420111 |
| `adamic_adar` | 0.230472 | 0.714056 | 0.502139 |
| `personalized_pagerank` | 0.239297 | 0.791845 | 0.551049 |

In this run, `personalized_pagerank` gives the best overall retrieval performance.

## Visualization Gallery

The figures below are generated from the current experiment outputs. Copies are stored in `docs/figures/` so they can render directly in this README.

| Metrics Summary | Precision@K Curve |
| --- | --- |
| ![Metrics Summary](docs/figures/metrics_summary.png) | ![Precision@K Curve](docs/figures/precision_at_k_curve.png) |

| Precision Heatmap | Score Distribution |
| --- | --- |
| ![Precision Heatmap](docs/figures/precision_at_k_heatmap.png) | ![Score Distribution](docs/figures/score_distribution_positive_negative.png) |

| Training Degree Distribution | Retrieval Network Example |
| --- | --- |
| ![Training Degree Distribution](docs/figures/train_degree_distribution.png) | ![Retrieval Network Example](docs/figures/user_30_ppr_retrieval_network.png) |

## Project Architecture

```text
.
├── net_rag/                         # Lightweight bridge for python -m net_rag.cli
├── scripts/
│   └── visualize_results.py         # Generates result visualizations
├── src/net_rag/
│   ├── candidates.py                # Builds positive and negative candidate sets
│   ├── cli.py                       # Command-line interface
│   ├── constants.py                 # Default parameters
│   ├── data.py                      # Dataset parsing and artifact I/O
│   ├── evaluation.py                # Metrics and built-in algorithm comparison plot
│   ├── retrieval.py                 # Single-user context retrieval and explanations
│   ├── scorers.py                   # Graph scoring algorithms
│   └── split.py                     # Stratified random train/test split
├── tests/                           # Unit and end-to-end smoke tests
├── GraphRAG_Context_Retrieval_Topic2.md
├── Wiki-Vote.txt
├── requirements.txt
└── sitecustomize.py
```

The code is organized as a small Python package under `src/net_rag/`. The command-line interface in `cli.py` connects the full workflow: data preparation, evaluation, and retrieval.

## How the Project Works

The pipeline has four main stages:

1. **Prepare**: parse `Wiki-Vote.txt`, split edges into training and testing sets, and save the split artifacts.
2. **Evaluate**: construct candidate targets for each test source node, score them with each graph algorithm, rank the results, and calculate metrics.
3. **Retrieve**: for a selected user, recommend Top-K related users that do not already have visible training edges from that source.
4. **Visualize**: generate plots that explain performance, data split behavior, score distributions, graph degree patterns, and one retrieval example network.

This structure matches the assignment goal: the project is not a full LLM application, but a working GraphRAG retrieval module that can identify relevant context from graph structure.

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

## Running the Project

Prepare the train/test split:

```powershell
.\.venv\Scripts\python -m net_rag.cli prepare --input Wiki-Vote.txt --seed 42
```

Evaluate all algorithms:

```powershell
.\.venv\Scripts\python -m net_rag.cli evaluate --k 10
```

Retrieve context for one user:

```powershell
.\.venv\Scripts\python -m net_rag.cli retrieve --user-id 30 --algo ppr --k 10
```

Generate visualizations:

```powershell
.\.venv\Scripts\python scripts\visualize_results.py --output-dir outputs --max-k 20 --user-id 30 --algo ppr
```

Run tests:

```powershell
.\.venv\Scripts\python -m pytest
```

## Outputs

Main generated files:

- `outputs/metrics/summary.csv`: evaluation metrics for all algorithms.
- `outputs/rankings/*.csv`: ranked candidate edges for each algorithm.
- `outputs/retrievals/user_30_ppr.csv`: example Top-K context retrieval output.
- `outputs/figures/*.png`: visual analysis results.

Visualization files include:

- `algorithm_compare.png`
- `metrics_summary.png`
- `train_test_split.png`
- `precision_at_k_curve.png`
- `hit_rate_at_k_curve.png`
- `precision_at_k_heatmap.png`
- `first_hit_rank_cdf.png`
- `score_distribution_positive_negative.png`
- `test_positives_per_source_distribution.png`
- `train_degree_distribution.png`
- `user_30_ppr_retrieval_network.png`

All visualizations use the required color palette:

```text
#1c2b7c
#42559a
#6e7eb7
#9da8d4
#cfd3f1
```

## What to Look For in the Results

- The metrics table shows which graph algorithm retrieves hidden links most accurately.
- The Precision@K and HitRate@K curves show how performance changes as more recommendations are allowed.
- The score distribution plot shows whether an algorithm separates true hidden edges from random negative candidates.
- The degree distribution plot shows that the training graph is highly skewed, which is typical for real networks.
- The retrieval network plot gives a small example of how graph paths can explain a recommendation.

## Testing

The test suite checks:

- raw file parsing
- deterministic train/test splitting
- negative candidate sampling
- scoring output shape and numeric validity
- small end-to-end CLI workflow

These tests are meant to show that the project can be rerun from raw data and that the main workflow is reproducible.
