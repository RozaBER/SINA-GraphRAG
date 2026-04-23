# Net RAG：基于 Wiki-Vote 的简化 GraphRAG 检索

[English](README.md) | 简体中文

本项目实现了一个面向 Wikipedia Vote Network 的简化版 GraphRAG 风格检索模块。它不构建完整的 LLM 问答系统，而是把链路预测视为上下文检索：如果系统预测用户 `A` 很可能连接到用户 `B`，那么用户 `B` 就被视为用户 `A` 的相关图上下文。

## 项目功能

- 解析原始 SNAP `Wiki-Vote.txt` 数据集。
- 构建可复现的分层随机训练/测试划分。
- 为高效链路预测评估采样负候选边。
- 使用 4 种图算法为候选链接打分。
- 使用 `Precision@K`、`HitRate@K` 和 `MRR` 评估 Top-K 检索质量。
- 从多个分析角度生成丰富的结果可视化。
- 提供简单的 `retrieve` 命令，用于单用户上下文检索，并给出结构化解释。

## 项目结构

```text
.
├── net_rag/                         # 支持 python -m net_rag.cli 的轻量桥接包
├── scripts/
│   └── visualize_results.py         # 可视化脚本
├── src/net_rag/
│   ├── candidates.py                # 正/负候选样本构造
│   ├── cli.py                       # prepare / evaluate / retrieve 命令
│   ├── constants.py                 # 默认参数
│   ├── data.py                      # 数据集与产物 I/O
│   ├── evaluation.py                # 指标计算与内置对比图
│   ├── retrieval.py                 # 单用户检索与解释
│   ├── scorers.py                   # 图打分算法
│   └── split.py                     # 分层随机划分
├── tests/                           # 单元测试与烟雾测试
├── GraphRAG_Context_Retrieval_Topic2.md
├── Wiki-Vote.txt
├── requirements.txt
└── sitecustomize.py
```

生成文件会写入 `outputs/`。

## 环境配置

创建本地虚拟环境并安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

仓库包含一个轻量桥接包，因此可以在仓库根目录直接运行命令：

```powershell
.\.venv\Scripts\python -m net_rag.cli --help
```

## 工作流程

### 1. 准备训练/测试划分

```powershell
.\.venv\Scripts\python -m net_rag.cli prepare --input Wiki-Vote.txt --seed 42
```

输出文件：

- `outputs/splits/train_edges.csv`
- `outputs/splits/test_edges.csv`
- `outputs/splits/split_meta.json`

划分方式按源节点分层进行。对于至少有 2 条出边的源节点，大约 20% 的出边会被隐藏为测试边，同时至少保留 1 条训练边。

### 2. 评估检索算法

```powershell
.\.venv\Scripts\python -m net_rag.cli evaluate --k 10
```

输出文件：

- `outputs/metrics/summary.csv`
- `outputs/rankings/common_neighbors.csv`
- `outputs/rankings/jaccard.csv`
- `outputs/rankings/adamic_adar.csv`
- `outputs/rankings/personalized_pagerank.csv`
- `outputs/figures/algorithm_compare.png`

### 3. 为单个用户检索上下文

```powershell
.\.venv\Scripts\python -m net_rag.cli retrieve --user-id 30 --algo ppr --k 10
```

输出文件：

- `outputs/retrievals/user_30_ppr.csv`

检索输出包含推荐目标节点、分数、排名、算法名，以及简短的图结构解释，例如 2-hop 路径。

### 4. 生成可视化

```powershell
.\.venv\Scripts\python scripts\visualize_results.py --output-dir outputs --max-k 20 --user-id 30 --algo ppr
```

输出文件：

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

所有可视化统一使用以下配色：

```text
#1c2b7c
#42559a
#6e7eb7
#9da8d4
#cfd3f1
```

## 算法

项目比较 4 种打分算法：

- `common_neighbors`
- `jaccard`
- `adamic_adar`
- `personalized_pagerank`

前三种局部相似度方法在训练图的无向投影上计算。这可以减少稀疏有向图中大量候选对得分为零的问题。

`personalized_pagerank` 使用基于随机游走的近似实现，而不是对每个源节点分别运行一次精确 PageRank。这样可以让完整 Wiki-Vote 评估保持可运行，同时保留“以源节点为中心进行图相关性传播”的核心思想。

## 评估协议

对于每个出现在测试集中的源节点：

1. 使用该源节点的全部隐藏测试边作为正候选。
2. 随机采样不存在的边作为负候选。
3. 使用每个算法为候选目标节点打分。
4. 按分数对候选节点排序。
5. 计算 Top-K 检索指标。

主指标是宏平均 `Precision@10`。项目同时报告 `HitRate@10` 和 `MRR`。

## 当前完整数据集结果

使用 `Wiki-Vote.txt`、`seed=42`、`K=10`，并为每个源节点采样 `100` 个负样本：

| 算法 | Precision@10 | HitRate@10 | MRR |
| --- | ---: | ---: | ---: |
| `common_neighbors` | 0.229641 | 0.711373 | 0.493400 |
| `jaccard` | 0.219742 | 0.698766 | 0.420111 |
| `adamic_adar` | 0.230472 | 0.714056 | 0.502139 |
| `personalized_pagerank` | 0.239297 | 0.791845 | 0.551049 |

当前划分包含：

- `82,822` 条训练边
- `20,867` 条测试边
- `103,689` 条原始唯一边

## 测试

运行测试套件：

```powershell
.\.venv\Scripts\python -m pytest
```

测试覆盖：

- 原始边解析
- 确定性的训练/测试划分
- 候选样本采样
- 打分行为
- 小规模端到端 CLI 工作流

## 备注

- `outputs/` 会被 Git 忽略，因为它包含生成的实验产物。
- 代码和文档默认使用相对路径，除非目标文件位于工作目录之外。
- 项目刻意保持脚本化和轻量化，方便检查、复现和扩展。
