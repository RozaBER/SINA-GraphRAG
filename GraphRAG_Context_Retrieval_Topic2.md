# Selected Topic 2: GraphRAG and Context Retrieval

## Description

Large Language Models (LLMs) such as ChatGPT and Gemini have shown impressive capabilities, but they still have significant limitations-specifically, they often "hallucinate" when lacking specific context or access to private data. A common solution is RAG (Retrieval-Augmented Generation), which retrieves relevant documents to ground the model's answer.

However, standard RAG relies heavily on keyword similarity and often fails at complex reasoning tasks. For example, if you ask: "How is the research of Professor A indirectly influenced by the theories of Scientist B?", a simple keyword search might fail if A and B never co-authored a paper.

This is the core idea behind GraphRAG. By structuring data as a graph, we can trace the "hidden" path from A to B through intermediate collaborators, effectively finding context that keyword search misses.

In this project, you will build a simplified "Retrieval Module" for a GraphRAG system. We will frame this as a Link Prediction problem on the Wikipedia Vote Network. The intuition is simple: if your algorithm predicts a strong link between User A and User B, it means User B is highly relevant "context" for User A. You will need to implement and compare different network analysis algorithms (such as Common Neighbors, PageRank, or Adamic/Adar) to calculate these relevance scores and predict missing connections.

## Datasets

You will use the Wikipedia Vote Network, a publicly available dataset representing the trust and administrative structure of Wikipedia.

- **Nodes:** Represent Wikipedia users (editors).
- **Edges:** A directed edge from node *i* to node *j* represents that user *i* voted for user *j* (indicating trust or relevance).

### Data Processing Requirement:

Since real-world AI systems must handle raw data, you are responsible for processing the dataset. You must implement a "Time Split" or "Random Split" strategy to create your own evaluation environment:

- **Training Set:** A subgraph containing roughly 80% of the edges (used to build your model).
- **Testing Set:** The remaining 20% of edges (masked/hidden, used to verify your predictions).

### Data Download

The dataset is available openly at the Stanford Network Analysis Project (SNAP).

- **Download Link:** https://snap.stanford.edu/data/wiki-Vote.html
- **Instructions:** Download the `wiki-Vote.txt.gz` file, unzip it, and use the text file as your input.

## Evaluations

The evaluation method focuses on the **Accuracy of Context Retrieval**, following standard Link Prediction protocols.

1. **Implementation:** You must implement at least 3 different scoring algorithms (e.g., Jaccard, Adamic/Adar, PageRank) to assign relevance scores to non-existent edges in your Training graph.
2. **Prediction:** For each node in your self-generated test set, your algorithm should rank and recommend the Top-K (e.g., Top 10) most likely links. (**Performance Tip:** The graph has ~7,000 nodes. To avoid long computation times, do not calculate scores for all possible pairs. You only need to calculate scores for the edges in your Test Set and a random sample of non-existent edges.)
3. **Metric:** We calculate the Intersection (Precision@K) of your predicted edges against the true edges in the Testing set.
4. **Metric Logic:** The higher the intersection, the better your "GraphRAG engine" is at predicting relevant context.
