from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Sequence

import networkx as nx


Edge = tuple[int, int]


def load_raw_edges(path: str | Path) -> list[Edge]:
    file_path = Path(path)
    edges: list[Edge] = []
    seen: set[Edge] = set()

    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            src_str, dst_str = stripped.split()[:2]
            edge = (int(src_str), int(dst_str))
            if edge in seen:
                continue
            seen.add(edge)
            edges.append(edge)

    return edges


def build_digraph(edges: Iterable[Edge]) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_edges_from(edges)
    return graph


def collect_nodes(edges: Iterable[Edge]) -> list[int]:
    return sorted({node for edge in edges for node in edge})


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_edges_csv(path: str | Path, edges: Sequence[Edge]) -> None:
    file_path = Path(path)
    ensure_directory(file_path.parent)
    with file_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_node", "target_node"])
        writer.writerows(edges)


def read_edges_csv(path: str | Path) -> list[Edge]:
    file_path = Path(path)
    with file_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [(int(row["source_node"]), int(row["target_node"])) for row in reader]


def write_rows_csv(path: str | Path, fieldnames: Sequence[str], rows: Sequence[dict[str, object]]) -> None:
    file_path = Path(path)
    ensure_directory(file_path.parent)
    with file_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str | Path, payload: dict[str, object]) -> None:
    file_path = Path(path)
    ensure_directory(file_path.parent)
    file_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
