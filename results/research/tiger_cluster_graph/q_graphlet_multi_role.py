"""Per-edge Graphlet roles on the quantized KNN graph.

The q KNN graph is the pure-Euclidean TIGER q, K=10, symmetrised union.
PGD supplies triangle / K4 / 4-cycle / 4-path; K1,3 star comes from the exact
verified counter, never PGD's local 3-star.

Questions:
  1. How often does one KNN edge carry several Graphlet roles?
  2. For real behaviour edges retained in q-KNN, how do their q Graphlet roles
     compare with their behaviour-graph roles?

    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        results/research/tiger_cluster_graph/q_graphlet_multi_role.py
"""
from __future__ import annotations

import gzip
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
Q_VECTORS = REPO / "results" / "research" / "tiger_cluster_graph" / "tiger_vectors.npz"
BEHAVIOUR_CSV = REPO / "results" / "research" / "tiger_cluster_graph" / "pgd_behavior_edges.csv"
BEHAVIOUR_MICRO = REPO / "results" / "research" / "tiger_cluster_graph" / "pgd_behavior.micro"
BEHAVIOUR_K13 = REPO / "results" / "research" / "tiger_cluster_graph" / "k13_edges.csv"
PGD = REPO / "tools" / "research" / "third_party" / "PGD" / "pgd"
K13 = REPO / "results" / "research" / "tiger_cluster_graph" / "k13_exact"
OUT_DIR = REPO / "results" / "research" / "q_graphlet_multi_role"
Q_EDGES = OUT_DIR / "q_knn_k10_edges.csv"
Q_MICRO = OUT_DIR / "q_knn_k10.micro"
Q_K13 = OUT_DIR / "q_knn_k10_k13.csv"
Q_COUNTS = OUT_DIR / "q_knn_k10_edge_graphlet_counts.csv.gz"
RETAINED = OUT_DIR / "retained_behaviour_graphlet_counts.csv"
OUT_JSON = OUT_DIR / "q_graphlet_multi_role.json"
OUT_MD = OUT_DIR / "q_graphlet_multi_role.md"
K = 10
ROLES = ("triangle", "k4", "cycle", "star", "chain")
ROLE_LABELS = {"triangle": "triangle", "k4": "K4", "cycle": "4-cycle",
               "star": "K1,3 star", "chain": "4-path"}
PGD_COLUMNS = {"triangle": 2, "k4": 4, "cycle": 7, "chain": 9}


def read_edge_csv(path: Path, header: bool = False) -> np.ndarray:
    rows = []
    with path.open() as handle:
        for line in handle:
            if line.startswith("%") or line.startswith("src"):
                continue
            values = line.replace(",", " ").split()
            if len(values) < 2:
                continue
            left, right = int(values[0]), int(values[1])
            if left != right:
                rows.append((min(left, right), max(left, right)))
    return np.unique(np.asarray(rows, dtype=np.int64), axis=0)


def build_q_knn(q: np.ndarray) -> np.ndarray:
    model = NearestNeighbors(n_neighbors=K + 1, metric="euclidean", algorithm="auto", n_jobs=-1)
    model.fit(q)
    raw = model.kneighbors(q, return_distance=False)
    directed = np.empty((len(q), K), dtype=np.int64)
    for index, row in enumerate(raw):
        row = row[row != index]
        if len(row) < K:
            raise ValueError(f"only {len(row)} non-self q neighbours for {index}")
        directed[index] = row[:K]
    left = np.repeat(np.arange(len(q), dtype=np.int64), K)
    right = directed.reshape(-1)
    return np.unique(np.sort(np.column_stack([left, right]), axis=1), axis=0)


def write_edges(path: Path, edges: np.ndarray) -> None:
    with path.open("w") as handle:
        for left, right in edges.tolist():
            handle.write(f"{int(left)},{int(right)}\n")


def parse_micro(path: Path, edge_path: Path, exact_path: Path,
                edge_array: np.ndarray) -> pd.DataFrame:
    lookup, inverse = {}, []
    with edge_path.open() as handle:
        for line in handle:
            values = line.replace(",", " ").split()
            if len(values) < 2:
                continue
            left, right = int(values[0]), int(values[1])
            for vertex in (left, right):
                if vertex not in lookup:
                    lookup[vertex] = len(inverse)
                    inverse.append(vertex)
    micro = pd.read_csv(path, skiprows=1, header=None)
    counts = {}
    for values in micro.itertuples(index=False, name=None):
        left = inverse[int(values[0]) - 1]
        right = inverse[int(values[1]) - 1]
        pair = (left, right) if left < right else (right, left)
        counts[pair] = {name: int(values[column]) for name, column in PGD_COLUMNS.items()}
    exact = pd.read_csv(exact_path)
    for left, right, value in exact[["src", "dst", "k13"]].itertuples(index=False):
        pair = (int(left), int(right)) if int(left) < int(right) else (int(right), int(left))
        counts[pair]["star"] = int(value)
    rows = []
    for left, right in edge_array:
        pair = (int(left), int(right))
        values = counts.get(pair)
        if values is None:
            raise ValueError(f"missing graphlet edge {pair}")
        rows.append({"left": int(left), "right": int(right),
                     **{role: values[role] for role in ROLES}})
    return pd.DataFrame(rows)


def rank_within_bins(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    output = np.zeros(len(values), dtype=float)
    for value in np.unique(bins):
        mask = bins == value
        order = np.argsort(values[mask], kind="mergesort")
        sorted_values = values[mask][order]
        starts = np.concatenate([[0], np.flatnonzero(np.diff(sorted_values) != 0) + 1])
        ends = np.concatenate([starts[1:], [len(sorted_values)]]) - 1
        ranks = np.empty(mask.sum(), dtype=float)
        for start, end in zip(starts, ends):
            ranks[order[start:end + 1]] = (start + end) / 2.0
        output[mask] = ranks / max(len(ranks) - 1, 1)
    return output


def high_role_summary(frame: pd.DataFrame, edges: np.ndarray) -> dict:
    degree = np.bincount(edges.ravel(), minlength=int(edges.max()) + 1)
    dmax = np.maximum(degree[frame["left"].to_numpy()], degree[frame["right"].to_numpy()])
    bins = pd.qcut(dmax, 5, labels=False, duplicates="drop")
    high = np.column_stack([
        rank_within_bins(frame[role].to_numpy(dtype=float), np.asarray(bins)) >= 0.75
        for role in ROLES
    ])
    role_count = high.sum(axis=1)
    combinations = Counter(tuple(row) for row in high.tolist())
    return {
        "role_count_histogram": {str(int(n)): int((role_count == n).sum()) for n in range(len(ROLES) + 1)},
        "multi_role_edges": int((role_count >= 2).sum()),
        "at_least_three_roles": int((role_count >= 3).sum()),
        "top_combinations": [
            {"roles": role_string(bits), "edges": int(count)}
            for bits, count in combinations.most_common(12)
        ],
        "high_bits": high,
    }


def role_string(bits: tuple[bool, ...]) -> str:
    names = [ROLE_LABELS[role] for role, present in zip(ROLES, bits) if present]
    return " + ".join(names) if names else "none"


def combination_summary(frame: pd.DataFrame) -> dict:
    bits = np.column_stack([(frame[role].to_numpy() > 0) for role in ROLES])
    role_count = bits.sum(axis=1)
    combinations = Counter(tuple(row) for row in bits.tolist())
    top = []
    for bits_tuple, count in combinations.most_common(20):
        top.append({"roles": role_string(bits_tuple), "edges": int(count)})
    return {
        "edges": int(len(frame)),
        "role_count_histogram": {str(int(n)): int((role_count == n).sum()) for n in range(len(ROLES) + 1)},
        "multi_role_edges": int((role_count >= 2).sum()),
        "at_least_three_roles": int((role_count >= 3).sum()),
        "top_combinations": top,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not Q_EDGES.exists():
        write_edges(Q_EDGES, build_q_knn(np.load(Q_VECTORS)["q"].astype(np.float32)))
    if not Q_MICRO.exists():
        with (OUT_DIR / "q_knn_k10.pgd.log").open("w") as log:
            subprocess.run([str(PGD), "-f", str(Q_EDGES), "-m", str(Q_MICRO),
                            "-w", "8", "-v"], stdout=log, stderr=subprocess.STDOUT, check=True)
    if not Q_K13.exists():
        subprocess.run([str(K13), str(Q_EDGES), str(Q_K13),
                        str(OUT_DIR / "q_knn_k10_k13_summary.txt")], check=True)

    q_edges = read_edge_csv(Q_EDGES)
    behaviour_edges = read_edge_csv(BEHAVIOUR_CSV)
    q_counts = parse_micro(Q_MICRO, Q_EDGES, Q_K13, q_edges)
    b_counts = parse_micro(BEHAVIOUR_MICRO, BEHAVIOUR_CSV, BEHAVIOUR_K13, behaviour_edges)
    q_set = {(int(row.left), int(row.right)) for row in q_counts.itertuples(index=False)}
    b_set = {(int(row.left), int(row.right)) for row in b_counts.itertuples(index=False)}

    q_counts.to_csv(Q_COUNTS, index=False, compression="gzip")
    q_roles = combination_summary(q_counts)
    b_roles = combination_summary(b_counts)
    q_high = high_role_summary(q_counts, q_edges)
    b_high = high_role_summary(b_counts, behaviour_edges)

    retained_mask = np.array([(int(row.left), int(row.right)) in q_set
                              for row in b_counts.itertuples(index=False)])
    retained_indices = np.flatnonzero(retained_mask)
    q_lookup = {(int(row.left), int(row.right)): row for row in q_counts.itertuples(index=False)}
    b_lookup = {(int(row.left), int(row.right)): row for row in b_counts.itertuples(index=False)}
    q_high_bits = q_high.pop("high_bits")
    b_high_bits = b_high.pop("high_bits")
    b_high_count = b_high_bits.sum(axis=1)
    retention_by_high_roles = {}
    for number in range(len(ROLES) + 1):
        mask = b_high_count == number
        retention_by_high_roles[str(number)] = {
            "n": int(mask.sum()), "retained": int(retained_mask[mask].sum()),
            "rate": float(retained_mask[mask].mean()) if mask.any() else float("nan"),
        }
    retained_rows = []
    for index in retained_indices:
        row = b_counts.iloc[index]
        pair = (int(row.left), int(row.right))
        q_value = q_lookup[pair]
        b_value = b_lookup[pair]
        retained_rows.append({
            "left": pair[0], "right": pair[1],
            **{f"b_{role}": int(getattr(b_value, role)) for role in ROLES},
            **{f"q_{role}": int(getattr(q_value, role)) for role in ROLES},
        })
    retained = pd.DataFrame(retained_rows)
    retained.to_csv(RETAINED, index=False)

    # Retention by the number of roles in the behaviour graph.
    b_bits = np.column_stack([(b_counts[role].to_numpy() > 0) for role in ROLES])
    b_role_count = b_bits.sum(axis=1)
    retention_by_roles = {}
    for number in range(len(ROLES) + 1):
        mask = b_role_count == number
        retention_by_roles[str(number)] = {
            "n": int(mask.sum()),
            "retained": int(retained_mask[mask].sum()),
            "rate": float(retained_mask[mask].mean()) if mask.any() else float("nan"),
        }
    retention_by_role = {}
    for role in ROLES:
        mask = b_counts[role].to_numpy() > 0
        retention_by_role[role] = {
            "n": int(mask.sum()),
            "retained": int(retained_mask[mask].sum()),
            "rate": float(retained_mask[mask].mean()) if mask.any() else float("nan"),
        }

    # Counts of retained edges: q vs behaviour, per role.
    retained_compare = {}
    for role in ROLES:
        b_values = retained[f"b_{role}"].to_numpy(dtype=float)
        q_values = retained[f"q_{role}"].to_numpy(dtype=float)
        retained_compare[role] = {
            "n": int(len(retained)),
            "behavior_mean": float(b_values.mean()),
            "q_mean": float(q_values.mean()),
            "behavior_median": float(np.median(b_values)),
            "q_median": float(np.median(q_values)),
            "spearman": float(pd.Series(b_values).corr(pd.Series(q_values), method="spearman"))
            if len(set(b_values)) > 1 and len(set(q_values)) > 1 else float("nan"),
            "q_presence_rate": float(np.mean(q_values > 0)),
            "behavior_presence_rate": float(np.mean(b_values > 0)),
        }

    result = {
        "q_graph_edges": int(len(q_edges)),
        "behaviour_graph_edges": int(len(behaviour_edges)),
        "retained_edges": int(len(retained)),
        "retention_rate": float(retained_mask.mean()),
        "roles": {role: ROLE_LABELS[role] for role in ROLES},
        "q_role_summary": q_roles,
        "behavior_role_summary": b_roles,
        "retention_by_behavior_role_count": retention_by_roles,
        "retention_by_behavior_role": retention_by_role,
        "high_role_q": q_high,
        "high_role_behavior": b_high,
        "retention_by_high_role_count": retention_by_high_roles,
        "retained_count_comparison": retained_compare,
        "q_counts_file": str(Q_COUNTS.relative_to(REPO)),
        "retained_file": str(RETAINED.relative_to(REPO)),
    }
    OUT_JSON.write_text(json.dumps(result, indent=2))

    lines = [
        "# Per-edge Graphlet roles on the quantized KNN graph",
        "",
        "q graph: pure-Euclidean TIGER q, K=10, symmetrised union. Triangle/K4/4-cycle/4-path",
        "from PGD micro; K1,3 star from the exact verified counter; PGD local 3-star not used.",
        "",
        f"q graph edges **{len(q_edges)}**; behaviour edges **{len(behaviour_edges)}**; "
        f"retained **{len(retained)}** ({retained_mask.mean():.3%}).",
        "",
        "## Multi-role prevalence",
        "",
        "| graph | role-count 0 | 1 | 2 | 3 | 4 | 5 | multi-role |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, summary in (("q KNN", q_roles), ("behaviour", b_roles)):
        histogram = summary["role_count_histogram"]
        lines.append(f"| {name} | " + " | ".join(str(histogram[str(n)]) for n in range(6)) +
                     f" | {summary['multi_role_edges']} |")
    lines += [
        "",
        "## Top role combinations (raw presence)",
        "",
        "| graph | combination | edges |",
        "|---|---|---:|",
    ]
    for name, summary in (("q KNN", q_roles), ("behaviour", b_roles)):
        for item in summary["top_combinations"][:10]:
            lines.append(f"| {name} | {item['roles']} | {item['edges']} |")
    lines += [
        "",
        "## High-involvement roles (top quartile within dmax bins)",
        "",
        "| graph | 0 | 1 | 2 | 3 | 4 | 5 | multi-role |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, summary in (("q KNN", q_high), ("behaviour", b_high)):
        histogram = summary["role_count_histogram"]
        lines.append(f"| {name} | " + " | ".join(str(histogram[str(n)]) for n in range(6)) +
                     f" | {summary['multi_role_edges']} |")
    lines += [
        "",
        "## Retention by number of behaviour-graph roles",
        "",
        "| behaviour roles | n | retained | rate |",
        "|---|---:|---:|---:|",
    ]
    for number, value in retention_by_roles.items():
        lines.append(f"| {number} | {value['n']} | {value['retained']} | {value['rate']:.4f} |")
    lines += [
        "",
        "## Retention by high-involvement role count",
        "",
        "| high roles | n | retained | rate |",
        "|---|---:|---:|---:|",
    ]
    for number, value in retention_by_high_roles.items():
        lines.append(f"| {number} | {value['n']} | {value['retained']} | {value['rate']:.4f} |")
    lines += [
        "",
        "## Retained edges: q vs behaviour counts",
        "",
        "| role | behaviour mean | q mean | Spearman | q presence |",
        "|---|---:|---:|---:|---:|",
    ]
    for role in ROLES:
        value = retained_compare[role]
        lines.append(f"| {ROLE_LABELS[role]} | {value['behavior_mean']:.3f} | {value['q_mean']:.3f} | "
                     f"{value['spearman']:+.3f} | {value['q_presence_rate']:.3f} |")
    lines += [
        "",
        "`q_counts_file` contains all q-edge counts; `retained_file` contains the retained",
        "behaviour-edge counts before/after quantization.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_JSON} and {OUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
