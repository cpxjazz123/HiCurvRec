"""Structure retention under the project's own structure definition: ORCA 5-node orbits.

The earlier section built its "structures" by matching small patterns by hand -- a triangle, a star,
a four-cycle.  That is not the structure definition this repository uses: the 213 audit and
`structure_retention.py` define structure by **ORCA's node orbits of graphlets up to five nodes** on
the train behaviour graph, grouped into six roles by NMF, and attribute every behaviour edge to the
role of its first endpoint (the history item).  This script keeps that definition and puts the
distance-based retention on top of it, so the retention table and the correlation are comparable with
`structure_retention_5node.json` rather than with a hand-built taxonomy.

Per pair (a Stage2 SID table plus the Stage3 recall produced from it), per role:

    prefix@1/2/3   both endpoints agree on the same SID position (the canonical retention)
    win rate       the behaviour neighbour (u,v) is closer than a random non-neighbour w of u in the
                   SID tree distance; a tie scores 0.5, so 0.5 is the chance line

and then the correlation of each series with `test_recall@10`: Spearman, Pearson, both permutation
p-values, plus the partial coefficient controlling the level-3 collision rate.

    python3 analysis_sid_recall/role_distance_vs_recall.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ROLES_MODULE = REPO / "results" / "research" / "issue213" / "graphlet_roles.py"
LEDGER = REPO / "results" / "research" / "issue139" / "r2_ledger.json"
MIRROR = HERE / "r2"
ROLE_CACHE5 = HERE / "structure_roles5.npz"
OUT = HERE / "role_distance_vs_recall.json"
RECALL = "test_recall@10"
LEVELS = ("prefix@1", "prefix@2", "prefix@3")
N_ROLES = 6
SEED = 42
PERMUTATIONS = 10_000


def load_roles_module():
    spec = importlib.util.spec_from_file_location("graphlet_roles", ROLES_MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    change = np.flatnonzero(np.diff(sorted_values) != 0) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(values)]]) - 1
    average = (starts + ends) / 2.0
    ranks_sorted = np.repeat(average, ends - starts + 1)
    ranks = np.empty(len(values))
    ranks[order] = ranks_sorted
    return ranks


def spearman(first: np.ndarray, second: np.ndarray) -> float:
    rank_a = average_ranks(first)
    rank_b = average_ranks(second)
    rank_a = rank_a - rank_a.mean()
    rank_b = rank_b - rank_b.mean()
    denominator = float(np.sqrt((rank_a ** 2).sum() * (rank_b ** 2).sum()))
    return float((rank_a * rank_b).sum() / denominator) if denominator else float("nan")


def pearson(first: np.ndarray, second: np.ndarray) -> float:
    centred_a = first - first.mean()
    centred_b = second - second.mean()
    denominator = float(np.sqrt((centred_a ** 2).sum() * (centred_b ** 2).sum()))
    return float((centred_a * centred_b).sum() / denominator) if denominator else float("nan")


def partial(first: np.ndarray, second: np.ndarray, control: np.ndarray, rank: bool) -> float:
    if rank:
        first, second, control = average_ranks(first), average_ranks(second), average_ranks(control)
    design = np.column_stack([np.ones_like(control), control])
    residual_a = first - design @ np.linalg.lstsq(design, first, rcond=None)[0]
    residual_b = second - design @ np.linalg.lstsq(design, second, rcond=None)[0]
    return pearson(residual_a, residual_b)


def permutation_p(first: np.ndarray, second: np.ndarray, observed: dict) -> dict:
    rng = np.random.default_rng(SEED)
    beats = {"spearman": 0, "pearson": 0}
    for _ in range(PERMUTATIONS):
        shuffled = rng.permutation(second)
        if abs(spearman(first, shuffled)) >= abs(observed["spearman"]):
            beats["spearman"] += 1
        if abs(pearson(first, shuffled)) >= abs(observed["pearson"]):
            beats["pearson"] += 1
    return {key: (value + 1) / (PERMUTATIONS + 1) for key, value in beats.items()}


def lcp_matrix(tokens: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Longest common prefix length of each (left, right) pair, vectorised over the pairs."""
    total = np.zeros(left.shape[0], dtype=np.int64)
    alive = np.ones(left.shape[0], dtype=bool)
    for level in range(3):
        alive &= (tokens[left, level] == tokens[right, level]) & (tokens[left, level] >= 0)
        total += alive.astype(np.int64)
    return total


def main() -> int:
    gr = load_roles_module()
    report = json.loads((HERE / "structure_retention_5node.json").read_text())
    ledger = json.loads(LEDGER.read_text())

    edges = np.asarray(gr.edges_of(), dtype=np.int64)
    n_items = 24587
    nodes = np.load(ROLE_CACHE5)["nodes"]
    roles = np.load(ROLE_CACHE5)["role"]
    role_arr = np.full(n_items, -1, dtype=np.int64)
    for node, role in zip(nodes, roles):
        if 0 <= int(node) < n_items:
            role_arr[int(node)] = int(role)

    u, v = edges[:, 0], edges[:, 1]
    keep = (u < n_items) & (v < n_items) & (role_arr[u] >= 0)
    u, v = u[keep], v[keep]
    role_of_u = role_arr[u]

    # one random non-neighbour per edge, drawn once and reused by every pair: the null must not drift
    rng = np.random.default_rng(SEED)
    adjacency = [set() for _ in range(n_items)]
    for left, right in edges.tolist():
        adjacency[int(left)].add(int(right))
        adjacency[int(right)].add(int(left))
    partner = np.full(u.shape[0], -1, dtype=np.int64)
    for index in range(u.shape[0]):
        anchor = int(u[index])
        for _ in range(32):
            candidate = int(rng.integers(n_items))
            if candidate != anchor and candidate not in adjacency[anchor]:
                partner[index] = candidate
                break

    pairs = []
    for number in sorted(p.name for p in MIRROR.iterdir() if p.is_dir()):
        entry = ledger.get(number) or {}
        if entry.get("retired"):
            continue
        recall_path, sid_path = MIRROR / number / "recall.json", MIRROR / number / "item_sids.json"
        if not (recall_path.exists() and sid_path.exists()):
            continue
        value = json.loads(recall_path.read_text()).get(RECALL)
        if value is None:
            continue
        attested = (entry.get("files") or {}).get("item_sids.json")
        digest = hashlib.sha256(sid_path.read_bytes()).hexdigest()
        if attested is not None and digest != attested:
            print(f"  skip {number}: sha mismatch against the ledger")
            continue
        pairs.append({"number": number, "recall": float(value), "sid_path": sid_path,
                      "arm": entry.get("arm") or entry.get("pair", {}).get("variant") or number})

    print(f"{len(pairs)} pairs with a recall and a SID that matches the ledger")
    rows = []
    for pair in pairs:
        payload = json.loads(pair["sid_path"].read_text())
        tokens = np.full((n_items, 3), -1, dtype=np.int64)
        for key, row in payload.items():
            item = int(key)
            if 0 <= item < n_items:
                tokens[item] = [int(code) for code in row[:3]]
        present = (tokens[u, 0] >= 0) & (tokens[v, 0] >= 0) & (partner >= 0)
        valid = present & (partner >= 0)
        close = 1.0 - lcp_matrix(tokens, u, v) / 3.0
        far = 1.0 - lcp_matrix(tokens, u, partner) / 3.0
        win = np.where(close < far, 1.0, np.where(close == far, 0.5, 0.0))
        triples = {(int(row[0]), int(row[1]), int(row[2]))
                   for row in tokens[tokens[:, 0] >= 0]}
        collision = 1.0 - len(triples) / max(int((tokens[:, 0] >= 0).sum()), 1)

        row = {"number": pair["number"], "arm": pair["arm"], "recall": pair["recall"],
               "l3_collision": collision, "roles": {}, "overall": {}}
        overall_mask = valid.copy()
        row["overall"] = {
            "edges": int(overall_mask.sum()),
            "win_rate": float(win[overall_mask].mean()) if overall_mask.any() else None,
        }
        for level in LEVELS:
            depth = int(level[-1])
            match = np.ones(u.shape[0], dtype=bool)
            for index in range(depth):
                match &= tokens[u, index] == tokens[v, index]
                match &= tokens[u, index] >= 0
            row["overall"][level] = float(match[overall_mask].mean()) if overall_mask.any() else None
        for role in range(N_ROLES):
            mask = overall_mask & (role_of_u == role)
            if not mask.any():
                row["roles"][str(role)] = {"edges": 0}
                continue
            entry = {"edges": int(mask.sum()), "win_rate": float(win[mask].mean())}
            for level in LEVELS:
                depth = int(level[-1])
                match = np.ones(u.shape[0], dtype=bool)
                for index in range(depth):
                    match &= tokens[u, index] == tokens[v, index]
                    match &= tokens[u, index] >= 0
                entry[level] = float(match[mask].mean())
            row["roles"][str(role)] = entry
        rows.append(row)
        print(f"  {pair['number']}: recall {pair['recall']:.6f}  "
              f"win {row['overall']['win_rate']:.3f}  prefix@1 {row['overall']['prefix@1']:.3f}  "
              f"collision {collision:.3f}")

    def correlate(values: list[float]) -> dict:
        pairs_values = [(value, row["recall"], row["l3_collision"])
                        for value, row in zip(values, rows) if value is not None]
        if len(pairs_values) < 4:
            return {"n": len(pairs_values), "spearman": None}
        first = np.array([a for a, _, _ in pairs_values])
        second = np.array([b for _, b, _ in pairs_values])
        control = np.array([c for _, _, c in pairs_values])
        observed = {"spearman": spearman(first, second), "pearson": pearson(first, second)}
        p_values = permutation_p(first, second, observed)
        return {"n": len(pairs_values), **observed,
                "p_spearman": p_values["spearman"], "p_pearson": p_values["pearson"],
                "partial_spearman_given_collision": partial(first, second, control, True),
                "partial_pearson_given_collision": partial(first, second, control, False)}

    result = {"n_pairs": len(rows), "recall_metric": RECALL,
              "structure_definition": "ORCA node orbits of 5-node graphlets, six NMF roles, edge "
                                      "attributed to the role of its first endpoint",
              "correlations": {}, "rows": rows}
    series = {"overall/win_rate": [row["overall"].get("win_rate") for row in rows]}
    for level in LEVELS:
        series[f"overall/{level}"] = [row["overall"].get(level) for row in rows]
    for role in range(N_ROLES):
        series[f"role{role}/win_rate"] = [row["roles"].get(str(role), {}).get("win_rate")
                                          for row in rows]
        for level in LEVELS:
            series[f"role{role}/{level}"] = [row["roles"].get(str(role), {}).get(level)
                                             for row in rows]
    for name, values in series.items():
        result["correlations"][name] = correlate(values)

    OUT.write_text(json.dumps(result, indent=1, default=float))
    print("\n=== Spearman / Pearson against test_recall@10, ORCA 5-node roles "
          f"(n={len(rows)}) ===")
    print(f"  {'series':<22}{'n':>5}{'spearman':>10}{'p':>8}{'pearson':>10}{'p':>8}"
          f"{'partialS':>10}{'partialP':>10}")
    for name, value in result["correlations"].items():
        if value.get("spearman") is None:
            continue
        print(f"  {name:<22}{value['n']:>5}{value['spearman']:>+10.3f}{value['p_spearman']:>8.3f}"
              f"{value['pearson']:>+10.3f}{value['p_pearson']:>8.3f}"
              f"{value['partial_spearman_given_collision']:>+10.3f}"
              f"{value['partial_pearson_given_collision']:>+10.3f}")
    # cross-check against the canonical prefix table
    canonical = report["correlations"]
    print("\ncross-check of prefix retention against structure_retention_5node.json:")
    for name in ("overall/prefix@1", "overall/prefix@2", "overall/prefix@3"):
        mine = result["correlations"].get(name, {})
        theirs = canonical.get(name, {})
        print(f"  {name:<20} mine rho {mine.get('spearman', float('nan')):+.3f}  "
              f"canonical rho {theirs.get('rho', float('nan')):+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
