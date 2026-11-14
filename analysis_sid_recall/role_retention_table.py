"""Per-role structure retention across the archived arms, with no interpretation of what a role is.

The six roles come straight from ORCA's 73 five-node orbit counts and the NMF fit that already exists
(`structure_roles5.npz`).  Nothing here names them, maps them to a graphlet, or reads their defining
features: the question is only whether a role's local behaviour relations survive in the SID, how much
that varies across the 67 archived arms, and how strongly it tracks `test_recall@10`.

Per role r and arm a, over the behaviour edges whose history item carries role r:

    win rate      the next item is closer in the SID tree than a random item; ties score 0.5, so the
                  chance line is 0.500
    percentile    the same question with 200 random items instead of one: the share of them that are
                  closer, so 0.500 is chance and 0.000 would be "nearest item in the catalogue"
    prefix@1      both endpoints share the level-1 code (the repository's canonical retention)

Retention is reported as a mean over arms with its spread, and each role's series across arms is
correlated with recall: Spearman and Pearson with permutation p-values, plus partial coefficients
controlling the level-3 collision rate.

    python3 analysis_sid_recall/role_retention_table.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
LEDGER = REPO / "results" / "research" / "issue139" / "r2_ledger.json"
MIRROR = HERE / "r2"
ROLE_CACHE5 = HERE / "structure_roles5.npz"
BEHAVIOUR = REPO / "results" / "research" / "issue215" / "graphs" / "behaviour.in"
OUT = HERE / "role_retention_table.json"
RECALL = "test_recall@10"
SEED = 42
CANDS = 200
EDGES_PER_ROLE = 3000
PERMUTATIONS = 10_000


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
    a, b = average_ranks(first), average_ranks(second)
    a, b = a - a.mean(), b - b.mean()
    denominator = float(np.sqrt((a ** 2).sum() * (b ** 2).sum()))
    return float((a * b).sum() / denominator) if denominator else float("nan")


def pearson(first: np.ndarray, second: np.ndarray) -> float:
    a, b = first - first.mean(), second - second.mean()
    denominator = float(np.sqrt((a ** 2).sum() * (b ** 2).sum()))
    return float((a * b).sum() / denominator) if denominator else float("nan")


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


def lcp_between(tokens: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    total = np.zeros(np.broadcast(left, right).shape, dtype=np.int64)
    alive = np.ones(np.broadcast(left, right).shape, dtype=bool)
    for level in range(3):
        alive &= (tokens[left, level] == tokens[right, level]) & (tokens[left, level] >= 0)
        total += alive.astype(np.int64)
    return total


def main() -> int:
    with BEHAVIOUR.open() as handle:
        n, m = map(int, handle.readline().split())
        edges = np.atleast_2d(np.loadtxt(handle, dtype=np.int64, max_rows=m))
    u, v = edges[:, 0].astype(np.int64), edges[:, 1].astype(np.int64)

    role_of = np.full(n, -1, dtype=np.int64)
    cached = np.load(ROLE_CACHE5)
    for node, role in zip(cached["nodes"], cached["role"]):
        if 0 <= int(node) < n:
            role_of[int(node)] = int(role)
    role_of_u = role_of[u]

    rng = np.random.default_rng(SEED)
    partner = np.full(u.shape[0], -1, dtype=np.int64)
    adjacency_sets = [set() for _ in range(n)]
    for left, right in edges.tolist():
        adjacency_sets[int(left)].add(int(right))
        adjacency_sets[int(right)].add(int(left))
    for index in range(u.shape[0]):
        anchor = int(u[index])
        for _ in range(32):
            candidate = int(rng.integers(n))
            if candidate != anchor and candidate not in adjacency_sets[anchor]:
                partner[index] = candidate
                break

    role_edges = {}
    for role in range(6):
        indices = np.flatnonzero(role_of_u == role)
        if indices.size > EDGES_PER_ROLE:
            picks = np.sort(rng.choice(indices.size, size=EDGES_PER_ROLE, replace=False))
            indices = indices[picks]
        role_edges[role] = indices

    candidates = {role: rng.integers(0, n, size=(indices.size, CANDS))
                  for role, indices in role_edges.items()}

    ledger = json.loads(LEDGER.read_text())
    arms = []
    for number in sorted(p.name for p in MIRROR.iterdir() if p.is_dir()):
        entry = ledger.get(number) or {}
        if entry.get("retired"):
            continue
        sid_path, recall_path = MIRROR / number / "item_sids.json", MIRROR / number / "recall.json"
        if not (sid_path.exists() and recall_path.exists()):
            continue
        recall = json.loads(recall_path.read_text()).get(RECALL)
        if recall is None:
            continue
        attested = (entry.get("files") or {}).get("item_sids.json")
        if attested is not None and hashlib.sha256(sid_path.read_bytes()).hexdigest() != attested:
            continue
        arms.append({"number": number, "recall": float(recall), "sid_path": sid_path,
                     "arm": entry.get("arm") or number})
    print(f"{len(arms)} arms with a ledger-verified SID and a recall", flush=True)

    rows = []
    for count, arm in enumerate(arms, start=1):
        payload = json.loads(arm["sid_path"].read_text())
        tokens = np.full((n, 3), -1, dtype=np.int64)
        for key, row in payload.items():
            item = int(key)
            if 0 <= item < n:
                tokens[item] = [int(code) for code in row[:3]]
        present = tokens[u, 0] >= 0
        triples = {(int(row[0]), int(row[1]), int(row[2])) for row in tokens[tokens[:, 0] >= 0]}
        collision = 1.0 - len(triples) / max(int((tokens[:, 0] >= 0).sum()), 1)

        close = 1.0 - lcp_between(tokens, u, v) / 3.0
        far = 1.0 - lcp_between(tokens, u, partner) / 3.0
        win = np.where(close < far, 1.0, np.where(close == far, 0.5, 0.0))
        prefix1 = (tokens[u, 0] == tokens[v, 0]) & (tokens[u, 0] >= 0)

        entry = {"number": arm["number"], "arm": arm["arm"], "recall": arm["recall"],
                 "l3_collision": collision, "roles": {}}
        for role in range(6):
            indices = role_edges[role]
            mask = indices[present[indices] & (partner[indices] >= 0)]
            if mask.size == 0:
                entry["roles"][str(role)] = None
                continue
            draws = candidates[role][:indices.size]
            keep = present[indices] & (partner[indices] >= 0)
            draws = draws[keep]
            anchors = u[indices][keep]
            truth = v[indices][keep]
            truth_distance = 1.0 - lcp_between(tokens, anchors, truth) / 3.0
            candidate_distance = 1.0 - lcp_between(tokens, np.repeat(anchors[:, None], CANDS, axis=1),
                                                    draws) / 3.0
            closer = (candidate_distance < truth_distance[:, None]).sum(axis=1)
            equal = (candidate_distance == truth_distance[:, None]).sum(axis=1)
            percentile = (closer + 0.5 * equal) / CANDS
            entry["roles"][str(role)] = {
                "edges": int(mask.size),
                "win_rate": float(win[mask].mean()),
                "percentile": float(np.mean(percentile)),
                "prefix@1": float(prefix1[mask].mean()),
            }
        rows.append(entry)
        print(f"  [{count}/{len(arms)}] {arm['number']} recall {arm['recall']:.6f}  " +
              " ".join(f"r{role}:{entry['roles'][str(role)]['percentile']:.3f}"
                       if entry["roles"][str(role)] else f"r{role}:n/a" for role in range(6)),
              flush=True)

    def series(role: int, key: str) -> np.ndarray:
        return np.array([row["roles"][str(role)][key] if row["roles"][str(role)] else np.nan
                         for row in rows], dtype=np.float64)

    recalls = np.array([row["recall"] for row in rows])
    collisions = np.array([row["l3_collision"] for row in rows])

    result = {"n_arms": len(rows), "recall_metric": RECALL, "candidates_per_edge": CANDS,
              "edges_per_role": EDGES_PER_ROLE,
              "note": "roles are ORCA 5-node orbits + the existing NMF fit, relabelled 1-6 for reading; "
                      "no role is named or mapped to a graphlet",
              "roles": {}}
    print(f"\n=== per-role retention across {len(rows)} arms, and its correlation with recall ===")
    print(f"  {'role':<6}{'items':>8}{'edges':>7}{'retention':>18}{'min':>7}{'max':>7}"
          f"{'rho':>8}{'p':>7}{'r':>8}{'p':>7}{'partS':>9}{'partP':>9}")
    for role in range(6):
        members = int((role_of == role).sum())
        block = {"label": f"Role {role + 1}", "members": members}
        for key in ("win_rate", "percentile", "score", "prefix@1"):
            values = 1.0 - series(role, "percentile") if key == "score" else series(role, key)
            finite = values[np.isfinite(values)]
            block[key] = {"mean": float(finite.mean()) if finite.size else None,
                          "sd": float(finite.std()) if finite.size else None,
                          "min": float(finite.min()) if finite.size else None,
                          "max": float(finite.max()) if finite.size else None}
        percentile = series(role, "percentile")
        score = 1.0 - percentile
        usable = np.isfinite(score)
        observed = {"spearman": spearman(score[usable], recalls[usable]),
                    "pearson": pearson(score[usable], recalls[usable])}
        p_values = permutation_p(score[usable], recalls[usable], observed)
        block["correlation"] = {**observed, "p_spearman": p_values["spearman"],
                                "p_pearson": p_values["pearson"],
                                "partial_spearman_given_collision":
                                    partial(score[usable], recalls[usable],
                                            collisions[usable], True),
                                "partial_pearson_given_collision":
                                    partial(score[usable], recalls[usable],
                                            collisions[usable], False),
                                "n": int(usable.sum())}
        prefix = series(role, "prefix@1")
        prefix_usable = np.isfinite(prefix)
        block["correlation_prefix"] = {
            "spearman": spearman(prefix[prefix_usable], recalls[prefix_usable]),
            "pearson": pearson(prefix[prefix_usable], recalls[prefix_usable])}
        result["roles"][str(role)] = block
        retention = block["score"]
        correlation = block["correlation"]
        print(f"  Role {role + 1:<2}{members:>8}{int(np.isfinite(percentile).sum()):>7}"
              f"{retention['mean']:>9.3f}+-{retention['sd']:.3f}"
              f"{retention['min']:>7.3f}{retention['max']:>7.3f}"
              f"{correlation['spearman']:>+8.3f}{correlation['p_spearman']:>7.3f}"
              f"{correlation['pearson']:>+8.3f}{correlation['p_pearson']:>7.3f}"
              f"{correlation['partial_spearman_given_collision']:>+9.3f}"
              f"{correlation['partial_pearson_given_collision']:>+9.3f}")
    OUT.write_text(json.dumps(result, indent=1, default=float))
    print(f"\nwritten {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
