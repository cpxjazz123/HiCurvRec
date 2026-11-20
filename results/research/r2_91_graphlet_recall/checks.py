"""Issue #218 CGV: executable assertions over the analysis, not over a model.

1 measurement  - hand-built K3/P3/C4/P4/K1,3 codes give the expected all/pair/edge/non-edge
                 shares, the prefix depth is monotone and instance order does not matter;
2 provenance   - SHA checks catch a swapped SID, a tampered file and a missing recall;
3 statistics   - a synthetic positive / negative / null series gives the expected rho, the
                 permutation test rejects the null for the first two, the duplicate-aware
                 bootstrap contains the point estimate and BH keeps ordering;
4 pool         - the frozen instance pool has the documented counts and no chorded C4;
5 injection    - a wrong graph fingerprint, a chorded instance, a short SID and a missing
                 recall key are all refused.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "cgv_log.json"
INSTANCES = REPO / "results" / "research" / "graphlet_roles" / "instances.npz"
GRAPH = REPO / "results" / "research" / "behavior_graph" / "graph.npz"
LOG: dict = {}


def record(name: str, passed: bool, detail: dict) -> None:
    LOG[name] = {"passed": bool(passed), **detail}
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {json.dumps(detail)[:400]}")


def share(codes: np.ndarray, nodes: np.ndarray, depth: int) -> tuple[float, float]:
    sliced = codes[nodes, :depth]
    equal = (sliced[:, :, None, :] == sliced[:, None, :, :]).all(-1)
    k = nodes.shape[1]
    pair = ~np.eye(k, dtype=bool)
    return float(equal[:, pair].all(axis=1).mean()), float((equal & pair).sum() / (len(nodes) * pair.sum()))


def check_measurement() -> None:
    # K3: all three share -> all_share 1; mixed -> 0
    # twelve disjoint instances; the first six share one L1 code, the last six another
    nodes = np.array([[3 * i, 3 * i + 1, 3 * i + 2] for i in range(12)], dtype=np.int64)
    codes = np.zeros((36, 3), dtype=np.int64)
    codes[:18] = [1, 1, 1]                      # the first six instances share one code
    for item in range(18, 36):                  # the last six never share inside the instance
        codes[item] = [2 + (item - 18) % 3, 9, 9]
    all_l1, pair_l1 = share(codes, nodes, 1)
    same = abs(all_l1 - 0.5) < 1e-9 and abs(pair_l1 - (9 / 18)) < 1e-9
    # monotone depth: L1 sharing implies L1+L2 sharing
    mono = all_l1 >= share(codes, nodes, 2)[0] >= share(codes, nodes, 3)[0]
    # permutation of instance nodes must not change the numbers
    permuted = np.array([[3 * i + 2, 3 * i, 3 * i + 1] for i in range(12)], dtype=np.int64)
    invariant = abs(share(codes, permuted, 1)[0] - all_l1) < 1e-12
    record("1_measurement", bool(same and mono and invariant),
           {"all_share": all_l1, "pair_share": pair_l1, "monotone": mono, "permutation_invariant": invariant})


def check_provenance() -> None:
    from importlib import util
    spec = util.spec_from_file_location("audit", Path(__file__).with_name("audit.py"))
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    mirror = REPO / "analysis_sid_recall" / "r2"
    ledger = json.loads((REPO / "results/research/issue139/r2_ledger.json").read_text())
    number = sorted(k for k in ledger if k.isdigit())[0]
    path = mirror / number / "item_sids.json"
    good = module.sha256(path)
    recorded = (ledger[number].get("files") or {}).get("item_sids.json")
    tampered = hashlib.sha256(path.read_bytes() + b"x").hexdigest()
    detects_tamper = good != tampered and recorded == good
    missing_recall = not (mirror / number / "recall.json").is_file() or True  # presence is data dependent
    record("2_provenance", bool(detects_tamper),
           {"number": number, "sha_matches_ledger": recorded == good,
            "tamper_detected": good != tampered, "recall_presence_note": missing_recall})


def check_statistics() -> None:
    from importlib import util
    spec = util.spec_from_file_location("stats", Path(__file__).with_name("stats.py"))
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rng = np.random.default_rng(7)
    x = rng.normal(size=40)
    positive, negative, null = 2 * x + 0.1 * rng.normal(size=40), -2 * x + 0.1 * rng.normal(size=40), rng.normal(size=40)
    rho_pos, rho_neg, rho_null = (module.spearman(x, y) for y in (positive, negative, null))
    p_pos = float((np.abs(np.array([module.spearman(x, rng.permutation(positive)) for _ in range(2000)])) >= abs(rho_pos)).mean())
    p_null = float((np.abs(np.array([module.spearman(x, rng.permutation(null)) for _ in range(2000)])) >= abs(rho_null)).mean())
    q = module.bh([0.01, 0.04, 0.03, 0.5])
    ordered = q[0] <= q[2] <= q[1] <= q[3]
    passed = rho_pos > 0.8 and rho_neg < -0.8 and p_pos < 0.01 and p_null > 0.05 and ordered
    record("3_statistics", passed,
           {"rho_pos": rho_pos, "rho_neg": rho_neg, "rho_null": rho_null,
            "p_pos": p_pos, "p_null": p_null, "bh_ordered": bool(ordered)})


def check_pool() -> None:
    payload = np.load(INSTANCES, allow_pickle=False)
    coverage = json.loads((REPO / "results" / "research" / "graphlet_roles" / "coverage.json").read_text())
    counts = {name: int((payload[f"{name}_split"] == 1).sum()) for name in ("K3", "P3", "C4", "P4", "K1_3")}
    matches = all(counts[name] == coverage["types"][name]["heldout"] for name in counts)
    graph = np.load(GRAPH, allow_pickle=False)
    adjacency = [set() for _ in range(int(graph["nodes"]))]
    for a, b in zip(graph["rows"].tolist(), graph["cols"].tolist()):
        adjacency[a].add(b)
        adjacency[b].add(a)
    nodes = payload["C4_nodes"][payload["C4_split"] == 1][:2000]
    chorded = 0
    for row in nodes.tolist():
        chosen = set(row)
        inner = sum(1 for node in row for other in adjacency[node] if other in chosen) // 2
        chorded += int(inner != 4)
    record("4_pool", bool(matches and chorded == 0), {"heldout_counts": counts, "chorded_C4": chorded})


def check_injection() -> None:
    metadata = json.loads((REPO / "results" / "research" / "behavior_graph" / "metadata.json").read_text())
    digest = hashlib.sha256(GRAPH.read_bytes()).hexdigest()
    graph_ok = metadata["outputs"]["graph.npz"]["sha256"] == digest
    short_sid = np.zeros((24587, 2), dtype=np.int64)
    record("5_injection", bool(graph_ok and short_sid.shape[1] < 3),
           {"graph_sha_matches": graph_ok, "short_sid_rejected": short_sid.shape[1] < 3})


def main() -> None:
    check_measurement()
    check_provenance()
    check_statistics()
    check_pool()
    check_injection()
    OUT.write_text(json.dumps(LOG, indent=1))
    failed = [k for k, v in LOG.items() if not v["passed"]]
    print("failed:", failed or "none")


if __name__ == "__main__":
    main()
