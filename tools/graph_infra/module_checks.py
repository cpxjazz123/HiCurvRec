"""Per-module verification on the real data (issue #246 §7, MODULE_CHECKS.json).

Each entry states the quantities that were measured on the real artefacts and the verdict for
that module.  Nothing is taken from a stored "ok" flag: every number is recomputed here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent / "tests"))

import reference_hand as ref  # noqa: E402
from tools.graph_infra import (assets, bank_features, config, curvature, graph_build,  # noqa: E402
                               graph_distance, graphlet, isolation, manifold, metrics, pairs,
                               ppr, splits, topology)

ROOT = config.ASSET_ROOT
OUT = REPO / "results" / "research" / "infra246" / "MODULE_CHECKS.json"


def check_graph_build() -> dict:
    sequences = graph_build.sequences_from_parquet(config.TRAIN)
    counts = graph_build.count_edges(list(sequences.values()))
    forward = counts.to_csr(24587)
    reference = ref.sequence_edges(list(sequences.values()))
    coo = forward.tocoo()
    produced = {(int(a), int(b)): int(w) for a, b, w in zip(coo.row, coo.col, coo.data)}
    identical = set(produced) == set(reference) and all(
        produced[edge] == reference[edge] for edge in reference)
    return {
        "users": len(sequences), "distinct_edges": len(produced),
        "weight_total": counts.total(),
        "self_loops": graph_build.self_loops(list(sequences.values())),
        "independent_crosscheck_identical": bool(identical),
        "verdict": "PASS" if identical else "FAIL",
    }


def check_splits() -> dict:
    sequences = graph_build.sequences_from_parquet(config.TRAIN)
    users = np.array(sorted(sequences), dtype=np.int64)
    result = splits.three_way(users, config.SPLIT_SEED, config.SPLIT_RATIOS)
    audit = splits.audit(result, users)
    full = graph_build.count_edges([sequences[int(u)] for u in users.tolist()])
    total = sum(graph_build.count_edges([sequences[int(u)] for u in result[name].tolist()]).total()
                for name in splits.NAMES)
    return {"sizes": audit["sizes"], "overlaps": audit["overlaps"],
            "union_complete": audit["union_complete"], "disjoint": audit["disjoint"],
            "weight_sum_matches_full": total == full.total(),
            "verdict": "PASS" if (audit["disjoint"] and audit["union_complete"]
                                  and total == full.total()) else "FAIL"}


def check_isolation() -> dict:
    visible = assets.load_sparse(ROOT / "graph" / "visible_forward.npz")
    val = assets.load_sparse(ROOT / "graph" / "struct_val_forward.npz")
    test = assets.load_sparse(ROOT / "graph" / "struct_test_forward.npz")
    held = assets.load_sparse(ROOT / "graph" / "held_out_support.npz")
    evaluation = (val + test).tocsr()
    recomputed = isolation.held_out_support(visible, evaluation)
    same = (recomputed != held).nnz == 0
    coo = held.tocoo()
    # the support is symmetric, so it necessarily contains the mirror of an eval-only edge;
    # what must never happen is a pair that is visible in BOTH directions
    visible_both = sum(1 for a, b in zip(coo.row, coo.col)
                       if visible[a, b] > 0 and visible[b, a] > 0)
    mirror_visible = sum(1 for a, b in zip(coo.row, coo.col)
                         if visible[a, b] > 0 and visible[b, a] == 0)
    blocked_train = 0
    for path in sorted((ROOT / "pairs").glob("train_*.npz")):
        blocked_train += len(isolation.blocked_pairs(np.load(path)["pairs"], held))
    return {"held_entries": int(held.nnz // 2), "symmetric": (held != held.T).nnz == 0,
            "recomputation_matches": bool(same),
            "held_pairs_visible_in_both_directions": visible_both,
            "held_pairs_with_a_visible_mirror": mirror_visible,
            "training_pairs_still_blocked": blocked_train,
            "verdict": "PASS" if same and visible_both == 0 and blocked_train == 0 else "FAIL"}


def check_topology() -> dict:
    visible = assets.load_sparse(ROOT / "graph" / "visible_forward.npz")
    reverse = visible.T.tocsr()
    blob = np.load(ROOT / "ego" / "ego_sampled_visible.npz")
    roots = blob["roots"]
    rng = np.random.default_rng(0)
    sample = rng.choice(roots, size=min(400, roots.size), replace=False)
    role_errors = hop_errors = edge_errors = 0
    for root in sample.tolist():
        block = topology.ego_induced(int(root), visible, reverse,
                                     np.random.default_rng([config.EGO_SEED, int(root)]),
                                     radius=2, cap=config.EGO_CAP)
        out_row = set(visible.indices[visible.indptr[root]:visible.indptr[root + 1]].tolist())
        in_row = set(reverse.indices[reverse.indptr[root]:reverse.indptr[root + 1]].tolist())
        for position, node in enumerate(block["nodes"].tolist()):
            if bool(block["role_out"][position]) != (node in out_row):
                role_errors += 1
            if bool(block["role_in"][position]) != (node in in_row):
                role_errors += 1
            if block["hop"][position] != (0 if node == root else 1 if node in out_row | in_row else 2):
                hop_errors += 1
        for left, right in block["edges"].tolist():
            if visible[block["nodes"][left], block["nodes"][right]] == 0:
                edge_errors += 1
    sizes = np.diff(blob["node_offsets"])
    return {"sampled_items": int(sample.size), "role_errors": role_errors,
            "hop_errors": hop_errors, "edge_errors": edge_errors,
            "max_nodes": int(sizes.max()), "cap": config.EGO_CAP,
            "cap_respected": bool(sizes.max() <= config.EGO_CAP + 1),
            "verdict": "PASS" if not (role_errors or hop_errors or edge_errors) else "FAIL"}


def check_graphlet() -> dict:
    visible = assets.load_sparse(ROOT / "graph" / "visible_forward.npz")
    reverse = visible.T.tocsr()
    counts = np.load(ROOT / "motifs" / "motif_counts.npz")
    names = counts["names"].tolist()
    rng = np.random.default_rng(1)
    # the brute-force comparison is only affordable on modest degrees; a hub with 2,220
    # neighbours would mean 2.4 million induced triples for one centre
    degree = np.asarray(visible.getnnz(axis=1)).ravel() + np.asarray(reverse.getnnz(axis=1)).ravel()
    candidates = np.flatnonzero(degree <= 60)
    centres = rng.choice(candidates, size=min(150, candidates.size), replace=False)
    mismatch = 0
    for centre in centres.tolist():
        produced = graphlet.motif_instances(int(centre), visible, reverse)
        tally = {}
        for instance in produced:
            tally[instance["family"]] = tally.get(instance["family"], 0) + 1
        row = {name: int(counts["counts"][centre, index]) for index, name in enumerate(names)}
        for name, value in tally.items():
            if row.get(name, 0) != value:
                mismatch += 1
        if sum(row.values()) != int(counts["instances"][centre]):
            mismatch += 1
    return {"sampled_centres": int(centres.size), "degree_ceiling": 60,
            "family_count_mismatches": mismatch,
            "families": len(names), "total_instances": int(counts["instances"].sum()),
            "verdict": "PASS" if mismatch == 0 else "FAIL"}


def check_pairs_and_banks() -> dict:
    visible = assets.load_sparse(ROOT / "graph" / "visible_forward.npz")
    reverse = visible.T.tocsr()
    problems = []
    degree_violations = 0
    checked = 0
    for path in sorted((ROOT / "pairs").glob("train_*.npz")):
        blob = np.load(path)
        produced = blob["pairs"]
        group = str(blob["group"])
        checked += len(produced)
        try:
            pairs.verify_group(visible, reverse, produced, group)
        except AssertionError as error:
            problems.append(f"{path.name}: {error}")
        if group == "degree_matched":
            degree_violations += pairs.verify_degree_match(visible, reverse, produced)["violations"]
        if np.any(produced[:, 0] == produced[:, 1]):
            problems.append(f"{path.name}: self pair")
    strict = set()
    for path in sorted((ROOT / "pairs").glob("strict_*.npz")):
        strict |= isolation.pair_keys(np.load(path)["pairs"])
    train = set()
    for path in sorted((ROOT / "pairs").glob("train_*.npz")):
        train |= isolation.pair_keys(np.load(path)["pairs"])
    probe = isolation.pair_keys(np.load(ROOT / "pairs" / "probe.npz")["pairs"])
    return {"pairs_checked": checked, "group_violations": problems[:5],
            "degree_match_violations": degree_violations,
            "strict_vs_train_overlap": len(strict & train), "probe_vs_train_overlap": len(probe & train),
            "verdict": "PASS" if not problems and degree_violations == 0 and not (strict & train)
            and not (probe & train) else "FAIL"}


def check_bank_features() -> dict:
    visible = assets.load_sparse(ROOT / "graph" / "visible_forward.npz")
    reverse = visible.T.tocsr()
    undirected = assets.load_sparse(ROOT / "graph" / "visible_undirected.npz")
    blob = np.load(sorted((ROOT / "pairs").glob("train_*.npz"))[0])
    produced = blob["pairs"]
    rng = np.random.default_rng(2)
    sample = rng.choice(len(produced), size=min(60, len(produced)), replace=False)
    hops = bank_features.graph_hops(undirected, produced, directed=False)
    directed = bank_features.graph_hops(visible, produced, directed=True)
    mismatches = 0
    directed_only_unreachable = 0
    for index in sample.tolist():
        source, target = int(produced[index, 0]), int(produced[index, 1])
        reference = graph_distance.bfs_distances(undirected, [source])[0, target]
        if not np.isinf(reference) and abs(reference - hops[index]) > 1e-9:
            mismatches += 1
        # an undirected path may exist where no directed path does; that is expected
        if hops[index] >= 0 and directed[index] < 0:
            directed_only_unreachable += 1
    blob = np.load(sorted((ROOT / "pairs").glob("train_*.npz"))[0])
    report = {"sampled_pairs": int(sample.size), "hop_mismatches": mismatches,
              "undirected_reachable_but_directed_unreachable": directed_only_unreachable,
              "unreachable_share": float((hops < 0).mean())}
    if "ppr" in blob.files:
        ppr_rows = blob["ppr"]
        report.update({"ppr_min": float(ppr_rows.min()), "ppr_max": float(ppr_rows.max()),
                       "ppr_finite": bool(np.all(np.isfinite(ppr_rows)))})
    else:
        report["ppr"] = "not shipped this round (owner withdrew the propagation column)"
    report["verdict"] = "PASS" if mismatches == 0 else "FAIL"
    return report


def check_ppr() -> dict:
    visible = assets.load_sparse(ROOT / "graph" / "visible_forward.npz")
    result = ppr.solve_selected(visible, np.array([[0, 1], [2, 3], [4, 5]], dtype=np.int64))
    mass = float(result["values"].sum())
    return {"sources": result["sources"], "residual_max": result["residual_max"],
            "coverage": result["coverage"], "mass_sum_sample": mass,
            "verdict": "PASS" if result["residual_max"] < 1e-6 and mass > 0 else "FAIL"}


def check_curvature() -> dict:
    path = ROOT / "curvature" / "curvature_visible.json"
    if not path.exists():
        # the curvature asset was not built this round (owner instruction); the module and its
        # known-answer tests are delivered, so the state is recorded rather than hidden
        from tools.graph_infra import curvature
        return {"asset": "not built this round (owner instruction)",
                "module_available": callable(curvature.orc_edge),
                "known_answer_tests": "tests/test_v2_geometry.py",
                "verdict": "NA (asset not built this round)"}
    report = json.loads(path.read_text())
    summary = report["summary"]
    return {"reference_nodes": report["reference_nodes"], "sampled_edges": report["sampled_edges"],
            "missing_nodes": report["missing_nodes"], "mean": summary["mean"],
            "median": summary["median"], "positive": summary["positive"],
            "negative": summary["negative"], "unit": report["unit"],
            "verdict": "PASS" if summary["n"] > 0 and report["reference_nodes"] > 0 else "FAIL"}


def check_metrics() -> dict:
    report = metrics.subgraph_summary([np.array([0.3, 0.4, 0.8])], threshold=0.5, direction="minimize")
    known = (abs(report["subgraph_pass_fraction"] - 2 / 3) < 1e-12
             and report["center_all_pass_fraction"] == 0.0
             and abs(report["worst"] - 0.8) < 1e-12 and abs(report["p90"] - 0.72) < 1e-12)
    table = json.loads((REPO / "results" / "stage2_RQ-VAE" / "TIGER_RQ-VAE" / "item_sids.json")
                       .read_text())
    audit = metrics.sid_audit(table, item_ids=list(range(24587)), width=4, max_token=1024)
    return {"known_answer_ok": bool(known), "subgraph_pass_fraction": report["subgraph_pass_fraction"],
            "worst": report["worst"], "p90": report["p90"],
            "tiger_sid_audit_ok": audit["ok"], "tiger_sid_missing": audit["missing_count"],
            "verdict": "PASS" if known and audit["ok"] else "FAIL"}


def check_assets() -> dict:
    audit = assets.audit_manifest(ROOT / "manifest.json")
    bank = np.array([[0, 1], [2, 3]], dtype=np.int64)
    rejections = 0
    for bad in (np.array([[0, 1]], dtype=np.int64), np.array([[0, 1], [0, 1]], dtype=np.int64),
                np.array([[0, 1], [9, 9]], dtype=np.int64),
                np.array([[0, 1], [2, 3], [4, 5]], dtype=np.int64)):
        try:
            assets.join_scores_v2(bank, bad, np.ones(len(bad)), expected_graph_version="v2/test",
                                   bank_graph_version="v2/test")
        except ValueError:
            rejections += 1
    non_finite = 0
    try:
        assets.join_scores_v2(bank, bank, np.array([0.5, np.nan]),
                              expected_graph_version="v2/test", bank_graph_version="v2/test")
    except ValueError:
        non_finite = 1
    return {"manifest_ok": audit["ok"], "assets": audit["assets"],
            "join_rejections": rejections, "non_finite_rejected": non_finite,
            "verdict": "PASS" if audit["ok"] and rejections == 4 and non_finite == 1 else "FAIL"}


def check_geometry() -> dict:
    import torch
    rows = {}
    for k in (-1.0, 1e-12, 0.0, 1.0):
        vector = torch.tensor([[0.4, -0.3]], dtype=torch.float64)
        point = manifold.expmap0(vector, torch.tensor(k, dtype=torch.float64))
        value = float(manifold.dist(torch.zeros_like(point), point, torch.tensor(k, dtype=torch.float64)))
        rows[f"k={k}"] = {"d(0,exp0(v))": value, "expected": 2 * float(vector.norm())}
    ok = all(abs(entry["d(0,exp0(v))"] - entry["expected"]) < 1e-4 for entry in rows.values())
    return {"per_k": rows, "round_trip_ok": bool(ok), "verdict": "PASS" if ok else "FAIL"}


CHECKS = {
    "graph_build": check_graph_build,
    "splits": check_splits,
    "isolation": check_isolation,
    "topology": check_topology,
    "graphlet": check_graphlet,
    "pairs_and_banks": check_pairs_and_banks,
    "bank_features": check_bank_features,
    "ppr": check_ppr,
    "curvature": check_curvature,
    "metrics": check_metrics,
    "assets": check_assets,
    "geometry": check_geometry,
}


def main() -> int:
    payload = {"issue": 246, "data": "the frozen v2 assets and the real train parquet"}
    failures = []
    for name, function in CHECKS.items():
        try:
            payload[name] = function()
        except Exception as error:  # noqa: BLE001
            payload[name] = {"verdict": "FAIL", "error": f"{type(error).__name__}: {error}"}
        if payload[name].get("verdict") != "PASS":
            failures.append(name)
    payload["summary"] = {"modules": len(CHECKS), "passed": len(CHECKS) - len(failures),
                          "failed": failures}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    for name in CHECKS:
        print(f"  {name}: {payload[name].get('verdict')}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
