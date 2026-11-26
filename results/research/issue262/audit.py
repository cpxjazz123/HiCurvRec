"""Issue #262: audit #261 identities, missing claims, validation records and strict banks."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/research/issue262"
sys.path.insert(0, str(ROOT))

from tools.iteration import analysis as ianalysis  # noqa: E402
from tools.issue252 import archive_audit  # noqa: E402
from tools.graph_infra import assets  # noqa: E402

ARMS = {
    100: "TIGER_gnn_attention_sectional_rq",
    101: "TIGER_three_expert_attention_rq",
    102: "TIGER_module_budget_prefix_rq",
    103: "TIGER_periodic_budget_prefix_rq",
    104: "TIGER_role_conditioned_l1_rq",
    105: "TIGER_role_conditioned_l3_rq",
    106: "TIGER_role_conditioned_l2_rq",
    107: "TIGER_topology_residual_l2_rq",
}
ROUND = {102: "r01", 103: "r02", 104: "r03", 105: "r04"}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def identity(path: Path) -> dict:
    return {"path": str(path), "exists": path.is_file(),
            "sha256": sha(path) if path.is_file() else None,
            "bytes": path.stat().st_size if path.is_file() else None}


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def find_final(arm: str) -> Path:
    candidates = sorted((ROOT / "results/stage3_T5Train" / arm).glob("**/test_final.json"))
    if len(candidates) != 1:
        raise RuntimeError(f"{arm}: expected one test_final, found {len(candidates)}")
    return candidates[0]


def validation_record(n: int, arm: str) -> dict:
    """Read and independently reduce an identity-bound Stage3 validation record."""
    if n not in ROUND:
        return {"status": "NA", "reason": "no identity-bound recommendation-validation record for this arm"}
    round_dir = ROOT / "results/research/issue256/rounds" / ROUND[n] / "recommendation_validation"
    record = round_dir / "VALIDATION.json"
    if not record.is_file():
        return {"status": "NA", "reason": "missing " + str(record)}
    payload = read_json(record)
    result = {"status": "PENDING", "record": identity(record),
              "schema": payload.get("schema"), "top_status": payload.get("status"),
              "training": payload.get("training"), "test_access": payload.get("test_access"),
              "used_for_checkpoint_selection": payload.get("used_for_checkpoint_selection")}
    identity_payload = payload.get("identity", {})
    final = find_final(arm)
    final_obj = read_json(final)
    checks = {}
    for key, actual in (
            ("test_final", final),
            ("checkpoint", Path(identity_payload["checkpoint"]["path"])),
            ("sid", Path(final_obj["sid_path"])),
            ("valid", ROOT / "results/stage0_build_parquet/valid.parquet")):
        want = identity_payload.get(key, {})
        got = identity(actual)
        checks[key] = {"want": want, "got": got,
                       "status": "PASS" if want.get("sha256") == got["sha256"]
                       and Path(want.get("path", "")).resolve() == actual.resolve() else "FAIL"}
    # metric identity: validation is a distinct population, never test_final.
    result["identity_checks"] = checks
    shards = sorted(round_dir.glob("rank[0-3].json"))
    rows = sorted(round_dir.glob("rank[0-3].jsonl"))
    result["shards"] = [identity(p) for p in shards]
    result["prediction_rows"] = [identity(p) for p in rows]
    result["row_counts"] = {p.name: sum(1 for line in p.open() if line.strip()) for p in rows}
    reduced = ianalysis.reduce_predictions(rows, int(payload["metrics"]["n_eval"]))
    reported = payload["metrics"]
    metric_checks = {}
    for key in ("validation_recall@5", "validation_recall@10",
                "validation_ndcg@5", "validation_ndcg@10"):
        reduced_key = key.replace("validation_", "test_")
        metric_checks[key] = {
            "reported": reported.get(key),
            "recomputed": reduced.get(reduced_key),
            "status": "PASS" if reported.get(key) is not None
            and reduced.get(reduced_key) is not None
            and abs(float(reported[key]) - float(reduced[reduced_key])) <= 1e-10 else "FAIL",
        }
    result["metrics"] = metric_checks
    result["reduced"] = reduced
    result["all_identity_checks_pass"] = all(x["status"] == "PASS" for x in checks.values())
    result["all_metrics_pass"] = all(x["status"] == "PASS" for x in metric_checks.values())
    result["status"] = "PASS" if (result["all_identity_checks_pass"]
                                  and result["all_metrics_pass"]
                                  and result["training"] is False
                                  and result["test_access"] is False
                                  and result["used_for_checkpoint_selection"] is False
                                  and all(count <= 14360 for count in result["row_counts"].values())
                                  and len(result["row_counts"]) == 4
                                  and sum(result["row_counts"].values()) == reduced["n_eval"]) else "FAIL"
    return result


def graph_bank_audit() -> dict:
    """Recompute directed labels on the graph that generated each strict bank."""
    from tools.graph_infra import pairs as pair_ops

    base = ROOT / "results/research/infra246/assets"
    out = {}
    for split in ("struct_test", "struct_val"):
        forward = assets.load_sparse(base / "graph" / f"{split}_forward.npz")
        reverse = assets.load_sparse(base / "graph" / f"{split}_reverse.npz")
        visible_forward = assets.load_sparse(base / "graph" / "visible_forward.npz")
        visible_reverse = assets.load_sparse(base / "graph" / "visible_reverse.npz")
        for group in ("shared_in", "shared_out"):
            bank = np.load(base / "pairs" / f"strict_{split}_{group}.npz", allow_pickle=False)
            pair_rows = np.asarray(bank["pairs"], dtype=np.int64)
            true_j_in, true_j_out, visible_j_in, visible_j_out = [], [], [], []
            witness_ok = direct_ok = 0
            for u, v in pair_rows.tolist():
                pu = set(int(x) for x in reverse.indices[reverse.indptr[u]:reverse.indptr[u + 1]])
                pv = set(int(x) for x in reverse.indices[reverse.indptr[v]:reverse.indptr[v + 1]])
                su = set(int(x) for x in forward.indices[forward.indptr[u]:forward.indptr[u + 1]])
                sv = set(int(x) for x in forward.indices[forward.indptr[v]:forward.indptr[v + 1]])
                tu = pu | pv
                to = su | sv
                true_j_in.append(len(pu & pv) / len(tu) if tu else None)
                true_j_out.append(len(su & sv) / len(to) if to else None)
                vpu = set(int(x) for x in visible_reverse.indices[
                    visible_reverse.indptr[u]:visible_reverse.indptr[u + 1]])
                vpv = set(int(x) for x in visible_reverse.indices[
                    visible_reverse.indptr[v]:visible_reverse.indptr[v + 1]])
                vsu = set(int(x) for x in visible_forward.indices[
                    visible_forward.indptr[u]:visible_forward.indptr[u + 1]])
                vsv = set(int(x) for x in visible_forward.indices[
                    visible_forward.indptr[v]:visible_forward.indptr[v + 1]])
                vtu = vpu | vpv
                vto = vsu | vsv
                visible_j_in.append(len(vpu & vpv) / len(vtu) if vtu else None)
                visible_j_out.append(len(vsu & vsv) / len(vto) if vto else None)
                witness_ok += int(bool(pu & pv) if group == "shared_in" else bool(su & sv))
                direct_ok += int((forward[u, v] == 0 and forward[v, u] == 0))
            def summary(values):
                vals = np.asarray([x for x in values if x is not None], dtype=float)
                return {"count": int(len(vals)), "missing": int(len(values) - len(vals)),
                        "mean": float(vals.mean()) if len(vals) else None,
                        "median": float(np.median(vals)) if len(vals) else None,
                        "positive_fraction": float((vals > 0).mean()) if len(vals) else None}
            out[f"{split}_{group}"] = {
                "bank": identity(base / "pairs" / f"strict_{split}_{group}.npz"),
                "pairs": int(len(pair_rows)),
                "true_graph_relation": {"witness_positive": int(witness_ok),
                                        "witness_fraction": float(witness_ok / len(pair_rows)) if len(pair_rows) else None,
                                        "no_direct_edge": int(direct_ok)},
                "true_jaccard_in": summary(true_j_in),
                "true_jaccard_out": summary(true_j_out),
                "visible_jaccard_in": summary(visible_j_in),
                "visible_jaccard_out": summary(visible_j_out),
                "feature_source_mismatch": "bank_features.compute used the visible graph; pairs came from the split graph",
            }
    return out


def main() -> None:
    inventory = read_json(ROOT / "results/research/issue261/INVENTORY.json")
    audit = {"schema": "genrec.issue262.audit/1", "arms": {},
             "validation": {}, "strict_banks": graph_bank_audit(),
             "claim_checks": {}}
    for n, arm in ARMS.items():
        final = find_final(arm)
        fd = read_json(final)
        checkpoint = Path(fd["stage2_checkpoint"])
        sid = Path(fd["sid_path"])
        binding = Path(fd["archive_binding"])
        export = Path(fd["stage2_export_manifest"])
        source = Path(read_json(binding)["source_tree"])
        train_metrics = sid.parent / "logs/training_metrics.jsonl"
        rows = [json.loads(x) for x in train_metrics.open()] if train_metrics.is_file() else []
        train_events = [x for x in rows if x.get("event") == "train"]
        record = {"arm": arm, "run_id": fd.get("stage2_run_id"),
                  "final": identity(final), "checkpoint": identity(checkpoint),
                  "sid": identity(sid), "archive_binding": identity(binding),
                  "export_manifest": identity(export), "source_tree": str(source),
                  "training": {"path": str(train_metrics), "events": len(train_events),
                               "complete": bool(train_events and len(train_events) == 3000
                                                and train_events[0].get("epoch") == 1
                                                and train_events[-1].get("epoch") == 3000)},
                  "stage3": {"test_recall@10": fd.get("test_recall@10"),
                             "n_eval": fd.get("n_eval"), "epochs": fd.get("stage3_input_config", {}).get("num_epochs")},
                  "r2_source_tar_local": "NA: #261 retained only recall.json; no source tar present in _r2",
                  "validation": validation_record(n, arm)}
        # Candidate identity: archive binding validation is kept as a fail-closed
        # observation, but a historical infra hash mismatch is not candidate drift.
        try:
            archive_audit.validate_binding(binding, sid, checkpoint, source)
            record["archive_binding_validation"] = {"status": "PASS"}
        except Exception as exc:  # noqa: BLE001
            record["archive_binding_validation"] = {"status": "FAIL", "error": repr(exc)}
        try:
            actual = archive_audit.source_hashes(source)
            expected = read_json(binding).get("candidate_sources")
            record["candidate_sources"] = {"status": "PASS" if actual == expected else "FAIL",
                                           "files": len(actual)}
        except Exception as exc:  # noqa: BLE001
            record["candidate_sources"] = {"status": "FAIL", "error": repr(exc)}
        audit["arms"][str(n)] = record
        if record["validation"]["status"] != "NA":
            audit["validation"][str(n)] = record["validation"]

    # Correct #261's blanket validation NA using the identity-bound records above.
    statuses = {n: audit["validation"].get(str(n), {}).get("status") for n in ARMS}
    audit["claim_checks"]["stage3_validation"] = {
        "261_claim": "No identity-bound validation final artifact exists",
        "found": {str(n): statuses[n] for n in ARMS if statuses[n] not in (None, "NA")},
        "remaining_na": [str(n) for n in ARMS if statuses[n] in (None, "NA")],
        "status": "CORRECTED" if any(x == "PASS" for x in statuses.values()) else "CONFIRMED",
        "scope": "R01-R04 recommendation_validation only; test was never used as validation",
    }
    # #261's +0.958 row is a cross-arm correlation, not a within-arm fidelity.
    full = read_json(ROOT / "results/research/issue261/FULL_METRICS.json")
    corr = read_json(ROOT / "results/research/issue261/CORRELATIONS.json")
    exact = [c for c in corr["correlations"]
             if c["target"] == "test_recall@10" and c["pearson"] is not None
             and abs(c["pearson"] - 0.9584480770975934) <= 1e-12]
    audit["claim_checks"]["shared_in_0958"] = {
        "source": exact[0] if exact else None,
        "level": "cross-arm correlation of a within-arm distance/label Spearman",
        "per_arm_predictor": "hard_q.strict_heldout.banks.strict_struct_test_shared_in.topology.jaccard_in.distance_spearman",
        "per_arm_values": {a: v["metrics"].get(
            "hard_q.strict_heldout.banks.strict_struct_test_shared_in.topology.jaccard_in.distance_spearman")
            for a, v in full.items()},
        "interpretation": "not a shared-in fidelity value; the per-arm values are near zero and negative",
    }
    (OUT / "AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    # Compact CSV of validation records.
    with (OUT / "VALIDATION.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "index", "arm", "status", "validation_recall@5", "validation_recall@10",
            "validation_ndcg@5", "validation_ndcg@10", "n_eval"])
        writer.writeheader()
        for n, arm in ARMS.items():
            v = audit["arms"][str(n)]["validation"]
            row = {"index": n, "arm": arm, "status": v.get("status")}
            for key in ("validation_recall@5", "validation_recall@10",
                        "validation_ndcg@5", "validation_ndcg@10"):
                row[key] = (v.get("reduced") or {}).get(key.replace("validation_", "test_")) if v.get("status") == "PASS" else None
            writer.writerow(row)
    print(json.dumps({"arms": len(audit["arms"]), "validation": statuses,
                      "claim_checks": audit["claim_checks"]["stage3_validation"],
                      "shared_in_0958": audit["claim_checks"]["shared_in_0958"]["source"],
                      "strict_banks": {k: v["pairs"] for k, v in audit["strict_banks"].items()}},
                     indent=2))


if __name__ == "__main__":
    main()
