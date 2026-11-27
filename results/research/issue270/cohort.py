"""Issue #270: resolve a verified artifact binding for every recoverable arm."""
from __future__ import annotations
import hashlib, json
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results/research/issue270"
S2 = REPO / "results/stage2_RQ-VAE"
S3 = REPO / "results/stage3_T5Train"
EXPECTED = 57439

FAMILY = {
    "TIGER_RQ-VAE": "baseline",
    "TIGER_euclid_local_curve_supervision_rq": "curvature_point",
    "TIGER_relative_curvature_ranking_rq": "curvature_rank",
    "TIGER_calibrated_kappa_rq": "curvature_calibrated",
    "TIGER_three_expert_attention_rq": "attention_geometry",
    "TIGER_mixed_rq_F0": "mixed_curvature",
    "TIGER_role_geom_rq": "role_geometry",
    "TIGER_role_geom_global_rq": "role_geometry",
    "TIGER_role_geom_shuffled_rq": "role_geometry",
    "TIGER_g4_closure_A": "graphlet_closure", "TIGER_g4_closure_B": "graphlet_closure",
    "TIGER_g4_closure_C": "graphlet_closure", "TIGER_g4_closure_D": "graphlet_closure",
    "TIGER_graphlet_codebook_rq": "graphlet_codebook",
    "TIGER_graphlet_codebook_global_rq": "graphlet_codebook",
    "TIGER_graphlet_role_E": "graphlet_role", "TIGER_graphlet_role_Q": "graphlet_role",
    "TIGER_graphlet_role_QR": "graphlet_role", "TIGER_graphlet_role_R": "graphlet_role",
    "TIGER_graphlet_role_T": "graphlet_role",
    "TIGER_frozen_topology_gnn_rq": "gnn",
    "TIGER_gnn_attention_sectional_rq": "gnn",
    "TIGER_topology_residual_l2_rq": "topology_residual",
    "TIGER_behavior_edge_contrastive_rq": "behavior_contrast",
    "TIGER_curve_contrastive_rq": "curve_contrast",
    "TIGER_cos_contrastive_rq": "curve_contrast",
    "TIGER_af3_curvature_drive_rq": "af3_curvature",
    "TIGER_af3_curvature_factor_mass_rq": "af3_curvature",
    "TIGER_af3_graphgeo_rq": "af3_geometry",
    "TIGER_kappa_sk_only_rq": "curvature_budget",
    "TIGER_degree_temp_rq": "control",
    "TIGER_baseline_recal_rq": "control",
    "TIGER_Uniform_KNN_RQ": "control", "TIGER_Uniform_KNN_control": "control",
    "TIGER_chain_align_rq": "alignment", "TIGER_g237_sem_rq": "semantic",
    "TIGER_i234_A1": "i234", "TIGER_i234_A2": "i234", "TIGER_i234_A3": "i234",
    "TIGER_i234_A4": "i234", "TIGER_i234_A5": "i234", "TIGER_i234_A6": "i234",
    "TIGER_role_conditioned_l1_rq": "role_condition", "TIGER_role_conditioned_l2_rq": "role_condition",
    "TIGER_role_conditioned_l3_rq": "role_condition",
    "TIGER_module_budget_prefix_rq": "budget", "TIGER_periodic_budget_prefix_rq": "budget",
}


def sha(path: Path) -> str:
    d = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def resolve(path: str | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    if p.is_file():
        return p
    alt = REPO / str(path).replace("/home/wlia0047/ar57/wenyu/GeneRec/", "")
    return alt if alt.is_file() else None


def main() -> None:
    rows = []
    for arm_dir in sorted(S3.iterdir()):
        if not arm_dir.is_dir():
            continue
        finals = sorted(arm_dir.glob("**/test_final.json"),
                        key=lambda p: datetime.strptime(p.parent.name, "%b-%d-%Y_%H-%M-%S"))
        if not finals:
            continue
        final = finals[-1]
        block = json.loads(final.read_text())
        preds = sorted(final.parent.glob("predictions/rank*.jsonl"))
        lines = 0
        for p in preds:
            with p.open("rb") as f:
                lines += sum(1 for _ in f)
        sid = resolve(block.get("sid_path"))
        ckpt = resolve(block.get("stage2_checkpoint"))
        binding = "immutable_run"
        if sid is None:
            arm_sid = S2 / arm_dir.name / "item_sids.json"
            sid = arm_sid if arm_sid.is_file() else None
            binding = "arm_dir_legacy"
        if ckpt is None:
            arm_ck = S2 / arm_dir.name / "rqvae_best.pth"
            ckpt = arm_ck if arm_ck.is_file() else None
        verified = bool(sid and block.get("sid_sha256") and sha(sid) == block["sid_sha256"]
                        and ckpt and block.get("stage2_checkpoint_sha256")
                        and sha(ckpt) == block["stage2_checkpoint_sha256"])
        rows.append({
            "arm": arm_dir.name,
            "family": FAMILY.get(arm_dir.name, "other"),
            "stage3_final": str(final),
            "stage2_run_id": block.get("stage2_run_id"),
            "recall@10": block.get("test_recall@10"),
            "ndcg@10": block.get("test_ndcg@10"),
            "n_eval": block.get("n_eval"),
            "sid": str(sid) if sid else None, "sid_sha256": sha(sid) if sid else None,
            "checkpoint": str(ckpt) if ckpt else None,
            "checkpoint_sha256": sha(ckpt) if ckpt else None,
            "binding": binding if sid or ckpt else "missing",
            "record_verified": verified,
            "predictions": len(preds), "prediction_rows": lines,
            "predictions_complete": lines == EXPECTED,
            "usable_geometry": bool(sid and ckpt and lines == EXPECTED),
        })
    # duplicate SIDs across arms
    seen = {}
    for r in rows:
        if r["sid_sha256"]:
            seen.setdefault(r["sid_sha256"], []).append(r["arm"])
    dup = {k: v for k, v in seen.items() if len(v) > 1}
    for r in rows:
        r["duplicate_sid"] = bool(r["sid_sha256"] and r["sid_sha256"] in dup)
    OUT.joinpath("COHORT.json").write_text(json.dumps(
        {"schema": "genrec.issue270.cohort/1", "rows": rows, "duplicate_sids": dup},
        indent=2, sort_keys=True) + "\n")
    usable = [r for r in rows if r["usable_geometry"]]
    print("rows", len(rows), "usable", len(usable), "verified", sum(r["record_verified"] for r in usable))
    from collections import Counter
    print("families", Counter(r["family"] for r in usable))
    print("duplicate groups", len(dup))
    for r in usable:
        print("%-50s %-22s %d r10=%.6f %s" % (r["arm"], r["family"], r["record_verified"], r["recall@10"], r["binding"]))


if __name__ == "__main__":
    main()
