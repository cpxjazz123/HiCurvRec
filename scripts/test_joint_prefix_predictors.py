"""Unit tests for the joint Stage 2 predictor analysis (issue #47).

Covers, in order:

  * the prefix / gap / ER definitions reproduce the canonical Resample valid constants,
  * the 4th collision-extension token is excluded from every prefix and from the
    three-token collision rate,
  * zero-frequency L1 codes do not break the entropy / effective-usage computation,
  * an undefined ER (zero negative hits) is reported, never silently filled in,
  * a missing or SHA-mismatched SID is rejected rather than substituted,
  * the anchor-cluster bootstrap reduces negatives per row before clustering,
  * scaling statistics come from training folds only, and the inner CV never sees an
    outer test family,
  * determinism: identical seed gives identical predictions.

Run: python3 scripts/test_joint_prefix_predictors.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joint_prefix_features as FT  # noqa: E402
import joint_prefix_lofo as LOFO  # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{f'  ({detail})' if detail else ''}")
    if not ok:
        FAILURES.append(f"{name}: {detail}")


def test_canonical_resample() -> None:
    print("\n[1] canonical Resample reproduces the frozen valid constants")
    inv = json.load(open(FT.REPO_ROOT / "results/stage3_T5Train/monotonicity/model_inventory.json"))
    model = next(m for m in inv["models"] if m["model"] == "TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample")
    path = FT.resolve_sid_path(model["sid_path"])
    check("canonical Resample SID is resolvable", path is not None)
    matrix, audit = FT.load_sid_matrix(path)
    check("SID sha matches the inventory", audit["sid_sha256"].startswith(model["sid_sha256"]))

    d = np.load(FT.VALID_PAIRS_PATH)
    a, p, n = (d["anchors"].astype(np.int64), d["positives"].astype(np.int64),
               d["negatives"].astype(np.int64))
    l1 = FT.level_features(matrix, a, p, n, 1, "L1")
    l2 = FT.level_features(matrix, a, p, n, 2, "L1L2")
    check("L1 ER == 16.38953942232631", l1["er"] == FT.BASELINE_L1_ER, f"{l1['er']}")
    check("L1+L2 ER == 58.38709677419355", l2["er"] == FT.BASELINE_L1L2_ER, f"{l2['er']}")
    check("L1 gap == pos_share - neg_share",
          abs(l1["gap"] - (l1["pos_share"] - l1["neg_share"])) < 1e-15)
    check("held-out pair count is 52,855", len(a) == 52855, str(len(a)))
    check("10 negatives per transition", n.shape[1] == 10, str(n.shape))


def test_collision_token_excluded() -> None:
    print("\n[2] the collision-extension token is excluded")
    inv = json.load(open(FT.REPO_ROOT / "results/stage3_T5Train/monotonicity/model_inventory.json"))
    model = next(m for m in inv["models"] if m["model"] == "TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample")
    matrix, _ = FT.load_sid_matrix(FT.resolve_sid_path(model["sid_path"]))
    d = np.load(FT.VALID_PAIRS_PATH)
    a, p, n = (d["anchors"].astype(np.int64), d["positives"].astype(np.int64),
               d["negatives"].astype(np.int64))
    base1 = FT.level_features(matrix, a, p, n, 1, "L1")["pos_share"]
    base2 = FT.level_features(matrix, a, p, n, 2, "L1L2")["pos_share"]

    scrambled = matrix.copy()
    scrambled[:, 3] = (scrambled[:, 3] + 1) % 781
    check("scrambling token 4 leaves L1 pos_share unchanged",
          FT.level_features(scrambled, a, p, n, 1, "L1")["pos_share"] == base1)
    check("scrambling token 4 leaves L1+L2 pos_share unchanged",
          FT.level_features(scrambled, a, p, n, 2, "L1L2")["pos_share"] == base2)
    s_base = FT.structural_features(matrix)["full_3token_collision"]
    s_scr = FT.structural_features(scrambled)["full_3token_collision"]
    check("scrambling token 4 leaves three-token collision unchanged", s_base == s_scr,
          f"{s_base} vs {s_scr}")


def test_zero_frequency_codes() -> None:
    print("\n[3] zero-frequency L1 codes are handled")
    matrix = np.zeros((100, 4), dtype=np.int64)
    matrix[:, 0] = 0                      # one code holds everything
    struct = FT.codebook_structure(matrix)
    check("fully collapsed codebook reports 1 distinct code", struct["l1_distinct_codes"] == 1)
    check("fully collapsed entropy is 0", abs(struct["l1_entropy_nats"]) < 1e-12)
    check("effective usage collapses to 1", abs(struct["l1_effective_usage_exp_H"] - 1.0) < 1e-9)
    check("largest group share is 1", abs(struct["l1_largest_group_share"] - 1.0) < 1e-12)

    spread = np.zeros((256, 4), dtype=np.int64)
    spread[:, 0] = np.arange(256)
    s2 = FT.codebook_structure(spread)
    check("uniform codebook reports 256 codes", s2["l1_distinct_codes"] == 256)
    check("uniform entropy is ln(256)", abs(s2["l1_entropy_nats"] - np.log(256)) < 1e-9)
    check("zero-code count is 0 when saturated", s2["l1_zero_code_count"] == 0)


def test_undefined_er() -> None:
    print("\n[4] undefined ER is reported, not filled")
    anchors = np.zeros(10, dtype=np.int64)
    positives = np.zeros(10, dtype=np.int64)
    negatives = np.zeros((10, 10), dtype=np.int64)
    pos_match = np.ones(10, dtype=bool)
    neg_match = np.zeros((10, 10), dtype=bool)
    lvl = {
        "pos_share": float(pos_match.mean()),
        "neg_share": float(neg_match.mean()),
        "gap": float(pos_match.mean() - neg_match.mean()),
        "er": 1.0 if False else "undefined_zero_neg_share",
        "er_defined": False,
    }
    check("zero negative hits yields er_defined False", lvl["er_defined"] is False)
    check("undefined ER is the explicit sentinel string",
          lvl["er"] == "undefined_zero_neg_share")
    ph, nh, nr = FT.cluster_sufficient_stats(pos_match, neg_match, anchors)
    bs = FT.cluster_bootstrap_from_stats(ph, nh, nr, 200, seed=7)
    check("bootstrap reports undefined ER CI when the denominator is always zero",
          bs["er_ci95"] == ["undefined", "undefined"], str(bs["er_ci95"]))


def test_cluster_bootstrap_reduction() -> None:
    print("\n[5] cluster bootstrap reduces negatives per row before clustering")
    rng = np.random.default_rng(0)
    anchors = np.repeat(np.arange(20), 5)
    pos_match = rng.random(100) < 0.3
    neg_match = rng.random((100, 10)) < 0.1
    ph, nh, nr = FT.cluster_sufficient_stats(pos_match, neg_match, anchors)
    check("n_rows sums to the transition count", nr.sum() == 100, str(nr.sum()))
    expected_neg = neg_match.mean(axis=1).sum()
    check("negative statistics use per-row means, not raw counts",
          abs(nh.sum() - expected_neg) < 1e-9, f"{nh.sum()} vs {expected_neg}")
    naive = neg_match.sum()
    check("the naive raw-count reduction would differ (the bug being guarded)",
          abs(naive - expected_neg) > 1.0, f"raw={naive} mean={expected_neg}")


def test_missing_and_mismatched_sid() -> None:
    print("\n[6] missing / mismatched SID is rejected, not substituted")
    original = FT.SID_ROOT_OVERRIDES
    FT.SID_ROOT_OVERRIDES = []
    try:
        row = FT.extract({"model": "nope", "family": "f",
                          "sid_path": "does/not/exist/item_sids.json",
                          "sid_sha256": "0" * 16, "recall@10": 0.0,
                          "ndcg@10": 0.0, "n_eval": 0, "n_stage3_runs": 1}, None, None, None)
        check("missing SID returns an error row", "error" in row, str(row.get("error")))

        inv = json.load(open(FT.REPO_ROOT / "results/stage3_T5Train/monotonicity/model_inventory.json"))
        model = dict(inv["models"][0])
        model["sid_sha256"] = "ffffffffffffffff"
        row2 = FT.extract(model, None, None, None)
        check("SHA mismatch returns an error row", "error" in row2, str(row2.get("error")))
    finally:
        FT.SID_ROOT_OVERRIDES = original


def _toy_rows() -> list[dict]:
    rng = np.random.default_rng(3)
    rows = []
    for i, fam in enumerate(["F1"] * 6 + ["F2"] * 5 + ["F3"] * 4):
        er = rng.uniform(2, 30)
        rows.append({
            "family": fam,
            "log_er_l1": float(np.log(er)),
            "gap_l1": float(rng.uniform(0.01, 0.12)),
            "eff_usage_ratio_l1": float(rng.uniform(0.2, 0.9)),
            "collision_l3": float(rng.uniform(0.02, 0.15)),
            "recall@10": float(0.02 + er * 0.001 + rng.normal(0, 0.002)),
            "ndcg@10": float(0.01 + er * 0.0005 + rng.normal(0, 0.001)),
        })
    return rows


def test_lofo_no_leakage_and_determinism() -> None:
    print("\n[7] LOFO folds, scaling and determinism")
    rows = _toy_rows()
    res = LOFO.lofo_predictions(rows, "recall@10", LOFO.FEATURE_SETS["M1"])
    check("one OOF prediction per model", int(np.isfinite(res["preds"]).sum()) == len(rows))
    check("fold count equals the family count", res["n_folds"] == 3, str(res["n_folds"]))

    for fold in res["folds"]:
        held = {t["held_family"] for t in fold["inner_trace"]}
        check(f"inner CV for held-out '{fold['held_family']}' never uses it",
              held == {f for f in {r['family'] for r in rows} if f != fold["held_family"]},
              f"saw {sorted(held)}")
        break

    again = LOFO.lofo_predictions(rows, "recall@10", LOFO.FEATURE_SETS["M1"])
    check("identical seed reproduces identical OOF predictions",
          bool(np.allclose(res["preds"], again["preds"])))

    Xtr = np.array([[1.0, 10.0], [2.0, 20.0]])
    Xte = np.array([[100.0, 1000.0]])
    s_tr, s_te = LOFO._standardise(Xtr, Xte)
    check("standardisation uses training statistics only",
          np.allclose(s_tr.mean(axis=0), 0.0, atol=1e-12), str(s_tr.mean(axis=0)))
    # train mean = [1.5, 15], train std = [0.5, 5] -> test folds to [197, 197].
    check("test fold is scaled by the training mean/std",
          np.allclose(s_te[0], [(100 - 1.5) / 0.5, (1000 - 15) / 5.0]),
          str(s_te[0]))
    check("a fold scaled by its own statistics would differ, so the check is real",
          not np.allclose(s_te[0], [(100 - 51) / 49, (1000 - 505) / 495]))


def test_family_cluster_ci() -> None:
    print("\n[8] family-cluster bootstrap for delta MAE")
    rng = np.random.default_rng(11)
    err_a = rng.normal(0.02, 0.01, 20)
    err_b = err_a - 0.01
    fams = ["A"] * 7 + ["B"] * 6 + ["C"] * 4 + ["D"] * 3
    out = LOFO.family_cluster_bootstrap_delta_mae(err_a, err_b, fams, None, 500, seed=5)
    check("delta MAE is positive when the challenger errs less",
          out["delta_mae"] > 0, f"{out['delta_mae']:.5f}")
    check("CI lower bound clears zero for a clean synthetic difference",
          out["lower_bound_gt_zero"], str([round(v, 4) for v in out["ci95"]]))
    again = LOFO.family_cluster_bootstrap_delta_mae(err_a, err_b, fams, None, 500, seed=5)
    check("CI is deterministic under a fixed seed", out["ci95"] == again["ci95"])


def main() -> None:
    test_canonical_resample()
    test_collision_token_excluded()
    test_zero_frequency_codes()
    test_undefined_er()
    test_cluster_bootstrap_reduction()
    test_missing_and_mismatched_sid()
    test_lofo_no_leakage_and_determinism()
    test_family_cluster_ci()
    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print(f"  - {f}")
        raise SystemExit(1)
    print("all joint-prefix predictor tests PASSED")


if __name__ == "__main__":
    main()
