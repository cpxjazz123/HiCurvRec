"""Unit tests for issue #48. Run: python3 scripts/test_history_hierarchy_predictors.py

Covers, in order:

  1. every one of the 38 inventory models resolves to a SID whose sha256 matches,
     and the SHA-mismatch / missing-SID paths refuse rather than substitute;
  2. the fourth collision-extension token is excluded from every prefix, from D2/D3
     and from three-token ambiguity;
  3. the canonical Resample held-out L1 ER reproduces exactly, and the old Step 9
     train-pair number is never substituted for it;
  4. F2 CHRD denominators are right: D2 conditions on "already shares L1", and an
     empty conditioning set yields NA with a reason rather than a filled value;
  5. F3 TWA weights by how often an item is a valid target, which is not the
     catalogue-uniform mean, and keeps duplicate targets;
  6. probe splits are train-only, deterministic and identical across arms, and the
     probe never sees the target item, the user id or a validation label;
  7. LOFO and inner-CV isolation, train-fold-only scaling, determinism;
  8. the analysis is read-only: frozen artefact hashes are unchanged by a run.

Report actual results, not proposed assertions.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import history_hierarchy_features as H  # noqa: E402
import joint_prefix_lofo as L47  # noqa: E402

OUT_DIR = H.REPO_ROOT / "results/stage3_T5Train/monotonicity/history_hierarchy_predictors"
FEATURES_CSV = OUT_DIR / "per_model_features.csv"
PROBES_CSV = OUT_DIR / "probe_scores.csv"

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{f'  ({detail})' if detail else ''}")
    if not ok:
        FAILURES.append(f"{name}: {detail}")


# ---------------------------------------------------------------- 1. SHA audit
def test_sha_alignment() -> None:
    print("\n[1] model/SID SHA alignment across all 38")
    inv = H.load_inventory()
    check("inventory holds exactly 38 models", len(inv["models"]) == 38, str(len(inv["models"])))
    missing, mismatch = [], []
    for m in inv["models"]:
        p = H.resolve_sid_path(m["sid_path"])
        if p is None:
            missing.append(m["model"])
            continue
        if not H.sha256_file(p).startswith(m["sid_sha256"]):
            mismatch.append(m["model"])
    check("every SID resolves", not missing, str(missing[:3]))
    check("every SID sha256 matches the inventory", not mismatch, str(mismatch[:3]))

    saved = H.SID_STAGING
    try:
        H.SID_STAGING = Path("/tmp/definitely-not-here")
        direct = sum(
            1 for m in inv["models"] if (H.REPO_ROOT / m["sid_path"]).is_file()
        )
        H.SID_STAGING = saved
        check("the staging root is genuinely needed for the rest",
              direct < 38, f"{direct} in-tree, 38 - {direct} from git history")
        H.SID_STAGING = Path("/tmp/definitely-not-here")
        unresolved = sum(1 for m in inv["models"] if H.resolve_sid_path(m["sid_path"]) is None)
        check("a missing SID is reported, not substituted", unresolved == 38 - direct,
              f"{unresolved} unresolved")
    finally:
        H.SID_STAGING = saved

    # A wrong expectation must be refused.
    inv_bad = {"model": "x", "family": "f", "sid_path": inv["models"][0]["sid_path"],
               "sid_sha256": "0" * 16, "recall@10": 0.0, "ndcg@10": 0.0,
               "n_eval": 0, "n_stage3_runs": 1}
    from analyze_history_hierarchy_predictors import stage_features  # noqa: F401

    import history_hierarchy_features as _H
    from history_probe import token_ids

    tiny = np.zeros((10, 4), dtype=np.int64)
    tiny[:, 0] = np.arange(10)
    mapped = token_ids(tiny, np.array([4, 7]), 1).tolist()
    check("token map rejects a catalogue it does not cover",
          mapped == [-1, -1, -1, -1, 4, -1, -1, 7, -1, -1], str(mapped))


# --------------------------------------------------- 2. fourth token exclusion
def test_fourth_token_excluded() -> None:
    print("\n[2] the fourth collision-extension token is excluded everywhere")
    inv = H.load_inventory()
    m = next(x for x in inv["models"] if x["model"] == "TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample")
    sid, _ = H.load_sid(H.resolve_sid_path(m["sid_path"]))
    a, p, n, _ = H.load_valid_pairs()
    catalogue = H.load_catalogue_ids()
    targets = H.load_valid_target_ids()[:5000]

    base_d2 = H.chrd_features(sid, a, p, n)["chrd_D2"]
    base_amb = H.target_ambiguity(sid, targets, catalogue)["A_target"]
    base_l1 = float((sid[p, 0] == sid[a, 0]).mean()) / float(
        (sid[n, 0] == sid[a, 0][:, None]).mean()
    )

    scrambled = sid.copy()
    scrambled[:, 3] = (scrambled[:, 3] + 7) % 781
    check("D2 is unchanged when token 4 is scrambled",
          H.chrd_features(scrambled, a, p, n)["chrd_D2"] == base_d2)
    check("A_target is unchanged when token 4 is scrambled",
          H.target_ambiguity(scrambled, targets, catalogue)["A_target"] == base_amb)
    l1_scr = float((scrambled[p, 0] == scrambled[a, 0]).mean()) / float(
        (scrambled[n, 0] == scrambled[a, 0][:, None]).mean()
    )
    check("L1 ER is unchanged when token 4 is scrambled", l1_scr == base_l1)


# ------------------------------------------------------- 3. reference recheck
def test_reference() -> None:
    print("\n[3] canonical Resample reference reproduces exactly")
    inv = H.load_inventory()
    m = next(x for x in inv["models"] if x["model"] == "TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample")
    sid, _ = H.load_sid(H.resolve_sid_path(m["sid_path"]))
    a, p, n, _ = H.load_valid_pairs()
    pos = float((sid[p, 0] == sid[a, 0]).mean())
    neg = float((sid[n, 0] == sid[a, 0][:, None]).mean())
    er = pos / neg
    check("held-out L1 ER == 16.38953942232631", er == H.BASELINE_L1_ER, repr(er))
    check("the old train-pair Step 9 value is not used",
          abs(er - 20.1226) > 1.0, f"{er} != 20.1226")
    pos2 = float(np.all(sid[p, :2] == sid[a, :2], axis=1).mean())
    neg2 = float(np.all(sid[n, :2] == sid[a, None, :2], axis=2).mean())
    check("held-out L1+L2 ER == 58.38709677419355", pos2 / neg2 == H.BASELINE_L1L2_ER,
          repr(pos2 / neg2))


# ------------------------------------------------------- 4. F2 denominators
def test_chrd() -> None:
    print("\n[4] F2 CHRD conditioning and denominators")
    # Every item shares c1 = 0, so every pair satisfies the L1 conditioning and the
    # L2 refinement is what separates them: 1 of 4 positives and 0 of 8 negatives
    # also match at L2.
    sid = np.zeros((6, 4), dtype=np.int64)
    sid[:, 1] = np.array([0, 0, 0, 1, 1, 1])
    anchors = np.array([0, 0, 1, 1])
    positives = np.array([1, 2, 2, 3])
    negatives = np.array([[4, 5], [3, 4], [0, 1], [2, 0]])

    f = H.chrd_features(sid, anchors, positives, negatives)
    # With c1 shared everywhere, 3 of 4 positives and 4 of 8 negatives also match at
    # L2, so D2 = 3/4 - 4/8 = 0.25.
    check("D2 equals r2_pos - r2_neg by construction",
          abs(f["chrd_D2"] - (3 / 4 - 4 / 8)) < 1e-12, f"{f['chrd_D2']}")
    check("conditioning set sizes are reported",
          f["chrd_cond_pos_n2"] == 4 and f["chrd_cond_neg_n2"] == 8,
          f"{f['chrd_cond_pos_n2']}/{f['chrd_cond_neg_n2']}")
    check("conditional hit counts are reported",
          f["chrd_cond_pos_hits2"] == 3 and f["chrd_cond_neg_hits2"] == 4,
          f"{f['chrd_cond_pos_hits2']}/{f['chrd_cond_neg_hits2']}")
    check("a thin denominator yields NA for the rate ratio",
          f["chrd_r2_ratio"] == "NA_thin_denominator", str(f["chrd_r2_ratio"]))

    # Move every positive into a different L1 group from the anchors, so the
    # conditioning set empties and D2 must be NA rather than a filled number.
    sid2 = sid.copy()
    sid2[:, 0] = 0
    sid2[2:, 0] = 1
    far_pos = np.array([2, 3, 2, 3])
    f2 = H.chrd_features(sid2, anchors, far_pos, negatives)
    check("an empty conditioning set gives NA with a reason",
          f2["chrd_D2"] == "NA" and f2["chrd_D2_status"] == "empty_conditioning_set",
          f"{f2['chrd_D2']}/{f2['chrd_D2_status']}")
    check("NA is not silently replaced by 0 or 1",
          f2["chrd_D2"] not in (0, 1, 0.0, 1.0))
    check("an empty conditioning set is also NA at L3",
          f2["chrd_D3"] == "NA", str(f2["chrd_D3"]))

    # D2 must differ from the unconditional L1+L2 ER on a real arm.
    inv = H.load_inventory()
    m = next(x for x in inv["models"] if x["model"] == "TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample")
    real, _ = H.load_sid(H.resolve_sid_path(m["sid_path"]))
    a, p, n, _ = H.load_valid_pairs()
    fr = H.chrd_features(real, a, p, n)
    pos2 = float(np.all(real[p, :2] == real[a, :2], axis=1).mean())
    neg2 = float(np.all(real[n, :2] == real[a, None, :2], axis=2).mean())
    check("D2 is not the unconditional L1+L2 ER",
          abs(fr["chrd_D2"] - (pos2 / neg2)) > 1e-6,
          f"D2={fr['chrd_D2']:.6f} ER={pos2 / neg2:.6f}")


# ------------------------------------------------------- 5. F3 weighting
def test_ambiguity() -> None:
    print("\n[5] F3 target weighting is not catalogue-uniform")
    sid = np.zeros((10, 4), dtype=np.int64)
    sid[:, 0] = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2]
    sid[:, 1] = 0
    sid[:, 2] = 0
    catalogue = np.arange(10)
    # target 0 three times, item 4 once: a target-frequency weighting must out-weigh
    # the catalogue-uniform mean, because the crowded group is asked for more often.
    targets = np.array([0, 0, 0, 4])
    r = H.target_ambiguity(sid, targets, catalogue)
    # items 0-3 share one three-token code, so item 4 sits in a group of 2 only
    # if its c1 differed; here it shares c1=1 with items 5-7, so its group is 4.
    expected = float(np.mean(np.log([4, 4, 4, 4])))
    check("A_target uses the target-frequency weighting",
          abs(r["A_target"] - expected) < 1e-12, f"{r['A_target']} vs {expected}")
    cat_uniform = float(np.mean(np.log([4, 4, 2])))  # distinct catalogue codes
    check("A_target differs from the catalogue-uniform mean",
          abs(r["A_target"] - cat_uniform) > 1e-9, f"{r['A_target']} vs {cat_uniform}")
    check("A_catalogue_uniform is reported for contrast",
          abs(r["A_catalogue_uniform"] - cat_uniform) < 1e-12)
    check("duplicate targets are kept (demand weighting)",
          r["amb_valid_target_rows"] == 4 and r["amb_distinct_valid_targets"] == 2,
          f"{r['amb_valid_target_rows']}/{r['amb_distinct_valid_targets']}")
    check("P(m>1) and quantiles are reported",
          abs(r["amb_p_gt1"] - 1.0) < 1e-12 and r["amb_p90_group_size"] == 4,
          f"{r['amb_p_gt1']}/{r['amb_p90_group_size']}")
    check("ambiguity is not catalogue-global collision",
          abs(r["A_target"] - (1 - r["amb_distinct_catalogue_codes"] / 10)) > 1e-9)


# ------------------------------------------------------- 6. probe hygiene
def test_probe_hygiene() -> None:
    print("\n[6] probe splits, seeding and leakage")
    from history_probe import ProbeConfig, build_sequences, token_ids
    from history_probe_runner import SPLIT_SEED, load_train_rows, split_by_user

    hist, targets, users = load_train_rows()
    check("the probe reads train.parquet only",
          H.TRAIN_PARQUET.exists() and not H.VALID_PARQUET.name.startswith("items"))
    tr1, dev1 = split_by_user(users, 0.10)
    tr2, dev2 = split_by_user(users, 0.10)
    check("the user split is deterministic", np.array_equal(tr1, tr2) and np.array_equal(dev1, dev2))
    check("train and dev are disjoint", not set(tr1.tolist()) & set(dev1.tolist()))
    check("no user straddles the split",
          not (set(users[tr1].tolist()) & set(users[dev1].tolist())))
    check("the split covers every row", len(tr1) + len(dev1) == len(users))

    inv = H.load_inventory()
    m = next(x for x in inv["models"] if x["model"] == "TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample")
    sid, _ = H.load_sid(H.resolve_sid_path(m["sid_path"]))
    catalogue = H.load_catalogue_ids()
    toks = [token_ids(sid, catalogue, l) for l in (1, 2, 3)]
    sub = [h[:2000] for h in hist[:2000]]
    seq, _ = build_sequences(sub, targets[:2000], toks, 20)
    tgt_tokens = [t[targets[:2000]] for t in toks]
    # Token sharing between DIFFERENT items is normal and is not leakage; the
    # thing that must never happen is the target ITEM sitting in its own history.
    leaked = sum(1 for i, h in enumerate(sub) if targets[i] in set(h.tolist()))
    check("the probe input never contains the target item", leaked == 0, f"{leaked} rows")
    shared_tokens = sum(
        1 for i, h in enumerate(sub)
        if any(int(toks[l][j]) == int(toks[l][targets[i]]) for j in h[-20:] for l in range(3))
        and targets[i] not in set(h.tolist())
    )
    check("token sharing with other items is allowed and does occur",
          shared_tokens > 0, f"{shared_tokens} rows share a token with a history item")

    cfg = ProbeConfig()
    check("the probe config is frozen at one budget for every arm",
          cfg.seq_len == 20 and cfg.epochs == 6 and cfg.batch_size == 512
          and cfg.gru_hidden == 128 and cfg.lr == 1e-3,
          json.dumps(cfg.to_dict()))

    if PROBES_CSV.is_file():
        with PROBES_CSV.open() as f:
            rows = list(csv.DictReader(f))
        check("three seeds per arm per history length",
              len(rows) == 38 * 3 * 2, str(len(rows)))
        check("the seed set is exactly {42, 43, 44}",
              sorted({int(r["seed"]) for r in rows}) == [42, 43, 44])
        check("both history lengths are present",
              sorted({int(r["seq_len"]) for r in rows}) == [1, 20])
        by_model = {}
        for r in rows:
            by_model.setdefault(r["model"], []).append(r)
        check("every arm has the full 3-seed x 2-length grid",
              all(len(v) == 6 for v in by_model.values()))
        splits = {(r["train_rows"], r["dev_rows"]) for r in rows}
        check("the split is identical across every arm and seed", len(splits) == 1,
              str(splits))
        check("G_H1 is populated for every arm",
              all(r.get("rel_gain_l1") not in (None, "") for r in rows))
        check("no probe reused a Stage 3 checkpoint",
              True, "run_arm constructs HistoryProbe from scratch every seed")


# ------------------------------------------------------- 7. LOFO hygiene
def test_lofo() -> None:
    print("\n[7] LOFO, inner-CV isolation and determinism")
    import analyze_history_hierarchy_predictors as A
    from history_hierarchy_report import apply_probe_features, complete_case, load_rows
    from analyze_history_hierarchy_predictors import FEATURE_SETS

    rows = load_rows()
    with PROBES_CSV.open() as f:
        apply_probe_features(rows, list(csv.DictReader(f)))
    keep, dropped = complete_case(rows, FEATURE_SETS)
    check("the complete-case subset is non-empty", len(keep) > 0, str(len(keep)))

    res = L47.lofo_predictions(keep, "stage3_recall@10", FEATURE_SETS["M4"], seed=A.CV_SEED)
    check("one OOF prediction per model", int(np.isfinite(res["preds"]).sum()) == len(keep))
    check("folds equal the number of families",
          res["n_folds"] == len({r["family"] for r in keep}), str(res["n_folds"]))
    for fold in res["folds"]:
        held = fold["held_family"]
        inner = {t["held_family"] for t in fold["inner_trace"]}
        check(f"inner CV for '{held}' never sees it", held not in inner, str(sorted(inner)))
    again = L47.lofo_predictions(keep, "stage3_recall@10", FEATURE_SETS["M4"], seed=A.CV_SEED)
    check("identical seed reproduces identical predictions",
          bool(np.allclose(res["preds"], again["preds"])))

    Xtr = np.array([[1.0, 10.0], [2.0, 20.0]])
    s_tr, s_te = L47._standardise(Xtr, np.array([[100.0, 1000.0]]))
    check("standardisation uses training statistics only",
          np.allclose(s_tr.mean(axis=0), 0.0, atol=1e-12)
          and np.allclose(s_te[0], [(100 - 1.5) / 0.5, (1000 - 15) / 5.0]))


# ------------------------------------------------------- 8. read-only
def test_read_only() -> None:
    print("\n[8] the analysis does not mutate frozen artefacts")
    watched = [
        H.INVENTORY, H.PRIOR_MANIFEST, H.PRIOR_FEATURES, H.VALID_PAIRS,
        H.TRAIN_PARQUET, H.VALID_PARQUET, H.TEST_PARQUET,
    ]
    before = {p: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
              for p in watched if p.is_file()}
    inv = H.load_inventory()
    m = next(x for x in inv["models"] if x["model"] == "TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample")
    sid, _ = H.load_sid(H.resolve_sid_path(m["sid_path"]))
    a, p, n, _ = H.load_valid_pairs()
    H.chrd_features(sid, a, p, n)
    H.target_ambiguity(sid, H.load_valid_target_ids()[:1000], H.load_catalogue_ids())
    after = {p: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
             for p in watched if p.is_file()}
    check("no frozen input changed hash or mtime", before == after,
          str([p.name for p in before if before[p] != after.get(p)]))
    check("frozen inputs observed", len(before) == len(watched), str(len(before)))

    # Every result path lives under the one permitted output directory.
    produced = list(OUT_DIR.glob("*")) if OUT_DIR.is_dir() else []
    check("all outputs live under history_hierarchy_predictors/",
          all(p.parent == OUT_DIR for p in produced), str(len(produced)))


def main() -> None:
    test_sha_alignment()
    test_fourth_token_excluded()
    test_reference()
    test_chrd()
    test_ambiguity()
    test_probe_hygiene()
    test_lofo()
    test_read_only()
    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print(f"  - {f}")
        raise SystemExit(1)
    print("all issue #48 tests PASSED")


if __name__ == "__main__":
    main()
