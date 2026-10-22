"""The historical evidence baseline's tests (issue #109, section 4).

The baseline decides whether the proxy scores may be used to rank a portfolio, so the tests
are about the properties that decision depends on: the seed-42-only rule, the family-held-out
protocol, the pairing of the errors, the counterexamples, the out-of-support flags, and the
refusal to compute anything when the archives disagree.

    /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 -m pytest \
        tools/harness2_genrec/tests/test_historical_evidence.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.harness2_genrec import historical_evidence as HE  # noqa: E402


def _write_archives(
    root: Path,
    *,
    models: int = 15,
    families: int = 5,
    seed: int = 0,
    drop_from: str = "",
    label_shift: float = 0.0,
) -> Path:
    """A synthetic archive with the same four files and the same columns as the real one."""
    history = root / HE.HISTORY
    (history / "joint_prefix_predictors").mkdir(parents=True, exist_ok=True)
    (history / "history_hierarchy_predictors_fix48").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    names = [f"model_{index:03d}" for index in range(models)]
    family_names = [f"family_{index}" for index in range(families)]
    family_of = {
        name: family_names[index % families] for index, name in enumerate(names)
    }
    recall = rng.uniform(0.05, 0.08, size=models)
    ndcg = rng.uniform(0.02, 0.05, size=models)
    log_er = rng.uniform(0.03, 2.8, size=models)
    d2 = rng.uniform(0.0, 0.2, size=models)

    inventory = pd.DataFrame(
        {
            "model": names,
            "family": [family_of[name] for name in names],
            "recall@10": recall + label_shift,
            "ndcg@10": ndcg,
            "n_eval": 57439,
            "sid_sha256": [f"sha{index:03d}" for index in range(models)],
            "status": "ok",
            "sid_item_ids_match_catalogue": True,
        }
    )
    features = pd.DataFrame(
        {
            "model": names,
            "family": [family_of[name] for name in names],
            "log_er_l1": log_er,
            # the label shift is applied to the INVENTORY only, so the two archives disagree
            "stage3_recall@10": recall,
            "stage3_ndcg@10": ndcg,
        }
    )
    per_model = pd.DataFrame(
        {"model": names, "chrd_D2": d2, "log_er_l1": log_er, "er_l1_recheck": np.exp(log_er)}
    )
    probe_rows = []
    for index, name in enumerate(names):
        prior = 5.0 + 0.1 * index
        probe_ce = prior * (1.0 - 0.05)
        relative = (prior - probe_ce) / prior
        for probe_seed in (42, 43, 44):
            # the seed-42 row is the one the baseline must use; the others are decoys with a
            # different value, so a three-seed mean would be visibly different
            scale = 1.0 if probe_seed == 42 else 2.0
            probe_rows.append(
                {
                    "model": name,
                    "seed": probe_seed,
                    "seq_len": 20,
                    "prior_ce_l1": prior,
                    "probe_ce_l1": probe_ce if probe_seed == 42 else probe_ce * scale,
                    "rel_gain_l1": relative if probe_seed == 42 else relative * scale,
                    "prior_top1_acc_l1": 0.1,
                    "probe_top1_acc_l1": 0.2,
                    "prior_top5_acc_l1": 0.3,
                    "probe_top5_acc_l1": 0.4,
                    "prior_top20_acc_l1": 0.5,
                    "probe_top20_acc_l1": 0.6,
                }
            )
    probe = pd.DataFrame(probe_rows)

    if drop_from:
        keep = [name for name in names if name != drop_from]
        inventory = inventory[inventory["model"].isin(keep)]
        features = features[features["model"].isin(keep)]
        per_model = per_model[per_model["model"].isin(keep)]
        probe = probe[probe["model"].isin(keep)]

    inventory.to_csv(history / "model_inventory.csv", index=False)
    features.to_csv(history / "joint_prefix_predictors" / "stage2_valid_features.csv", index=False)
    per_model.to_csv(
        history / "history_hierarchy_predictors_fix48" / "per_model_features.csv", index=False
    )
    probe.to_csv(history / "history_hierarchy_predictors_fix48" / "probe_scores.csv", index=False)
    return root


def test_the_seed_42_rows_are_used_and_not_the_three_seed_mean(tmp_path):
    """The published means average three seeds; the baseline must use seed 42 only."""
    root = _write_archives(tmp_path)
    table = HE.historical_table(root, expected_models=15)
    probe = pd.read_csv(
        root / HE.HISTORY / "history_hierarchy_predictors_fix48" / "probe_scores.csv"
    )
    seed42 = probe[probe["seed"] == 42].set_index("model")
    mean_of_all = probe.groupby("model")["rel_gain_l1"].mean()
    for name in table["model"].head(5):
        expected = (seed42.loc[name, "prior_ce_l1"] - seed42.loc[name, "probe_ce_l1"]) / (
            seed42.loc[name, "prior_ce_l1"]
        )
        assert table.set_index("model").loc[name, "G_H1"] == pytest.approx(expected)
        # the three-seed mean would be a different number, and it is not what is used
        assert table.set_index("model").loc[name, "G_H1"] != pytest.approx(mean_of_all[name])


def test_each_family_is_held_out_exactly_once_and_never_predicts_itself(tmp_path):
    """No model's prediction may come from a model of its own family."""
    root = _write_archives(tmp_path)
    table = HE.historical_table(root, expected_models=15)
    predictions = HE.lofo_predictions(table, HE.FEATURE_SETS["ER_plus_G_H1"], "recall")
    folds = predictions.groupby("held_out_family")["model"].nunique()
    assert set(folds.index) == set(table["family"].unique())
    assert len(folds) == 5
    # the held-out family of a row is the row's own family, and the row is in the test fold
    for _, row in predictions.iterrows():
        assert row["held_out_family"] == row["family"]
    # every model is predicted exactly once per feature set
    assert len(predictions) == len(table)


def test_the_errors_are_paired_on_the_same_models(tmp_path):
    """Every feature set must be scored on the same models, or the delta means nothing."""
    root = _write_archives(tmp_path)
    table = HE.historical_table(root, expected_models=15)
    baseline = HE.lofo_predictions(table, HE.FEATURE_SETS["ER"], "recall")
    proxy = HE.lofo_predictions(table, HE.FEATURE_SETS["ER_plus_G_H1"], "recall")
    assert set(baseline["model"]) == set(proxy["model"])
    merged = proxy.merge(baseline[["model", "abs_error"]], on="model", suffixes=("", "_er"))
    assert len(merged) == len(baseline)


def test_a_counterexample_is_reported(tmp_path):
    """A model the proxy predicts worse than the ER alone must be named."""
    root = _write_archives(tmp_path)
    table = HE.historical_table(root, expected_models=15)
    report = HE.analyse(root, expected_models=15)
    rows = {row["feature_set"]: row for row in report["results"] if row["target"] == "recall"}
    assert "counterexamples" in rows["ER_plus_G_H1"]
    # the count and the names agree with each other
    row = rows["ER_plus_G_H1"]
    assert len(row["counterexample_models"]) == row["counterexamples"]
    assert set(row["counterexample_models"]).issubset(set(table["model"]))


def test_a_value_outside_the_training_support_is_flagged(tmp_path):
    """A feature outside the training folds' range must be flagged, not silently used."""
    root = _write_archives(tmp_path)
    table = HE.historical_table(root, expected_models=15)
    predictions = HE.lofo_predictions(table, HE.FEATURE_SETS["ER_plus_G_H1"], "recall")
    flagged = predictions[predictions["outside_training_support"].apply(bool)]
    assert len(flagged) > 0
    for _, row in flagged.iterrows():
        assert row["outside_training_support"]


def test_a_provenance_failure_is_refused(tmp_path):
    """A model missing from one archive, or a label that disagrees, stops the analysis."""
    root = _write_archives(tmp_path, drop_from="model_003")
    with pytest.raises(HE.EvidenceError):
        HE.historical_table(root, expected_models=15)

    root = _write_archives(tmp_path / "shifted", label_shift=0.01)
    with pytest.raises(HE.EvidenceError):
        HE.historical_table(root, expected_models=15)


def test_the_real_archives_reproduce_the_historical_audit(tmp_path):
    """The real archives: 38 models, the ER-only MAE the reference value."""
    report = HE.analyse(REPO_ROOT)
    assert report["historical_models"] == 38
    assert report["historical_families"] == 5
    assert report["complete_case_models"] == 38
    assert report["er_only_reproduces_the_historical_audit"] is True
    assert report["er_only_recall_mae"] == pytest.approx(HE.ER_ONLY_RECALL_MAE, abs=HE.ER_ONLY_MAE_TOLERANCE)
    assert report["g_h1_source"]["seed"] == 42
    assert report["g_h1_source"]["three_seed_means_used"] is False
    # the proxies are explicitly not certified for screening
    assert report["abstention"]["certified_for_screening"] is False
    # the six feature sets are all present for both targets
    names = {row["feature_set"] for row in report["results"]}
    assert names == set(HE.FEATURE_SETS)
    assert {row["target"] for row in report["results"]} == {"recall", "ndcg"}
