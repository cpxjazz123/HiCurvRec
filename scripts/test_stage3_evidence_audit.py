"""Regression tests for scripts/stage3_evidence_audit.py.

The tests never run Stage2/Stage3, re-read frozen pairs, or write into archives.
"""
import numpy as np
import pandas as pd
import pytest
from scripts.stage3_evidence_audit import ROOT, historical, campaign, lofo, evaluate


def test_inventory_and_corrected_probes_match_exactly():
    data = historical(ROOT)
    assert len(data) == 38
    assert data.model.is_unique
    assert data.family.nunique() == 5
    assert data["G_H1"].notna().all()
    assert len(data) == data["fix_er"].notna().sum()
    resample = data.loc[data.model == "TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample"].iloc[0]
    assert resample["er"] == pytest.approx(16.38953942232631, abs=1e-11)
    assert resample["l1l2_er"] == pytest.approx(58.38709677419355, abs=1e-11)


def test_historical_ER_LOFO_reproduces_47():
    data = historical(ROOT)
    out = lofo(data, ("log_er_l1",), "recall@10")
    assert len(out) == 38
    assert out.model.is_unique
    assert np.mean(out.abs_error) == pytest.approx(0.003752853292309109, abs=2e-8)


def test_held_out_labels_cannot_change_their_own_predictions():
    # Labels from family A are never seen by the model fitting/testing family A.
    rows = []
    for j, family in enumerate(("A", "B", "C", "D")):
        for i in range(5):
            x = 1.0 + j * 0.2 + i * 0.02
            rows.append({"model":f"{family}_{i}","group":family,
                         "log_er_l1":x,"recall@10":0.03 + x*0.01 + (j%2)*0.002})
    source = pd.DataFrame(rows)
    first = lofo(source, ("log_er_l1",), "recall@10")
    revised = source.copy()
    revised.loc[revised.group == "A","recall@10"] += 0.5
    second = lofo(revised, ("log_er_l1",), "recall@10")
    p1 = first.loc[first.family=="A"].set_index("model").pred.sort_index()
    p2 = second.loc[second.family=="A"].set_index("model").pred.sort_index()
    np.testing.assert_allclose(p1, p2, rtol=0, atol=1e-12)


def test_paired_campaign_is_not_interpreted_as_all_wins_or_failures():
    archive, _ = campaign(ROOT)
    assert "r8_control" in set(archive.model)
    assert archive.paired.sum() >= 6
    assert not archive.joint_success.any()
    assert not archive.loc[archive["recall@10"].isna(), "paired"].any()


def test_report_can_be_built_offline(tmp_path):
    report = evaluate(ROOT, tmp_path/"analysis")
    assert report["historical_models"] == 38
    assert report["historical_families"] == 5
    assert report["campaign_paired_including_control"] >= 6
    assert report["campaign_joint_wins"] == 0
    for path in ("historical_oof.csv","historical_lofo.csv",
                 "campaign_manifest_audit.csv","campaign_holdout.csv",
                 "campaign_mechanism_diagnostics.csv","evidence.json","README.md"):
        assert (tmp_path/"analysis"/path).is_file()
    holdout = pd.read_csv(tmp_path/"analysis"/"campaign_holdout.csv")
    assert holdout["not_certified"].all()
    assert holdout.groupby("target").model.nunique().min() >= 6
