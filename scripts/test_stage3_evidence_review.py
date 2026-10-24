"""Unit tests for retrospective cluster uncertainty, independent from Stage2/Stage3."""
import pandas as pd
import pytest
from scripts.stage3_evidence_review import assess


def fixture_outputs(tmp_path):
    rows=[]
    for i,fam in enumerate(("A","B","C","D","E")):
        for j in range(2):
            model=fam+str(j)
            y=.04+.003*i+.0001*j
            rows.append({"model":model,"family":fam,"target":"recall@10",
                "y":y,"pred":y+.003,"abs_error":.003,"alpha":1.0,"model_set":"ER"})
            rows.append({"model":model,"family":fam,"target":"recall@10",
                "y":y,"pred":y+.002,"abs_error":.002,"alpha":1.0,"model_set":"ER_plus_D2"})
    pd.DataFrame(rows).to_csv(tmp_path/"historical_oof.csv",index=False)
    pd.DataFrame([{"model":"new","target":"recall@10","er":17.0,
      "predicted":.06,"observed":.058,"abs_error":.002,
      "er_outside_historical_range":True,"not_certified":True}]).to_csv(
        tmp_path/"campaign_holdout.csv",index=False)
    pd.DataFrame([{"model":"new","paired":True,"er":17.0,
        "gini":.3,"branches":64,"sequence_excess_in_sample":.10,
        "recall@10":.058}]).to_csv(tmp_path/"campaign_mechanism_diagnostics.csv",index=False)


def test_retrospective_ci_never_certifies_selection(tmp_path):
    fixture_outputs(tmp_path)
    review=assess(tmp_path,repeats=100)
    assert review["certificate"]["auto_screening_gate_permitted"] is False
    item=review["uncertainty"][0]
    assert item["delta_mae"]==pytest.approx(.001)
    assert item["ci_lower_positive"] is True
    assert item["certified"] is False
    assert review["campaign_holdout"]["n_out_of_historical_ER_range"]==1
    assert (tmp_path/"uncertainty_ci.csv").is_file()


def test_mismatched_historical_labels_are_rejected(tmp_path):
    fixture_outputs(tmp_path)
    p=tmp_path/"historical_oof.csv"
    d=pd.read_csv(p)
    d.loc[(d.model=="A0")&(d.model_set=="ER_plus_D2"),"y"] = .2
    d.to_csv(p,index=False)
    with pytest.raises(ValueError,match="label mismatch"):
        assess(tmp_path,repeats=10)


def test_incomplete_case_does_not_steal_full_cohort_baseline(tmp_path):
    fixture_outputs(tmp_path)
    p=tmp_path/"historical_oof.csv"
    d=pd.read_csv(p)
    d=d.loc[~((d.model_set=="ER_plus_D2")&(d.model=="A0"))]
    d.to_csv(p,index=False)
    result=assess(tmp_path,repeats=10)
    assert result["uncertainty"][0]["delta_ci95"] is None
    assert result["uncertainty"][0]["certified"] is False
