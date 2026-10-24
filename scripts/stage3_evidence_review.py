#!/usr/bin/env python3
"""Second-stage uncertainty/diagnostic audit, using ONLY stage3_evidence_audit.py outputs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

def assess(folder: Path, seed: int = 20261022, repeats: int = 2000) -> dict:
    oof = pd.read_csv(folder/"historical_oof.csv")
    campaign = pd.read_csv(folder/"campaign_holdout.csv")
    diagnosis = pd.read_csv(folder/"campaign_mechanism_diagnostics.csv")
    rng = np.random.default_rng(seed)
    uncertainty = []
    for target in sorted(oof.target.unique()):
        base=oof[(oof.target==target)&(oof.model_set=="ER")]
        for model in sorted(oof.model_set.unique()):
            if model=="ER": continue
            newer=oof[(oof.target==target)&(oof.model_set==model)]
            merged=base[["model","family","y","abs_error"]].merge(
                newer[["model","family","y","abs_error"]],on="model",
                suffixes=("_er","_new"),validate="one_to_one")
            if not (merged.family_er==merged.family_new).all():
                raise ValueError(f"model family mismatch in paired CI for {model}")
            if not np.allclose(merged.y_er, merged.y_new):
                raise ValueError(f"target label mismatch in paired CI for {model}")
            diff=(merged.abs_error_er-merged.abs_error_new).to_numpy(float)
            families=merged.family_er.unique()
            family_deltas={str(f):float(np.mean(diff[merged.family_er==f])) for f in families}
            if len(merged)!=len(base):
                # Full 38 ER predictions cannot substitute for baseline REFITTED on
                # a feature's smaller complete-case subset.
                uncertainty.append({"target":target,"model":model,"n":len(merged),
                    "status":"missing_same_subset_baseline_predictions",
                    "delta_ci95":None,"certified":False})
                continue
            samples=[]
            for _ in range(repeats):
                chosen=rng.choice(families,size=len(families),replace=True)
                values=np.concatenate([diff[np.asarray(merged.family_er==f)] for f in chosen])
                samples.append(float(np.mean(values)))
            lo,hi=np.quantile(samples,[.025,.975])
            uncertainty.append({"target":target,"model":model,"n":len(merged),
                "families":len(families),"delta_mae":float(diff.mean()),
                "delta_ci95":[float(lo),float(hi)],
                "family_deltas":family_deltas,
                "ci_lower_positive":bool(lo>0),
                "status":"retrospective_5_family_exploratory_only","certified":False})
    correlations=[]
    sub=diagnosis.loc[diagnosis.paired==True]
    if len(sub):
        for field in ("er","gini","branches","sequence_excess_in_sample"):
            for target in ("recall@10",):
                mask=sub[[field,target]].dropna()
                if len(mask)>=3:
                    corr=spearmanr(mask[field],mask[target])
                    correlations.append({"metric":field,"target":target,"n":len(mask),
                        "rho":float(corr.statistic),"p_nominal":float(corr.pvalue),
                        "label":"descriptive; small, dependent set; no multiple-test confirmation"})
    er_transfer=campaign.loc[campaign.target=="recall@10"]
    if len(er_transfer)==0:raise ValueError("no campaign holdout pairs")
    review={"uncertainty":uncertainty,"campaign_correlations":correlations,
        "campaign_holdout":{"n":int(er_transfer.model.nunique()),
            "n_out_of_historical_ER_range":int(er_transfer.er_outside_historical_range.sum()),
            "recall_mae":float(er_transfer.abs_error.mean()),
            "mean_bias_pred_minus_observed":float((er_transfer.predicted-er_transfer.observed).mean()),
            "stage3_labels_used_for_training":False,
            "provenance":"new campaign frozen archive labels; independent from historical fit"},
        "certificate":{"auto_screening_gate_permitted":False,
            "rationale":"historical family-cluster retrospective tests only; no independent new-family proof, 8 of 10 campaign ER observations extrapolate"},
        "constraints":["do not select a best feature from the same #107 holdout it was assessed on",
          "diagnostic sequence information computed in-sample is not a TIGER Stage3 proxy",
          "zero missing Stage3 outcomes were imputed as failures"]}
    (folder/"uncertainty_review.json").write_text(json.dumps(review,indent=2,ensure_ascii=False),encoding="utf8")
    pd.DataFrame(uncertainty).drop(columns=["family_deltas"],errors="ignore").to_csv(folder/"uncertainty_ci.csv",index=False)
    pd.DataFrame(correlations).to_csv(folder/"diagnostic_correlations.csv",index=False)
    return review

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--folder",type=Path,default=Path("results/harness3_runs/stage3_evidence_audit"))
    a=p.parse_args()
    z=assess(a.folder)
    print(json.dumps({"holdout":z["campaign_holdout"],"certificate":z["certificate"]},indent=2))
