#!/usr/bin/env python3
"""Read-only Stage2→Stage3 evidence audit.

Usage: python scripts/stage3_evidence_audit.py --out results/harness3_runs/stage3_evidence_audit
No training, frozen-pair reads, or source artifact modifications.
New #107 outcomes are NEVER used for training, feature selection, or hyperparameter tuning.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
HIST = Path("results/stage3_T5Train/monotonicity")
STORE = Path("results/harness3_runs/store/runs")
TRANSFER = Path("results/harness3_runs/transfer_analysis/stage3_transfer_report.json")
ER_GATE, R_GATE, N_GATE = 16.38953942232631, 0.06406796775709883, 0.03530651701940677
FEATURES = {
    "ER": ("log_er_l1",),
    "ER_plus_gap": ("log_er_l1", "gap"),
    "ER_plus_structure": ("log_er_l1", "struct_l1_effective_usage_ratio", "struct_full_3token_collision"),
    "structure_only": ("struct_l1_effective_usage_ratio", "struct_full_3token_collision"),
    "ER_plus_D2": ("log_er_l1", "chrd_D2"),
    "ER_plus_G_H1": ("log_er_l1", "G_H1"),
    "ER_plus_L12_ER": ("log_er_l1", "log_er_l12"),
    "L12_ER_only": ("log_er_l12",),
}
ALPHAS = (.1, 1., 10., 100.)

def require_csv(p):
    if not p.is_file():
        raise FileNotFoundError(f"missing original archive: {p}")
    return pd.read_csv(p)

def historical(root):
    inv = require_csv(root/HIST/"model_inventory.csv")
    feat = require_csv(root/HIST/"joint_prefix_predictors/stage2_valid_features.csv")
    fix = require_csv(root/HIST/"history_hierarchy_predictors_fix48/per_model_features.csv")
    probe = require_csv(root/HIST/"history_hierarchy_predictors_fix48/probe_scores.csv")
    for name, data in (("inventory",inv),("#47",feat),("#49",fix)):
        if data.model.duplicated().any():
            raise ValueError(f"duplicate model ID in {name}")
    if len(inv)!=38 or set(inv.model)!=set(feat.model) or set(inv.model)!=set(fix.model):
        raise ValueError("history inventory / frozen features coverage mismatch")
    inv = inv[["model","family","recall@10","ndcg@10","n_eval","sid_sha256"]]
    feat = feat.rename(columns={"family":"feature_family", "sid_sha256":"feature_sid_sha",
        "stage3_recall@10":"feature_recall", "stage3_ndcg@10":"feature_ndcg"})
    d = inv.merge(feat, on="model", validate="one_to_one")
    if not (d.family==d.feature_family).all():
        raise ValueError("model family mismatch")
    if not np.allclose(d["recall@10"],d.feature_recall,atol=1e-10):
        raise ValueError("historical Recall label mismatch")
    if not np.allclose(d["ndcg@10"],d.feature_ndcg,atol=1e-10):
        raise ValueError("historical NDCG label mismatch")
    if not all(str(full).startswith(str(prefix)) for full,prefix in zip(d.feature_sid_sha,d.sid_sha256)):
        raise ValueError("historical SID sha mismatch")
    if not (d.n_eval==57439).all() or not (d.status=="ok").all() or not d.sid_item_ids_match_catalogue.all():
        raise ValueError("historical Stage3/SID protocol changed")
    fix = fix[["model","chrd_D2","log_er_l1","er_l1_recheck"]].rename(
        columns={"log_er_l1":"fix_log_er","er_l1_recheck":"fix_er"})
    d = d.merge(fix,on="model",validate="one_to_one")
    if not np.allclose(d.log_er_l1,d.fix_log_er,atol=1e-8) or not np.allclose(d.er,d.fix_er,atol=1e-8):
        raise ValueError("independent frozen ER mismatch")
    # The corrected #49 probe has 3 seeds at seq_len=20; #48 is INVALIDATED.
    probe = probe.loc[probe.seq_len==20]
    seeds = probe.groupby("model").seed.nunique()
    if set(seeds.index)!=set(inv.model) or not (seeds==3).all():
        raise ValueError("corrected 3-seed history probe incomplete")
    gh = probe.groupby("model",as_index=False).rel_gain_l1.mean().rename(columns={"rel_gain_l1":"G_H1"})
    d = d.merge(gh,on="model",validate="one_to_one")
    deep = pd.to_numeric(d.l1l2_er,errors="coerce")
    d["log_er_l12"] = np.log(deep.where(deep>0))
    d["group"] = d.family
    return d

def campaign(root):
    records = {}
    paths = sorted((root/STORE).glob("*/manifest.json"))
    if not paths: raise ValueError("no archival run manifests")
    for path in paths:
        m = json.loads(path.read_text(encoding="utf8"))
        if m.get("status")!="succeeded" or not m.get("mechanism_id"): continue
        name=m["mechanism_id"]
        row=records.setdefault(name,{"model":name})
        metrics=m.get("metrics") or {}
        ev=metrics.get("frozen_evaluation") or {}
        er=ev.get("hard_l1_er",ev.get("l1_er"))
        if er is not None:
            if "er" in row and not np.isclose(row["er"],er,atol=1e-8):
                raise ValueError(f"ER manifest conflict: {name}")
            row["er"]=float(er)
            row["log_er_l1"]=float(np.log(er)) if er>0 else np.nan
            row["seed"]=m.get("seed")
        s3=metrics.get("stage3") or {}
        r=s3.get("test_recall@10"); n=s3.get("test_ndcg@10")
        if r is not None and n is not None:
            if "recall@10" in row and not np.isclose(row["recall@10"],r,atol=1e-10):
                raise ValueError(f"Stage3 manifest conflict: {name}")
            row["recall@10"]=float(r);row["ndcg@10"]=float(n)
            row["n_eval"]=s3.get("n_eval")
    d=pd.DataFrame(list(records.values()))
    if "r8_control" not in set(d.model):
        raise ValueError("missing frozen control; forbidden to retrain it")
    c=d.loc[d.model=="r8_control"].iloc[0]
    if not np.isclose(c.er,16.334505666275888,atol=1e-10) or not np.isclose(c["recall@10"],R_GATE,atol=1e-10):
        raise ValueError("archived control mismatch")
    d["paired"]=d[["er","recall@10","ndcg@10"]].notna().all(axis=1)
    d["passes_ER"]=d.er>ER_GATE
    d["joint_success"]=d.passes_ER & (d["recall@10"]>R_GATE) & (d["ndcg@10"]>N_GATE)
    return d, len(paths)

def fit_predict(train,test,features,target,alpha):
    sc=StandardScaler().fit(train[list(features)])
    model=Ridge(alpha=alpha).fit(sc.transform(train[list(features)]),train[target])
    return model.predict(sc.transform(test[list(features)]))

def alpha_from_training(train,features,target):
    # Inner grouped CV: all preprocessing and alpha selection inside outer train.
    groups=train.group.to_numpy()
    if train.group.nunique()<3: raise ValueError("too few inner training groups")
    scores=[]
    for a in ALPHAS:
        err=[]
        for tr,te in LeaveOneGroupOut().split(train,groups=groups):
            y=fit_predict(train.iloc[tr],train.iloc[te],features,target,a)
            err.extend(abs(y-train.iloc[te][target].to_numpy()))
        scores.append(float(np.mean(err)))
    return ALPHAS[int(np.argmin(scores))]

def lofo(data,features,target):
    d=data.dropna(subset=[*features,target]).copy()
    if len(d)<10 or d.group.nunique()<3: return None
    out=[]
    for tr,te in LeaveOneGroupOut().split(d,groups=d.group):
        train,test=d.iloc[tr],d.iloc[te]
        a=alpha_from_training(train,features,target)
        pred=fit_predict(train,test,features,target,a)
        out.extend({"model":m,"family":g,"target":target,"y":float(y),"pred":float(p),
                    "abs_error":float(abs(y-p)),"alpha":a}
                   for m,g,y,p in zip(test.model,test.group,test[target],pred))
    return pd.DataFrame(out)

def evaluate(root,out):
    history=historical(root)
    new,n_manifests=campaign(root)
    out.mkdir(parents=True,exist_ok=True)
    scores=[]; predictions=[]
    for target in ("recall@10","ndcg@10"):
        for name,features in FEATURES.items():
            p=lofo(history,features,target)
            if p is None: continue
            baseline=lofo(history.loc[history.model.isin(p.model)],FEATURES["ER"],target)
            mae=p.abs_error.mean()
            y=p.y.to_numpy();yp=p.pred.to_numpy()
            scores.append({"target":target,"model":name,"n":len(p),"families":history.loc[history.model.isin(p.model),"family"].nunique(),
                           "mae":float(mae),"delta_mae_vs_ER_same_rows":float(baseline.abs_error.mean()-mae),
                           "oof_r2":float(1-sum((y-yp)**2)/sum((y-y.mean())**2)),
                           "status":"exploratory_not_certified"})
            p["model_set"]=name;predictions.append(p)
    results=pd.DataFrame(scores)
    # Detect accidental regression in the canonical historical ER-only computation.
    chk=results.loc[(results.target=="recall@10")&(results.model=="ER")].iloc[0]
    if not np.isclose(chk.mae,0.003752853292309109,rtol=0,atol=2e-5):
        raise ValueError("historical ER-only LOFO fails #47 reproducibility check")
    pd.concat(predictions).to_csv(out/"historical_oof.csv",index=False)
    results.to_csv(out/"historical_lofo.csv",index=False)
    new.to_csv(out/"campaign_manifest_audit.csv",index=False)
    # Campaign is independent holdout; all predictors fitted on history only.
    pairs=new.loc[new.paired].copy()
    predictions=[]
    for target in ("recall@10","ndcg@10"):
        feat=FEATURES["ER"]
        a=alpha_from_training(history,feat,target)
        yp=fit_predict(history,pairs,feat,target,a)
        for (_,row),pred in zip(pairs.iterrows(),yp):
            predictions.append({"model":row.model,"target":target,"er":row.er,
                "observed":row[target],"predicted":float(pred),"abs_error":float(abs(row[target]-pred)),
                "er_outside_historical_range":bool(row.er<history.er.min() or row.er>history.er.max()),
                "not_certified":True})
    future=pd.DataFrame(predictions)
    future.to_csv(out/"campaign_holdout.csv",index=False)
    # Mechanism diagnosis is descriptive; never treated as an experimentally proven cause.
    path=root/TRANSFER
    diag=[]
    if path.is_file():
        structure=json.loads(path.read_text(encoding="utf8")).get("arms",{})
        for _,row in new.iterrows():
            x=structure.get(row.model,{})
            bal=x.get("token_balance",{}).get("level_1",{})
            seq=x.get("sequential_predictability",{}).get("level_1",{})
            branches=x.get("prefix_structure",{}).get("l1_to_l2",{})
            diag.append({"model":row.model,"er":row.get("er"),"recall@10":row.get("recall@10"),
                "gini":bal.get("gini"),"branches":branches.get("mean_branches"),
                "sequence_excess_in_sample":seq.get("excess_over_the_marginal"),
                "paired":bool(row.paired)})
    diag=pd.DataFrame(diag)
    diag.to_csv(out/"campaign_mechanism_diagnostics.csv",index=False)
    report={"historical_models":len(history),"historical_families":int(history.family.nunique()),
      "campaign_manifests":n_manifests,"campaign_arms":len(new),
      "campaign_paired_including_control":int(new.paired.sum()),
      "campaign_joint_wins":int(new.joint_success.sum()),
      "campaign_ER_range": [float(pairs.er.min()),float(pairs.er.max())],
      "historical_ER_range":[float(history.er.min()),float(history.er.max())],
      "holdout_ood":int(future.loc[future.target=="recall@10","er_outside_historical_range"].sum()),
      "models":scores,
      "limitations":["#48 invalidated; only corrected #49 used",
        "#50 prose/CSV mismatch requires separate audit before using its semantic scores",
        "No historical Stage3 labels used during #107 holdout prediction",
        "Only 5 historical families; retrospective model choice prohibits a confirmatory claim",
        "ER read from frozen post-Stage2 artifacts, not a train-time available feature",
        "Missing Stage3 outcomes are missing, not negative labels",
        "Sequence-excess uses in-sample transition statistics; not a validated predictor",
        "Mechanism indicators are associations, not causal interventions",
        "No predictor is certified for automatic training rejection"]}
    (out/"evidence.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf8")
    md=["# Historical Stage3 evidence audit","","Read-only; no cold run or frozen evaluation was executed.","",
       f"History: {len(history)} models / {history.family.nunique()} families. Campaign: {new.paired.sum()} paired (including control), {new.joint_success.sum()} joint wins.","",
       "| Target | Predictors | n | LOFO MAE | Delta vs ER on same rows |",
       "|---|---|---:|---:|---:|"]
    for x in scores:
        md.append(f"| {x['target']} | {x['model']} | {x['n']} | {x['mae']:.6f} | {x['delta_mae_vs_ER_same_rows']:+.6f} |")
    md+=["","**Not certified for future screening.** Retrospective feature discovery and 5 families mean even promising MAE reductions require a prospective independent mechanism-family test.",
      "","# Files","","- historical_oof.csv: true out-of-family predictions",
      "- historical_lofo.csv: feature ablation on comparable samples",
      "- campaign_manifest_audit.csv: immutable evidence joins, includes missing outcomes",
      "- campaign_holdout.csv: history-only ER predictions on separate campaign",
      "- campaign_mechanism_diagnostics.csv: associations, not causal diagnosis",
      "- evidence.json: provenance counts and restrictions",""]
    (out/"README.md").write_text("\n".join(md),encoding="utf8")
    return report

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--out",type=Path,default=Path("results/harness3_runs/stage3_evidence_audit"))
    args=parser.parse_args()
    print(json.dumps(evaluate(args.root,args.out),indent=2))
