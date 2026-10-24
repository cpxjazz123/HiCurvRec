#!/usr/bin/env python3
"""Read-only Stage2 SID indicator audit for every recoverable Stage3 training run.

Each output row is one Stage3 test_final.json. Frozen held-out behaviour pairs
are read once; no Stage2/Stage3 training, checkpoint mutation, or test re-evaluation.
Historic blobs are matched to the existing 38-model provenance inventory by SHA.
"""
from __future__ import annotations
import collections
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import historical_er_backfill as h

ROOT = h.ROOT
OUT = ROOT / "results/research/historical_sid_indicator_screen"
SID_BASE = "results/stage2_RQ-VAE/"
FROZEN = ROOT / "results/stage3_T5Train/monotonicity"
PAIR_FILE = FROZEN / "behavior_pairs_valid_s42.npz"
META_FILE = FROZEN / "behavior_pairs_valid_s42.meta.json"
INV_FILE = FROZEN / "history_hierarchy_predictors_fix48/per_model_features.csv"
_RUN_REF_CACHE = {}
_SID_CACHE = {}

def sha(blob):
    return hashlib.sha256(blob).hexdigest()

def history_version(path, expected_sha):
    key = (path, expected_sha)
    if key in _RUN_REF_CACHE:
        return _RUN_REF_CACHE[key]
    cmd = ["git", "log", "--all", "--format=%H", "--", path]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=70)
    if proc.returncode:
        raise RuntimeError("git log failed for " + path + ":" + proc.stderr[:160])
    attempts = 0
    for commit in proc.stdout.splitlines():
        data = h.git_show(commit, path)
        attempts += 1
        if data is not None and sha(data).startswith(expected_sha):
            result = (data, commit + ":" + path)
            _RUN_REF_CACHE[key] = result
            return result
        if attempts >= 90:
            break
    raise FileNotFoundError("No SHA-matching archived SID found: " + path + " sha " + expected_sha[:16])

def to_inventory():
    with INV_FILE.open(encoding="utf-8", newline="") as f:
        return {r["sid_path"]:r for r in csv.DictReader(f)}

def sid_at(runid, path, inventory):
    if runid in h.HISTORIC_REFS:
        b = h.git_show(h.HISTORIC_REFS[runid], path)
        if b is None:
            raise FileNotFoundError("Pinned historical SID absent: " + path)
        return b, "pinned_git:" + h.HISTORIC_REFS[runid] + ":" + path
    if path in inventory:
        expected = inventory[path]["sid_sha256"]
        f = ROOT / path
        if f.is_file() and sha(f.read_bytes()).startswith(expected):
            return f.read_bytes(), "sha_matched_worktree:" + path
        return history_version(path, expected)
    f = ROOT / path
    if f.is_file():
        return f.read_bytes(), "working_tree:" + path
    return history_version(path, "")  # Unknown old path; follow independent validation below

def sid_metrics(raw, anchors, positives, negatives):
    obj = json.loads(raw)
    sid = {int(k):tuple(v) for k,v in obj.items()}
    if len(sid) != 24587 or set(sid) != set(range(24587)):
        raise ValueError("SID item IDs do not match frozen catalogue")
    if any(len(v)<3 for v in sid.values()):
        raise ValueError("insufficient SID levels")
    get = sid.__getitem__
    l1 = {i:t[0] for i,t in sid.items()}
    l12 = {i:t[:2] for i,t in sid.items()}
    l123 = {i:t[:3] for i,t in sid.items()}
    n = len(anchors)
    # Counts are exactly paired under one fixed train-excluded valid set.
    ph1=nh1=ph2=nh2=ph3=nh3=0
    for idx, (a,b) in enumerate(zip(anchors, positives)):
        a=int(a); b=int(b)
        if l1[a]==l1[b]: ph1+=1
        if l12[a]==l12[b]: ph2+=1
        if l123[a]==l123[b]: ph3+=1
        for k in range(h.gate.EXPECTED_NEGATIVES):
            c=int(negatives[idx*10+k])
            if l1[a]==l1[c]: nh1+=1
            if l12[a]==l12[c]: nh2+=1
            if l123[a]==l123[c]: nh3+=1
    counter1=collections.Counter(l1.values())
    counter12=collections.Counter(l12.values())
    counter123=collections.Counter(l123.values())
    def H(counts):
        denom=sum(counts.values())
        return -sum((n/denom)*math.log(n/denom) for n in counts.values())
    def gini(counts):
        x=sorted(counts.values())
        t=sum(x); n=len(x)
        return sum((2*(i+1)-n-1)*v for i,v in enumerate(x))/(n*t)
    def safe(a,b):
        return a/b if b else None
    pos1=ph1/n; neg1=nh1/(n*10)
    pos2=ph2/n; neg2=nh2/(n*10)
    pos3=ph3/n; neg3=nh3/(n*10)
    H1,H2,H3=H(counter1),H(counter12),H(counter123)
    return {
      "l1_pos_rate":pos1, "l1_neg_rate":neg1,
      "l1_gap":pos1-neg1, "l1_er":safe(pos1,neg1),
      "l12_pos_rate":pos2,"l12_neg_rate":neg2,
      "l12_gap":pos2-neg2,"l12_er":safe(pos2,neg2),
      "l123_pos_rate":pos3,"l123_neg_rate":neg3,
      "l123_gap":pos3-neg3,"l123_er":safe(pos3,neg3),
      "d2_conditional":(ph2/ph1 - nh2/nh1) if ph1 and nh1 else None,
      "d3_conditional":(ph3/ph2 - nh3/nh2) if ph2 and nh2 else None,
      "l1_pos_hits":ph1,"l1_neg_hits":nh1,
      "l12_pos_hits":ph2,"l12_neg_hits":nh2,
      "l123_pos_hits":ph3,"l123_neg_hits":nh3,
      "l1_distinct":len(counter1),"l2_distinct":len(counter12),
      "l3_distinct":len(counter123),
      "l3_collision":1-len(counter123)/len(sid),
      "l1_entropy":H1, "l2_entropy":H2, "l3_entropy":H3,
      "l1_effective_usage":math.exp(H1),"l2_effective_usage":math.exp(H2),
      "l1_effective_ratio":math.exp(H1)/256,
      "l1_largest_cluster_share":max(counter1.values())/len(sid),
      "l2_largest_cluster_share":max(counter12.values())/len(sid),
      "l1_gini":gini(counter1),"l2_gini":gini(counter12),
      "l12_branch_per_l1":len(counter12)/len(counter1),
      "l123_branch_per_l12":len(counter123)/len(counter12),
    }

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for label,p,reference in [
      ("valid pair NPZ",PAIR_FILE,h.gate.FROZEN_VALID_PAIRS_BLOB_SHA1),
      ("valid pair meta",META_FILE,h.gate.FROZEN_VALID_META_BLOB_SHA1),
    ]:
        if h.gate._git_blob_sha1(p)!=reference:
            raise RuntimeError("Frozen source identity mismatch: " + label)
    A,P,N = h.gate._pairs(PAIR_FILE,META_FILE)
    inventory=to_inventory()
    test_files=sorted((ROOT/"results/stage3_T5Train").rglob("test_final.json"))
    allpaths=[p.relative_to(ROOT).as_posix() for p in test_files]
    if len(allpaths)!=77 or allpaths[0]!=h.EXPECTED_FIRST or allpaths[-1]!=h.EXPECTED_LAST:
        raise RuntimeError("Run order changed; review historic source mappings")
    baseline_path=SID_BASE+"TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample/item_sids.json"
    base_metrics=sid_metrics((ROOT/baseline_path).read_bytes(),A,P,N)
    if abs(base_metrics["l1_er"]-16.38953942232631)>1e-9 or base_metrics["l2_distinct"]!=16145:
        raise RuntimeError("frozen resample baseline mismatched")
    rows=[]
    for index,testfile in enumerate(test_files):
        rec={"run_id":"main_%03d"%index,
             "test_final_path":allpaths[index]}
        try:
            test=json.loads(testfile.read_text(encoding="utf-8"))
            if test.get("n_eval")!=57439:
                raise ValueError("Unexpected n_eval")
            rec["recall10"]=test["test_recall@10"]
            rec["ndcg10"]=test["test_ndcg@10"]
            lp=testfile.with_name("HG_Rec.log")
            with lp.open(encoding="utf-8",errors="replace") as f:
                config="".join(f.readline() for _ in range(5))
            sidpath=h.canonical_sid_path(index,config)
            rec["sid_path"]=sidpath
            raw,provenance=sid_at(index,sidpath,inventory)
            rec["sid_provenance"]=provenance
            rec["sid_sha256"]=sha(raw)
            rec.update(sid_metrics(raw,A,P,N))
            # For archived original 38, refuse a mistaken version or invalid history.
            if index not in h.HISTORIC_REFS and sidpath in inventory and not rec["sid_sha256"].startswith(inventory[sidpath]["sid_sha256"]):
                raise ValueError("SID SHA differs from previously audited 38-model inventory")
            if index in h.EXPECTED_MISSING_L2 and rec["l2_distinct"]!=h.EXPECTED_MISSING_L2[index]:
                raise ValueError("historical prefix mismatch")
            rec["status"]="VALID"
        except Exception as exc:
            rec["status"]="UNRESOLVED"
            rec["error"]=str(exc)
        rows.append(rec)
        print("SID_RUN", json.dumps({"id":rec["run_id"],"status":rec["status"],"sid":rec.get("sid_path"),"er":rec.get("l1_er"),"L2":rec.get("l2_distinct"),"error":rec.get("error")}),flush=True)
    # Issue #117 separately archived finished Stage3 results, without main test_final.
    for model,recall,ndcg in [
      ("crowd_posattract_w3",0.06426,None),
      ("crowd_couser_w3",0.06353,None),
    ]:
        row={"run_id":"issue117_"+model,"recall10":recall,"ndcg10":ndcg,
             "sid_path":SID_BASE+"TIGER_BEHAVIOR_SAMPLE_RQ-VAE/"+model+"/item_sids.json"}
        try:
            raw=(ROOT/row["sid_path"]).read_bytes()
            row["sid_sha256"]=sha(raw)
            row["sid_provenance"]="working_tree:"+row["sid_path"]
            row.update(sid_metrics(raw,A,P,N))
            row["status"]="VALID"
        except Exception as exc:
            row["status"]="UNRESOLVED";row["error"]=str(exc)
        rows.append(row)
        print("ISSUE_RUN",json.dumps({"id":row["run_id"],"status":row["status"],"error":row.get("error")}),flush=True)
    cols=["run_id","recall10","ndcg10","test_final_path","sid_path","sid_provenance","sid_sha256","status","error"]+list(sid_metrics((ROOT/baseline_path).read_bytes(),A,P,N))
    with (OUT/"metrics_by_run.csv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=cols,extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    result={"total":len(rows),"valid":sum(r["status"]=="VALID" for r in rows),
            "unresolved":[{"id":r["run_id"],"error":r.get("error")} for r in rows if r["status"]!="VALID"],
            "frozen_baseline_er":base_metrics["l1_er"],"frozen_baseline_l2":base_metrics["l2_distinct"],
            "expected_metrics":len(cols)-9}
    (OUT/"summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print("METRIC_SUMMARY",json.dumps(result),flush=True)

if __name__=="__main__":
    main()
