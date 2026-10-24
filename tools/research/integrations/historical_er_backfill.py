#!/usr/bin/env python3
"""Read-only frozen validation Hard-L1 ER audit of Stage3 runs.
One row per actual Stage3 test_final.json; NEVER deduplicate Stage3 executions.
Historic overwritten SID JSONs are read at pinned git commits (without checkout).
"""
from __future__ import annotations
import csv
import hashlib
import importlib.util
import json
import math
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
GATE_FILE = ROOT / "tools/genrec-mcp/stage2_quality_gate.py"
spec = importlib.util.spec_from_file_location("frozen_quality_gate", str(GATE_FILE))
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)

# Indexed by the sorted canonical paths of all Stage3 test_final.jsons on main.
# Index -> immutable old commit that actually supplied this run's SID.
# These refs are *data*, not model code; "git show" is read-only.
HISTORIC_REFS = {
  0:"fb7b3d2cf0b5",1:"bc3811514742",2:"ba230c6e5f91",
  3:"127983facb17",4:"f02273b6b051",5:"c1b4692fdeae",
  6:"fb7b3d2cf0b5",7:"fb7b3d2cf0b5",8:"fb7b3d2cf0b5",
  9:"fb7b3d2cf0b5",10:"fb7b3d2cf0b5",11:"fb7b3d2cf0b5",
  12:"fb7b3d2cf0b5",13:"fb7b3d2cf0b5",
  15:"bc96625d591a",33:"bbd89044c3a9",34:"a92ff89d1208",
  36:"c3d92a5f88a6",42:"748bafc511f8",43:"9f284cdc0f7f",
  44:"ff43ae2cbd63",45:"ff43ae2cbd63",46:"a6d8630428ba",
  58:"8a2ffec40d93",59:"05a4b360e644",60:"05a4b360e644",
  61:"ee783f696f87",63:"bf8140521314",64:"bf8140521314",
  65:"bf8140521314",66:"bf8140521314",
}
# Previous verified SID-prefix counts are required for each missing ER.
# Any mismatch stops that observation being counted: protect against stale files.
EXPECTED_MISSING_L2 = {
  0:9369,1:11398,2:24530,3:24531,4:24548,5:24540,
  6:9369,7:5198,8:14402,9:14473,10:12130,11:5945,
  12:14353,13:9440,33:7789,36:15748,42:27,43:9902,
  44:85,45:1,46:97,58:11398,59:12596,60:9301,61:783,
  62:4831,63:17112,64:16653,65:13883,66:9461,67:369,
  69:16145,72:5919,73:6376,
}
# For early repeated-overwrite runs, the Stage3 config path alone is not
# enough to identify the old SID version; these paths are pinned by ref above.
SPECIAL = {
  0:"curvature_RQ-VAE/item_sids.json",
  1:"curvature_RQ-VAE/item_sids.json",
  2:"curvature_RQ-VAE/item_sids.json",
  3:"curvature_RQ-VAE/item_sids.json",
  4:"curvature_RQ-VAE/item_sids.json",
  5:"curvature_RQ-VAE/item_sids.json",
  6:"curvature_RQ-VAE/item_sids.json",
  7:"curvature_RQ-VAE/versions/B/item_sids.json",
  8:"curvature_RQ-VAE/versions/C/item_sids.json",
  9:"curvature_RQ-VAE/versions/D/item_sids.json",
  10:"curvature_RQ-VAE/versions/E/item_sids.json",
  11:"curvature_RQ-VAE/versions/Baseline/item_sids.json",
  12:"curvature_RQ-VAE/versions/BehaviorOnly/item_sids.json",
  13:"curvature_RQ-VAE/versions/CurvatureOnly/item_sids.json",
  15:"curvature_RQ-VAE/item_sids.json",
}
# All 77 results are sorted against this prefix. This hash prevents accidental
# re-indexing if someone adds/removes a Stage3 result on the analysis branch.
EXPECTED_FIRST = "results/stage3_T5Train/curvature_RQ-VAE/logs/Amazon_2023_Instruments/Oct-01-2026_15-23-32/test_final.json"
EXPECTED_LAST = "results/stage3_T5Train/unknown_variant/logs/Amazon_2023_Instruments/Sep-24-2026_16-17-58/test_final.json"
CODE_PATH_RE = re.compile(r"['\"]code_path['\"]:\s*['\"]([^'\"]+)['\"]")
SID_ROOT = "results/stage2_RQ-VAE/"
OUT = ROOT / "results/research/historical_er_backfill"

def git_show(commit, path):
    try:
        return subprocess.run(
            ["git","show",commit + ":" + path],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=True, timeout=35).stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None

def canonical_sid_path(run_number, log):
    if run_number in SPECIAL:
        return SID_ROOT + SPECIAL[run_number]
    match = CODE_PATH_RE.search(log)
    if match is None:
        raise ValueError("Stage3 configuration has no code_path")
    value = match.group(1)
    if "/results/stage2_RQ-VAE/" in value:
        return SID_ROOT + value.split("/results/stage2_RQ-VAE/", 1)[1]
    if value.startswith(SID_ROOT):
        return value
    return value.lstrip("/")

def load_sid_source(number, path):
    ref = HISTORIC_REFS.get(number)
    if ref:
        blob = git_show(ref, path)
        if blob is None:
            raise FileNotFoundError("historical Git blob absent: " + ref + ":" + path)
        return blob, ref + ":" + path
    file = ROOT / path
    if not file.is_file():
        raise FileNotFoundError("SID file absent: " + path)
    return file.read_bytes(), "working-tree:" + path

def calc_sid(raw, anchors, positives, negatives):
    obj = json.loads(raw)
    if not isinstance(obj, dict) or not obj:
        raise ValueError("SID is not a nonempty mapping")
    sid = {int(k): tuple(v) for k, v in obj.items()}
    if len(sid) != 24587 or set(sid) != set(range(24587)):
        raise ValueError("SID item universe mismatch (want all 24587 items)")
    if any(len(v) < 3 or any(type(x) is not int or x < 0 for x in v) for v in sid.values()):
        raise ValueError("SID malformed")
    l2 = len({v[:2] for v in sid.values()})
    m = gate._measures(sid, anchors, positives, negatives)
    return l2, m

def audit():
    OUT.mkdir(parents=True, exist_ok=True)
    pairs_file = ROOT / "results/stage3_T5Train/monotonicity/behavior_pairs_valid_s42.npz"
    meta_file = ROOT / "results/stage3_T5Train/monotonicity/behavior_pairs_valid_s42.meta.json"
    baseline_file = ROOT / SID_ROOT / "TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample/item_sids.json"
    for title, path, expected in (
      ("validation npz",pairs_file,gate.FROZEN_VALID_PAIRS_BLOB_SHA1),
      ("validation meta",meta_file,gate.FROZEN_VALID_META_BLOB_SHA1),
      ("resample baseline",baseline_file,gate.FROZEN_BASELINE_BLOB_SHA1)):
        got = gate._git_blob_sha1(path)
        if got != expected:
            raise RuntimeError(title + " frozen Git blob mismatch " + got + " != " + expected)
    anchors, positives, negatives = gate._pairs(pairs_file, meta_file)
    assert len(anchors) == 52855 and len(negatives) == 528550
    ref_l2, ref = calc_sid(baseline_file.read_bytes(), anchors, positives, negatives)
    if ref_l2 != 16145 or not math.isclose(ref["valid_l1_hard_er"],16.38953942232631,rel_tol=0,abs_tol=1e-10):
        raise RuntimeError("frozen ER baseline check failed")
    paths = sorted((ROOT/"results/stage3_T5Train").rglob("test_final.json"))
    relative = [p.relative_to(ROOT).as_posix() for p in paths]
    if len(paths) != 77 or relative[0]!=EXPECTED_FIRST or relative[-1]!=EXPECTED_LAST:
        raise RuntimeError("77 run indexing drift: re-audit historical refs before use; detected " + str(len(paths)))
    rows=[]
    for i,p in enumerate(paths):
        row={"run_id":"main_%03d"%i, "stage3_test_final_path":relative[i]}
        try:
            test=json.loads(p.read_text(encoding="utf-8"))
            row["recall10"]=test["test_recall@10"]
            row["ndcg10"]=test.get("test_ndcg@10")
            row["n_eval"]=test.get("n_eval")
            if row["n_eval"]!=57439:
                raise ValueError("unexpected Stage3 evaluation sample size")
            logpath=p.with_name("HG_Rec.log")
            if not logpath.is_file():
                raise FileNotFoundError("missing Stage3 run configuration log")
            with logpath.open(encoding="utf-8",errors="replace") as f:
                config="".join(f.readline() for _ in range(5))
            sidpath=canonical_sid_path(i,config)
            row["sid_path"]=sidpath
            raw,provenance=load_sid_source(i,sidpath)
            row["sid_source"]=provenance
            row["sid_sha256"]=hashlib.sha256(raw).hexdigest()
            l2,measures=calc_sid(raw,anchors,positives,negatives)
            row["l2_prefixes"]=l2
            if i in EXPECTED_MISSING_L2 and l2 != EXPECTED_MISSING_L2[i]:
                raise ValueError("historic SID identity mismatch: expected L2=%d got=%d"%(EXPECTED_MISSING_L2[i],l2))
            row["er"]=measures["valid_l1_hard_er"]
            row["pos_hits"]=measures["positive_hits"]
            row["neg_hits"]=measures["negative_hits"]
            row["status"]="VALID"
        except Exception as exc:
            row["status"]="UNRESOLVED"
            row["error"]=str(exc)
        rows.append(row)
        if i in EXPECTED_MISSING_L2:
            print("BACKFILL",json.dumps(row,ensure_ascii=False),flush=True)
    columns=["run_id","stage3_test_final_path","recall10","ndcg10","n_eval","sid_path","sid_source","sid_sha256","l2_prefixes","er","pos_hits","neg_hits","status","error"]
    with (OUT/"all_stage3_runs.csv").open("w",encoding="utf-8",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=columns,extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)
    missing=[r for r in rows if int(r["run_id"].split("_")[1]) in EXPECTED_MISSING_L2]
    complete=[r for r in missing if r["status"]=="VALID" and r.get("er") is not None]
    summary={
      "branch_git_sha":subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,capture_output=True,text=True).stdout.strip(),
      "frozen_reference_l1_er":ref["valid_l1_hard_er"],
      "frozen_reference_l2":ref_l2,
      "stage3_files":len(rows),
      "total_er_computed":sum(r["status"]=="VALID" for r in rows),
      "historic_er_missing_input":len(missing),
      "historic_er_newly_resolved":len(complete),
      "historic_er_still_unresolved":[{"id":r["run_id"],"error":r.get("error"),"sid":r.get("sid_path")} for r in missing if r["status"]!="VALID"],
      "other_unresolved":[{"id":r["run_id"],"error":r.get("error")} for r in rows if r["status"]!="VALID" and r not in missing],
      "method":"Same frozen independent valid-s42 npz, positive/negative L1 SID equality; historical Git blob pinned per Stage3 execution; no training/evaluation rerun.",
    }
    (OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print("AUDIT_SUMMARY",json.dumps(summary,ensure_ascii=False),flush=True)
    return 0 if len(complete)==len(missing) else 2

if __name__=="__main__":
    raise SystemExit(audit())
