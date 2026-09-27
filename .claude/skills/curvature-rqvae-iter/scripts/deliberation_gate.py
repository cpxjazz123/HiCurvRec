#!/usr/bin/env python3
"""Main-only JSON workflow gate for curvature-RQ-VAE iterations."""
from __future__ import annotations
import json, re
from pathlib import Path

PRE_STAGE2=[
"source_snapshot_iter{n}.json","protocol_manifest_iter{n}.json",
"hypothesis_iter{n}.json","mechanism_manifest_iter{n}.json",
"mechanism_contract_iter{n}.json","one_factor_diff_iter{n}.json",
"preflight_contract_iter{n}.log","mvg_check_iter{n}.log"]
POST_STAGE3=[
"sid_geometry_iter{n}.json","stage3_protocol_gate_iter{n}.log",
"stage3_outcome_iter{n}.json","failure_attribution_iter{n}.json",
"gate_decision_iter{n}.json","git_closure_iter{n}.json"]

def fail(m): raise RuntimeError(f"WORKFLOW_GATE_FAIL: {m}")
def read(p):
    if not p.is_file(): fail(f"missing file: {p}")
    return p.read_text(encoding="utf-8",errors="replace")
def read_json(p):
    try: v=json.loads(read(p))
    except json.JSONDecodeError as e: fail(f"invalid JSON: {p}: {e}")
    if not isinstance(v,dict): fail(f"JSON root must be object: {p}")
    return v
def iter_id(w):
    m=re.fullmatch(r"curvature_RQ-VAE_iter(\d+)",w.name)
    if not m: fail(f"run from curvature_RQ-VAE_iter<N>, got: {w}")
    return m.group(1)
def reject_runtime_markdown(logs):
    if not logs.exists(): return
    bad=sorted(p for p in logs.rglob("*.md") if p.is_file())
    if bad:
        shown=", ".join(str(p.relative_to(logs)) for p in bad[:20])
        fail(f"runtime Markdown is forbidden; migrate/remove: {shown}")
def require_files(logs,n,templates):
    for t in templates:
        p=logs/t.format(n=n)
        read_json(p) if p.suffix==".json" else read(p)
def require_protocol(logs,n):
    d=read_json(logs/f"protocol_manifest_iter{n}.json")
    keys=["protocol_id","stage3_trainer_path","stage3_trainer_sha256","stage3_seed",
          "stage3_epochs","stage3_early_stop","stage3_no_eval","stage3_skip_test",
          "stage3_beam_size","stage3_code_path","stage3_log_path","stage3_save_path"]
    miss=[k for k in keys if k not in d]
    if miss: fail("protocol manifest missing keys: "+", ".join(miss))
    if str(d["stage3_early_stop"]).upper()!="DISABLED":
        fail("stage3_early_stop must be DISABLED")
def require_pre(logs,n):
    if "MECHANISM_CONTRACT_PASS" not in read(logs/f"preflight_contract_iter{n}.log"):
        fail("preflight missing MECHANISM_CONTRACT_PASS")
    if "MVG PASS" not in read(logs/f"mvg_check_iter{n}.log"):
        fail("MVG missing MVG PASS")
def require_result(logs,n):
    f=read_json(logs/f"failure_attribution_iter{n}.json")
    g=read_json(logs/f"gate_decision_iter{n}.json")
    status=str(f.get("mechanism_status","")).upper()
    allowed={"ACTIVE_POSITIVE","ACTIVE_NEUTRAL","ACTIVE_NEGATIVE","IMPLEMENTATION_INVALID",
             "CONTRACT_INVALID","PIPELINE_INVALID","MECHANISM_INACTIVE"}
    if status not in allowed: fail(f"invalid or missing mechanism_status: {status!r}")
    promotion=str(g.get("promotion_status","")).upper()
    if promotion not in {"PROMOTION_PASS","PROMOTION_FAIL"}:
        fail(f"invalid or missing promotion_status: {promotion!r}")
    if not g.get("autonomous_next_action"): fail("gate decision missing autonomous_next_action")
def require_abort(logs,n):
    d=read_json(logs/f"iteration_abort_iter{n}.json")
    if d.get("status")!="ITERATION_ABORTED_INFEASIBLE": fail("abort record has wrong status")
    for k in ("abort_stage","abort_evidence","why_same_iteration_repair_invalid","next_iteration_constraints"):
        if k not in d: fail(f"abort record missing {k}")
def main():
    w=Path.cwd().resolve(); n=iter_id(w); logs=w/"logs"
    reject_runtime_markdown(logs)
    abort=logs/f"iteration_abort_iter{n}.json"
    if abort.is_file():
        require_abort(logs,n)
        print("WORKFLOW_ABORT_CONFIRMED"); print(f"iter={n}"); print("phase=ABORTED"); print("owner=MAIN")
        return
    require_files(logs,n,PRE_STAGE2); require_protocol(logs,n); require_pre(logs,n)
    phase="PRE_STAGE2"
    closure=any((logs/t.format(n=n)).is_file() for t in (
        "stage3_outcome_iter{n}.json","gate_decision_iter{n}.json","git_closure_iter{n}.json"))
    if closure:
        require_files(logs,n,POST_STAGE3)
        if "STAGE3_PROTOCOL_PASS" not in read(logs/f"stage3_protocol_gate_iter{n}.log"):
            fail("Stage3 protocol gate did not pass")
        require_result(logs,n); phase="CLOSURE"
    review=logs/f"global_review_after_iter{n}.json"
    if review.is_file():
        if phase!="CLOSURE": fail("global review exists before closure")
        read_json(review); phase="GLOBAL_REVIEW"
    print("WORKFLOW_GATE_PASS"); print(f"iter={n}"); print(f"phase={phase}"); print("owner=MAIN")
if __name__=="__main__": main()
