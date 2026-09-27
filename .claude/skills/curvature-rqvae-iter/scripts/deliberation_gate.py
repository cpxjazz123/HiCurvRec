#!/usr/bin/env python3
"""Agent-minimal workflow gate for curvature-RQ-VAE iterations.

Only P01_RESEARCH_DESIGN, P05_RESULT_DECISION, and conditional
P07_GLOBAL_REVIEW require Research Agent records. All other phases are
validated from deterministic artifacts/checker outputs.
"""
from __future__ import annotations
import re
from pathlib import Path

DESIGN_ARTIFACTS=[
"source_snapshot_iter{n}.md","protocol_manifest_iter{n}.md","hypothesis_iter{n}.md",
"mechanism_manifest_iter{n}.md","mechanism_contract_iter{n}.json",
"one_factor_diff_iter{n}.md","preflight_contract_iter{n}.log","mvg_check_iter{n}.log"]
CLOSURE_ARTIFACTS=[
"sid_geometry_iter{n}.md","stage3_protocol_gate_iter{n}.log","stage3_outcome_iter{n}.md",
"failure_attribution_iter{n}.md","gate_decision_iter{n}.md","git_closure_iter{n}.md"]
AGENT_PHASES={
"P01_RESEARCH_DESIGN":["hypothesis_iter{n}.md","mechanism_manifest_iter{n}.md","mechanism_contract_iter{n}.json","one_factor_diff_iter{n}.md"],
"P05_RESULT_DECISION":["failure_attribution_iter{n}.md","gate_decision_iter{n}.md"],
"P07_GLOBAL_REVIEW":["global_review_after_iter{n}.md"]}

def fail(m): raise RuntimeError(f"DELIBERATION_GATE_FAIL: {m}")
def read(p):
    if not p.is_file(): fail(f"missing file: {p}")
    return p.read_text(encoding="utf-8",errors="replace")
def iter_id(w):
    m=re.fullmatch(r"curvature_RQ-VAE_iter(\d+)",w.name)
    if not m: fail(f"run from curvature_RQ-VAE_iter<N>, got: {w}")
    return m.group(1)
def highest_round(d):
    rs=[]
    for c in d.iterdir() if d.is_dir() else []:
        m=re.fullmatch(r"round_(\d+)",c.name)
        if m and c.is_dir(): rs.append((int(m.group(1)),c))
    if not rs: fail(f"no stage round found under {d}")
    return sorted(rs)[-1]
def require_files(logs,n,templates):
    names=[]
    for t in templates:
        name=t.format(n=n); read(logs/name); names.append(name)
    return names
def require_agent_phase(logs,root,n,phase):
    r,rd=highest_round(root/phase); packet=rd/"source_packet.md"; agent=rd/"agent.md"; decision=rd/"decision.md"
    read(packet); at=read(agent)
    for x in ("ROLE=RESEARCH_AGENT",f"STAGE_ID={phase}","SOURCE_PACKET="):
        if x not in at: fail(f"{agent} missing {x}")
    dt=read(decision)
    if f"STAGE_ID={phase}" not in dt: fail(f"{decision} wrong/missing STAGE_ID")
    m=re.search(r"^VERDICT=(ACCEPT|REPAIR_AND_RERUN|ABORT_ITERATION)\s*$",dt,re.M)
    if not m: fail(f"{decision} missing valid VERDICT")
    v=m.group(1)
    for x in ("HARD_GATE=","CANONICAL_DECISION=","CANONICAL_ARTIFACT=","CONFIDENCE=",
              "USER_INPUT_REQUIRED=NO","ITERATION_PURPOSE=PERFORMANCE_SEEKING_MECHANISM",
              "SWEEP_OR_REPLICATION_ITERATION=NO","ROOT_CAUSE_ITERATION=NO","AUTONOMOUS_NEXT_ACTION="):
        if x not in dt: fail(f"{decision} missing {x}")
    if v=="REPAIR_AND_RERUN":
        if "SAME_ITERATION_REPAIR=AUTHORIZED" not in dt: fail(f"{decision} repair lacks authorization")
        fail(f"{phase} still requires repair/rerun")
    if v=="ACCEPT" and "HARD_GATE=PASS" not in dt: fail(f"{decision} ACCEPT requires HARD_GATE=PASS")
    if v=="ABORT_ITERATION":
        for x in ("ABORT_REASON=","ABORT_EVIDENCE=","NEXT_ITERATION_CONSTRAINTS="):
            if x not in dt: fail(f"{decision} abort missing {x}")
        return r,v
    names=require_files(logs,n,AGENT_PHASES[phase]); dec=re.search(r"^CANONICAL_ARTIFACT=(.+)\s*$",dt,re.M)
    if dec:
        for name in names:
            if name not in dec.group(1): fail(f"{decision} does not declare {name}")
    return r,v
def protocol_manifest(logs,n):
    t=read(logs/f"protocol_manifest_iter{n}.md")
    for x in ("PROTOCOL_ID=","CANONICAL_BASELINE_ITER=","CANONICAL_BASELINE_TEST_FINAL=",
              "STAGE2_SEED=","STAGE2_MAX_STEPS=","STAGE3_TRAINER_PATH=","STAGE3_TRAINER_SHA256=",
              "STAGE3_SEED=","STAGE3_EPOCHS=","STAGE3_EARLY_STOP=","STAGE3_NO_EVAL=",
              "STAGE3_SKIP_TEST=","STAGE3_BEAM_SIZE=","STAGE3_SCREEN_BASELINE_LOG=",
              "STAGE3_CODE_PATH=","STAGE3_LOG_PATH=","STAGE3_SAVE_PATH="):
        if x not in t: fail(f"protocol manifest missing {x}")
def deterministic_pre_stage2(logs,n):
    if "MECHANISM_CONTRACT_PASS" not in read(logs/f"preflight_contract_iter{n}.log"):
        fail("preflight missing MECHANISM_CONTRACT_PASS")
    if "MVG PASS" not in read(logs/f"mvg_check_iter{n}.log"):
        fail("MVG missing MVG PASS")
def validate_abort(logs,n):
    t=read(logs/f"iteration_abort_iter{n}.md")
    for x in ("STATUS=ITERATION_ABORTED_INFEASIBLE","ABORT_STAGE=","ABORT_EVIDENCE=","NEXT_ITERATION_CONSTRAINTS="):
        if x not in t: fail(f"abort artifact missing {x}")
def legacy_gate(logs,n):
    root=logs/"deliberation"; req=["S00_SOURCE_TRUTH","S01_PROTOCOL_LOCK","S02_HYPOTHESIS","S03_PROVENANCE","S04_CONTRACT","S05_ONE_FACTOR","S06_IMPLEMENTATION","S07_PREFLIGHT","S08_MVG","S09_STAGE2_EXECUTION"]
    closure=(logs/f"gate_decision_iter{n}.md").is_file() or (logs/f"failure_attribution_iter{n}.md").is_file()
    if closure: req+=["S10_STAGE2_ANALYSIS","S11_STAGE3_EVALUATION","S12_RESULT_CLASSIFICATION","S13_GIT_CLOSURE"]
    for stage in req:
        _,rd=highest_round(root/stage); jt=read(rd/"judge.md")
        if not re.search(r"^VERDICT=(ACCEPT_A|ACCEPT_B|MERGE_AB|ABORT_ITERATION)\s*$",jt,re.M):
            fail(f"legacy {stage} has no terminal verdict")
    print("DELIBERATION_GATE_PASS"); print(f"iter={n}"); print(f"phase={'CLOSURE' if closure else 'PRE_STAGE2'}"); print("mode=LEGACY_COMPAT")
def main():
    w=Path.cwd().resolve(); n=iter_id(w); logs=w/"logs"; root=logs/"stage_records"
    if not (root/"P01_RESEARCH_DESIGN").is_dir():
        if (logs/"deliberation").is_dir(): legacy_gate(logs,n); return
        fail("missing P01_RESEARCH_DESIGN stage record")
    if (logs/f"iteration_abort_iter{n}.md").is_file():
        validate_abort(logs,n); print("DELIBERATION_ABORT_CONFIRMED"); print(f"iter={n}"); print("phase=ABORTED"); return
    require_files(logs,n,DESIGN_ARTIFACTS); protocol_manifest(logs,n); deterministic_pre_stage2(logs,n)
    dr,dv=require_agent_phase(logs,root,n,"P01_RESEARCH_DESIGN")
    if dv!="ACCEPT": fail("P01_RESEARCH_DESIGN not accepted")
    closure=(logs/f"stage3_outcome_iter{n}.md").is_file() or (logs/f"gate_decision_iter{n}.md").is_file() or (root/"P05_RESULT_DECISION").is_dir()
    phase="PRE_STAGE2"
    if closure:
        require_files(logs,n,CLOSURE_ARTIFACTS)
        if "STAGE3_PROTOCOL_PASS" not in read(logs/f"stage3_protocol_gate_iter{n}.log"): fail("Stage3 protocol gate did not pass")
        rr,rv=require_agent_phase(logs,root,n,"P05_RESULT_DECISION")
        if rv!="ACCEPT": fail("P05_RESULT_DECISION not accepted")
        phase="CLOSURE"
    if (logs/f"global_review_after_iter{n}.md").is_file():
        if phase!="CLOSURE": fail("global review exists before closure")
        gr,gv=require_agent_phase(logs,root,n,"P07_GLOBAL_REVIEW")
        if gv!="ACCEPT": fail("P07_GLOBAL_REVIEW not accepted")
        phase="GLOBAL_REVIEW"
    print("DELIBERATION_GATE_PASS"); print(f"iter={n}"); print(f"phase={phase}"); print(f"P01_RESEARCH_DESIGN: round_{dr} ACCEPT")
    if phase in {"CLOSURE","GLOBAL_REVIEW"}: print(f"P05_RESULT_DECISION: round_{rr} ACCEPT")
    if phase=="GLOBAL_REVIEW": print(f"P07_GLOBAL_REVIEW: round_{gr} ACCEPT")
if __name__=="__main__": main()
