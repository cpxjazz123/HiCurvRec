#!/usr/bin/env python3
"""Single-agent stage gate for curvature-RQ-VAE iterations.

New/reopened stages use logs/stage_records/<STAGE_ID>/round_<R>/ with
source_packet.md, agent.md, decision.md, and optional repair_record.md.
Historical logs/deliberation A/B/Judge rounds remain readable for compatibility
only and must not be generated for new work. Existing output strings are kept
for caller compatibility.
"""
from __future__ import annotations
import re
from pathlib import Path

PRE_STAGE2=[("S00_SOURCE_TRUTH",["source_snapshot_iter{n}.md"]),("S01_PROTOCOL_LOCK",["protocol_manifest_iter{n}.md"]),("S02_HYPOTHESIS",["hypothesis_iter{n}.md"]),("S03_PROVENANCE",["mechanism_manifest_iter{n}.md"]),("S04_CONTRACT",["mechanism_contract_iter{n}.json"]),("S05_ONE_FACTOR",["one_factor_diff_iter{n}.md"]),("S06_IMPLEMENTATION",["implementation_plan_iter{n}.md"]),("S07_PREFLIGHT",["preflight_contract_iter{n}.log"]),("S08_MVG",["mvg_check_iter{n}.log"]),("S09_STAGE2_EXECUTION",["stage2_execution_plan_iter{n}.md"])]
POST_STAGE2=[("S10_STAGE2_ANALYSIS",["sid_geometry_iter{n}.md"]),("S11_STAGE3_EVALUATION",["stage3_evaluation_plan_iter{n}.md","stage3_outcome_iter{n}.md"]),("S12_RESULT_CLASSIFICATION",["failure_attribution_iter{n}.md","gate_decision_iter{n}.md"]),("S13_GIT_CLOSURE",["git_closure_iter{n}.md"])]
GLOBAL_REVIEW=[("S14_GLOBAL_REVIEW",["global_review_after_iter{n}.md"])]
LEGACY_ALLOWED={"ACCEPT_A","ACCEPT_B","MERGE_AB","ABORT_ITERATION"}

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
def packet_match(path,packet):
    t=read(path); m=re.search(r"^SOURCE_PACKET=(.+)\s*$",t,re.M)
    if not m: fail(f"{path} missing SOURCE_PACKET")
    v=m.group(1).strip(); pn=packet.as_posix()
    if v not in {pn,str(packet)} and not pn.endswith(v): fail(f"{path} SOURCE_PACKET mismatch")
def require_agent(path,stage,packet):
    t=read(path)
    for x in ("ROLE=RESEARCH_AGENT",f"STAGE_ID={stage}","SOURCE_PACKET="):
        if x not in t: fail(f"{path} missing {x}")
    packet_match(path,packet)
def forbidden_positive(action,pattern):
    for m in re.finditer(pattern,action,re.I):
        pre=action[max(0,m.start()-64):m.start()].lower()
        if re.search(r"(?:do\s+not|don['’]?t|must\s+not|should\s+not|never|without|forbid(?:den)?|avoid)\b[^.;:]{0,48}$",pre,re.I): continue
        return True
    return False
def common(path,t):
    for x in ("CANONICAL_DECISION=","CANONICAL_ARTIFACT=","CONFIDENCE=","USER_INPUT_REQUIRED=NO","ITERATION_PURPOSE=PERFORMANCE_SEEKING_MECHANISM","SWEEP_OR_REPLICATION_ITERATION=NO","ROOT_CAUSE_ITERATION=NO","AUTONOMOUS_NEXT_ACTION="):
        if x not in t: fail(f"{path} missing {x}")
    a=re.search(r"^AUTONOMOUS_NEXT_ACTION=(.+)\s*$",t,re.M)
    if not a or not a.group(1).strip(): fail(f"{path} empty AUTONOMOUS_NEXT_ACTION")
    for p in [r"ask\s+the\s+user",r"wait\s+for\s+(the\s+)?user",r"need\s+user\s+decision",r"pause\s+for\s+direction",r"which\s+option\s+do\s+you\s+want",r"should\s+i\s+continue",r"do\s+you\s+want\s+me\s+to",r"what\s+should\s+the\s+next\s+iteration"]:
        if re.search(p,t,re.I): fail(f"{path} contains forbidden user-decision escalation")
    act=a.group(1).strip()
    for p in [r"\bparameter\s+sweep\b",r"\bgrid\s+search\b",r"\brandom\s+search\b",r"\bbayesian\s+(optimization|search)\b",r"\bmulti[- ]seed\b",r"\bmatched[- ]seed\b",r"\bseed\s+replication\b",r"\breplicat(e|ion)\b.*\bseed",r"\bnoise\s+estimation\b",r"\bvariance\s+estimation\b",r"\bsensitivity\s+(study|analysis|test)\b",r"\bablation[- ]only\b",r"\broot[- ]cause\b",r"\bwhy\s+.*(failed|worked|improved|dropped)\b",r"\bmicro[- ]delta\b"]:
        if forbidden_positive(act,p): fail(f"{path} proposes forbidden iteration type")
def require_decision(path,stage,names):
    t=read(path)
    if f"STAGE_ID={stage}" not in t: fail(f"{path} wrong/missing STAGE_ID")
    m=re.search(r"^VERDICT=(ACCEPT|REPAIR_AND_RERUN|ABORT_ITERATION)\s*$",t,re.M)
    if not m: fail(f"{path} missing valid VERDICT")
    v=m.group(1)
    if "HARD_GATE=" not in t: fail(f"{path} missing HARD_GATE")
    common(path,t)
    if v=="REPAIR_AND_RERUN":
        if "SAME_ITERATION_REPAIR=AUTHORIZED" not in t: fail(f"{path} repair lacks authorization")
        fail(f"{stage} latest round still requires repair/rerun")
    if v=="ACCEPT" and "HARD_GATE=PASS" not in t: fail(f"{path} ACCEPT requires HARD_GATE=PASS")
    if v=="ABORT_ITERATION":
        for x in ("ABORT_REASON=","ABORT_EVIDENCE=","NEXT_ITERATION_CONSTRAINTS="):
            if x not in t: fail(f"{path} abort missing {x}")
    d=re.search(r"^CANONICAL_ARTIFACT=(.+)\s*$",t,re.M)
    if d:
        for name in names:
            if name not in d.group(1): fail(f"{path} missing canonical artifact declaration {name}")
    return v
def require_repair(path):
    t=read(path)
    for x in ("ROUND_TYPE=OPERATIONAL_REPAIR","LOCKED_SCIENCE_CHANGED=NO"):
        if x not in t: fail(f"{path} missing {x}")
    if "PARALLEL_EXECUTION=YES" not in t:
        m=re.search(r"^SERIALIZATION_REASON=(.+)\s*$",t,re.M)
        if not m or not m.group(1).strip(): fail(f"{path} needs PARALLEL_EXECUTION or SERIALIZATION_REASON")
    if "CHECKS_PASS=YES" not in t: fail(f"{path} CHECKS_PASS!=YES")
def abort_artifact(logs,n):
    p=logs/f"iteration_abort_iter{n}.md"; t=read(p)
    for x in ("STATUS=ITERATION_ABORTED_INFEASIBLE","ABORT_STAGE=","ABORT_EVIDENCE=","WHY_SAME_ITERATION_REPAIR_INVALID=","NEXT_ITERATION_CONSTRAINTS="):
        if x not in t: fail(f"{p} missing {x}")
    return p
def check_new(logs,root,n,stage,templates):
    r,rd=highest_round(root/stage); packet=rd/"source_packet.md"; read(packet); require_agent(rd/"agent.md",stage,packet)
    if (rd/"repair_record.md").is_file(): require_repair(rd/"repair_record.md")
    paths=[logs/t.format(n=n) for t in templates]; names=[p.name for p in paths]; dec=rd/"decision.md"; raw=read(dec)
    if "VERDICT=ABORT_ITERATION" in raw:
        ap=abort_artifact(logs,n); v=require_decision(dec,stage,[ap.name]); return stage,r,v,[ap.name]
    v=require_decision(dec,stage,names)
    for p in paths:
        if not p.is_file(): fail(f"{stage} canonical artifact missing: {p}")
    return stage,r,v,names
def legacy_worker(path,role,stage):
    t=read(path)
    for x in (f"ROLE={role}",f"STAGE_ID={stage}","SOURCE_PACKET="):
        if x not in t: fail(f"{path} missing legacy {x}")
def legacy_judge(path,stage,names):
    t=read(path); m=re.search(r"^VERDICT=(ACCEPT_A|ACCEPT_B|MERGE_AB|REJECT_BOTH|REPAIR_AND_RERUN|ABORT_ITERATION)\s*$",t,re.M)
    if not m: fail(f"{path} missing legacy VERDICT")
    v=m.group(1)
    if v in {"REJECT_BOTH","REPAIR_AND_RERUN"} or v not in LEGACY_ALLOWED: fail(f"{stage} unresolved legacy verdict {v}")
    for x in ("CANONICAL_ARTIFACT=","USER_INPUT_REQUIRED=NO","AUTONOMOUS_NEXT_ACTION="):
        if x not in t: fail(f"{path} missing legacy {x}")
    return v
def check_legacy(logs,root,n,stage,templates):
    r,rd=highest_round(root/stage); legacy_worker(rd/"agent_a.md","AGENT_A",stage); legacy_worker(rd/"agent_b.md","AGENT_B",stage)
    paths=[logs/t.format(n=n) for t in templates]; names=[p.name for p in paths]; j=rd/"judge.md"; raw=read(j)
    if "VERDICT=ABORT_ITERATION" in raw:
        ap=abort_artifact(logs,n); v=legacy_judge(j,stage,[ap.name]); return stage,r,v,[ap.name]
    v=legacy_judge(j,stage,names)
    for p in paths:
        if not p.is_file(): fail(f"{stage} legacy canonical artifact missing: {p}")
    return stage,r,v,names
def check_stage(logs,new,old,n,stage,templates):
    if (new/stage).is_dir(): return check_new(logs,new,n,stage,templates)
    if (old/stage).is_dir(): return check_legacy(logs,old,n,stage,templates)
    fail(f"missing stage record for {stage}")
def main():
    w=Path.cwd().resolve(); n=iter_id(w); logs=w/"logs"; new=logs/"stage_records"; old=logs/"deliberation"; stages=list(PRE_STAGE2); phase="PRE_STAGE2"
    if (logs/f"gate_decision_iter{n}.md").is_file() or (logs/f"failure_attribution_iter{n}.md").is_file(): stages+=POST_STAGE2; phase="CLOSURE"
    if (logs/f"global_review_after_iter{n}.md").is_file():
        if phase!="CLOSURE": fail("global review exists before closure artifacts")
        stages+=GLOBAL_REVIEW; phase="GLOBAL_REVIEW"
    summary=[]
    for stage,templates in stages:
        x=check_stage(logs,new,old,n,stage,templates); summary.append(x)
        if x[2]=="ABORT_ITERATION":
            print("DELIBERATION_ABORT_CONFIRMED"); print(f"iter={n}"); print("phase=ABORTED"); print(f"abort_stage={stage}")
            for sid,r,v,names in summary: print(f"{sid}: round_{r} {v} -> {','.join(names)}")
            return
    print("DELIBERATION_GATE_PASS"); print(f"iter={n}"); print(f"phase={phase}")
    for sid,r,v,names in summary: print(f"{sid}: round_{r} {v} -> {','.join(names)}")
if __name__=="__main__": main()
