#!/usr/bin/env python3
"""Deterministic frozen-Stage3 protocol gate.

Run with NO CLI arguments from stage2_RQ-VAE/curvature_RQ-VAE_iter<N>/.
No scientific judgment is performed: exact mismatches block Stage3.
"""
from __future__ import annotations
import ast, hashlib, re
from pathlib import Path

def fail(m): raise RuntimeError(f"PROTOCOL_GATE_FAIL: {m}")
def read(p):
    if not p.is_file(): fail(f"missing file: {p}")
    return p.read_text(encoding="utf-8",errors="replace")
def sha256(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()
def iter_id(w):
    m=re.fullmatch(r"curvature_RQ-VAE_iter(\d+)",w.name)
    if not m: fail(f"run from curvature_RQ-VAE_iter<N>, got {w}")
    return m.group(1)
def kv(text,key):
    m=re.search(rf"^{re.escape(key)}=(.*)$",text,re.M)
    if not m: fail(f"missing {key}")
    return m.group(1).strip()
def bval(v):
    x=v.lower()
    if x in {"true","1","yes"}: return True
    if x in {"false","0","no"}: return False
    fail(f"invalid boolean {v}")
def literals(path):
    tree=ast.parse(read(path),filename=str(path)); out={}
    for node in tree.body:
        if isinstance(node,ast.Assign):
            try: value=ast.literal_eval(node.value)
            except Exception: continue
            for t in node.targets:
                if isinstance(t,ast.Name): out[t.id]=value
        elif isinstance(node,ast.AnnAssign) and isinstance(node.target,ast.Name):
            try: out[node.target.id]=ast.literal_eval(node.value)
            except Exception: pass
    return out
def overrides(path):
    tree=ast.parse(read(path),filename=str(path)); out={}
    for node in ast.walk(tree):
        if not isinstance(node,(ast.Assign,ast.AnnAssign)): continue
        targets=node.targets if isinstance(node,ast.Assign) else [node.target]
        for t in targets:
            if isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name) and t.value.id=="trainer":
                try: out[t.attr]=ast.literal_eval(node.value)
                except Exception: out[t.attr]="<dynamic>"
    return out
def main():
    work=Path.cwd().resolve(); n=iter_id(work); repo=work.parents[1]
    claude=read(repo/"CLAUDE.md"); manifest=read(work/"logs"/f"protocol_manifest_iter{n}.md")
    expected={
        "NUM_EPOCHS":int(kv(claude,"STAGE3_NUM_EPOCHS")),
        "EARLY_STOP":kv(claude,"STAGE3_EARLY_STOP"),
        "NO_EVAL":bval(kv(claude,"STAGE3_NO_EVAL")),
        "SKIP_TEST":bval(kv(claude,"STAGE3_SKIP_TEST")),
        "SEED":int(kv(claude,"STAGE3_SEED")),
        "BEAM_SIZE":int(kv(claude,"STAGE3_BEAM_SIZE")),
        "SCREEN_BASELINE_LOG":kv(claude,"STAGE3_SCREEN_BASELINE_LOG"),
    }
    manifest_pairs={
        "NUM_EPOCHS":kv(manifest,"STAGE3_EPOCHS"),
        "EARLY_STOP":kv(manifest,"STAGE3_EARLY_STOP"),
        "NO_EVAL":kv(manifest,"STAGE3_NO_EVAL"),
        "SKIP_TEST":kv(manifest,"STAGE3_SKIP_TEST"),
        "SEED":kv(manifest,"STAGE3_SEED"),
        "BEAM_SIZE":kv(manifest,"STAGE3_BEAM_SIZE"),
        "SCREEN_BASELINE_LOG":kv(manifest,"STAGE3_SCREEN_BASELINE_LOG"),
    }
    expected_text={
        "NUM_EPOCHS":str(expected["NUM_EPOCHS"]),"EARLY_STOP":str(expected["EARLY_STOP"]),
        "NO_EVAL":str(expected["NO_EVAL"]).lower(),"SKIP_TEST":str(expected["SKIP_TEST"]).lower(),
        "SEED":str(expected["SEED"]),"BEAM_SIZE":str(expected["BEAM_SIZE"]),
        "SCREEN_BASELINE_LOG":str(expected["SCREEN_BASELINE_LOG"]),
    }
    for k,v in expected_text.items():
        if manifest_pairs[k].lower()!=v.lower(): fail(f"manifest {k}={manifest_pairs[k]!r} != repository lock {v!r}")

    trainer=repo/kv(manifest,"STAGE3_TRAINER_PATH"); vals=literals(trainer)
    for k in ("NUM_EPOCHS","NO_EVAL","SKIP_TEST","SEED","BEAM_SIZE","SCREEN_BASELINE_LOG"):
        if k not in vals: fail(f"trainer missing literal {k}")
        if vals[k]!=expected[k]: fail(f"trainer {k}={vals[k]!r} != locked {expected[k]!r}")
    if str(expected["EARLY_STOP"]).upper()=="DISABLED":
        if vals.get("EARLY_STOP") not in (None,False,0): fail(f"trainer EARLY_STOP={vals.get('EARLY_STOP')!r}; locked protocol requires DISABLED")
    elif str(vals.get("EARLY_STOP"))!=str(expected["EARLY_STOP"]):
        fail(f"trainer EARLY_STOP={vals.get('EARLY_STOP')!r} != locked {expected['EARLY_STOP']!r}")

    actual_sha=sha256(trainer); locked_sha=kv(manifest,"STAGE3_TRAINER_SHA256")
    if actual_sha!=locked_sha: fail(f"trainer SHA drift: locked={locked_sha} actual={actual_sha}")

    launcher=work/"scripts"/f"run_stage3_iter{n}.py"; ov=overrides(launcher)
    allowed={"CODE_PATH","RQVAE_VARIANT","LOG_PATH","SAVE_PATH"}
    forbidden=sorted(set(ov)-allowed)
    if forbidden: fail(f"launcher overrides frozen Stage3 fields: {forbidden}")
    missing=sorted(allowed-set(ov))
    if missing: fail(f"launcher missing required wiring fields: {missing}")

    expected_paths={
        "CODE_PATH":str(repo/"results"/"stage2_RQ-VAE"/f"curvature_RQ-VAE_iter{n}"/"item_sids.json"),
        "LOG_PATH":str(repo/"results"/"stage3_T5Train"/f"curvature_RQ-VAE_iter{n}"/"logs")+"/",
        "SAVE_PATH":str(repo/"results"/"stage3_T5Train"/f"curvature_RQ-VAE_iter{n}"/"ckpt")+"/",
    }
    for k,v in expected_paths.items():
        if ov.get(k)!=v: fail(f"{k}={ov.get(k)!r} != expected {v!r}")
        if kv(manifest,f"STAGE3_{k}")!=v: fail(f"manifest STAGE3_{k} != expected {v}")
    if not isinstance(ov.get("RQVAE_VARIANT"),str) or not ov["RQVAE_VARIANT"]: fail("RQVAE_VARIANT must be a non-empty literal string")
    print("STAGE3_PROTOCOL_PASS"); print(f"iter={n}"); print(f"trainer_sha256={actual_sha}"); print(f"launcher={launcher}"); print(f"variant={ov['RQVAE_VARIANT']}")
if __name__=="__main__": main()
