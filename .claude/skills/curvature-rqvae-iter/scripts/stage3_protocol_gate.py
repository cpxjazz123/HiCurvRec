#!/usr/bin/env python3
"""Stateless frozen-Stage3 protocol gate.

Run with no CLI arguments from stage2_RQ-VAE/curvature_RQ-VAE/.
The gate reads only root CLAUDE.md, the shared trainer, and the current launcher.
It creates no workflow artifacts and requires no manifest.
"""
from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path

def fail(message):
    raise RuntimeError(f"PROTOCOL_GATE_FAIL: {message}")

def read(path):
    if not path.is_file():
        fail(f"missing file: {path}")
    return path.read_text(encoding="utf-8", errors="replace")

def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

def work_dir(work):
    """The single in-place working directory; iteration numbers are gone.

    Iterations now edit stage2_RQ-VAE/curvature_RQ-VAE in place and roll back
    through Git, so there is exactly one accepted directory name.
    """
    if work.name != "curvature_RQ-VAE":
        fail(f"run from stage2_RQ-VAE/curvature_RQ-VAE/, got {work}")
    return "curvature_RQ-VAE"

def kv(text,key):
    m=re.search(rf"^{re.escape(key)}=(.*)$",text,re.M)
    if not m:
        fail(f"missing repository lock {key}")
    return m.group(1).strip()

def bval(value):
    value=value.lower()
    if value in {"true","1","yes"}:
        return True
    if value in {"false","0","no"}:
        return False
    fail(f"invalid boolean {value}")

def literals(path):
    tree=ast.parse(read(path),filename=str(path))
    out={}
    for node in tree.body:
        if isinstance(node,ast.Assign):
            try:
                value=ast.literal_eval(node.value)
            except Exception:
                continue
            for target in node.targets:
                if isinstance(target,ast.Name):
                    out[target.id]=value
        elif isinstance(node,ast.AnnAssign) and isinstance(node.target,ast.Name):
            try:
                out[node.target.id]=ast.literal_eval(node.value)
            except Exception:
                pass
    return out

def overrides(path):
    tree=ast.parse(read(path),filename=str(path))
    out={}
    for node in ast.walk(tree):
        if not isinstance(node,(ast.Assign,ast.AnnAssign)):
            continue
        targets=node.targets if isinstance(node,ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target,ast.Attribute) and isinstance(target.value,ast.Name) and target.value.id=="trainer":
                try:
                    out[target.attr]=ast.literal_eval(node.value)
                except Exception:
                    out[target.attr]="<dynamic>"
    return out

def main():
    work=Path.cwd().resolve()
    variant_dir=work_dir(work)
    repo=work.parents[1]
    claude=read(repo/"CLAUDE.md")

    expected={
        "NUM_EPOCHS":int(kv(claude,"STAGE3_NUM_EPOCHS")),
        "EARLY_STOP":kv(claude,"STAGE3_EARLY_STOP"),
        "NO_EVAL":bval(kv(claude,"STAGE3_NO_EVAL")),
        "SKIP_TEST":bval(kv(claude,"STAGE3_SKIP_TEST")),
        "SEED":int(kv(claude,"STAGE3_SEED")),
        "BEAM_SIZE":int(kv(claude,"STAGE3_BEAM_SIZE")),
        "SCREEN_BASELINE_LOG":kv(claude,"STAGE3_SCREEN_BASELINE_LOG"),
    }

    trainer=repo/"stage3_T5Train"/"train_HG-Rec.py"
    values=literals(trainer)

    for key in ("NUM_EPOCHS","NO_EVAL","SKIP_TEST","SEED","BEAM_SIZE","SCREEN_BASELINE_LOG"):
        if key not in values:
            fail(f"trainer missing literal {key}")
        if values[key]!=expected[key]:
            fail(f"trainer {key}={values[key]!r} != locked {expected[key]!r}")

    if str(expected["EARLY_STOP"]).upper()=="DISABLED":
        if values.get("EARLY_STOP") not in (None,False,0):
            fail(f"trainer EARLY_STOP={values.get('EARLY_STOP')!r}; locked protocol requires DISABLED")
    elif str(values.get("EARLY_STOP"))!=str(expected["EARLY_STOP"]):
        fail(f"trainer EARLY_STOP={values.get('EARLY_STOP')!r} != locked {expected['EARLY_STOP']!r}")

    launcher = work / "scripts" / "run_stage3_curvature.py"
    ov = overrides(launcher)
    allowed={"CODE_PATH","RQVAE_VARIANT","LOG_PATH","SAVE_PATH"}
    forbidden=sorted(set(ov)-allowed)
    if forbidden:
        fail(f"launcher overrides frozen Stage3 fields: {forbidden}")
    missing=sorted(allowed-set(ov))
    if missing:
        fail(f"launcher missing required wiring fields: {missing}")

    expected_paths={
        "CODE_PATH":str(repo/"results"/"stage2_RQ-VAE"/variant_dir/"item_sids.json"),
        "LOG_PATH":str(repo/"results"/"stage3_T5Train"/variant_dir/"logs")+"/",
        "SAVE_PATH":str(repo/"results"/"stage3_T5Train"/variant_dir/"ckpt")+"/",
    }
    for key,value in expected_paths.items():
        if ov.get(key)!=value:
            fail(f"{key}={ov.get(key)!r} != expected {value!r}")

    if not isinstance(ov.get("RQVAE_VARIANT"),str) or not ov["RQVAE_VARIANT"]:
        fail("RQVAE_VARIANT must be a non-empty literal string")

    print("STAGE3_PROTOCOL_PASS")
    print(f"variant_dir={variant_dir}")
    print(f"trainer_sha256={sha256(trainer)}")
    print(f"launcher={launcher}")
    print(f"variant={ov['RQVAE_VARIANT']}")

if __name__=="__main__":
    main()
