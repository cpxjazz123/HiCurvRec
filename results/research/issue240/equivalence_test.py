"""C5: prove the issue-240 instrumentation is side-effect free.

Two full runs of the real entry point, identical except for ``DIAG_ENABLED``, must
produce bit-identical training losses, final parameters and final SID.  A third run
repeats the disabled configuration to separate "the switch did something" from "the
recipe is not deterministic".

The runs use a temporary copy of the diagnostic tree with EPOCHS=1 and small batches
so the check is cheap; the copy is deleted afterwards.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "stage2_RQ-VAE" / "TIGER_BEHAVIOR_SAMPLE_RQ-VAE_diag"
WORK = REPO / "results" / "research" / "issue240" / "equivalence"
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9"

PATCHES = {
    "EPOCHS              = 3000": "EPOCHS              = 1",
    "BATCH_SIZE_PER_RANK = 4096": "BATCH_SIZE_PER_RANK = 64",
    "WARMUP_EPOCHS       = 50": "WARMUP_EPOCHS       = 1",
    "PAIR_BATCH_SIZE_PER_RANK = 2048": "PAIR_BATCH_SIZE_PER_RANK = 32",
    "EVAL_INTERVAL       = 50": "EVAL_INTERVAL       = 1",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tensors(path: Path) -> str:
    import torch

    payload = torch.load(path, map_location="cpu")
    digest = hashlib.sha256()
    for name in sorted(payload["state_dict"]):
        digest.update(name.encode())
        digest.update(payload["state_dict"][name].contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def build(name: str, enabled: bool) -> Path:
    tree = REPO / "stage2_RQ-VAE" / name
    if tree.exists():
        shutil.rmtree(tree)
    shutil.copytree(SRC, tree, ignore=shutil.ignore_patterns("__pycache__", "logs"))
    trainer = tree / "train_rqvae.py"
    text = trainer.read_text(encoding="utf-8")
    for old, new in PATCHES.items():
        if old not in text:
            raise RuntimeError(f"patch anchor missing in {trainer}: {old!r}")
        text = text.replace(old, new)
    trainer.write_text(text, encoding="utf-8")
    diag = tree / "diag_trace.py"
    diag_text = diag.read_text(encoding="utf-8")
    diag_text = diag_text.replace("DIAG_ENABLED = True", f"DIAG_ENABLED = {enabled}")
    diag.write_text(diag_text, encoding="utf-8")
    return tree


def run(tree: Path, name: str, ranks: int, use_torchrun: bool = False) -> dict:
    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(ranks)),
        "OMP_NUM_THREADS": "1",
    })
    if ranks == 1 and not use_torchrun:
        cmd = [PYTHON, "train_rqvae.py"]
    else:
        cmd = [
            PYTHON, "-m", "torch.distributed.run", "--standalone",
            f"--nproc_per_node={ranks}", "--master_port=50299", "train_rqvae.py",
        ]
    result = subprocess.run(cmd, cwd=tree, env=env, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        raise RuntimeError(f"{name} failed:\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}")
    out = REPO / "results" / "stage2_RQ-VAE" / tree.name / "resample"
    record = {
        "name": name,
        "ranks": ranks,
        "sid_sha256": sha256_file(out / "item_sids.json"),
        "params_sha256": sha256_tensors(out / "rqvae_best.pth"),
        "metrics": [
            json.loads(line) for line in
            (tree / "logs" / "resample" / "training_metrics.jsonl")
            .read_text(encoding="utf-8").splitlines()
        ]
        if (tree / "logs" / "resample" / "training_metrics.jsonl").exists()
        else [],
    }
    steps = REPO / "results" / "research" / "issue240" / "raw" / "steps.jsonl"
    if steps.exists() and "on" in name:
        rows = [json.loads(line) for line in steps.read_text(encoding="utf-8").splitlines()]
        record["identity_rel_error_max"] = max(
            (row["identity"].get("rel_error") or 0.0) for row in rows
        )
        record["identity_rows"] = len(rows)
        record["steps_sha256"] = sha256_file(steps)
        steps.unlink()
    return record


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    report = {"single": [], "ddp1": [], "ddp4": []}
    for label, suffix, ranks, torchrun in (
        ("single", "1", 1, False), ("ddp1", "d1", 1, True), ("ddp4", "4", 4, True),
    ):
        tree_off = build(f"_diag_equiv_off{suffix}", False)
        tree_on = build(f"_diag_equiv_on{suffix}", True)
        tree_off2 = build(f"_diag_equiv_off{suffix}b", False)
        report[label] = [
            run(tree_off, f"off{suffix}", ranks, torchrun),
            run(tree_on, f"on{suffix}", ranks, torchrun),
            run(tree_off2, f"off{suffix}b", ranks, torchrun),
        ]
        for tree in (tree_off, tree_on, tree_off2):
            shutil.rmtree(tree)
            shutil.rmtree(REPO / "results" / "stage2_RQ-VAE" / tree.name, ignore_errors=True)
    (WORK / "equivalence.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    ok = True
    for mode, runs in report.items():
        off_a, on, off_b = runs
        deterministic = (
            off_a["sid_sha256"] == off_b["sid_sha256"]
            and off_a["params_sha256"] == off_b["params_sha256"]
        )
        equivalent = (
            off_a["sid_sha256"] == on["sid_sha256"]
            and off_a["params_sha256"] == on["params_sha256"]
        )
        print(f"{mode}: deterministic={deterministic} diag_equivalent={equivalent}")
        ok = ok and deterministic and equivalent
    print("EQUIVALENCE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
