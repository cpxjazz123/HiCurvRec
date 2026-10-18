"""Capture GREEN for every R5 arm against its own tree."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9"
ARMS = [f"r5_{n:02d}" for n in range(1, 11)]
PARENT = REPO_ROOT / "stage2_RQ-VAE/TIGER_r4_08"


def run(arm: str):
    script = REPO_ROOT / "results/stage2_RQ-VAE" / arm / f"test_{arm}.py"
    env = dict(os.environ)
    env.pop("R5_TEST_TREE", None)
    env.pop("R4_TEST_TREE", None)
    done = subprocess.run(
        [PYTHON, str(script)], cwd=REPO_ROOT, capture_output=True, text=True,
        timeout=7200, env=env,
    )
    return done.returncode, done.stdout + done.stderr, f"{PYTHON} results/stage2_RQ-VAE/{arm}/test_{arm}.py"


def main() -> int:
    failed = []
    for arm in ARMS:
        code, output, command = run(arm)
        path = REPO_ROOT / "results/stage2_RQ-VAE" / arm / "tdd_red_green.md"
        text = path.read_text(encoding="utf-8") if path.is_file() else f"# {arm.upper()} TDD record\n"
        text += "\n## GREEN\n\n"
        text += "Command (run against this arm's own tree, mechanism implemented):\n\n"
        text += f"    {command}\n\nExit code: {code}\n\n```\n{output.rstrip()}\n```\n\n"
        diff = subprocess.run(
            ["diff", "-u", str(PARENT / "train_rqvae.py"),
             str(REPO_ROOT / "stage2_RQ-VAE" / f"TIGER_{arm}" / "train_rqvae.py")],
            capture_output=True, text=True,
        ).stdout
        text += "## Single delta\n\n```diff\n" + diff + "\n```\n"
        path.write_text(text, encoding="utf-8")
        cfg_path = REPO_ROOT / "results/stage2_RQ-VAE" / arm / "config.json"
        if cfg_path.is_file():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            trainer = REPO_ROOT / "stage2_RQ-VAE" / f"TIGER_{arm}" / "train_rqvae.py"
            cfg["train_rqvae_sha256"] = hashlib.sha256(trainer.read_bytes()).hexdigest()
            cfg["single_delta_diff"] = diff
            cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        status = "GREEN OK" if code == 0 else "GREEN FAIL"
        print(f"{arm}: exit={code} {status}", flush=True)
        if code != 0:
            failed.append(arm)
            for ln in output.splitlines():
                if ln.startswith(("FAIL:", "ERROR:")):
                    print(f"    {ln[:170]}", flush=True)
    print("\nFAILED ARMS:", failed if failed else "none")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
