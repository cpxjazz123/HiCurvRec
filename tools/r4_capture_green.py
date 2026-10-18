"""Run every R4 arm's test file against its OWN tree and record the GREEN phase.

Appends a ``## GREEN`` block (plus the single-delta diff summary) to each arm's
``tdd_red_green.md``. Exits non-zero if any arm's suite does not pass.

    python3 tools/r4_capture_green.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9"
ARMS = [f"r4_{n:02d}" for n in range(1, 11)]
BASELINE_TREE = REPO_ROOT / "stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE"


def run(arm: str) -> tuple[int, str, str]:
    script = REPO_ROOT / "results/stage2_RQ-VAE" / arm / f"test_{arm}.py"
    env = dict(os.environ)
    env.pop("R4_TEST_TREE", None)
    done = subprocess.run(
        [PYTHON, str(script)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=7200,
        env=env,
    )
    command = f"{PYTHON} results/stage2_RQ-VAE/{arm}/test_{arm}.py"
    return done.returncode, done.stdout + done.stderr, command


def diff_for(arm: str) -> tuple[str, str]:
    tree = f"stage2_RQ-VAE/TIGER_{arm}"
    rel_files = ["train_rqvae.py", "model/layers.py", "model/model.py"]
    stat_lines = []
    diffs = []
    for rel in rel_files:
        left = f"stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE/{rel}"
        right = f"{tree}/{rel}"
        full = subprocess.run(
            ["diff", "-u", left, right],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        if full.stdout.strip():
            stat_lines.append(rel)
            diffs.append(full.stdout)
    body = "\n".join(diffs)
    stat = "\n".join(stat_lines) if stat_lines else "(none)"
    return f"### Files\n\n```\n{stat}\n```\n\n### Diff\n\n```diff\n{body}\n```\n", body


def main() -> int:
    failed = []
    for arm in ARMS:
        code, output, command = run(arm)
        path = REPO_ROOT / "results/stage2_RQ-VAE" / arm / "tdd_red_green.md"
        text = path.read_text(encoding="utf-8") if path.is_file() else f"# {arm.upper()} TDD record\n"
        text += "\n## GREEN\n\n"
        text += f"Command (run against this arm's own tree, mechanism implemented):\n\n"
        text += f"    {command}\n\nExit code: {code}\n\n```\n{output.rstrip()}\n```\n\n"
        diff_md, raw_diff = diff_for(arm)
        text += "## Single delta\n\n" + diff_md + "\n"
        path.write_text(text, encoding="utf-8")
        cfg_path = REPO_ROOT / "results/stage2_RQ-VAE" / arm / "config.json"
        if cfg_path.is_file():
            import hashlib, json
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            trainer_file = REPO_ROOT / "stage2_RQ-VAE" / f"TIGER_{arm}" / "train_rqvae.py"
            cfg["train_rqvae_sha256"] = hashlib.sha256(trainer_file.read_bytes()).hexdigest()
            cfg["single_delta_diff"] = raw_diff
            cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        status = "GREEN OK" if code == 0 else "GREEN FAIL"
        print(f"{arm}: exit={code} {status}", flush=True)
        if code != 0:
            failed.append(arm)
            for line in output.splitlines():
                if line.startswith(("FAIL:", "ERROR:")):
                    print(f"    {line[:170]}", flush=True)
    print("\nFAILED ARMS:", failed if failed else "none")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
