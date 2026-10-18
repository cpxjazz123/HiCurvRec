"""Fork the canonical Resample tree into the ten R4 arms and rewire identity only.

Purely mechanical: copy the tree, point the arm guard at the new arm name, and
redirect the artifact directory to the issue-mandated ``results/stage2_RQ-VAE/r4_NN/``.
No mechanism code is touched here; that is a separate, per-arm edit.

    python3 tools/r4_fork_arms.py
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
SRC_TREE = REPO_ROOT / "stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE"
ARMS = [f"r4_{n:02d}" for n in range(1, 11)]
SKIP = ("logs", "__pycache__")


def fork(arm: str) -> Path:
    tree = REPO_ROOT / "stage2_RQ-VAE" / f"TIGER_{arm}"
    if tree.exists():
        shutil.rmtree(tree)
    shutil.copytree(SRC_TREE, tree, symlinks=True)
    for path in sorted(tree.rglob("__pycache__")) + sorted(tree.rglob("logs")):
        if path.is_dir():
            shutil.rmtree(path)
    (REPO_ROOT / "results/stage2_RQ-VAE" / arm).mkdir(parents=True, exist_ok=True)
    return tree


def rewire(arm: str, tree: Path) -> list[str]:
    """Apply the three identity edits and return the changed line numbers."""
    entry = tree / "train_rqvae.py"
    text = entry.read_text(encoding="utf-8")
    original = text

    text, n_arm = re.subn(
        r'^EXPERIMENT_ARM = "resample"(.*)$',
        f'EXPERIMENT_ARM = "{arm}"\\1',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    text, n_guard = re.subn(
        r'if EXPERIMENT_ARM != "resample":',
        f'if EXPERIMENT_ARM != "{arm}":',
        text,
        count=1,
    )
    text, n_msg = re.subn(
        r"only the 'resample' mechanism may run in this tree",
        f"only the '{arm}' mechanism may run in this tree",
        text,
        count=1,
    )
    text, n_out = re.subn(
        r'^_OUT = OUT_ROOT / TREE_NAME / EXPERIMENT_ARM$',
        "_OUT = OUT_ROOT / EXPERIMENT_ARM",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if (n_arm, n_guard, n_msg, n_out) != (1, 1, 1, 1):
        raise SystemExit(
            f"{arm}: identity rewrite incomplete (arm={n_arm} guard={n_guard} "
            f"msg={n_msg} out={n_out})"
        )
    if text == original:
        raise SystemExit(f"{arm}: rewrote nothing")
    entry.write_text(text, encoding="utf-8")
    return ["EXPERIMENT_ARM", "guard", "message", "_OUT"]


def main() -> int:
    for arm in ARMS:
        tree = fork(arm)
        fields = rewire(arm, tree)
        print(f"{arm}: forked at {tree} | rewired {', '.join(fields)}")
    # Prove the canonical tree is untouched.
    diff = subprocess.run(
        ["git", "status", "--short", "--", str(SRC_TREE.relative_to(REPO_ROOT))],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if diff.stdout.strip():
        raise SystemExit(f"canonical tree was modified:\n{diff.stdout}")
    print("canonical Resample tree unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
