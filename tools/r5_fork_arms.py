"""Fork TIGER_r4_08 into the ten R5 arms and rewire identity only.

Parent is exactly issue #58 (commit bc96625). No mechanism code is touched here.

    python3 tools/r5_fork_arms.py
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
SRC_TREE = REPO_ROOT / "stage2_RQ-VAE/TIGER_r4_08"
ARMS = [f"r5_{n:02d}" for n in range(1, 11)]


def fork(arm: str) -> Path:
    tree = REPO_ROOT / "stage2_RQ-VAE" / f"TIGER_{arm}"
    if tree.exists():
        shutil.rmtree(tree)
    shutil.copytree(SRC_TREE, tree, symlinks=True)
    for path in list(tree.rglob("__pycache__")) + list(tree.rglob("logs")):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
    for pyc in tree.rglob("*.pyc"):
        pyc.unlink(missing_ok=True)
    (REPO_ROOT / "results/stage2_RQ-VAE" / arm).mkdir(parents=True, exist_ok=True)
    return tree


def rewire(arm: str, tree: Path) -> None:
    entry = tree / "train_rqvae.py"
    text = entry.read_text(encoding="utf-8")
    text, n_arm = re.subn(
        r'^EXPERIMENT_ARM = "r4_08"(.*)$',
        f'EXPERIMENT_ARM = "{arm}"\\1',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    text, n_guard = re.subn(
        r'if EXPERIMENT_ARM != "r4_08":',
        f'if EXPERIMENT_ARM != "{arm}":',
        text,
        count=1,
    )
    text, n_msg = re.subn(
        r"only the 'r4_08' mechanism may run in this tree",
        f"only the '{arm}' mechanism may run in this tree",
        text,
        count=1,
    )
    if (n_arm, n_guard, n_msg) != (1, 1, 1):
        raise SystemExit(
            f"{arm}: identity rewrite incomplete (arm={n_arm} guard={n_guard} msg={n_msg})"
        )
    # provenance comment near the arm flag
    text = text.replace(
        f'EXPERIMENT_ARM = "{arm}"',
        f'EXPERIMENT_ARM = "{arm}"  # R5 sole delta over pinned parent TIGER_r4_08 @ bc96625',
        1,
    )
    entry.write_text(text, encoding="utf-8")


def main() -> int:
    if not SRC_TREE.is_dir():
        raise SystemExit(f"missing parent tree {SRC_TREE}")
    for arm in ARMS:
        tree = fork(arm)
        rewire(arm, tree)
        print(f"{arm}: forked from TIGER_r4_08 -> {tree}")
    diff = subprocess.run(
        ["git", "status", "--short", "--", str(SRC_TREE.relative_to(REPO_ROOT))],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if diff.stdout.strip():
        raise SystemExit(f"parent TIGER_r4_08 was modified:\n{diff.stdout}")
    print("parent TIGER_r4_08 unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
