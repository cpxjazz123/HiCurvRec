"""Materialise one directed arm tree from the template (Issue #221).

Two constants differ: ARM_NAME in train_rqvae.py and ARM_VARIANT in drq.py.
"""
from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "template"
REPO = HERE.parents[2]
ARMS = ("D0", "D1", "D2", "D3", "D4")


def digest(root: Path) -> dict:
    out = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            out[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def main(arm: str) -> None:
    if arm not in ARMS:
        raise SystemExit(f"arm must be one of {ARMS}")
    name = f"TIGER_drq_{arm}"
    target = REPO / "stage2_RQ-VAE" / name
    if target.exists():
        raise SystemExit(f"{target} already exists")
    shutil.copytree(TEMPLATE, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "logs"))
    trainer = target / "train_rqvae.py"
    text = trainer.read_text().replace('ARM_NAME = "TIGER_RQ-VAE"', f'ARM_NAME = "{name}"')
    trainer.write_text(text)
    module = target / "drq.py"
    text = module.read_text().replace('ARM_VARIANT = "D0"', f'ARM_VARIANT = "{arm}"')
    module.write_text(text)
    reference, produced = digest(TEMPLATE), digest(target)
    differing = sorted(key for key in reference if reference[key] != produced.get(key))
    assert set(differing) <= {"drq.py", "train_rqvae.py"}, differing
    assert "train_rqvae.py" in differing
    print(f"{target} ready; differing files {differing}")


if __name__ == "__main__":
    main(sys.argv[1])
