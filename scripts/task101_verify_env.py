#!/usr/bin/env python3
"""
Task #101 — Reproducibility Environment Verifier

Checks that the conda env `grid_toys` is active, the dataset exists,
and key packages import. Prints a one-shot summary and exits 0 only
if all checks pass.

Usage:
    python3 scripts/task101_verify_env.py [--data-dir data/amazon_data/musical_instruments]

No network. No fallback (R2). Failure => raise.
"""

from __future__ import annotations
import argparse
import importlib
import sys
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")

_REQUIRED = [
    "torch", "torch.cuda",
    "pytorch_lightning",
    "transformers",
    "hydra",
    "omegaconf",
]


def check_packages() -> None:
    """Assert all required packages import."""
    missing: list[str] = []
    for name in _REQUIRED:
        try:
            importlib.import_module(name)
            print(f"  [OK] {name}")
        except ImportError as e:
            missing.append(f"{name}: {e}")
            print(f"  [FAIL] {name}: {e}")
    if missing:
        raise RuntimeError(
            "Required packages missing:\n  " + "\n  ".join(missing)
            + "\n  Activate the env: conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys"
        )


def check_dataset(data_dir: Path) -> None:
    """Assert the Musical_Instruments 5-core CSV exists."""
    csv = data_dir / "Musical_Instruments_5core.csv.gz"
    if not csv.exists():
        raise FileNotFoundError(
            f"Dataset CSV not found at {csv}.\n"
            "Run: bash scripts/download_amazon_musical_instruments.sh\n"
            "Or manually place the CSV at this path (see REPRODUCE.md §2)."
        )
    size = csv.stat().st_size
    if size < 1_000_000:
        raise RuntimeError(
            f"Dataset {csv} is suspiciously small ({size} bytes). "
            "Expected ≥ 1 MB compressed for the 5-core Musical_Instruments dataset."
        )
    print(f"  [OK] Dataset CSV at {csv} ({size:,} bytes)")


def check_repo_layout() -> None:
    """Assert key project directories exist."""
    for d in ("src", "configs", "data", "scripts", "papers", "verdicts"):
        if not (REPO / d).exists():
            raise FileNotFoundError(f"Repo layout missing: {REPO / d}")
    print("  [OK] Repo layout (src/configs/data/scripts/papers/verdicts)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default=str(REPO / "data" / "amazon_data" / "musical_instruments"),
    )
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip dataset CSV check (used by Task #107 CI to avoid pulling 100s MB)",
    )
    parser.add_argument(
        "--skip-packages",
        action="store_true",
        help="Skip torch/transformers/hydra import check (used by Task #107 CI which only needs pyyaml)",
    )
    args = parser.parse_args()

    print("=== Task #101 Environment Verification ===")
    check_repo_layout()
    if args.skip_packages:
        print("  [SKIP] Package import check (--skip-packages flag; CI mode)")
    else:
        check_packages()
    if args.skip_data:
        print("  [SKIP] Dataset CSV check (--skip-data flag; CI mode)")
    else:
        check_dataset(Path(args.data_dir))
    print("=== ALL CHECKS PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
