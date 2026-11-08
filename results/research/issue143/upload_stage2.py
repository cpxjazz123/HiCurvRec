"""Archive a finished Stage 2 arm into the numbered layout, with its own record.

These four arms are Stage 2 only -- Issues #140-#143 forbid a Stage 3 -- so the
fourth entry of a numbered directory is the mandatory layered SID record rather
than a recall block. The other three are the same as everywhere else: the
mechanism's source, the SID table and the checkpoint that exported it.

    python3 results/research/issue143/upload_stage2.py M4_post_sid_af3
    python3 results/research/issue143/upload_stage2.py M4_post_sid_af3 --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

RESEARCH = Path(__file__).resolve().parent
REPO = RESEARCH.parents[2]
sys.path.insert(0, str(REPO / "results" / "research" / "issue139"))

from r2_layout import Ledger, rclone, upload_directory  # noqa: E402

RESULT = REPO / "results" / "stage2_RQ-VAE"
TREE = REPO / "stage2_RQ-VAE"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def arm_directory(arm: str) -> str:
    """The arm's directory name, accepting either 'X' or the full 'TIGER_X'."""
    return arm if arm.startswith("TIGER_") else "TIGER_" + arm


def stage(arm: str, staging: Path) -> dict:
    result = RESULT / arm_directory(arm)
    tree = TREE / arm_directory(arm)
    sid = result / "item_sids.json"
    checkpoint = result / "rqvae_best.pth"
    missing = [name for name, path in (("item_sids.json", sid), ("rqvae_best.pth", checkpoint),
                                       ("source tree", tree if tree.is_dir() else None))
               if path is None or not Path(path).exists()]
    if not sid.is_file():
        missing.append("item_sids.json")
    code = staging / arm
    code.mkdir(parents=True, exist_ok=True)
    for relative in ("train_rqvae.py", "math_spec.json", "reference_contract.json"):
        source = tree / relative
        if source.is_file():
            shutil.copy2(source, code / relative)
    for directory in ("model", "tests"):
        if (tree / directory).is_dir():
            shutil.copytree(tree / directory, code / directory,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    if sid.is_file():
        shutil.copy2(sid, staging / "item_sids.json")
    if checkpoint.is_file():
        shutil.copy2(checkpoint, staging / "rqvae_best.pth")
    record = None
    if sid.is_file():
        target = result / "stage2_metrics" / (digest(sid) + ".json")
        if target.is_file():
            shutil.copy2(target, staging / "stage2_record.json")
            record = json.loads(target.read_text())
    files = {str(path.relative_to(staging)): digest(path)
             for path in sorted(staging.rglob("*")) if path.is_file()}
    return {"files": files, "missing": missing, "key": "stage2:" + arm,
            "record_keys": sorted(record) if isinstance(record, dict) else None}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("arm")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    staging = Path("/tmp") / ("r2_stage2_" + args.arm)
    shutil.rmtree(staging, ignore_errors=True)
    try:
        bundle = stage(args.arm, staging)
        ledger = Ledger()
        number = ledger.assign(bundle["key"])
        target = upload_directory(staging, number, dry_run=args.dry_run)
        ledger.record(number, bundle["key"], bundle["files"],
                      {"arm": args.arm, "missing": bundle["missing"],
                       "record_keys": bundle["record_keys"], "target": target,
                       "stage": "stage2-only"})
        if not args.dry_run:
            ledger.save()
        print(json.dumps({"number": number, "target": target, "files": len(bundle["files"]),
                          "missing": bundle["missing"]}, indent=2))
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
