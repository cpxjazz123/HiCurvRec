"""Archive one finished Stage 3 run into the numbered layout.

The rule (CLAUDE.md 6b): every time a Stage 3 run finishes, the four pieces are
archived together -- the Stage 2 code it used, the SID table, the checkpoint that
exported that table, and the complete recall block. The bucket's first level is a
two-digit number and each number holds exactly those four entries:

    r2:genrec/<NN>/stage2_code/        train_rqvae.py, model/, math_spec.json,
                                       reference_contract.json
    r2:genrec/<NN>/item_sids.json
    r2:genrec/<NN>/rqvae_best.pth
    r2:genrec/<NN>/recall.json         every key of test_final.json

What the number refers to, and the checksum of every uploaded file, goes into
`r2_ledger.json` beside this script.

    python3 results/research/issue139/upload_stage3.py F0
    python3 results/research/issue139/upload_stage3.py F0 --dry-run
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

RESEARCH = Path(__file__).resolve().parent
REPO = RESEARCH.parents[2]
sys.path.insert(0, str(RESEARCH))

from r2_layout import Ledger, sha256, upload_directory  # noqa: E402

RESULTS = REPO / "results"
TREES = REPO / "stage2_RQ-VAE"
STAGE3_LOGS = RESULTS / "stage3_T5Train"


def stage(arm: str, staging: Path) -> tuple[dict, str]:
    """Stage the four entries; return the checksums and the ledger key."""
    result = RESULTS / "stage2_RQ-VAE" / arm
    sid = result / "item_sids.json"
    checkpoint = result / "rqvae_best.pth"
    source = TREES / arm
    if not source.is_dir():
        for candidate in sorted(TREES.glob("*/" + arm)):
            if candidate.is_dir():
                source = candidate
                break
    stage3 = STAGE3_LOGS / arm / "test_final.json"

    missing = [name for name, path in (("item_sids.json", sid),
                                       ("rqvae_best.pth", checkpoint),
                                       ("stage2 source tree", source if source.is_dir() else None),
                                       ("stage3 test_final.json", stage3)) if path is None]
    if missing:
        raise SystemExit("%s is incomplete, not archiving: %s" % (arm, ", ".join(missing)))

    staging.mkdir(parents=True, exist_ok=True)
    code = staging / "stage2_code"
    code.mkdir(exist_ok=True)
    for relative in ("train_rqvae.py", "math_spec.json", "reference_contract.json"):
        if (source / relative).is_file():
            shutil.copy2(source / relative, code / relative)
    if (source / "model").is_dir():
        shutil.copytree(source / "model", code / "model",
                        ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(sid, staging / "item_sids.json")
    shutil.copy2(checkpoint, staging / "rqvae_best.pth")
    block = json.loads(stage3.read_text())
    (staging / "recall.json").write_text(json.dumps(block, indent=2, sort_keys=True) + "\n")

    files = {str(path.relative_to(staging)): sha256(path)
             for path in sorted(staging.rglob("*")) if path.is_file()}
    return files, arm


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("arm")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()

    staging = Path("/tmp") / ("r2_arm_%s" % args.arm.replace("/", "__"))
    shutil.rmtree(staging, ignore_errors=True)
    try:
        files, key = stage(args.arm, staging)
        ledger = Ledger()
        number = ledger.assign(key)
        target = upload_directory(staging, number, dry_run=args.dry_run)
        ledger.record(number, key, files, {"arm": args.arm,
                                           "recall_keys": sorted(json.loads(
                                               (staging / "recall.json").read_text()))})
        if not args.dry_run:
            ledger.save()
        print("%s: %d files, recall keys %d -> %s" % (
            args.arm, len(files), len(json.loads((staging / "recall.json").read_text())), target))
        print(("would upload to " if args.dry_run else "uploaded to ") + target)
    finally:
        if not args.keep:
            shutil.rmtree(staging, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
