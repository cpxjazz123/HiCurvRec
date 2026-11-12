"""Archive one finished Stage 3 run into the numbered layout.

The rule (CLAUDE.md 6b): every time a Stage 3 run finishes, the four pieces are
archived together -- the Stage 2 code it used, the SID table, the checkpoint that
exported that table, and the complete recall block. The bucket's first level is a
two-digit number and each number holds exactly those four entries:

    r2:genrec/<NN>/<mechanism>/        train_rqvae.py, model/, math_spec.json,
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

#: Entries of a Stage2 tree that are run artifacts, not the mechanism's source.
_NOT_SOURCE = {"logs", "__pycache__", "dataset", "ckpt", "results", ".git"}
_NOT_SOURCE_SUFFIXES = {".pth", ".pt", ".npy", ".npz", ".parquet", ".log", ".pyc", ".tmp"}


def _latest_test_final(arm: str) -> Path | None:
    """Newest test_final.json of this arm; nested arms are stored with ``__``."""
    for name in (arm, arm.replace("/", "__")):
        flat = STAGE3_LOGS / name / "test_final.json"
        if flat.is_file():
            return flat
        nested = sorted((STAGE3_LOGS / name).glob("logs/*/*/test_final.json"),
                        key=lambda path: path.stat().st_mtime, reverse=True)
        if nested:
            return nested[0]
    return None


def archive_key(arm: str, sid_path: Path) -> str:
    """The ledger key: the arm and the digest of the SID table it archived.

    A re-run of the same arm does not reproduce the SID (the codebook initialisation
    is not seeded), so keying by the arm name alone would overwrite the earlier
    archive and its recall the moment a second run finished. The digest makes the key
    name what it holds, and an identical SID still maps to the same number.
    """
    return "%s/%s" % (arm.replace("/", "__"), sha256(Path(sid_path))[:12])


def stage(arm: str, staging: Path) -> tuple[dict, str]:
    """Stage the four entries; return the checksums and the ledger key."""
    result = RESULTS / "stage2_RQ-VAE" / arm
    sid = result / "item_sids.json"
    checkpoint = result / "rqvae_best.pth"
    if not checkpoint.is_file():
        found = sorted(result.rglob("rqvae_best.pth"))
        checkpoint = found[0] if found else checkpoint
    source = TREES / arm
    if not source.is_dir():
        source = TREES / arm.split("/")[0]
    if not source.is_dir():
        for candidate in sorted(TREES.glob("*/" + arm)):
            if candidate.is_dir():
                source = candidate
                break
    stage3 = _latest_test_final(arm)

    missing = [name for name, path in (("item_sids.json", sid),
                                       ("rqvae_best.pth", checkpoint),
                                       ("stage2 source tree", source if source.is_dir() else None),
                                       ("stage3 test_final.json", stage3)) if path is None]
    if missing:
        raise SystemExit("%s is incomplete, not archiving: %s" % (arm, ", ".join(missing)))

    staging.mkdir(parents=True, exist_ok=True)
    code = staging / arm.replace("/", "__")
    code.mkdir(exist_ok=True)
    # The mechanism's source is whatever the tree imports: copying a fixed three-file
    # list archived a trainer whose `losses/`, `distances/` and `train_config.py` were
    # missing, so the archived copy could not run. Everything that is source travels;
    # run artifacts, binaries and the dataset link do not.
    for entry in sorted(source.iterdir()):
        name = entry.name
        if name in _NOT_SOURCE or name.startswith("."):
            continue
        if entry.is_dir():
            shutil.copytree(entry, code / name,
                            ignore=shutil.ignore_patterns("__pycache__", "logs",
                                                          "*.pth", "*.npy", "*.parquet"))
        elif entry.suffix.lower() not in _NOT_SOURCE_SUFFIXES:
            shutil.copy2(entry, code / name)
    shutil.copy2(sid, staging / "item_sids.json")
    shutil.copy2(checkpoint, staging / "rqvae_best.pth")
    block = json.loads(stage3.read_text())
    (staging / "recall.json").write_text(json.dumps(block, indent=2, sort_keys=True) + "\n")

    files = {str(path.relative_to(staging)): sha256(path)
             for path in sorted(staging.rglob("*")) if path.is_file()}
    return files, archive_key(arm, sid)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("arm")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()

    staging = RESEARCH / "_stage" / ("r2_arm_%s" % args.arm.replace("/", "__"))
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
            # The ledger must cover the directory, not just this staging set:
            # a number that earlier uploads filled tree-and-all keeps those
            # objects, and they need a checksum too.
            result = ledger.reconcile(number)
            ledger.save()
            for relative in sorted(result["added"]):
                print("  recorded without uploading: %s/%s" % (number, relative))
            for relative in result["recorded_but_absent"]:
                print("  WARNING recorded but not in the bucket: %s/%s" % (number, relative))
        print("%s: %d files, recall keys %d -> %s" % (
            args.arm, len(files), len(json.loads((staging / "recall.json").read_text())), target))
        print(("would upload to " if args.dry_run else "uploaded to ") + target)
    finally:
        if not args.keep:
            shutil.rmtree(staging, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
