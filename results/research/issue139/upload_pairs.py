"""Upload every properly paired historical SID to R2, with whatever it has.

"Properly paired" is the audit's own VERIFIED verdict: the Stage 2 arm directory
and the Stage 3 variant directory carry the same name, which is the only pairing
CLAUDE.md 64b accepts. For each such pair this uploads what exists -- the SID
table, the codebook that exported it, the Stage 2 source, and the complete recall
block -- and records what does not exist in the manifest instead of pretending.

The historical arms mostly lose pieces to policy rather than to error: binaries
stay out of git (CLAUDE.md 46), and several SID tables were never kept. The run
summary counts each reason, so the archive is honest about its own gaps.

    python3 results/research/issue139/upload_pairs.py --dry-run
    python3 results/research/issue139/upload_pairs.py --limit 3
    python3 results/research/issue139/upload_pairs.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RESEARCH = Path(__file__).resolve().parent
sys.path.insert(0, str(RESEARCH))

from r2_layout import Ledger, upload_directory  # noqa: E402

AUDIT = REPO / "analysis_sid_recall" / "pair_audit.csv"
PREFIX = "issue139pairs"


def recall_block(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def build(row: dict, staging: Path) -> dict:
    """Stage exactly four entries: the code folder, the SID, the checkpoint, the recall.

    The layout is the owner's: no manifest, no extra names. The checksums and the
    identity of the directory go into the repository-side ledger instead, so the
    bucket keeps the four entries and the provenance stays auditable.
    """
    sid_path = REPO / row["sid_path"]
    arm = sid_path.parent.name
    candidates = [REPO / "stage2_RQ-VAE" / arm,
                  REPO / "stage2_RQ-VAE" / sid_path.parent.parent.name / arm]
    source = next((path for path in candidates if path.is_dir()), candidates[0])
    checkpoint = sid_path.parent / "rqvae_best.pth"
    recall_path = Path(row["recall_path"]) if row.get("recall_path") else None
    block = recall_block(recall_path) if recall_path else {}

    code = staging / row["stage2_rel"].replace("/", "__")
    staging.mkdir(parents=True, exist_ok=True)
    code.mkdir(exist_ok=True)
    for relative in ("train_rqvae.py", "math_spec.json", "reference_contract.json"):
        candidate = source / relative
        if candidate.is_file():
            shutil.copy2(candidate, code / relative)
    if (source / "model").is_dir():
        shutil.copytree(source / "model", code / "model",
                        ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(sid_path, staging / "item_sids.json")
    shutil.copy2(checkpoint, staging / "rqvae_best.pth")
    (staging / "recall.json").write_text(json.dumps(block, indent=2, sort_keys=True) + "\n")

    files = {}
    for path in sorted(staging.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(staging))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"files": files, "recall_keys": sorted(block),
            "key": "%s/%s" % (row["stage2_rel"], row["sid_sha256"][:12]),
            "pair": {"stage2_rel": row["stage2_rel"], "variant": row["variant"],
                     "sid_sha256": row["sid_sha256"], "commit_tier": row.get("commit_tier")}}


def pieces(row: dict) -> dict:
    """Which of the four required pieces exist for this pair."""
    sid = REPO / row["sid_path"]
    checkpoint = sid.parent / "rqvae_best.pth"
    source = REPO / "stage2_RQ-VAE" / sid.parent.name
    if not source.is_dir():
        source = REPO / "stage2_RQ-VAE" / sid.parent.parent.name / sid.parent.name
    block = recall_block(Path(row["recall_path"])) if row.get("recall_path") else {}
    return {"sid": sid.is_file(), "checkpoint": checkpoint.is_file(),
            "stage3": bool(block), "source": source.is_dir()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-incomplete", action="store_true",
                        help="also archive pairs that are missing a piece (default: no)")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()

    ledger = Ledger()
    verified = [row for row in csv.DictReader(open(AUDIT)) if row["verdict"] == "VERIFIED"]
    tiers = {}
    for row in verified:
        have = pieces(row)
        key = " ".join("%s%s" % (name, "+" if have[name] else "-") for name in have)
        tiers.setdefault(key, []).append(row)
    print("VERIFIED pairs: %d" % len(verified))
    for key, members in sorted(tiers.items(), key=lambda item: -len(item[1])):
        print("  %4d  %s" % (len(members), key))
    complete = [row for key, members in tiers.items() if all(p == "+" for p in
                [v for k, v in zip(key.split(), [True]*4)]) for row in members]
    # A pair is archivable only when all four pieces exist; that is the rule.
    complete = [row for row in verified if all(pieces(row).values())]
    rows = verified if args.include_incomplete else complete
    print("archivable (all four pieces): %d%s" % (len(complete),
          "" if args.include_incomplete else " -- the only ones archived"))
    if args.limit:
        rows = rows[:args.limit]
    uploaded, counts = 0, {}
    for row in rows:
        staging = Path("/tmp") / ("r2_pair_%s" % row["sid_sha256"][:12])
        shutil.rmtree(staging, ignore_errors=True)
        bundle = build(row, staging)
        number = ledger.assign(bundle["key"])
        try:
            target = upload_directory(staging, number, dry_run=args.dry_run)
            ledger.record(number, bundle["key"], bundle["files"], {"pair": bundle["pair"],
                                                                  "recall_keys": bundle["recall_keys"],
                                                                  "target": target})
            uploaded += 1
            if uploaded % 10 == 0 or uploaded == 1:
                print("  [%d/%d] %s -> %s" % (uploaded, len(rows), row["stage2_rel"][:52], target),
                      flush=True)
        except RuntimeError as exc:
            print("  FAILED %s: %s" % (row["stage2_rel"], str(exc)[:200]), flush=True)
            counts["upload failed"] = counts.get("upload failed", 0) + 1
        if not args.keep:
            shutil.rmtree(staging, ignore_errors=True)
    print("\n%s %d of %d pairs" % ("would upload" if args.dry_run else "uploaded", uploaded, len(rows)))
    print("gaps by reason:")
    for reason, count in sorted(counts.items(), key=lambda item: -item[1]):
        print("  %5d  %s" % (count, reason))
    ledger.save()
    Path(RESEARCH / "r2_pairs_upload.json").write_text(json.dumps(
        {"pairs": len(rows), "uploaded": uploaded, "gaps": counts,
         "dry_run": args.dry_run, "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
        indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
