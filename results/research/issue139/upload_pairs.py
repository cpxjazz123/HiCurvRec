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

from upload_stage3 import R2_REMOTE, RCLONE, credentials, sha256  # noqa: E402

AUDIT = REPO / "analysis_sid_recall" / "pair_audit.csv"
PREFIX = "genrec/issue139pairs"


def recall_block(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def build(row: dict, staging: Path) -> dict:
    sid_path = REPO / row["sid_path"]
    tree = sid_path.parent.parent           # results/stage2_RQ-VAE/<parent>
    source = REPO / "stage2_RQ-VAE" / tree.name
    if not source.is_dir():
        source = REPO / "stage2_RQ-VAE" / tree.parent.name / tree.name
    checkpoint = sid_path.parent / "rqvae_best.pth"
    recall_path = Path(row["recall_path"]) if row.get("recall_path") else None
    block = recall_block(recall_path) if recall_path else {}

    staging.mkdir(parents=True, exist_ok=True)
    manifest = {"pair": {"stage2_rel": row["stage2_rel"], "variant": row["variant"],
                         "sid_sha256": row["sid_sha256"], "verdict": row["verdict"],
                         "commit_tier": row.get("commit_tier")},
                "archived_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "recall_block": block, "missing": [], "files": {}}

    if sid_path.is_file():
        shutil.copy2(sid_path, staging / "item_sids.json")
        manifest["files"]["item_sids.json"] = sha256(sid_path)
    else:
        manifest["missing"].append("item_sids.json")
    if checkpoint.is_file():
        shutil.copy2(checkpoint, staging / "rqvae_best.pth")
        manifest["files"]["rqvae_best.pth"] = sha256(checkpoint)
    else:
        manifest["missing"].append("rqvae_best.pth (the codes cannot be decoded without it)")
    if source.is_dir():
        archive = staging / "stage2_source.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for relative in ("train_rqvae.py", "math_spec.json", "reference_contract.json"):
                candidate = source / relative
                if candidate.is_file():
                    tar.add(candidate, arcname=relative)
            if (source / "model").is_dir():
                tar.add(source / "model", arcname="model")
        manifest["files"]["stage2_source.tar.gz"] = sha256(archive)
    else:
        manifest["missing"].append("stage2 source tree")
    if block:
        if recall_path.is_file():
            shutil.copy2(recall_path, staging / "stage3_test_final.json")
            manifest["files"]["stage3_test_final.json"] = sha256(recall_path)
        manifest["recall_keys"] = sorted(block)
    else:
        manifest["missing"].append("stage3 result")
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def upload(staging: Path, target: str, *, dry_run: bool) -> None:
    env = dict(os.environ)
    env.update(credentials())
    command = [str(RCLONE), "copy", str(staging), target, "--s3-no-check-bucket"]
    if dry_run:
        command.append("--dry-run")
    done = subprocess.run(command, capture_output=True, text=True, env=env, timeout=1800)
    if done.returncode != 0:
        raise RuntimeError((done.stderr or done.stdout)[-400:])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()

    rows = [row for row in csv.DictReader(open(AUDIT)) if row["verdict"] == "VERIFIED"]
    if args.limit:
        rows = rows[:args.limit]
    print("VERIFIED pairs to archive:", len(rows), flush=True)
    uploaded, counts = 0, {}
    for row in rows:
        slug = re.sub(r"[^A-Za-z0-9._-]+", "__", row["stage2_rel"])
        target = "%s/%s/%s/%s/" % (R2_REMOTE, PREFIX, slug, row["sid_sha256"][:12])
        staging = Path("/tmp") / ("r2_pair_%s" % row["sid_sha256"][:12])
        shutil.rmtree(staging, ignore_errors=True)
        manifest = build(row, staging)
        for item in manifest["missing"]:
            counts[item] = counts.get(item, 0) + 1
        try:
            upload(staging, target, dry_run=args.dry_run)
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
    Path(RESEARCH / "r2_pairs_upload.json").write_text(json.dumps(
        {"pairs": len(rows), "uploaded": uploaded, "gaps": counts,
         "dry_run": args.dry_run, "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
        indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
