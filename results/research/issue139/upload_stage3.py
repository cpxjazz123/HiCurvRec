"""Archive one finished Stage 3 run to Cloudflare R2.

The rule (CLAUDE.md section 12, added 2026-11-08 on the owner's instruction): every
time a Stage 3 run finishes, upload the Stage 2 code it used, the SID table, the
checkpoint that exported that table, and the COMPLETE recall block of the Stage 3
result -- not `test_recall@10` alone.

Credentials are read from `r2_apikey` in the repository root and are never printed
or written into the manifest. rclone is configured through environment variables,
so nothing persistent is created on the host.

    python3 results/research/issue139/upload_stage3.py F0
    python3 results/research/issue139/upload_stage3.py F0 --dry-run
    python3 results/research/issue139/upload_stage3.py F0 --allow-incomplete
"""
from __future__ import annotations

import argparse
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
CREDENTIALS = REPO / "r2_apikey"
RCLONE = Path("/home/wlia0047/bin/rclone")
#: Where the archive lives. The bucket exists already; the prefix is ours.
R2_REMOTE = "r2:models"
R2_PREFIX = "genrec/issue139"


def credentials() -> dict:
    """The four values rclone needs, read from the owner's key file."""
    raw = CREDENTIALS.read_text()

    def grab(label: str) -> str:
        match = re.search(re.escape(label) + r"\s*=\s*(\S+)", raw)
        if not match:
            raise SystemExit("r2_apikey is missing the line %r" % label)
        return match.group(1)

    return {
        "RCLONE_CONFIG_R2_TYPE": "s3",
        "RCLONE_CONFIG_R2_PROVIDER": "Cloudflare",
        "RCLONE_CONFIG_R2_ACCESS_KEY_ID": grab("Access Key ID"),
        "RCLONE_CONFIG_R2_SECRET_ACCESS_KEY": grab("Secret Access Key"),
        "RCLONE_CONFIG_R2_ENDPOINT": grab("S3 API"),
        "RCLONE_CONFIG_R2_REGION": "auto",
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stage3_result(arm: str) -> tuple[Path | None, dict]:
    folder = REPO / "results" / "stage3_T5Train" / ("TIGER_mixed_rq_" + arm)
    found = sorted(folder.glob("logs/*/*/test_final.json")) if folder.is_dir() else []
    if not found:
        return None, {}
    path = found[-1]
    return path, json.loads(path.read_text())


def build_bundle(arm: str, staging: Path, *, allow_incomplete: bool) -> dict:
    """Copy what the rule names into one directory, and describe it."""
    tree = REPO / "stage2_RQ-VAE" / ("TIGER_mixed_rq_" + arm)
    result = REPO / "results" / "stage2_RQ-VAE" / ("TIGER_mixed_rq_" + arm)
    sid = result / "item_sids.json"
    checkpoint = result / "rqvae_best.pth"
    metrics = tree / "logs" / "training_metrics.jsonl"
    test_final, recall_block = stage3_result(arm)
    missing = [name for name, path in (("stage2 tree", tree), ("item_sids.json", sid),
                                       ("rqvae_best.pth", checkpoint),
                                       ("training_metrics.jsonl", metrics)) if not path.exists()]
    if test_final is None:
        missing.append("stage3 test_final.json")
    if missing and not allow_incomplete:
        raise SystemExit("nothing archived: missing " + ", ".join(missing)
                         + " (pass --allow-incomplete to archive anyway)")

    staging.mkdir(parents=True, exist_ok=True)
    manifest = {"arm": arm, "archived_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "recall_block": recall_block, "missing": missing, "files": {}}

    # 1. the Stage 2 code, as it stands in the tree that produced the SID
    if tree.is_dir():
        archive = staging / "stage2_source.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for relative in ("train_rqvae.py", "math_spec.json", "reference_contract.json"):
                candidate = tree / relative
                if candidate.is_file():
                    tar.add(candidate, arcname=relative)
            if (tree / "model").is_dir():
                tar.add(tree / "model", arcname="model")
        manifest["files"]["stage2_source.tar.gz"] = sha256(archive)
    # 2. the SID table
    if sid.is_file():
        shutil.copy2(sid, staging / "item_sids.json")
        manifest["files"]["item_sids.json"] = sha256(sid)
    # 3. the checkpoint that exported it
    if checkpoint.is_file():
        shutil.copy2(checkpoint, staging / "rqvae_best.pth")
        manifest["files"]["rqvae_best.pth"] = sha256(checkpoint)
    # 4. the Stage 2 run record
    if metrics.is_file():
        shutil.copy2(metrics, staging / "stage2_training_metrics.jsonl")
        manifest["files"]["stage2_training_metrics.jsonl"] = sha256(metrics)
    # 5. the complete Stage 3 record, every key of it
    if test_final is not None:
        shutil.copy2(test_final, staging / "stage3_test_final.json")
        manifest["files"]["stage3_test_final.json"] = sha256(test_final)
        for name, pattern in (("stage3_training_metrics.jsonl", "training_metrics.jsonl"),
                              ("stage3_run.log", "HG_Rec.log")):
            candidate = test_final.parent / pattern
            if candidate.is_file():
                shutil.copy2(candidate, staging / name)
                manifest["files"][name] = sha256(candidate)
    manifest["recall_keys"] = sorted(recall_block)
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest["files"]["manifest.json"] = sha256(staging / "manifest.json")
    return manifest


def upload(arm: str, staging: Path, *, dry_run: bool) -> str:
    target = "%s/%s/%s/%s/" % (R2_REMOTE, R2_PREFIX, arm,
                               time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    env = dict(os.environ)
    env.update(credentials())
    command = [str(RCLONE), "copy", str(staging), target, "--s3-no-check-bucket"]
    if dry_run:
        command.append("--dry-run")
    done = subprocess.run(command, capture_output=True, text=True, env=env, timeout=3600)
    if done.returncode != 0:
        raise SystemExit("upload failed: " + (done.stderr or done.stdout)[-800:])
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("arm")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--keep", action="store_true", help="keep the staging directory")
    args = parser.parse_args()

    staging = Path("/tmp") / ("r2_stage3_%s_%d" % (args.arm, int(time.time())))
    manifest = build_bundle(args.arm, staging, allow_incomplete=args.allow_incomplete)
    print("bundle for %s: %d files, recall keys %s"
          % (args.arm, len(manifest["files"]) - 1, manifest["recall_keys"]))
    if manifest["missing"]:
        print("  incomplete, missing:", manifest["missing"])
    target = upload(args.arm, staging, dry_run=args.dry_run)
    print(("would upload to " if args.dry_run else "uploaded to ") + target)
    if not args.keep:
        shutil.rmtree(staging, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
