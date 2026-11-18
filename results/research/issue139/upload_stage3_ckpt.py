"""Upload the Stage 3 checkpoint that produced each archived recall.

The numbered archive already holds the four required pieces (source, SID, Stage 2 codebook, recall
block).  The recall block names the `HG_Rec_best.pth` its test run reloaded, and until now that file
stayed only on the training host -- so the archive could not reproduce its own recall.  This adds it as
`<NN>/HG_Rec_best.pth`, records its checksum in the ledger next to the other four, and skips an entry
whose remote copy already matches.

    python3 results/research/issue139/upload_stage3_ckpt.py --dry-run
    python3 results/research/issue139/upload_stage3_ckpt.py --limit 2
    python3 results/research/issue139/upload_stage3_ckpt.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
#: the tree is reachable through two roots on this host; the archive must record one label
ROOTS = (str(REPO), "/home/wlia0047/ar57/wenyu/GeneRec")
RESEARCH = Path(__file__).resolve().parent
sys.path.insert(0, str(RESEARCH))

from r2_layout import BUCKET, Ledger, rclone, sha256  # noqa: E402

ARCHIVE = REPO / "analysis_sid_recall" / "r2"
NAME = "HG_Rec_best.pth"


def relative_label(path: Path) -> str:
    """The repository-relative label, whichever of the two roots the path came through."""
    text = str(path)
    for root in ROOTS:
        if text.startswith(root + "/"):
            return text[len(root) + 1:]
    return text


def remote_size(number: str) -> int | None:
    done = rclone(["lsf", "%s/%s" % (BUCKET, number), "--files-only", "--format", "sp"])
    if done.returncode != 0:
        raise SystemExit("listing of %s failed: %s" % (number, (done.stderr or done.stdout)[-400:]))
    for line in done.stdout.splitlines():
        if ";" not in line:
            continue
        size, relative = line.split(";", 1)
        if relative.strip() == NAME:
            return int(size)
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    ledger = Ledger()
    numbers = sorted(key for key in ledger.entries if key.isdigit())
    if args.limit:
        numbers = numbers[:args.limit]
    uploaded = skipped = missing = failed = 0
    for number in numbers:
        entry = ledger.entries[number]
        recall_path = ARCHIVE / number / "recall.json"
        if recall_path.is_file():
            block = json.loads(recall_path.read_text())
        else:
            # seven entries carry their recall only in the bucket (the local mirror predates them);
            # the remote block still names the checkpoint, so it is read back rather than guessed
            try:
                block = json.loads(ledger.remote_bytes(number, "recall.json").decode())
            except SystemExit:
                print(f"  {number}: no recall block locally or remotely", flush=True)
                missing += 1
                continue
        checkpoint = block.get("best_checkpoint")
        if not checkpoint or not Path(checkpoint).is_file():
            print(f"  {number}: recall names no existing checkpoint ({checkpoint})", flush=True)
            missing += 1
            continue
        path = Path(checkpoint)
        digest = sha256(path)
        recorded = entry.get("stage3_checkpoint", {}).get("sha256")
        size = path.stat().st_size
        existing = remote_size(number)
        if existing == size and recorded == digest:
            skipped += 1
            continue
        if existing == size and recorded != digest:
            # same size, different checksum: the local file is the authority, re-upload
            print(f"  {number}: remote size matches but ledger sha differs; re-uploading", flush=True)
        if args.verify_only:
            print(f"  {number}: remote size {existing} vs local {size} (sha {digest[:12]})", flush=True)
            continue
        target = "%s/%s/%s" % (BUCKET, number, NAME)
        command = ["copyto", str(path), target, "--s3-no-check-bucket"]
        if args.dry_run:
            command.append("--dry-run")
        done = rclone(command, timeout=1800)
        if done.returncode != 0:
            print(f"  {number}: upload failed: {(done.stderr or done.stdout)[-200:]}", flush=True)
            failed += 1
            continue
        if not args.dry_run:
            ledger.record(number, entry.get("key", ""), {NAME: digest},
                          {"stage3_checkpoint": {"source": relative_label(path),
                                                 "sha256": digest, "bytes": size,
                                                 "uploaded": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                           time.gmtime())}})
        uploaded += 1
        if uploaded % 10 == 1:
            print(f"  [{uploaded}] {number} {size/2**20:.1f} MiB -> {target}", flush=True)
    if not args.dry_run:
        ledger.save()
    print(f"\nnumbers {len(numbers)} | {'would upload' if args.dry_run else 'uploaded'} {uploaded} "
          f"| already there {skipped} | missing {missing} | failed {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
