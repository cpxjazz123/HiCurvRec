"""Mirror numbered archive entries from R2 back to the local archive.

Some numbers were uploaded straight from the bucket side (69-74) and the local mirror never received
them; 24 has its recall block only in the bucket.  This pulls the objects a number holds and checks each
downloaded file against the ledger's sha256, so the local mirror is the same archive rather than a
partial copy.

    python3 results/research/issue139/download_archive.py --numbers 69-74
    python3 results/research/issue139/download_archive.py --numbers 24 --files recall.json
    python3 results/research/issue139/download_archive.py --numbers all --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RESEARCH = Path(__file__).resolve().parent
sys.path.insert(0, str(RESEARCH))

from r2_layout import BUCKET, Ledger, rclone, sha256  # noqa: E402

ARCHIVE = REPO / "analysis_sid_recall" / "r2"


def parse_numbers(text: str, ledger: Ledger) -> list[str]:
    live = sorted(key for key in ledger.entries if key.isdigit())
    if text == "all":
        return live
    wanted = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if "-" in chunk:
            start, end = chunk.split("-")
            wanted.extend("%02d" % value for value in range(int(start), int(end) + 1))
        elif chunk:
            wanted.append("%02d" % int(chunk))
    return [number for number in wanted if number in ledger.entries]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--numbers", default="all")
    parser.add_argument("--files", default="",
                        help="comma list of relative paths to pull (default: everything)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ledger = Ledger()
    numbers = parse_numbers(args.numbers, ledger)
    only = [name.strip() for name in args.files.split(",") if name.strip()]
    pulled = verified = mismatch = 0
    for number in numbers:
        objects = ledger.remote_listing(number)
        entry = ledger.entries.get(number, {})
        recorded = entry.get("files", {})
        target_dir = ARCHIVE / number
        for relative, size in objects:
            if only and relative not in only:
                continue
            destination = target_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            command = ["copyto", "%s/%s/%s" % (BUCKET, number, relative), str(destination),
                       "--s3-no-check-bucket"]
            if args.dry_run:
                command.append("--dry-run")
            done = rclone(command, timeout=1800)
            if done.returncode != 0:
                print(f"  {number}/{relative}: download failed: "
                      f"{(done.stderr or done.stdout)[-160:]}", flush=True)
                continue
            pulled += 1
            if args.dry_run:
                continue
            digest = sha256(destination)
            expected = recorded.get(relative)
            if expected is not None and digest != expected:
                mismatch += 1
                print(f"  {number}/{relative}: sha mismatch (ledger {expected[:12]}, "
                      f"file {digest[:12]})", flush=True)
            else:
                verified += 1
        print(f"  {number}: {len(objects)} objects "
              f"{'(filtered)' if only else ''} listed", flush=True)
    print(f"\nn{' would pull' if args.dry_run else ' pulled'} {pulled} files; "
          f"sha verified {verified}; mismatches {mismatch}; numbers {len(numbers)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
