"""The archive layout: one numbered directory per archived mechanism.

The owner's layout (2026-11-08): the bucket's first level is `01`, `02`, `03`, ...
and each of those holds exactly four things --

    stage2_code/           the Stage 2 source that produced the identifier
    item_sids.json         the identifier table
    rqvae_best.pth         the checkpoint that exported it
    recall.json            every key of the Stage 3 result

No issue names, no arm names, no nesting beyond that. What a number refers to is
recorded here, in a ledger that lives in the repository, so the bucket stays
anonymous while the provenance stays auditable; the checksums of every uploaded
file are in the ledger too, because the four-entry layout has no room for a
manifest.

    from r2_layout import Ledger
    ledger = Ledger()
    number = ledger.assign("TIGER_successor_entropy_curvature/9e1a7c3fc3e7")
    ledger.save()
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LEDGER_PATH = Path(__file__).resolve().parent / "r2_ledger.json"
RCLONE = Path("/home/wlia0047/bin/rclone")
BUCKET = "r2:genrec"
CREDENTIALS = REPO / "r2_apikey"

#: The four entries every numbered directory carries, in this order.
STAGE2_CODE = "stage2_code"
SID = "item_sids.json"
CHECKPOINT = "rqvae_best.pth"
RECALL = "recall.json"
LAYOUT = (STAGE2_CODE, SID, CHECKPOINT, RECALL)


def credentials() -> dict:
    raw = CREDENTIALS.read_text()

    def grab(label: str) -> str:
        match = re.search(re.escape(label) + r"\s*=\s*(\S+)", raw)
        if not match:
            raise SystemExit("r2_apikey is missing the line %r" % label)
        return match.group(1)

    return {"RCLONE_CONFIG_R2_TYPE": "s3", "RCLONE_CONFIG_R2_PROVIDER": "Cloudflare",
            "RCLONE_CONFIG_R2_ACCESS_KEY_ID": grab("Access Key ID"),
            "RCLONE_CONFIG_R2_SECRET_ACCESS_KEY": grab("Secret Access Key"),
            "RCLONE_CONFIG_R2_ENDPOINT": grab("S3 API"),
            "RCLONE_CONFIG_R2_REGION": "auto"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rclone(args: list[str], timeout: int = 3600) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.update(credentials())
    return subprocess.run([str(RCLONE)] + args, capture_output=True, text=True,
                          env=env, timeout=timeout)


class Ledger:
    """Number to identity, and the checksum of everything uploaded."""

    def __init__(self) -> None:
        self.entries = json.loads(LEDGER_PATH.read_text()) if LEDGER_PATH.is_file() else {}

    def number(self) -> int:
        used = [int(key) for key in self.entries] or [0]
        return max(used) + 1

    def assign(self, key: str) -> str:
        """The two-digit number for this key, allocated once."""
        for number, entry in self.entries.items():
            if entry.get("key") == key:
                return number
        number = "%02d" % self.number()
        self.entries[number] = {"key": key}
        return number

    def record(self, number: str, key: str, files: dict, extra: dict | None = None) -> None:
        entry = {"key": key, "files": files}
        if extra:
            entry.update(extra)
        self.entries[number] = entry

    def save(self) -> Path:
        LEDGER_PATH.write_text(json.dumps(self.entries, indent=2, sort_keys=True) + "\n")
        return LEDGER_PATH


def upload_directory(local: Path, number: str, *, dry_run: bool = False) -> str:
    target = "%s/%s/" % (BUCKET, number)
    command = ["copy", str(local), target, "--s3-no-check-bucket"]
    if dry_run:
        command.append("--dry-run")
    done = rclone(command)
    if done.returncode != 0:
        raise SystemExit("upload of %s failed: %s" % (number, (done.stderr or done.stdout)[-500:]))
    return target
