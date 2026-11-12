"""The archive layout: one numbered directory per archived mechanism.

The owner's layout (2026-11-08): the bucket's first level is `01`, `02`, `03`, ...
and each of those holds exactly four things --

    <mechanism>/           the Stage 2 source that produced the identifier,
                           named after the mechanism itself
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

#: The four entries every numbered directory carries: the code folder carries the
#: mechanism's own name, so it is built per archive rather than fixed here.
SID = "item_sids.json"
CHECKPOINT = "rqvae_best.pth"
RECALL = "recall.json"


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


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def rclone(args: list[str], timeout: int = 3600, *, text: bool = True) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.update(credentials())
    return subprocess.run([str(RCLONE)] + args, capture_output=True, text=text,
                          env=env, timeout=timeout)


class Ledger:
    """Number to identity, and the checksum of everything uploaded."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else LEDGER_PATH
        self.entries = json.loads(self.path.read_text()) if self.path.is_file() else {}

    def number(self) -> int:
        used = [int(key) for key in self.entries if key.isdigit()] or [0]
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
        """Record what this upload staged, without forgetting what earlier ones put there.

        The archive holds more than any single staging set: numbers 48/49/50 were
        uploaded tree-and-all and later re-recorded from the narrow four-piece
        view, and a replacing write left 33 objects with no checksum. Merging keeps
        every path that was ever recorded and lets a re-upload refresh its own.
        """
        entry = self.entries.get(number, {})
        merged = dict(entry.get("files", {}))
        merged.update(files)
        entry["key"] = key
        entry["files"] = merged
        if extra:
            entry.update(extra)
        self.entries[number] = entry

    def remote_listing(self, number: str) -> list[tuple[str, int]]:
        """(path relative to the number, size) for every object under it."""
        done = rclone(["lsf", "%s/%s" % (BUCKET, number), "-R", "--files-only", "--format", "sp"])
        if done.returncode != 0:
            raise SystemExit("listing of %s failed: %s" % (number, (done.stderr or done.stdout)[-500:]))
        objects = []
        for line in done.stdout.splitlines():
            if not line.strip():
                continue
            size, relative = line.split(";", 1)
            objects.append((relative, int(size)))
        return objects

    def remote_bytes(self, number: str, relative: str) -> bytes:
        done = rclone(["cat", "%s/%s/%s" % (BUCKET, number, relative)], text=False)
        if done.returncode != 0:
            raise SystemExit("read of %s/%s failed: %s"
                             % (number, relative, (done.stderr or b"")[-300:]))
        return done.stdout

    def reconcile(self, number: str, *, listing=None, fetch=None,
                  dry_run: bool = False) -> dict:
        """Give every object in the bucket a checksum, and report the reverse gap.

        Returns the objects it added and the recorded paths the bucket does not
        have. Absent paths are reported, never deleted: the ledger is the record,
        and dropping a path because a listing came back short would destroy it.
        """
        listing = listing or self.remote_listing
        fetch = fetch or self.remote_bytes
        entry = self.entries.get(number, {})
        recorded = dict(entry.get("files", {}))
        objects = list(listing(number))
        names = [relative for relative, _ in objects]
        added = {}
        for relative, _size in objects:
            if relative in recorded:
                continue
            added[relative] = sha256_bytes(fetch(number, relative))
        if added and not dry_run:
            entry.setdefault("key", number)
            merged = dict(recorded)
            merged.update(added)
            entry["files"] = merged
            self.entries[number] = entry
        return {"added": added,
                "recorded_but_absent": sorted(p for p in recorded if p not in set(names))}

    def save(self) -> Path:
        self.path.write_text(json.dumps(self.entries, indent=2, sort_keys=True) + "\n")
        return self.path


def upload_directory(local: Path, number: str, *, dry_run: bool = False) -> str:
    target = "%s/%s/" % (BUCKET, number)
    command = ["copy", str(local), target, "--s3-no-check-bucket"]
    if dry_run:
        command.append("--dry-run")
    done = rclone(command)
    if done.returncode != 0:
        raise SystemExit("upload of %s failed: %s" % (number, (done.stderr or done.stdout)[-500:]))
    return target
