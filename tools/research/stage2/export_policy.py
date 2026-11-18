"""Decide whether a SID may enter Stage 3 given its export manifest.

The failure this guards against: a SID exported from the live in-memory module is
not the table the saved checkpoint produces.  Rotating a handful of near-tie
items is enough to change the downstream recall, so the archive's
(checkpoint, SID) pair stops being reproducible -- which is exactly what the
2026-11-18 audit found on 12 of 14 locally checkable arms.

Policy, in order:

* a manifest that names this SID and reports ``tokens_match`` -> VERIFIED, allow;
* a manifest that names another SID, is unreadable, or reports a mismatch ->
  REFUSED (a wrong table must not silently become a Stage 3 baseline);
* no manifest and the SID is in the R2 ledger -> UNVERIFIED, allow, because those
  numbers were archived before the guard existed and are already measured;
* no manifest and not archived -> REFUSED: run :mod:`export_check` first.

This module is standard library only so the MCP server can decide without torch.
"""
from __future__ import annotations

import json
from pathlib import Path

MANIFEST_NAME = "export_manifest.json"
SCHEMA = "genrec.stage2.export_manifest/1"
LEDGER_RELATIVE = Path("results") / "research" / "issue139" / "r2_ledger.json"


def manifest_path(sid: Path) -> Path:
    """The manifest sits beside the SID table it describes."""
    return Path(sid).parent / MANIFEST_NAME


def read_manifest(sid: Path) -> dict | None:
    path = manifest_path(sid)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"unreadable": True}


def archived_sids(repo: Path) -> set[str]:
    """The SID sha256 of every number already in the R2 ledger."""
    path = Path(repo) / LEDGER_RELATIVE
    if not path.is_file():
        return set()
    try:
        ledger = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    found = set()
    for entry in ledger.values():
        if not isinstance(entry, dict):
            continue
        sha = (entry.get("files") or {}).get("item_sids.json")
        if sha:
            found.add(sha)
    return found


def decide(sid_sha256: str, manifest: dict | None, archived: set[str]) -> dict:
    """One verdict per SID, with the reason a refusal happened."""
    if manifest is None:
        if sid_sha256 in archived:
            return {"allow": True, "state": "UNVERIFIED",
                    "reason": "archived before the export guard existed; the SID is authoritative"}
        return {"allow": False, "state": "REFUSED",
                "reason": ("no export_manifest.json beside the SID; run "
                           "tools/research/stage2/export_check.py first")}
    if manifest.get("unreadable"):
        return {"allow": False, "state": "REFUSED",
                "reason": "export_manifest.json is unreadable"}
    if manifest.get("schema") != SCHEMA:
        return {"allow": False, "state": "REFUSED",
                "reason": "unexpected manifest schema %r" % manifest.get("schema")}
    if manifest.get("sid_sha256") != sid_sha256:
        return {"allow": False, "state": "REFUSED",
                "reason": "the manifest describes a different SID table"}
    if not manifest.get("tokens_match"):
        message = ("the checkpoint does not reproduce the SID: %s of %s items differ"
                   % (manifest.get("original_differing_items", manifest.get("differing_items")),
                      manifest.get("items")))
        # An archived table is authoritative for the recall already measured on it;
        # the mismatch is reported, not turned into a second refusal.
        if sid_sha256 in archived:
            return {"allow": True, "state": "UNVERIFIED", "reason": message}
        return {"allow": False, "state": "REFUSED", "reason": message}
    return {"allow": True, "state": "VERIFIED",
            "reason": "the checkpoint reproduces the SID"}
