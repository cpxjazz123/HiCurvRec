"""Byte-exact SID attestation for Stage 3 (Issue #129, WP0).

The Stage 3 protocol is frozen and is not modified. This module is the verifier a
prospective run must call at start-up, at every checkpoint and before the final
test: the SID table is resolved once, its SHA-256 and its item-id order digest
are pinned, and every later use re-checks the bytes on disk and fails closed on a
mismatch. A matching vocabulary size is never accepted as identity, and a mutable
table is never re-read.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def digest_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def items_digest(table: dict) -> str:
    """A digest of the item ids in their mapping order, independent of the codes."""
    digest = hashlib.sha256()
    for item in table:
        digest.update(str(int(item)).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


class AttestedSID:
    """The SID table, pinned by hash and item order, loaded exactly once."""

    def __init__(self, path: Path, *, expected_sha: str | None = None,
                 expected_items: str | None = None) -> None:
        self.path = Path(path).resolve(strict=True)
        payload = self.path.read_bytes()
        self.sha = hashlib.sha256(payload).hexdigest()
        if expected_sha is not None and self.sha != expected_sha:
            raise ValueError(f"the SID hash changed: {self.sha} != {expected_sha}")
        self.table = json.loads(payload)
        self.items = items_digest(self.table)
        if expected_items is not None and self.items != expected_items:
            raise ValueError("the item mapping/order does not match the pinned digest")
        self._payload = payload

    def verify(self, *, expected_sha: str | None = None,
               expected_items: str | None = None) -> None:
        """Re-check the bytes on disk; a table that changed fails closed."""
        on_disk = digest_file(self.path)
        if on_disk != self.sha:
            raise ValueError("the SID file changed after it was attested")
        if expected_sha is not None and expected_sha != self.sha:
            raise ValueError("the caller's expected hash does not match the attested one")
        if expected_items is not None and expected_items != self.items:
            raise ValueError("the caller's expected item order does not match")
