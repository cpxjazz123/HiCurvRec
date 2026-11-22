"""The Stage 3 export gate: which SID may pass, and why it is refused.

The bug this pins: before 2026-11-18 a trainer exported the SID from the live
in-memory module, so the saved checkpoint could not reproduce the table (12 of 14
locally checkable arms differ on 0.3-0.5% of items; 12 of 65 archived numbers
differ on thousands).  The gate must pass only a table the checkpoint reproduces,
grandfather the numbers already in the R2 ledger, and refuse everything else.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from tools.research.stage2.export_policy import (SCHEMA, archived_sids, decide,
                                                 manifest_path, read_manifest)

SID_SHA = "a" * 64


def manifest(**overrides):
    base = {"schema": SCHEMA, "sid_sha256": SID_SHA, "tokens_match": True,
            "items": 10, "differing_items": 0}
    base.update(overrides)
    return base


def test_a_matching_manifest_passes() -> None:
    verdict = decide(SID_SHA, manifest(), set())
    assert verdict == {"allow": True, "state": "VERIFIED",
                       "reason": "the checkpoint reproduces the SID"}


def test_an_archived_table_with_a_mismatch_is_unverified_not_refused() -> None:
    verdict = decide(SID_SHA, manifest(tokens_match=False, differing_items=7), {SID_SHA})
    assert verdict["allow"] is True and verdict["state"] == "UNVERIFIED"
    assert "7 of 10" in verdict["reason"]


def test_a_mismatch_is_refused_with_the_count() -> None:
    verdict = decide(SID_SHA, manifest(tokens_match=False, differing_items=7), set())
    assert verdict["allow"] is False and verdict["state"] == "REFUSED"
    assert "7 of 10" in verdict["reason"]


def test_a_manifest_for_another_table_is_refused() -> None:
    verdict = decide(SID_SHA, manifest(sid_sha256="b" * 64), set())
    assert verdict["allow"] is False
    assert "different SID table" in verdict["reason"]


def test_an_unreadable_manifest_is_refused() -> None:
    verdict = decide(SID_SHA, {"unreadable": True}, set())
    assert verdict["allow"] is False
    assert "unreadable" in verdict["reason"]


def test_an_unknown_schema_is_refused() -> None:
    verdict = decide(SID_SHA, manifest(schema="something-else"), set())
    assert verdict["allow"] is False
    assert "schema" in verdict["reason"]


def test_no_manifest_passes_only_for_an_archived_number() -> None:
    refused = decide(SID_SHA, None, set())
    assert refused["allow"] is False
    assert "export_check" in refused["reason"]

    passed = decide(SID_SHA, None, {SID_SHA})
    assert passed["allow"] is True and passed["state"] == "UNVERIFIED"


def test_archived_sids_reads_the_ledger(tmp_path: Path) -> None:
    repo = tmp_path
    ledger_dir = repo / "results" / "research" / "issue139"
    ledger_dir.mkdir(parents=True)
    (ledger_dir / "r2_ledger.json").write_text(json.dumps({
        "75": {"files": {"item_sids.json": "c" * 64, "recall.json": "d" * 64}},
        "W56": {"files": {}},
        "77": {"arm": "x"},
    }), encoding="utf-8")
    assert archived_sids(repo) == {"c" * 64}


def test_manifest_lives_beside_the_sid(tmp_path: Path) -> None:
    sid = tmp_path / "item_sids.json"
    sid.write_text("{}", encoding="utf-8")
    assert manifest_path(sid) == tmp_path / "export_manifest.json"
    assert read_manifest(sid) is None
