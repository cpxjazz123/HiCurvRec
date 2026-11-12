"""The ledger must never shrink, and it must be able to catch up with the bucket.

Why this exists: numbers 48/49/50 were uploaded tree-and-all (9/23/26 files) and
later re-recorded from the narrow four-piece staging set. `Ledger.record` replaced
`files` wholesale, so 33 objects kept sitting in the bucket with no checksum in
the ledger, and nothing noticed. These tests pin the two halves of the fix --
`record` merges, and `reconcile` adds whatever the bucket has and the ledger does
not -- plus the staging directory rule (no /tmp).
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

import r2_layout  # noqa: E402

TMP = HERE / "_test_tmp"


def _fresh_ledger(**entries) -> r2_layout.Ledger:
    shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True)
    path = TMP / "ledger.json"
    path.write_text(json.dumps(entries))
    return r2_layout.Ledger(path)


def test_record_merges_instead_of_replacing() -> str:
    ledger = _fresh_ledger()
    ledger.record("48", "TIGER_BEHAVIOR_R48", {"a.py": "1", "b.py": "2"})
    ledger.record("48", "TIGER_BEHAVIOR_R48", {"b.py": "9", "c.py": "3"})
    files = ledger.entries["48"]["files"]
    expected = {"a.py": "1", "b.py": "9", "c.py": "3"}
    if files != expected:
        raise AssertionError("record replaced the file map: %r" % (files,))
    return "a re-record keeps the paths it did not stage, and refreshes the ones it did"


def test_record_keeps_provenance_it_was_not_given_again() -> str:
    ledger = _fresh_ledger()
    ledger.record("48", "TIGER_BEHAVIOR_R48", {"train_rqvae.py": "1"},
                  {"stage": "stage2-retrain", "recall_paired": False, "arm": "R48"})
    ledger.record("48", "TIGER_BEHAVIOR_R48", {"train_rqvae.py": "2"})
    entry = ledger.entries["48"]
    if entry.get("stage") != "stage2-retrain" or entry.get("recall_paired") is not False:
        raise AssertionError("re-record dropped provenance: %r" % (entry,))
    if entry["files"]["train_rqvae.py"] != "2":
        raise AssertionError("re-record did not refresh the checksum: %r" % (entry,))
    return "a re-record keeps the provenance fields it was not given again"


def test_reconcile_adds_objects_the_ledger_never_saw() -> str:
    ledger = _fresh_ledger()
    ledger.record("48", "TIGER_BEHAVIOR_R48", {"train_rqvae.py": "kept"})
    payload = b"print('grad check')\n"
    remote = {("scripts/grad_check.py", len(payload)): payload}

    result = ledger.reconcile(
        "48",
        listing=lambda number: sorted(remote),
        fetch=lambda number, relative: remote[(relative, len(payload))],
    )
    digest = hashlib.sha256(payload).hexdigest()
    if result["added"] != {"scripts/grad_check.py": digest}:
        raise AssertionError("reconcile did not add the missing object: %r" % (result,))
    if ledger.entries["48"]["files"].get("scripts/grad_check.py") != digest:
        raise AssertionError("reconcile did not write the checksum: %r" % (ledger.entries["48"],))
    if ledger.entries["48"]["files"].get("train_rqvae.py") != "kept":
        raise AssertionError("reconcile disturbed a recorded path")
    return "an object the ledger never saw gets a checksum, and recorded paths survive"


def test_reconcile_is_idempotent() -> str:
    ledger = _fresh_ledger()
    payload = b"x = 1\n"
    digest = hashlib.sha256(payload).hexdigest()
    ledger.record("48", "R48", {"scripts/grad_check.py": digest})
    calls = []
    result = ledger.reconcile(
        "48",
        listing=lambda number: [("scripts/grad_check.py", len(payload))],
        fetch=lambda number, relative: calls.append(relative) or payload,
    )
    if result["added"] or calls:
        raise AssertionError("reconcile refetched a recorded object: %r %r" % (result, calls))
    return "a second reconcile downloads nothing and adds nothing"


def test_reconcile_reports_recorded_files_that_are_absent() -> str:
    ledger = _fresh_ledger()
    ledger.record("48", "R48", {"gone.py": "abc"})
    result = ledger.reconcile("48", listing=lambda number: [], fetch=lambda n, r: b"")
    if result["recorded_but_absent"] != ["gone.py"]:
        raise AssertionError("an absent recorded file was not reported: %r" % (result,))
    if "gone.py" not in ledger.entries["48"]["files"]:
        raise AssertionError("reconcile deleted a recorded path instead of reporting it")
    return "a recorded path missing from the bucket is reported, never silently dropped"


def test_withdraw_retires_a_number_and_keeps_what_was_deleted() -> str:
    ledger = _fresh_ledger()
    ledger.record("51", "TIGER_bootstrap_code_stability__resample",
                  {"item_sids.json": "abc", "recall.json": "def"})
    ledger.record("74", "TIGER_l2k_rq", {"item_sids.json": "xyz"})
    ledger.withdraw("51", "recall measured on a destroyed SID")
    if "51" in ledger.entries:
        raise AssertionError("withdraw left the number in the live shelf")
    entry = ledger.entries.get("W51", {})
    if entry.get("retired") is not True or entry.get("retired_from") != "51":
        raise AssertionError("withdrawn entry is not marked: %r" % (entry,))
    if entry.get("deleted_files") != {"item_sids.json": "abc", "recall.json": "def"}:
        raise AssertionError("withdraw dropped the deleted checksums: %r" % (entry,))
    if "recall measured on a destroyed SID" not in entry.get("note", ""):
        raise AssertionError("withdraw dropped the reason: %r" % (entry,))
    if ledger.number() != 75:
        raise AssertionError("withdraw disturbed the number allocation: %r" % (ledger.number(),))
    return "a withdrawn number leaves the shelf, keeps its checksums, and frees no number for reuse"


def test_the_uploaders_stage_inside_the_repository() -> str:
    for name in ("upload_stage3.py", "upload_pairs.py"):
        source = (HERE / name).read_text()
        if '"/tmp"' in source or "'/tmp'" in source:
            raise AssertionError("%s still stages under /tmp (CLAUDE.md 86)" % name)
    return "both uploaders keep their staging directory inside the repository"


if __name__ == "__main__":
    failures = 0
    for test in (test_record_merges_instead_of_replacing,
                 test_record_keeps_provenance_it_was_not_given_again,
                 test_reconcile_adds_objects_the_ledger_never_saw,
                 test_reconcile_is_idempotent,
                 test_reconcile_reports_recorded_files_that_are_absent,
                 test_withdraw_retires_a_number_and_keeps_what_was_deleted,
                 test_the_uploaders_stage_inside_the_repository):
        try:
            note = test()
            print("ok   %s -- %s" % (test.__name__, note))
        except AssertionError as exc:
            failures += 1
            print("FAIL %s -- %s" % (test.__name__, exc))
    shutil.rmtree(TMP, ignore_errors=True)
    raise SystemExit(1 if failures else 0)
