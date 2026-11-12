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


def test_renumber_moves_a_number_and_rewrites_its_bucket_path() -> str:
    ledger = _fresh_ledger()
    ledger.record("57", "TIGER_struct_mixed_rq", {"item_sids.json": "a"},
                  {"target": "r2:genrec/57/", "arm": "TIGER_struct_mixed_rq"})
    ledger.renumber("57", "51")
    if "57" in ledger.entries:
        raise AssertionError("renumber left the old number behind")
    entry = ledger.entries.get("51", {})
    if entry.get("files") != {"item_sids.json": "a"}:
        raise AssertionError("renumber lost the file map: %r" % (entry,))
    if entry.get("target") != "r2:genrec/51/":
        raise AssertionError("renumber did not rewrite the bucket path: %r" % (entry,))
    if entry.get("renumbered_from") != "57":
        raise AssertionError("renumber did not record where it came from: %r" % (entry,))
    return "a renumbered entry keeps its files and points at its new directory"


def test_renumber_refuses_a_number_that_is_taken_or_retired() -> str:
    ledger = _fresh_ledger()
    ledger.record("51", "kept", {"item_sids.json": "k"})
    ledger.record("57", "moved", {"item_sids.json": "m"})
    try:
        ledger.renumber("57", "51")
    except RuntimeError as exc:
        if "51" not in str(exc):
            raise AssertionError("the refusal does not name the taken number: %s" % exc)
    else:
        raise AssertionError("renumber overwrote a live number")
    if ledger.entries["57"].get("files") != {"item_sids.json": "m"}:
        raise AssertionError("the refused move damaged the source entry")
    ledger.entries["W09"] = {"retired": True, "retired_from": "09"}
    try:
        ledger.renumber("W09", "09")
    except RuntimeError:
        pass
    else:
        raise AssertionError("renumber moved a retired record onto the live shelf")
    return "renumber refuses a taken target and a retired source"


def test_the_archive_key_names_the_sid_it_holds() -> str:
    """A re-run of an arm produces a new SID, so it must not overwrite the old entry."""
    import shutil
    import tempfile
    sys.path.insert(0, str(HERE))
    import upload_stage3

    sid = REPO / "results" / "stage2_RQ-VAE" / "TIGER_RQ-VAE" / "item_sids.json"
    if not sid.is_file():
        return "the baseline SID is not present; nothing to key"
    first = upload_stage3.archive_key("TIGER_RQ-VAE", sid)
    again = upload_stage3.archive_key("TIGER_RQ-VAE", sid)
    if first != again:
        raise AssertionError("the same SID produced two keys: %r vs %r" % (first, again))
    if not first.startswith("TIGER_RQ-VAE/"):
        raise AssertionError("the key does not name the arm: %r" % first)
    with tempfile.TemporaryDirectory(dir=REPO / "results" / "research") as folder:
        other = Path(folder) / "item_sids.json"
        other.write_text('{"0": [1, 2, 3, 4]}')
        different = upload_stage3.archive_key("TIGER_RQ-VAE", other)
    if different == first:
        raise AssertionError("two different SIDs shared one key: %r" % first)
    return "the key is the arm plus the SID digest, so a re-run gets its own number"


def test_the_staged_source_is_the_code_the_trainer_imports() -> str:
    """The archive must hold the tree's own modules, not a fixed three-file list."""
    import sys
    import tempfile
    sys.path.insert(0, str(HERE))
    import upload_stage3

    sid = REPO / "results" / "stage2_RQ-VAE" / "TIGER_RQ-VAE" / "item_sids.json"
    tree = REPO / "stage2_RQ-VAE" / "TIGER_RQ-VAE"
    if not sid.is_file() or not tree.is_dir():
        return "the baseline tree or its SID is absent; nothing to stage"
    with tempfile.TemporaryDirectory(dir=REPO / "results" / "research") as folder:
        files, _key = upload_stage3.stage("TIGER_RQ-VAE", Path(folder))
    staged = set(files)
    root = "TIGER_RQ-VAE/"
    required = [root + name for name in ("train_rqvae.py", "train_config.py",
                                         "config/train_config.json")]
    missing = [name for name in required if name not in staged]
    for prefix in ("losses/", "distances/", "model/"):
        if not any(name.startswith(root + prefix) for name in staged):
            missing.append(root + prefix + "*")
    if missing:
        raise AssertionError("the staged source misses %r, so the archive cannot run"
                             % (missing,))
    inside_source = [name for name in staged if name.startswith(root)]
    if any(name.endswith((".pth", ".pt", ".npy", ".parquet")) for name in inside_source):
        raise AssertionError("the staged source tree carries a binary artifact")
    return "the archive carries every module the trainer imports, and no artifacts"


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
                 test_the_archive_key_names_the_sid_it_holds,
                 test_the_staged_source_is_the_code_the_trainer_imports,
                 test_renumber_moves_a_number_and_rewrites_its_bucket_path,
                 test_renumber_refuses_a_number_that_is_taken_or_retired,
                 test_the_uploaders_stage_inside_the_repository):
        try:
            note = test()
            print("ok   %s -- %s" % (test.__name__, note))
        except AssertionError as exc:
            failures += 1
            print("FAIL %s -- %s" % (test.__name__, exc))
    shutil.rmtree(TMP, ignore_errors=True)
    raise SystemExit(1 if failures else 0)
