"""Acceptance tests for the immutable evidence store (issue #105, workstream A).

They run with the framework's interpreter, because the package needs ``harness2`` and
Python 3.12:

    /home/wlia0047/ar57_scratch/wenyu/harness2/.venv/bin/python -m pytest \
        tools/harness2_genrec/tests/test_evidence.py -q

Zero GPU, zero network, zero credentials. Every test here is about a REFUSAL: the store
exists so that a run cannot quietly reuse evidence, write over another run's artifacts, or
report success without a report.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.harness2_genrec import evidence  # noqa: E402


# --------------------------------------------------------------------------------------
# the store
# --------------------------------------------------------------------------------------
def _plan(tmp_path: Path, **overrides) -> evidence.RunPlan:
    plan = {
        "kind": "arm",
        "mechanism_id": "test_mechanism",
        "tree": None,
        "output_dir": tmp_path / "out",
        "seed": 42,
    }
    plan.update(overrides)
    return evidence.RunPlan(**plan)


def test_a_run_gets_an_id_a_directory_and_a_manifest(tmp_path):
    store = evidence.EvidenceStore(tmp_path / "store").open()
    run = store.admit(_plan(tmp_path))
    manifest = store.close(run, status="succeeded", metrics={"er": 1.0})
    assert manifest["run_id"] == run.run_id
    assert manifest["status"] == "succeeded"
    assert manifest["manifest_sha256"] == evidence.manifest_digest(manifest)
    assert store.manifest_path(run.run_id).is_file()
    ok, problems = store.verify(run.run_id)
    assert ok, problems


def test_a_manifest_is_immutable(tmp_path):
    store = evidence.EvidenceStore(tmp_path / "store").open()
    run = store.admit(_plan(tmp_path))
    store.close(run, status="succeeded")
    with pytest.raises(evidence.EvidenceError):
        store.close(run, status="succeeded")


def test_verify_detects_a_changed_artifact(tmp_path):
    store = evidence.EvidenceStore(tmp_path / "store").open()
    artifact = tmp_path / "artifact.json"
    artifact.write_text('{"a": 1}')
    run = store.admit(_plan(tmp_path))
    store.close(run, status="succeeded", artifacts=[artifact])
    artifact.write_text('{"a": 2}')
    ok, problems = store.verify(run.run_id)
    assert not ok
    assert any("changed since the manifest" in problem for problem in problems)


def test_verify_detects_an_edited_manifest(tmp_path):
    store = evidence.EvidenceStore(tmp_path / "store").open()
    run = store.admit(_plan(tmp_path))
    store.close(run, status="succeeded")
    path = store.manifest_path(run.run_id)
    manifest = json.loads(path.read_text())
    manifest["metrics"] = {"er": 99.0}
    path.write_text(json.dumps(manifest))
    ok, problems = store.verify(run.run_id)
    assert not ok
    assert any("digest does not match" in problem for problem in problems)


def test_a_run_refuses_an_existing_output_directory(tmp_path):
    store = evidence.EvidenceStore(tmp_path / "store").open()
    output = tmp_path / "out"
    output.mkdir(parents=True)
    (output / "leftover").write_text("x")
    with pytest.raises(evidence.EvidenceError):
        store.admit(_plan(tmp_path, output_dir=output))


def test_a_run_refuses_a_tree_that_changed_after_its_verdict(tmp_path):
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "train_rqvae.py").write_text("x = 1\n")
    store = evidence.EvidenceStore(tmp_path / "store").open()
    digest, _count = evidence.tree_digest(tree)
    (tree / "train_rqvae.py").write_text("x = 2\n")
    with pytest.raises(evidence.EvidenceError):
        store.admit(_plan(tmp_path, tree=tree), verdict_tree_digest=digest)


def test_overlapping_output_paths_are_refused(tmp_path):
    with pytest.raises(evidence.EvidenceError):
        evidence.refuse_overlapping_paths([tmp_path / "a", tmp_path / "a" / "b"])
    with pytest.raises(evidence.EvidenceError):
        evidence.refuse_overlapping_paths([tmp_path / "a", tmp_path / "a"])
    evidence.refuse_overlapping_paths([tmp_path / "a", tmp_path / "b"])


def test_lineage_is_recorded(tmp_path):
    store = evidence.EvidenceStore(tmp_path / "store").open()
    parent = store.admit(_plan(tmp_path, mechanism_id="parent"))
    store.close(parent, status="succeeded")
    child = store.admit(
        _plan(tmp_path, mechanism_id="child", output_dir=tmp_path / "child_out")
    )
    store.close(child, status="succeeded", parent_runs=[parent.run_id])
    assert store.lineage(child.run_id)[-1] == child.run_id
    assert parent.run_id in store.lineage(child.run_id)


# --------------------------------------------------------------------------------------
# fail-closed subprocess execution
# --------------------------------------------------------------------------------------
def test_a_nonzero_exit_is_a_failure_with_a_reason(tmp_path):
    result = evidence.run_checked(
        [sys.executable, "-c", "import sys; sys.exit(3)"], cwd=tmp_path
    )
    assert not result.ok
    assert "status 3" in (result.failure_reason or "")


def test_a_missing_report_is_a_failure(tmp_path):
    result = evidence.run_checked(
        [sys.executable, "-c", "print('done')"],
        cwd=tmp_path,
        report_path=tmp_path / "absent.json",
        report_keys=("ok",),
    )
    assert not result.ok
    assert "wrote no report" in (result.failure_reason or "")


def test_a_report_without_its_keys_is_a_failure(tmp_path):
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"ok": True}))
    result = evidence.run_checked(
        [sys.executable, "-c", "print('done')"],
        cwd=tmp_path,
        report_path=report,
        report_keys=("ok", "metrics"),
    )
    assert not result.ok
    assert "no 'metrics'" in (result.failure_reason or "")


def test_a_report_with_too_few_samples_is_a_failure(tmp_path):
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"ok": True, "samples": 3}))
    result = evidence.run_checked(
        [sys.executable, "-c", "print('done')"],
        cwd=tmp_path,
        report_path=report,
        report_keys=("ok",),
        min_samples=10,
    )
    assert not result.ok
    assert "fewer than the protocol's 10" in (result.failure_reason or "")


def test_a_missing_artifact_is_a_failure(tmp_path):
    result = evidence.run_checked(
        [sys.executable, "-c", "print('done')"],
        cwd=tmp_path,
        artifact_paths=[tmp_path / "missing.npy"],
    )
    assert not result.ok
    assert "expected artifact is missing" in (result.failure_reason or "")


def test_a_clean_run_is_ok(tmp_path):
    report = tmp_path / "report.json"
    result = evidence.run_checked(
        [
            sys.executable,
            "-c",
            f"import json; json.dump({{'ok': True, 'samples': 12}}, open({str(report)!r}, 'w'))",
        ],
        cwd=tmp_path,
        report_path=report,
        report_keys=("ok", "samples"),
        min_samples=10,
    )
    assert result.ok, result.failure_reason
    assert result.report["samples"] == 12
