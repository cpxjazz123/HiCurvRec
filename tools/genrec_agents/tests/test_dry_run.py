"""The dry run's tests (issue #110, section 7).

The dry run has to walk the whole flow, it has to mark the stages that are mocked, and it
must never present itself as a scientific success: the success predicate is asked, it
refuses, and the refusal is in the output.

    /home/wlia0047/ar57_scratch/wenyu/harness2/.venv/bin/python -m pytest \
        tools/genrec_agents/tests/test_dry_run.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# the dry run runs the geometry gates, which need torch
pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.genrec_agents import dry_run  # noqa: E402


def test_the_dry_run_walks_every_stage(tmp_path):
    report = dry_run.run_dry_run(out_dir=tmp_path)
    for stage in dry_run.STAGES:
        assert stage in report["stages"], stage
    assert report["stages"]["search"]["state"] == "searched"
    assert report["stages"]["prior_art"]["allowed"] is True
    assert report["stages"]["implementation"]["status"] == "implemented"
    assert report["stages"]["resume"]["reserved_experiments"] == 1


def test_the_gpu_stages_are_marked_as_mocked(tmp_path):
    report = dry_run.run_dry_run(out_dir=tmp_path)
    for stage in dry_run.MOCKED_STAGES:
        assert report["stages"][stage]["mocked"] is True, stage
        assert report["stages"][stage]["state"] == "mocked", stage


def test_the_dry_run_is_never_a_scientific_success(tmp_path):
    report = dry_run.run_dry_run(out_dir=tmp_path)
    assert report["dry_run"] is True
    assert report["is_scientific_success"] is False
    assert "must never be recorded as one" in report["warning"]
    # and the success predicate is asked and refuses
    assert report["stages"]["success_predicate"]["won"] is False
    assert report["stages"]["success_predicate"]["reason"]


def test_the_artefact_is_written(tmp_path):
    report = dry_run.run_dry_run(out_dir=tmp_path)
    document = json.loads((tmp_path / "dry_run.json").read_text())
    assert document["stages"] == report["stages"]
    assert document["is_scientific_success"] is False


def test_the_geometry_gates_run_on_the_synthetic_mechanism(tmp_path):
    report = dry_run.run_dry_run(out_dir=tmp_path)
    gates = report["stages"]["math_review"]["gates"]
    assert gates["H1"] == "pass"
    assert gates["H2"] == "pass", "the synthetic mechanism calls a genuinely hyperbolic operator"
    assert gates["H7"] == "pass"
