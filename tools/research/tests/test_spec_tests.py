"""A Math Reviewer's finding is answered with a test, and the test is forever.

The round does not accept a finding on prose. The reviewer writes a test that
fails on the specification carrying the defect; the framework proves it fails,
writes it to ``tools/research/rounds/spec_tests/`` and runs every retained test
before every later freeze. These tests prove the three parts of that contract:
a real test is kept, a test that passes on the defect is refused, and a finding
that arrives with no test is refused as well.
"""
from __future__ import annotations

import json
from pathlib import Path

from tools.research.rounds import spec_tests, worker

REPO = Path(__file__).resolve().parents[3]
#: The frozen specification of the round that produced the balancing eval is a
#: FIXTURE: the research state it came from is not part of the repository, and a
#: test that depends on a scratch directory stops running when it is cleaned.
FROZEN = (REPO / "tools" / "research" / "tests" / "fixtures"
          / "round_20261105_144909_130" / "math_spec.json")


def _spec() -> dict:
    return json.loads(FROZEN.read_text(encoding="utf-8"))


#: A test that refuses a specification whose lift statement is missing.
REAL_TEST = '''
def check(spec):
    statements = (spec.get("continuous_surface") or {}).get("statements") or []
    if not any("def lift" in line for line in statements):
        return ["the surface defines no lift"]
    return []
'''

#: A test that passes on every specification: not a test of anything.
TOOTHLESS = '''
def check(spec):
    return []
'''


def test_a_retained_test_passes_on_the_frozen_specification():
    result = spec_tests.run(_spec())
    assert result["passed"], result["failures"]
    assert result["checked"] >= 1


def test_a_retained_test_refuses_the_defect_it_names():
    """The suite's domain test must refuse a specification with no domain."""
    broken = _spec()
    for entry in broken["computations"]:
        if entry.get("name") == "distance":
            entry.pop("domain", None)
    result = spec_tests.run(broken)
    assert not result["passed"]
    assert any("domain" in problem for problem in result["findings"])


def test_every_retained_test_carries_its_finding_and_round():
    records = spec_tests.load()
    assert records, "the suite keeps at least one test"
    for record in records:
        assert record.get("finding"), record
        assert record.get("round"), record


def test_a_real_test_is_kept(tmp_path):
    kept = spec_tests.record(tmp_path, "round_x", "the surface defines no lift",
                             REAL_TEST)
    assert kept["written"] and Path(kept["path"]).exists()
    text = Path(kept["path"]).read_text(encoding="utf-8")
    assert "FINDING" in text and "def check(spec)" in text


def test_a_repeated_finding_does_not_duplicate_its_test(tmp_path):
    first = spec_tests.record(tmp_path, "round_x", "the surface defines no lift",
                              REAL_TEST)
    second = spec_tests.record(tmp_path, "round_x", "the surface defines no lift",
                               REAL_TEST)
    assert second["written"] is False
    assert second["path"] == first["path"]


def test_the_reviewer_must_prove_the_test_fails_on_the_specification(tmp_path,
                                                                    monkeypatch):
    """A test that passes on the defect is refused: it is not a test of it."""
    monkeypatch.setattr(spec_tests, "DIRECTORY", tmp_path)
    spec = _spec()
    broken = json.loads(json.dumps(spec))
    broken["continuous_surface"]["statements"] = [
        line for line in broken["continuous_surface"]["statements"]
        if "def lift" not in line]
    written, refused = worker._retain_spec_tests(
        tmp_path, "round_x",
        [{"verdict": "REVISE",
          "findings": ["the surface defines no lift"],
          "tests": [{"finding": "the surface defines no lift", "code": TOOTHLESS}]}],
        broken, log=lambda *a, **k: None)
    assert written == []
    assert refused and "does not refuse the defect" in refused[0]


def test_a_test_that_does_refuse_is_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(spec_tests, "DIRECTORY", tmp_path)
    broken = _spec()
    broken["continuous_surface"]["statements"] = [
        line for line in broken["continuous_surface"]["statements"]
        if "def lift" not in line]
    written, refused = worker._retain_spec_tests(
        tmp_path, "round_x",
        [{"verdict": "REVISE",
          "findings": ["the surface defines no lift"],
          "tests": [{"finding": "the surface defines no lift", "code": REAL_TEST}]}],
        broken, log=lambda *a, **k: None)
    assert not refused
    assert written and Path(written[0]["path"]).exists()


def test_a_finding_with_no_test_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(spec_tests, "DIRECTORY", tmp_path)
    written, refused = worker._retain_spec_tests(
        tmp_path, "round_x",
        [{"verdict": "REVISE", "findings": ["something is wrong"], "tests": []}],
        _spec(), log=lambda *a, **k: None)
    assert written == []
    assert refused and "came with no test" in refused[0]


def test_a_test_that_raises_is_a_failure_not_a_pass(tmp_path):
    path = tmp_path / "raises.py"
    path.write_text("def check(spec):\n    raise RuntimeError('boom')\n",
                    encoding="utf-8")
    result = spec_tests.run(_spec(), paths=[str(path)])
    assert not result["passed"]
    assert any("raised" in problem for problem in result["findings"])
