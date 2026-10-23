"""Agent I's tests (issue #110, WP1).

The implementation agent implements the mathematics the proposal froze and cannot quietly
substitute something easier. The tests cover the three rules: a changed mathematics needs a
new mechanism identity, the frozen files are not writable, and a repair is only for a
verified code bug under the same mathematics, is counted, and never runs a cold run.

    /home/wlia0047/ar57_scratch/wenyu/harness2/.venv/bin/python -m pytest \
        tools/harness2_genrec/tests/test_implementation.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.harness2_genrec import implementation as I  # noqa: E402


def _proposal():
    return {
        "mechanism_id": "arm_one",
        "equation": "L = d_c(x, y)^2",
        "symbols": {"x": "the source", "y": "the codebook"},
        "parameters": {"c": 1.0, "weight": 0.05},
        "gradient_targets": {"source": "zero", "target": "zero", "codebook": "nonzero"},
        "target_paths": ["codebook"],
    }


def _request(tmp_path):
    return I.ImplementationRequest(
        mechanism_id="arm_one",
        proposal=_proposal(),
        proposal_sha256="p" * 64,
        tree=tmp_path / "tree",
    )


def test_an_implementation_of_the_frozen_mathematics_is_accepted(tmp_path):
    request = _request(tmp_path)
    module = tmp_path / "tree" / "mechanism_arm_one.py"
    module.parent.mkdir(parents=True, exist_ok=True)
    module.write_text("def mechanism_loss(*a, **k):\n    return 0.0\n", encoding="utf-8")
    result = I.implement(request, _proposal(), written=[module])
    assert result.status == I.IMPLEMENTED
    assert result.implementation_sha256
    assert result.evidence["official_cold_run_performed"] is False
    assert result.evidence["mathematics_comparison"]["identical"] is True


def test_a_changed_equation_needs_a_new_mechanism_identity(tmp_path):
    request = _request(tmp_path)
    changed = dict(_proposal())
    changed["equation"] = "L = ||x - y||^2"
    result = I.implement(request, changed)
    assert result.status == I.REVISION_REQUIRED
    assert "equation" in result.revision_request["missing"]
    assert "new mechanism identity" in result.detail


def test_a_changed_gradient_target_needs_a_new_mechanism_identity(tmp_path):
    request = _request(tmp_path)
    changed = dict(_proposal())
    changed["gradient_targets"] = {"source": "nonzero", "target": "nonzero", "codebook": "nonzero"}
    result = I.implement(request, changed)
    assert result.status == I.REVISION_REQUIRED
    assert "gradient_targets" in result.revision_request["missing"]


def test_a_touched_frozen_file_is_refused(tmp_path):
    request = _request(tmp_path)
    result = I.implement(
        request, _proposal(), frozen_files_touched=["tools/cgv/contract.py"]
    )
    assert result.status == I.REFUSED
    assert "frozen" in result.detail


def test_a_frozen_path_is_not_writable(tmp_path):
    tree = tmp_path / "tree"
    (tree / "tools" / "cgv").mkdir(parents=True)
    assert not I.write_is_allowed(tree, tree / "tools" / "cgv" / "contract.py")
    assert I.write_is_allowed(tree, tree / "mechanism_arm_one.py")


def test_an_implementation_with_no_equation_requests_a_revision(tmp_path):
    request = _request(tmp_path)
    result = I.implement(request, {"symbols": {}})
    assert result.status == I.REVISION_REQUIRED
    assert "no equation" in result.detail


def test_a_repair_is_only_for_a_code_bug_under_the_same_mathematics(tmp_path):
    result = I.ImplementationResult(status=I.IMPLEMENTED, mechanism_id="a", proposal_sha256="p")
    allowed, reason = I.repair_allowed(result, is_code_bug=False, same_mathematics=True)
    assert not allowed
    assert "code bug" in reason
    allowed, reason = I.repair_allowed(result, is_code_bug=True, same_mathematics=False)
    assert not allowed
    assert "frozen mathematics" in reason
    allowed, reason = I.repair_allowed(result, is_code_bug=True, same_mathematics=True)
    assert allowed
    spent = I.ImplementationResult(
        status=I.IMPLEMENTED, mechanism_id="a", proposal_sha256="p", repairs_used=I.MAX_REPAIRS
    )
    allowed, reason = I.repair_allowed(spent, is_code_bug=True, same_mathematics=True)
    assert not allowed
    assert "budget" in reason


def test_a_repair_never_runs_an_official_cold_run(tmp_path):
    request = _request(tmp_path)
    module = tmp_path / "tree" / "mechanism_arm_one.py"
    module.parent.mkdir(parents=True, exist_ok=True)
    module.write_text("x = 1\n", encoding="utf-8")
    result = I.implement(request, _proposal(), written=[module])
    assert result.evidence["official_cold_run_performed"] is False
