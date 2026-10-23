"""The hyperbolic gates' tests (issue #110, section 2 and the section 7 acceptance list).

The gates are the independent check that a mechanism is genuinely hyperbolic. The tests cover
both directions: a renamed Euclidean tangent loss must be refused at H2 and cost no GPU, and
a mechanism that really uses the geometry must pass. They also cover the fail-closed rule:
a gate with no evidence is never a pass.

    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python -m pytest \
        tools/harness2_genrec/tests/test_hyperbolic_gates.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

torch = pytest.importorskip("torch")

from tools.hyperbolic_verifier import gates  # noqa: E402


def _declaration(**overrides):
    declaration = {
        "hyperbolic_model": "poincare_ball",
        "curvature": -1.0,
        "hyperbolic_operator": "geodesic distance",
        "basepoints": "the origin",
    }
    declaration.update(overrides)
    return declaration


EUCLIDEAN_SOURCE = '''
import torch

def mechanism_loss(source, target, codebook, mask, **parameters):
    distance = torch.cdist(source, codebook)            # Euclidean
    logits = -distance / parameters["temperature"]
    picked = logits.argmax(dim=-1)
    return (logits.gather(1, picked.unsqueeze(1)).squeeze(1) * mask).mean()
'''

HYPERBOLIC_SOURCE = '''
import torch
from math import sqrt, log

def _poincare_distance(x, y, c):
    x2 = (x * x).sum(-1)
    y2 = (y * y).sum(-1)
    xy = ((x - y) * (x - y)).sum(-1)
    ratio = (c * xy / ((1 - c * x2) * (1 - c * y2))).clamp(max=1 - 1e-12)
    return 2 / sqrt(c) * torch.atanh(ratio)

def mechanism_loss(source, target, codebook, mask, **parameters):
    distance = _poincare_distance(source.unsqueeze(1), codebook.unsqueeze(0), parameters["c"])
    logits = -distance
    picked = logits.argmax(dim=-1)
    return (logits.gather(1, picked.unsqueeze(1)).squeeze(1) * mask).mean()
'''


def test_a_renamed_euclidean_loss_is_refused_at_h2():
    """A Euclidean codeword loss with a hyperbolic title is refused, and costs no GPU."""
    certificate = gates.run_gates(
        _declaration(hyperbolic_operator="geodesic distance"),
        EUCLIDEAN_SOURCE,
        proposal_sha256="p" * 64,
        code_tree_sha256="c" * 64,
    )
    assert certificate.gates["H2"].state == "fail"
    assert "no hyperbolic operator" in certificate.gates["H2"].detail
    assert not certificate.passed


def test_a_mechanism_that_uses_the_geometry_passes_h2():
    certificate = gates.run_gates(
        _declaration(),
        HYPERBOLIC_SOURCE,
        proposal_sha256="p" * 64,
        code_tree_sha256="c" * 64,
    )
    assert certificate.gates["H2"].state == "pass"
    assert "geodesic_distance" in certificate.gates["H2"].measurements["hyperbolic_operators_called"]


def test_a_gate_without_evidence_is_never_a_pass():
    """Insufficient evidence blocks: the issue says so explicitly."""
    certificate = gates.run_gates(
        _declaration(),
        HYPERBOLIC_SOURCE,
        proposal_sha256="p" * 64,
        code_tree_sha256="c" * 64,
    )
    assert certificate.gates["H4"].state == "insufficient_evidence"
    assert certificate.gates["H6"].state == "insufficient_evidence"
    assert not certificate.passed


def test_the_curvature_dependence_is_measured():
    """The geometry must change with the curvature, and not only by a scale."""
    result = gates.gate_h3(_declaration())
    assert result.state == "pass"
    assert result.measurements["max_change"] > 1e-6


def test_the_numerics_survive_the_degenerate_cases():
    result = gates.gate_h5(_declaration())
    assert result.state == "pass"


def test_the_identities_hold_against_the_oracle():
    result = gates.gate_h7(_declaration())
    assert result.state == "pass"


def test_the_manifold_and_the_curvature_are_checked():
    assert gates.gate_h1(_declaration(), "").state == "pass"
    assert gates.gate_h1(_declaration(hyperbolic_model="euclidean"), "").state == "fail"
    assert gates.gate_h1(_declaration(curvature=0.5), "").state == "fail"
    assert gates.gate_h1(_declaration(curvature=None), "").state == "fail"


def test_a_collapsed_export_is_refused_at_h6():
    result = gates.gate_h6(
        {"operator_enabled": True, "changed_fraction": 0.9, "distinct_level_codes": 2}
    )
    assert result.state == "fail"
    assert "collapsed" in result.detail


def test_the_gradient_evidence_must_cover_the_declared_targets():
    result = gates.gate_h4(
        {"declared_targets": ["codebook"], "reached_targets": ["encoder"], "finite": True}
    )
    assert result.state == "fail"
    result = gates.gate_h4(
        {"declared_targets": ["codebook"], "reached_targets": ["codebook"], "finite": True}
    )
    assert result.state == "pass"


def test_the_certificate_is_bound_to_what_it_checked():
    certificate = gates.run_gates(
        _declaration(),
        HYPERBOLIC_SOURCE,
        proposal_sha256="p" * 64,
        code_tree_sha256="c" * 64,
    )
    document = certificate.as_dict()
    assert document["proposal_sha256"] == "p" * 64
    assert document["code_tree_sha256"] == "c" * 64
    assert set(document["summary"]) == set(gates.GATES)
    assert document["passed"] is False
