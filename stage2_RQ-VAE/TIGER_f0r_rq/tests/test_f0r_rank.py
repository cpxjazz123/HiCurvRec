"""Issue #204: the ranked teacher is not flat, the gates refuse rather than train, and the metric is the frozen one."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

TREE = Path(__file__).resolve().parents[1]
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from model import poincare_contract as pc, sdh_kernel as k


def test_the_ranked_teacher_is_strictly_ordered_and_normalised():
    distances = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    mask = torch.ones(1, 4, dtype=torch.bool)
    teacher = k.rank_teacher(distances, mask)
    assert float(teacher.sum()) == pytest.approx(1.0)
    values = teacher[0].tolist()
    assert values == sorted(values, reverse=True)          # nearer candidate carries more mass
    assert values[0] / values[-1] == pytest.approx(4.0)    # 1/(1+0) against 1/(1+3)
    assert max(values) / min(values) >= 2.0                # the acceptance threshold


def test_the_ranked_teacher_beats_a_uniform_one_on_entropy():
    distances = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
    mask = torch.ones(1, 5, dtype=torch.bool)
    ranked = k.rank_teacher(distances, mask)
    entropy_ranked = float(-(ranked * ranked.clamp_min(1e-12).log()).sum())
    saved = k.UNIFORM_TEACHER
    try:
        k.UNIFORM_TEACHER = True
        uniform = k.rank_teacher(distances, mask)
    finally:
        k.UNIFORM_TEACHER = saved
    entropy_uniform = float(-(uniform * uniform.clamp_min(1e-12).log()).sum())
    assert entropy_ranked < entropy_uniform


def test_the_support_is_the_only_place_with_mass():
    distances = torch.tensor([[0.1, 0.2, 0.3]])
    mask = torch.tensor([[True, False, True]])
    teacher = k.rank_teacher(distances, mask)
    assert float(teacher[0, 1]) == 0.0
    assert float(teacher.sum()) == pytest.approx(1.0)


def test_the_scale_gate_refuses_the_configuration_that_blocks_this_arm():
    """The numbers the one-epoch run produced: lambda 159.676 against L_SD 0.74997 and L_recon 1.89596."""
    lam, l_sd, l_recon = 159.67558553920395, 0.7499675750732422, 1.8959649801254272
    passes = (1e-6 <= lam <= 1e3) and (lam * l_sd <= 10.0 * l_recon)
    assert passes is False
    assert lam * l_sd == pytest.approx(119.75, abs=0.01)
    assert 10.0 * l_recon == pytest.approx(18.96, abs=0.01)


def test_the_metric_at_curvature_one_is_the_frozen_contract():
    torch.manual_seed(0)
    points = pc.project(torch.randn(3, 8) * 0.05, 1.0).double()
    codes = pc.project(torch.randn(5, 8) * 0.05, 1.0).double()
    candidates = codes.unsqueeze(0).expand(points.shape[0], codes.shape[0], codes.shape[1])
    mine = k.curvature_distance(points, candidates, torch.ones(3, dtype=torch.float64),
                                torch.ones(5, dtype=torch.float64))
    reference = pc.distance(points.unsqueeze(1), codes.unsqueeze(0), 1.0)
    assert torch.allclose(mine, reference, rtol=1e-6, atol=1e-8)
