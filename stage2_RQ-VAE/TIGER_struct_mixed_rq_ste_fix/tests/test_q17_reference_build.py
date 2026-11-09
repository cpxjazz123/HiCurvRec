"""The reference rebuilt from the declared statements must gradcheck at the
legitimate collision; only then is the declared surface executable."""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.research.stage2.reference import build_reference

CONTRACT = json.loads((Path(__file__).resolve().parents[1] / "reference_contract.json").read_text())
CALLS = {c["name"]: c for c in CONTRACT["calls"]}


def test_the_declared_distance_reference_gradchecks_at_the_collision():
    ref, _ = build_reference(CALLS["mixed_distance"])
    left = torch.zeros(1, 2, dtype=torch.float64).requires_grad_(True)
    right = torch.zeros(1, 2, dtype=torch.float64)
    f = lambda a: ref(left=a, right=right)
    assert torch.autograd.gradcheck(f, (left,), eps=1e-6, atol=1e-8, rtol=1e-6)
    g = torch.autograd.grad(ref(left=left, right=right), left, allow_unused=True)[0]
    assert torch.isfinite(g).all()


def test_the_declared_structure_reference_gradchecks_at_the_collision():
    ref, _ = build_reference(CALLS["structure_objective"])
    points = torch.zeros(1, 6, dtype=torch.float64).requires_grad_(True)
    u = torch.zeros(1, 3, dtype=torch.float64)
    scale = torch.ones(1, 1, dtype=torch.float64)
    f = lambda p: ref(points=p, u=u, scale=scale)
    assert torch.autograd.gradcheck(f, (points,), eps=1e-6, atol=1e-8, rtol=1e-6)
    g = torch.autograd.grad(ref(points=points, u=u, scale=scale), points, allow_unused=True)[0]
    assert torch.isfinite(g).all()


if __name__ == "__main__":
    test_the_declared_distance_reference_gradchecks_at_the_collision()
    test_the_declared_structure_reference_gradchecks_at_the_collision()
    print("ok")
