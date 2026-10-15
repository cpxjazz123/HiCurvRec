"""Checks for the Riemannian decoupled arm.

Two base points are used on purpose. Near the boundary of the ball the conformal
factor bites, so that is where the hyperbolic and Euclidean operators must visibly
differ. In the interior, with the curvature taken to zero, the hyperbolic metric
degenerates to the Euclidean one and the operator must reproduce the control's
radial removal -- that limit is what makes these two arms a pair rather than two
unrelated operators.
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tangent_grad_projection import (  # noqa: E402
    RIEMANN_CURVATURE, hyperbolic_inner, project_deep_supervision, riemannian_radial,
)

fails = []


def check(name, ok, detail=""):
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        fails.append(name)


def euclid_radial(g, x):
    """The control's radial removal: project onto the orthogonal complement of x."""
    return (g * x).sum(-1, keepdim=True) / (x * x).sum(-1, keepdim=True).clamp_min(1e-12) * x


torch.manual_seed(0)
g = torch.randn(5, 4)
interior = F.normalize(torch.randn(5, 4), dim=-1) * 0.3     # safely inside for c = 1
boundary = F.normalize(torch.randn(5, 4), dim=-1) * 0.9     # A = 1 - c|x|^2 = 0.19

# --- the c -> 0 limit is the Euclidean radial removal ---------------------
limit = riemannian_radial(g, interior, c=1e-8)
ref = euclid_radial(g, interior)
check("as c -> 0 the operator becomes the Euclidean radial removal",
      torch.allclose(limit, ref, atol=1e-5),
      f"max|diff| = {float((limit - ref).abs().max()):.2e}")

# --- orthogonality in the hyperbolic metric -------------------------------
c = RIEMANN_CURVATURE
projected = g - riemannian_radial(g, boundary, c)
inner = hyperbolic_inner(projected, boundary, boundary, c).abs().max()
check("the projected gradient is tangent at the base point (<v, x>_x = 0)",
      float(inner) < 1e-4, f"max |<v,x>_x| = {float(inner):.2e}")
check("and it is not Euclidean-orthogonal: the metrics genuinely differ",
      float((projected * boundary).sum(-1).abs().max()) > 1e-3,
      f"Euclidean dot = {float((projected * boundary).sum(-1).abs().max()):.3e}")

# --- at c = 1 the two operators differ where it matters -------------------
euclid_at_c = g - euclid_radial(g, boundary)
check("at c = 1 the Riemannian operator differs from the Euclidean one",
      not torch.allclose(projected, euclid_at_c, atol=1e-3),
      f"max|diff| = {float((projected - euclid_at_c).abs().max()):.4f}")


class FakeL1(torch.nn.Module):
    def __init__(self, k=6, d=4):
        super().__init__()
        self.emb = torch.nn.Embedding(k, d)
        self.n_embed = k

    def _distances(self, x):
        return (x ** 2).sum(-1, keepdim=True) - 2 * x @ self.emb.weight.t() \
            + (self.emb.weight ** 2).sum(-1).t()

    def _indices(self, distances, infer_use_sk):
        return distances.argmin(dim=-1)

    def get_code_embs(self):
        return self.emb.weight


class FakeModule(torch.nn.Module):
    def __init__(self, levels=3, n=6, d=4):
        super().__init__()
        rq = torch.nn.Module()
        rq.vq_layers = torch.nn.ModuleList([FakeL1(n, d) for _ in range(levels)])
        self.rq = rq


mod = FakeModule()
raw = torch.randn(3, 6, 4)
out = project_deep_supervision(mod, raw)
check("forward values pass through unchanged", torch.equal(out, raw),
      f"max|diff| = {float((out - raw).abs().max()):.2e}")

t = raw.clone().requires_grad_(True)
(project_deep_supervision(mod, t) ** 2).sum().backward()
check("the deep gradient is finite and non-zero",
      bool(torch.isfinite(t.grad).all()) and float(t.grad.abs().sum()) > 0,
      f"|grad|_1 = {float(t.grad.abs().sum()):.4f}")
book = mod.rq.vq_layers[0].emb.weight
check("the codebook received the removed radial component",
      book.grad is not None and float(book.grad.abs().sum()) > 0,
      f"|codebook grad|_1 = {float(book.grad.abs().sum()):.4f}")

print("\n" + ("all Riemannian checks passed" if not fails else f"FAILURES: {fails}"))
sys.exit(1 if fails else 0)
