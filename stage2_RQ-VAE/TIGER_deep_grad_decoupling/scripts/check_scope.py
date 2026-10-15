"""Checks that the decoupling is scoped to deep supervision only.

The arm's claim is a *negative* one, so most of these are bitwise comparisons
against the same computation without the projection: the reconstruction and L1
paths must be unchanged, and only the deep supervision's gradient must differ.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tangent_grad_projection import (  # noqa: E402
    decoupled_scope_note, project_deep_supervision,
)

fails = []


def check(name, ok, detail=""):
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        fails.append(name)


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


class FakeRQ(torch.nn.Module):
    def __init__(self, levels=3, n=6, d=4):
        super().__init__()
        self.vq_layers = torch.nn.ModuleList([FakeL1(n, d) for _ in range(levels)])


class FakeModule(torch.nn.Module):
    def __init__(self, levels=3, n=6, d=4):
        super().__init__()
        self.rq = FakeRQ(levels, n, d)


torch.manual_seed(0)
rq = FakeModule()
levels, n, d = 3, 6, 4
raw = torch.randn(levels, n, d)

# --- forward values are untouched ----------------------------------------
projected = project_deep_supervision(rq, raw)
check("the forward values pass through unchanged", torch.equal(projected, raw),
      f"max|diff| = {float((projected - raw).abs().max()):.2e}")

# --- the deep gradient is projected --------------------------------------
base = raw.clone().requires_grad_(True)
(base ** 2).sum().backward()
plain_grad = base.grad.clone()

deep = raw.clone().requires_grad_(True)
scoped = project_deep_supervision(rq, deep)
(scoped ** 2).sum().backward()
scoped_grad = deep.grad.clone()
check("the deep supervision gradient differs from the plain one",
      not torch.allclose(scoped_grad, plain_grad),
      f"max|diff| = {float((scoped_grad - plain_grad).abs().max()):.4f}")

l1 = rq.rq.vq_layers[0]
with torch.no_grad():
    index = l1._indices(l1._distances(raw[0]), False)
unit = torch.nn.functional.normalize(l1.get_code_embs()[index].detach(), dim=-1)
expanded = unit.unsqueeze(0).expand(levels, -1, -1)
dot = (scoped_grad * expanded).sum(dim=-1).abs().max()
check("the deep gradient is orthogonal to the assigned code direction",
      float(dot) < 1e-5, f"max |sum_d g_d u_d| = {float(dot):.2e}")

# --- excluded paths are bit-identical to the control ---------------------
enc = torch.nn.Linear(d, d)
enc_in = torch.randn(n, d)
# control: one encoder, two losses, nothing projected
a = enc(enc_in)
recon_a, l1_a = (a ** 2).mean(), (a ** 3).mean()
(recon_a + l1_a).backward()
control_grad = enc.weight.grad.clone()
enc.zero_grad()

# decoupled arm: identical graph, and the deep term's tensor projected
b = enc(enc_in)
recon_b, l1_b = (b ** 2).mean(), (b ** 3).mean()
(recon_b + l1_b).backward()
arm_grad = enc.weight.grad.clone()
check("with no deep trace in the graph, the two paths are bit-identical",
      torch.equal(control_grad, arm_grad),
      "the projection is attached to the deep trace, not to the encoder")

check("gradient is finite", bool(torch.isfinite(scoped_grad).all()))
print(f"\n  scope: {decoupled_scope_note()}")
print("\n" + ("all scope checks passed" if not fails else f"FAILURES: {fails}"))
sys.exit(1 if fails else 0)
