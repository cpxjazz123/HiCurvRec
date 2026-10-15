"""Per-loss gradient separation, which is what the scope claim rests on.

Three losses reach the encoder through different tensors, and the arm's promise is
about exactly which of them change:

  * reconstruction -- through the quantiser's own forward, never near the trace;
  * L1 behaviour -- through level 0 of the trace;
  * L2/L3 behaviour -- through levels 1 and 2 of the trace.

Each is back-propagated on its own, with and without the projection, and the two
gradient sets are compared. The test can fail: if the projection reached a path it
should not, the corresponding comparison would stop being bit-identical.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tangent_grad_projection import decoupled_scope_note, project_deep_supervision  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        fails.append(name)


class FakeL1(torch.nn.Module):
    def __init__(self, k=8, d=6):
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
    def __init__(self, levels=3, k=8, d=6):
        super().__init__()
        rq = torch.nn.Module()
        rq.vq_layers = torch.nn.ModuleList([FakeL1(k, d) for _ in range(levels)])
        self.rq = rq


torch.manual_seed(0)
levels, n, d = 3, 8, 6
mod = FakeModule()
encoder = torch.nn.Linear(d, d)
x = torch.randn(n, d)

# The three losses, expressed the way the trainer wires them.
def recon_loss(enc_out, trace):
    # reconstruction sees only the encoder output path
    return enc_out.pow(2).mean()


def behaviour_loss(trace, level):
    return trace[level].pow(2).mean()


def gradients(project: bool, which: str):
    """One loss, back-propagated on its own, with or without the projection."""
    enc = torch.nn.Linear(d, d)
    enc.weight.data.copy_(encoder.weight.data)
    enc.bias.data.copy_(encoder.bias.data)
    enc_in = x.clone().requires_grad_(True)
    out = enc(enc_in)
    # The trainer's trace starts at the encoder output, so every level here is a
    # function of it -- otherwise the behaviour losses would reach nothing and the
    # comparison would be vacuous.
    trace = torch.stack([out, out * 2.0, out * 3.0], dim=0)
    if project:
        trace = project_deep_supervision(mod, trace)
    if which == "recon":
        loss = out.pow(2).mean()
    elif which == "l1_behaviour":
        loss = trace[0].pow(2).mean()
    else:
        loss = trace[1].pow(2).mean() + trace[2].pow(2).mean()
    loss.backward()
    return enc.weight.grad.clone()


for which, label in (("recon", "reconstruction"),
                     ("l1_behaviour", "L1 behaviour"),
                     ("deep_behaviour", "L2/L3 behaviour")):
    control = gradients(False, which)
    decoupled = gradients(True, which)
    same = torch.equal(control, decoupled)
    if which == "deep_behaviour":
        check("the deep behaviour gradient is projected", not same,
              f"max|diff| = {float((control - decoupled).abs().max()):.4f}")
    else:
        check(f"the {label} gradient is bit-identical to the control", same,
              f"max|diff| = {float((control - decoupled).abs().max()):.2e}")

# the forward values still do not change, at any level
raw = torch.randn(levels, n, d)
check("no level's forward values change", torch.equal(project_deep_supervision(mod, raw), raw),
      "the arm changes gradients only")

print(f"\n  scope: {decoupled_scope_note()}")
print("\n" + ("all per-loss checks passed" if not fails else f"FAILURES: {fails}"))
sys.exit(1 if fails else 0)
