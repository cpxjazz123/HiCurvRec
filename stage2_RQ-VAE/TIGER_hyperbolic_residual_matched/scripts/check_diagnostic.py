"""The diagnostic must measure what its names say.

The previous version compared a Mobius accumulation against another Mobius
accumulation and called the second one the Euclidean control, so the decoder
comparison measured nothing. These checks pin the two comparisons apart.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyperbolic_residual import (  # noqa: E402
    MOBIUS_CURVATURE, decoder_input, reconstruction_deviation,
)

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

    def forward(self, x, infer_use_sk=False):
        idx = (x * 5.0).sum(-1).abs().long() % self.n_embed
        return self.emb(idx), torch.tensor(0.0), 0, idx

    def get_code_embs(self):
        return self.emb.weight


class FakeModule(torch.nn.Module):
    def __init__(self, levels=3, k=8, d=6):
        super().__init__()
        self.encoder = torch.nn.Linear(d, d)
        rq = torch.nn.Module()
        rq.vq_layers = torch.nn.ModuleList([FakeL1(k, d) for _ in range(levels)])
        self.rq = rq


torch.manual_seed(0)
mod = FakeModule()
embeddings = torch.randn(32, 6)

# --- the decoder comparison must use one walk and two accumulations --------
codes = [torch.randn(32, 6) * 0.05 for _ in range(3)]
reference_matched = decoder_input(codes, MOBIUS_CURVATURE, matched=True)
reference_euclid = decoder_input(codes, MOBIUS_CURVATURE, matched=False)
check("the reference Euclidean accumulation really is the sum",
      torch.allclose(reference_euclid, codes[0] + codes[1] + codes[2], atol=1e-6),
      f"max|diff| = {float((reference_euclid - (codes[0]+codes[1]+codes[2])).abs().max()):.2e}")
check("the matched accumulation differs from it on non-collinear codes",
      not torch.allclose(reference_matched, reference_euclid, atol=1e-4),
      f"max|diff| = {float((reference_matched - reference_euclid).abs().max()):.4f}")

stats = reconstruction_deviation(mod, embeddings)
check("the diagnostic reports the decoder comparison under its own name",
      "decoder_dev_rel" in stats and "residual_dev_rel" in stats,
      f"keys = {sorted(stats)}")
check("every value is finite (floats for ratios, an int for the code count)",
      all((isinstance(v, int) or torch.isfinite(torch.tensor(v)).item())
          for v in stats.values()),
      f"values = { {k: round(v, 6) for k, v in stats.items()} }")

# The decoder-only number must be non-zero on a real model, since the two rules
# genuinely differ; a zero here would mean the comparison is still degenerate.
check("the decoder-only comparison is non-degenerate",
      stats["decoder_dev_rel"] > 0 or stats["decoder_dev_max_row"] > 0,
      f"decoder_dev_rel = {stats['decoder_dev_rel']:.6f}")

# --- curvature -> 0 collapses the decoder comparison ----------------------
tiny = [torch.randn(32, 6) * 0.02 for _ in range(3)]
close = decoder_input(tiny, 1e-8, matched=True)
flat = decoder_input(tiny, 1e-8, matched=False)
check("as c -> 0 the matched accumulation becomes the Euclidean sum",
      torch.allclose(close, flat, atol=1e-5),
      f"max|diff| = {float((close - flat).abs().max()):.2e}")

# --- the residual comparison reports assignment changes -------------------
check("the residual comparison reports whether codes changed",
      "residual_codes_changed" in stats and isinstance(stats["residual_codes_changed"], int),
      f"codes_changed = {stats['residual_codes_changed']}")

# --- no state mutation ----------------------------------------------------
before = {n: p.clone() for n, p in mod.named_parameters()}
reconstruction_deviation(mod, embeddings)
after = dict(mod.named_parameters())
check("the diagnostic does not mutate the model",
      all(torch.equal(before[n], after[n]) for n in before))

print("\n" + ("all diagnostic checks passed" if not fails else f"FAILURES: {fails}"))
sys.exit(1 if fails else 0)
