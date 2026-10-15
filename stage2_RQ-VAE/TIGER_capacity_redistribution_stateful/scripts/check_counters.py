"""Checks for the corrected M9/M10 trees (CPU only, no training).

Four defects are exercised:
  1. the counter follows the index the level actually assigned, including when
     that index comes from Sinkhorn rather than argmin;
  2. the counter travels in state_dict();
  3. an untouched slot still decays (the EMA is over time);
  4. the wrapper delegates to VQLayer._indices, so Sinkhorn still decides.
"""
import sys
from pathlib import Path

import torch

ROOT = "/fs04/ar57/wenyu/GeneRec/stage2_RQ-VAE"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from usage_redistribution import install_usage_redistribution  # noqa: E402
from model.layers import VQLayer  # noqa: E402

failures = []


def check(name, ok, detail=""):
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


# ===================== M9: capacity redistribution =========================
print("M9 capacity_redistribution_stateful")
layer9 = VQLayer(codebook_size=64, codebook_dim=8, sk_epsilon=0.5, sk_iters=3)
layer9.train()
install_usage_redistribution(layer9, decay=0.9, strength=0.5)

check("counter is a registered buffer", "_usage_ema" in dict(layer9.named_buffers()),
      f"buffers={sorted(dict(layer9.named_buffers()))}")

# Feed a batch and see which codes the counter recorded.
latent = torch.randn(64, 8)
distances = layer9._distances(latent)
assigned = layer9._indices(distances, False)
usage = layer9._usage_ema
recorded = torch.nonzero(usage > 0).reshape(-1)
observed = torch.unique(assigned)
check("counter tracks the assigned codes, not the penalised argmin",
      torch.equal(torch.sort(recorded).values, torch.sort(observed).values),
      f"counter={recorded.tolist()[:8]}... assigned={observed.tolist()[:8]}...")

# On this layer Sinkhorn is active, so argmin and Sinkhorn disagree; the counter
# must follow Sinkhorn.
argmin_choice = distances.argmin(dim=-1)
disagree = int((argmin_choice != assigned).sum())
check("Sinkhorn was active for this layer", disagree > 0,
      f"{disagree}/{len(assigned)} rows where Sinkhorn differs from argmin")
check("counter does NOT follow argmin",
      not torch.equal(torch.sort(recorded).values, torch.sort(torch.unique(argmin_choice)).values)
      or disagree == 0,
      "only meaningful when Sinkhorn disagreed")

# An untouched slot must still decay. Seed it with a load the batch will not
# touch (slot 1 is not among the assigned codes), then watch it fade.
layer9._usage_ema[1] = 3.0
before = float(layer9._usage_ema[1].item())
for _ in range(5):
    layer9._indices(layer9._distances(latent), False)
after = float(layer9._usage_ema[1].item())
check("an untouched slot decays over steps", abs(after - before * 0.9 ** 5) < 1e-5,
      f"slot 1: {before:.4f} -> {after:.4f} (expected {before * 0.9 ** 5:.4f})")

# state_dict round-trip
saved = layer9.state_dict()
check("counter is in state_dict", "_usage_ema" in saved)
fresh9 = VQLayer(codebook_size=64, codebook_dim=8, sk_epsilon=0.5, sk_iters=3)
install_usage_redistribution(fresh9, decay=0.9, strength=0.5)
fresh9.load_state_dict(saved)
check("reload restores the counter",
      torch.allclose(fresh9._usage_ema, layer9._usage_ema),
      f"max|diff|={float((fresh9._usage_ema - layer9._usage_ema).abs().max()):.3e}")

# ===================== M10: prefix-aware ===================================
print("\nM10 prefix_aware_quantization_stateful")
layer10 = VQLayer(codebook_size=64, codebook_dim=8, sk_epsilon=0.5, sk_iters=3)
layer10.train()
install_prefix_aware(layer10, level=3, decay=0.9, strength=0.5, base=64)
check("counter table is a registered buffer", "_prefix_counts" in dict(layer10.named_buffers()),
      f"buffers={sorted(dict(layer10.named_buffers()))}")

layer10._prefix_key = torch.zeros(64, dtype=torch.long)
counts = layer10._prefix_counts
# Mark one slot as if it had accumulated load long ago, in a prefix this batch
# will never select.
stale_prefix = counts.shape[0] - 1
counts[stale_prefix, 5] = 3.0
stale_before = float(counts[stale_prefix, 5].item())
for _ in range(4):
    dist10 = layer10._distances(latent)
    layer10._indices(dist10, False)
stale_after = float(layer10._prefix_counts[stale_prefix, 5].item())
check("an unhit prefix slot decays", stale_after < stale_before,
      f"prefix {stale_prefix} slot 5: {stale_before:.4f} -> {stale_after:.4f}"
      f" (expected ~{stale_before * 0.9 ** 4:.4f})")

saved10 = layer10.state_dict()
check("table is in state_dict", "_prefix_counts" in saved10)
fresh10 = VQLayer(codebook_size=64, codebook_dim=8, sk_epsilon=0.5, sk_iters=3)
install_prefix_aware(fresh10, level=3, decay=0.9, strength=0.5, base=64)
fresh10.load_state_dict(saved10)
check("reload restores the table",
      torch.allclose(fresh10._prefix_counts, layer10._prefix_counts))

print("\n" + ("all M9/M10 checks passed" if not failures
              else f"FAILURES: {failures}"))
sys.exit(1 if failures else 0)
