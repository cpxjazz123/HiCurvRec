"""Same semantics check as the capacity arm, adapted to the prefix table.

The L3 prefix table is (base**2 x K), so a full-table decay on every quantiser
forward is both wrong -- the decay exponent becomes the number of calls rather
than the number of steps -- and expensive. The boundary makes it once per step.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import prefix_radius as pr  # noqa: E402
from model.layers import VQLayer  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        fails.append(name)


torch.manual_seed(0)
n_embed, d, n, base = 32, 8, 16, 256
DECAY = 0.9


def step(auxiliary_traces: int):
    layer = VQLayer(codebook_size=n_embed, codebook_dim=d, sk_epsilon=0.5, sk_iters=3)
    pr.install_prefix_radius(layer, level=2, decay=DECAY, base=base,
                             geometry=pr.GEOMETRY, c=pr.CURVATURE)
    layer.train()
    pr.reset_load_update_count()
    layer._prefix_key = torch.randint(0, base, (n,))

    chosen = layer._indices(layer._distances(torch.randn(n, d)), False)
    for _ in range(auxiliary_traces):
        with pr.suppress_load_updates():
            layer._indices(layer._distances(torch.randn(n, d)), False)
    return layer._prefix_counts.clone(), pr.load_update_count(), chosen


_, u0, _ = step(0)
_, u1, _ = step(1)
_, u3, c3 = step(3)

check("one main batch advances the table exactly once", u1 == 1, f"updates = {u1}")
check("auxiliary traces do not advance it", u3 == 1, f"updates after 3 traces = {u3}")
check("no auxiliary trace means exactly one update", u0 == 1, f"updates = {u0}")
check("suppressed calls still return real codes",
      c3.numel() == n and bool((c3 >= 0).all()) and bool((c3 < n_embed).all()))

layer = VQLayer(codebook_size=n_embed, codebook_dim=d)
pr.install_prefix_radius(layer, level=2, base=base, geometry=pr.GEOMETRY, c=pr.CURVATURE)
layer.train()
layer._prefix_key = torch.randint(0, base, (n,))
with pr.suppress_load_updates():
    layer._indices(layer._distances(torch.randn(n, d)), False)
before = pr.load_update_count()
layer._indices(layer._distances(torch.randn(n, d)), False)
check("updates resume after the block exits", pr.load_update_count() == before + 1,
      f"count = {pr.load_update_count()}")

print("\n" + ("all prefix EMA-boundary checks passed" if not fails else f"FAILURES: {fails}"))
sys.exit(1 if fails else 0)
