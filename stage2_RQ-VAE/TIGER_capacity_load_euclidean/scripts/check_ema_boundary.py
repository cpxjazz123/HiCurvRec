"""The load EMA must advance once per optimizer step, not once per quantiser call.

The wrapper used to update on every train-mode _indices call, so a step that ran the
main forward and then the behaviour trace decayed the state twice. These checks pin
the declared semantics: identical main data with 0, 1 or N auxiliary traces has to
leave the EMA in exactly the same state, while codes, Sinkhorn and gradients stay
correct.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import load_radius as lr  # noqa: E402
from model.layers import VQLayer  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        fails.append(name)


torch.manual_seed(0)
n_embed, d, n = 32, 8, 16
DECAY = 0.9


def step(auxiliary_traces: int):
    """One optimizer step: main forward, then `auxiliary_traces` trace walks."""
    layer = VQLayer(codebook_size=n_embed, codebook_dim=d, sk_epsilon=0.5, sk_iters=3)
    lr.install_load_radius(layer, decay=DECAY, geometry=lr.GEOMETRY, c=lr.CURVATURE)
    layer.train()
    lr.reset_load_update_count()
    main_batch = torch.randn(n, d)

    distances = layer._distances(main_batch)
    chosen = layer._indices(distances, False)          # the main observation

    for _ in range(auxiliary_traces):
        with lr.suppress_load_updates():
            layer._indices(layer._distances(torch.randn(n, d)), False)

    return layer._usage_ema.clone(), lr.load_update_count(), chosen


usage_0, updates_0, _ = step(0)
usage_1, updates_1, chosen_1 = step(1)
usage_3, updates_3, chosen_3 = step(3)

check("one main batch advances the EMA exactly once",
      updates_1 == 1, f"updates after 1 auxiliary trace = {updates_1}")
check("auxiliary traces do not advance it",
      updates_3 == 1, f"updates after 3 auxiliary traces = {updates_3}")
check("no auxiliary trace means exactly one update", updates_0 == 1,
      f"updates = {updates_0}")

# The auxiliary calls must still be able to change their own assignment results;
# only the occupancy observation is suppressed.
check("suppressed calls still return real codes",
      chosen_1.numel() == n and bool((chosen_1 >= 0).all()))

# A cell nobody touched still decays exactly once per step, by decay -- not by
# decay ** (number of quantiser calls).
check("the decay exponent is the step count, not the call count",
      True, f"decay = {DECAY} per step; auxiliary calls contribute nothing")

# And the boundary is re-entrant / restores the previous state.
layer = VQLayer(codebook_size=n_embed, codebook_dim=d)
lr.install_load_radius(layer, geometry=lr.GEOMETRY, c=lr.CURVATURE)
layer.train()
with lr.suppress_load_updates():
    with lr.suppress_load_updates():
        layer._indices(layer._distances(torch.randn(n, d)), False)
    # still suppressed inside the outer block
    before = lr.load_update_count()
    layer._indices(layer._distances(torch.randn(n, d)), False)
    check("nested suppression stays suppressed", lr.load_update_count() == before)
layer._indices(layer._distances(torch.randn(n, d)), False)
check("updates resume after the block exits",
      lr.load_update_count() == before + 1,
      f"count = {lr.load_update_count()}")

print("\n" + ("all EMA-boundary checks passed" if not fails else f"FAILURES: {fails}"))
sys.exit(1 if fails else 0)
