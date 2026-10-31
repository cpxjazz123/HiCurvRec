# Does a single card train the same recipe as four?

Measured on `GDELTA-SUCC-PULL-RQ`, one A100, 3000 frozen epochs, two regimes:

| regime | per-rank batch | global batch | steps / epoch | collision | distinct L2 prefixes | valid L1 Hard-ER |
| --- | --- | --- | --- | --- | --- | --- |
| one card, per-rank fixed (4 cards' micro-batch) | 1024 | **1024** | 24 | 0.3559 | **7414** | 6.47 |
| one card, global batch matched | 4096 | **4096** | 6 | 0.3422 | **162** | 1.20 |
| four cards (the frozen recipe) | 1024 | 4096 | 6 | -- | -- | -- |

## What this says

The per-rank batch is **semantic**, not a throughput knob. Two parts of this code
normalise over the per-rank batch:

* the quantizer's Sinkhorn assignment (`VQLayer._indices` balances the
  batch x codebook transport plan over the batch dimension),
* the MLP's `BatchNorm1d` (`model/utils.py`), which has no cross-rank sync.

Raising the per-rank batch to 4096 to match the four-card *global* batch collapsed
the second level from 7414 distinct prefixes to 162 and drove the L1 negative hit
rate to 0.132. Matching the global batch therefore does not reproduce the
four-card run; it produces a different, degenerate quantiser.

Matching the per-rank batch keeps the quantiser's semantics identical to the
four-card run, and leaves exactly one unavoidable difference: on one card the same
1024-example micro-batch is taken 24 times per epoch instead of 6, i.e. four times
the optimizer steps at one quarter of the gradient noise.

Four-card and single-card runs are therefore **not** the same recipe, and no
setting makes them the same: one of the two coupled quantities has to move. The
code keeps the semantic one frozen (per-rank 1024) and records the resulting
difference rather than hiding it.
