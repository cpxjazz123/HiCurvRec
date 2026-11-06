# Second shape of the community mechanism: the pair supervision collapses the codes

**What was tried.** The previous arm put the community information into the
level-1 *cost* and it improved the coarse level while costing the fine levels,
without moving the recall. The arm that came closest to the threshold in this
campaign (resample, `test_recall@10 = 0.06374`) had changed something else: it
resampled which pairs the encoder was contrasted on every epoch, and it left
the code space alone. So this arm translated the community partition into that
lever: every step, every item of the batch is an anchor, its positive is
another item drawn from the same community **fresh on every step**, the
negatives are the step's other positives, and a candidate that the partition
puts in the anchor's own community is masked out as an unlabelled positive.
The loss is a contrastive cross-entropy over the latents entering the
quantiser; the codebooks are left to the reconstruction.

Both scales were measured rather than chosen: the temperature is the median
latent distance of the first step, and the weight is the ratio that makes the
term equal to the reconstruction loss at that step. At epoch 1 the term
contributed 0.000484 against the reconstruction's 0.000463, so it was live and
at the intended magnitude, not a nominal term.

**What happened.** The code space collapsed, from the fiftieth epoch on.

| epoch | 50 | 300 | 700 | 1050 | 1250 | 1349 |
|---|---|---|---|---|---|---|
| learned three-level collision | 0.459 | 0.420 | 0.460 | 0.582 | 0.534 | 0.491 |
| unique SIDs (of 24587) | 13 311 | 14 261 | 13 287 | 10 271 | 11 450 | 12 508 |

The owner's gate refuses anything above 0.35 and the run never came near it;
the minimum over the whole trajectory was **0.420**. The run was stopped at
epoch 1349 rather than burning the remaining 1651 epochs: the verdict could not
change, and the collision is not a transient (it is ~0.45 at epoch 50, before
the encoder has learned anything, and it never recovers).

For reference, the same trainer family with the community term in the *cost*
(the previous arm) held the collision at 0.11–0.20 through the same epochs and
finished at 0.206, so the difference is the mechanism and not the tree.

**Why.** A contrastive term on the encoder's latents is a far stronger grouping
force than a cost term. The cost term only reorders which code an item
prefers; the contrastive term makes the *latents themselves* similar for every
community, and the quantiser — which quantises exactly those latents — has no
choice but to reuse codes. Prefix diversity is what the decoder conditions on,
so it dies with the latents. This is the sinkhorn arm's failure (0.771) in a
milder form, caused by the supervision instead of the assignment.

**What this closes.** Three shapes of the same community mechanism have now
been measured:

| shape | where the community acts | L3 collision | recall |
|---|---|---|---|
| cost at level 1 | the assignment's cost matrix | 0.3259 (passes the gate) | 0.05263 |
| standalone | nothing in the RQ-VAE | — | — (community quality only) |
| pair supervision | the encoder's latents | **0.42–0.58 (refused)** | not run |

The direction the campaign now supports is narrow and negative: **any force
that makes the code space community-clustered destroys prefix diversity**, and
the one shape that survives the gate (the cost term) does not move the recall.
The community information is real — it is the only signal in this campaign
that improved the coarse level's positive-hit rate (0.0864 → 0.0989) — but
using it as a grouping force on the codes is not a productive direction, and
the arm that came closest to the threshold used a *sampling* lever without
touching the code space at all.

**What is not concluded.** This is one run stopped early with a clear
mechanistic reading; it is not a sweep. A weaker contrast (a smaller measured
weight) might have stayed under the gate, but choosing that weight would be
tuning by trying values, which the experiment policy forbids, and the observed
trend — collapsed at epoch 50 — says the effect is structural, not a matter of
strength.

**Evidence.** `community_pairs_failed.json` (the trajectory, the gate, the
reason), the arm's tree `stage2_RQ-VAE/TIGER_community-pairs-rq/` with its seven
tests and its reference contract, and the machine admission that let it run:
all ten checks passed, including the reference agreement (values 0.0, gradients
2.2e-16, gradcheck on both sides, coupling displacement 0.69).
