# AF3-guided hierarchical supervision: semantics kept, discrimination lost

The question this round was asked to settle in Stage 2 alone:

> can the three-level RQ-VAE learn structures of different granularity directly
> from the behaviour graph and its AF3 curvature, while keeping its semantic
> reconstruction and its ability to tell items apart?

The answer the run gives is: **the reconstruction and the codebook usage survive,
the behaviour discrimination does not — it gets worse than with no supervision at
all**, and the mechanism that was supposed to sharpen it contracts the space
instead.

## What was trained

`TIGER_af3_hierarchy`, forked from the frozen baseline (`ARM_NAME` rewritten,
every frozen hyper-parameter untouched: 3000 epochs, 3×256 codes of width 32,
AdamW, batch 1024 per rank on four cards, seed 42). One new module,
`model/af3_hierarchy.py`, four terms, all acting on a **soft read-out of each
level's own assignment**; the baseline's balanced assignment is untouched.

| term | what it asks for | how |
|---|---|---|
| `L1` (AF3) | the edges the frozen AF3 convention calls locally dense should share the coarse group | `-w · log K`, `w = clip((AF3 − median)/(p90 − median), 0, 1)`, `K` the contract's **Poincaré kernel** of the two items' expected level-1 codewords |
| `L2` (behaviour) | inside the level-1 group, behaviour edges refine, non-edges separate | positives pulled together, non-edges pushed apart, both selected **by the current argmax group** |
| `L3` (discrimination) | related items stay distinguishable at the third level; unrelated items are not pulled together | hinge on related pairs above 0.5, hinge on unrelated pairs below 0.05 |
| `L4` (balance) | no level collapses onto a few codes | KL of each level's mean assignment from uniform |

The AF3 curvature is the single frozen table of the issue-#136 convention
(315391 training edges, **157386 above the median**; median −128, p90 −28),
exported once by `tools/research/science/community_curvature/export_edge_af3.py`
so the arm cannot carry a second implementation of the same quantity.

The four terms are introduced at epoch 1500 and ramped to 1 at epoch 3000, and
their combined gradient contribution is capped at 10% of the original
objective's, checked separately for the encoder and the codebook with the
stricter scaling applied.

Admission was the machine check: **10 of 10 PASS**, including the reference
contract (three declared surfaces agree with their SymPy rebuild to 2.2e-16 in
value and 1.8e-15 in gradient, `gradcheck` true on both sides) and the coupling
verdict ("the behaviour moves the computation", 0.439). The run took 620.5 s.

## The three criteria, measured inside training

Every 100 epochs the record computes the criterion on the **exported** codes —
`q` is the sum of the three codewords at the exported assignment, i.e. the
representation Stage 3 would read — against the held-out validation
interactions.

| epoch | ramp | reconstruction | held-out AUC | positive cos | negative cos | background cos | L1 codes | L2 codes |
|---|---|---|---|---|---|---|---|---|
| 1000 | 0.00 | 7.45e-05 | 0.6886 | — | — | — | 246 | 256 |
| 1400 | 0.00 | 7.12e-05 | 0.6877 | 0.602 | 0.465 | 0.456 | 249 | 256 |
| 1500 | 0.00 | 7.06e-05 | 0.6869 | 0.603 | 0.468 | 0.460 | 249 | 256 |
| 1700 | 0.13 | 6.98e-05 | 0.6852 | 0.613 | 0.484 | 0.474 | 252 | 256 |
| 1900 | 0.27 | 6.92e-05 | 0.6835 | 0.644 | 0.526 | 0.515 | 252 | 256 |
| 2100 | 0.40 | 6.86e-05 | 0.6803 | 0.682 | 0.578 | 0.568 | 252 | 256 |
| 2500 | 0.67 | 6.78e-05 | 0.6716 | — | — | — | 252 | 256 |
| 3000 | 1.00 | 6.90e-05 | **0.6636** | — | — | — | 252 | 244 |

**Criterion 1 — the behaviour discrimination — fails.** The arm's own
pre-ramp state is 0.6869 at epoch 1500 and the frozen baseline ends at 0.6836;
after the supervision is switched on the arm declines **monotonically to
0.6636**, twenty points below the baseline in the third decimal and the worst
value the whole run produced.

**Criterion 2 — the three levels each do their job — is mixed.** Every level
keeps using its codebook (L1 252, L2 244, L3 256 of 256), and the level-3 term
never bound, which means related items were already distinguishable at the third
level and stayed so. But the behaviour *structure* the levels encode is not
better: the neighbour enrichment falls at every depth against the frozen
baseline (L1 7.84 vs 7.34, L2 15.83 vs 17.75, L3 19.11 vs 25.70).

**Criterion 3 — the base capability — holds.** Reconstruction improves slightly
under supervision (7.06e-05 → 6.90e-05), the three-level collision ends at
0.1767 against the frozen baseline's 0.1775 and the baseline re-run's 0.1624 —
inside the baseline's own run-to-run range — and the level-2 prefix count is
7281 against 7414.

## Why it failed, and why it is the same failure as the last arm

The contraction is visible in the record. From the moment the ramp starts, the
**positive and the negative similarities rise together**:

```
epoch 1500: positive 0.603, negative 0.468, background 0.460
epoch 2100: positive 0.682, negative 0.578, background 0.568
```

The gap narrows (0.135 → 0.104) while everything moves closer. That is the same
signature the community-graph arm showed, from a completely different
supervision (AF3-weighted curved aggregation on each level instead of a
community-affinity target on the summed representation). Two mechanisms, one
failure mode: **any behaviour-graph supervision of this quantiser's codes raises
the global similarity of the representation and costs held-out discrimination.**

The reason is structural and does not depend on the details of either term. The
reconstruction objective already arranges the space so that a genuinely related
pair is closer than a random one (AUC 0.6836); a graph term that says "these
items should be closer" is, for the vast majority of its edges, asking for
something the space already does, and it has no way to say "and nothing else
should move". Every gradient it spends on pulling a related pair together also
compresses the background. The level-2 spikes (3.13 → 3.52 at epochs 2200 and
2500) and the balance term rising from 0.043 to 0.062 show the four terms
fighting each other rather than adding up.

## What this settles

1. **The supervision does reach every level.** All four terms are non-zero from
   epoch 1500, the encoder and the codebook both receive gradient, and the
   level-2 term had to be fixed twice before it fired at all (an absolute
   agreement threshold selected nothing; the filter now follows the current
   assignment). The mechanism is wired, and it is wired to the thing it claims.
2. **Wiring is not the problem, and neither is the AF3 signal's fidelity:**
   the arm learns what it is asked to learn (L1's loss falls 0.029 → 0.017) and
   the criterion still degrades. The failure is in *what is asked*.
3. **The next design must not be another graph-attraction term.** Both attempts
   so far rearranged similarity in the same direction. A mechanism that can only
   move related items closer will always pay for it in the background, and the
   criterion measures exactly that payment.
4. **The bar is still the baseline's 0.6836**, and it is now clear that it has
   to be cleared *without* raising the background similarity — the two arms
   that were tried both failed on that second clause before they failed on the
   first.

## Files

* `stage2_RQ-VAE/TIGER_af3_hierarchy/` — the arm, its 18 tests, its reference
  contract, its mechanism module
* `tools/research/science/community_curvature/export_edge_af3.py` — the frozen
  AF3 table
* `results/stage2_RQ-VAE/TIGER_af3_hierarchy/logs/training_metrics.jsonl` — the
  per-epoch record, including the in-training criterion curve
* `results/stage2_RQ-VAE/TIGER_af3_hierarchy/stage2_metrics/` — the mandatory
  layered record, written before any quality conclusion was drawn
