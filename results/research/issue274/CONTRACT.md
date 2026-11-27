# Issue #274 — contract (round 1): curvature-selected relative pair ranking on the final hard q

Frozen before the implementation and before the formal training. The issue body sha256 is
`d258950dc4586c498329c029008acf2819e7c98a463d30a0e1c74246a2a4d541` (`ISSUE_BODY.md`).

## 1. Scope

Parent = the **frozen TIGER baseline** (`stage2_RQ-VAE/baseline/TIGER_RQ-VAE`, the SID
`025ad8bd…`): the 768→32 bottleneck, three levels of 256 codes, Euclidean selection, the
residual/floor/last-level Sinkhorn, the reconstruction term and the seed. The candidate tree is
`stage2_RQ-VAE/TIGER_pairrank_rq`; only `ARM_NAME` and the new loss module differ from the parent.

Round 1 adds exactly one mechanism: the **relative pair-ranking loss** `L_geo` on the final hard q,
with the positive/negative pairs **selected offline** from the frozen real-graph sectional curvature.
Nothing else changes. In particular the candidate does **not** inherit any pointwise or difference
curvature supervision (the #269/#273 arms' mechanism), no global distance term (the #272 arm) and no
behaviour contrastive loss (the behaviour-sample arms).

## 2. Objects (kept separate, per issue section 1)

| symbol | meaning | enters the loss? |
|---|---|---|
| `y_t` | the raw graph sectional curvature of triplet `t` (the frozen assets' `curvature` column) | **no** — offline selection only |
| `kappa_t` | the fitted spatial curvature parameter (the #269/#273 family's `kappa_cache`) | **no** — not used at all |
| `(B, C)` | the positive outer pair: the closed-triangle triplet of the centre's group with the **highest** `y_t` | as indices only |
| `(D, E)` | the negative outer pair: the open-wedge triplet of the centre's group with the **lowest** `y_t` | as indices only |
| `q_i` | the final hard representation of item `i`: `sum_l c_l` over the three levels | **yes** |
| `m` | the fixed positive margin, identical for every valid sample | **yes** (as a constant) |

The curvature value never enters the loss value, the margin, a sample's loss weight, the distance
function, the code selection or the model inputs. The frozen pair index
(`results/research/issue274/pairs_index.npz`, sha256 `a3395e54…`) is read-only for the model: the
labels cannot change during training.

## 3. The loss

For a training batch of selected groups:

```
d_pos = || q_B - q_C ||_2
d_neg = || q_D - q_E ||_2
L_geo = mean over the groups of  max(0, d_pos - d_neg + m)
L_total = L_RQVAE + lambda * L_geo
```

`m = 0.2529585361480713`, frozen in advance as the median of the frozen TIGER's positive-pair
hard-q distances over the **training** groups (`baselines.json`). The same `m` is used by the
candidate and by the frozen TIGER baseline, so the acceptance comparison is at one caliber.

This is a **two-pair distance ranking**, not a shared-anchor triplet loss: the positive pair and the
negative pair share no anchor; the centre only selected them.

## 4. The gradient routing

The forward value of the supervised representation is exactly the final hard `q = sum_l c_l`
(the three levels' selected codeword embeddings, summed with weight 1). The gradient of `L_geo`
therefore reaches **the three codebooks** through the embedding lookup, and the encoder is not in
this loss's graph: the RQ selection is a non-differentiable argmin over the residual, so the pair
loss shapes the **discrete code geometry** while the reconstruction keeps training the encoder.

This is deliberate and is the mechanism's central causal claim: the #269–#273 family degraded the
*representation* by supervising curvature values pointwise (and by difference), which pushed the
encoder latent into a degenerate regime (the #269's variance ratio 0.42 and slope 0.019 against the
TIGER's 28.7 and 0.041; see `DIAGNOSIS.md`). A loss on the code geometry alone cannot reproduce that
degradation.

The reconstruction path, the residual, the floor and the Sinkhorn are untouched, so `L_RQVAE` is
bit-identical to the parent's when `lambda = 0`.

## 5. Shapes, boundaries, tolerances

| item | value / rule |
|---|---|
| the batch of groups | `B` groups per step; the indices address the items of the batch's hard q |
| `q` shape | `[items, 32]` (float32), the sum of the three `[256, 32]` codebook embeddings |
| `d_pos`, `d_neg` | float32 scalars per group, the Euclidean norm over the 32 dimensions |
| `m` | a fixed positive float32 constant, identical for every sample |
| `m < 0` | rejected at construction (the margin must be positive) |
| coincident outer pairs | `d_pos = d_neg = 0` -> the hinge is `max(0, m) = m`; the **gradient** is zero (the hinge is at its flat region), which is measured explicitly, not assumed |
| an inverted pair (`d_pos > d_neg + m`) | the hinge is positive and the gradient flows to both pairs |
| a satisfied pair (`d_pos + m <= d_neg`) | the hinge is exactly 0 and the gradient is exactly 0 |
| invalid indices | an index outside `[0, items)` raises; an item index whose code is missing raises |
| the curvature change with fixed labels | the loss is **bit-identical**: the curvature is not an input of the loss function (a dedicated test) |
| numerical stability | the loss and the gradients must be finite; a non-finite value aborts the run (the parent's behaviour) |
| the pair index | read-only; the model never re-derives or re-orders the labels |

## 6. Paper / official-implementation comparison

| module | paper position | official implementation | our choice |
|---|---|---|---|
| the graph sectional curvature `y_t` | GraphMoRE (arXiv 2412.11085v1), Appendix A.1, Eq. A.1: `kappa(m;b,c) = mean_{a in R(m)}[g(a,m)^2 + g(b,c)^2/4 - (g(a,b)^2 + g(a,c)^2)/2]` | the frozen assets (`assets/shards/*.npz`), computed on the undirected projection of the visible directed graph | used **only** as the offline ordering key; the fitted `kappa_t` is not used |
| the relative-distance supervision | no external position: the closed-vs-open outer-pair ranking is derived here from the graph's own adjacency | the repository's `TIGER_BEHAVIOR_SAMPLE_RQ-VAE` (`quant_rank_loss.py`) has a shared-anchor ranking on the summed quantized representation with `relu(pos - neg + margin)` | the same loss form and the same margin convention, but a **two-pair** ranking (no shared anchor) and a curvature-selected pair construction; the margin is calibrated in advance on the frozen baseline's scale instead of the fixed 0.05 constant |
| the hard q and the STE | — | `quantized_representation`: `sum_l w_l q_l + (z - z.detach())` (the encoder's identity path included) | the forward value is the plain `sum_l c_l`; the pair loss reaches the codebooks only, so the encoder is not shaped by this loss (section 4) |
| the base RQ-VAE | TIGER (the frozen baseline tree) | `stage2_RQ-VAE/baseline/TIGER_RQ-VAE` | unchanged except `ARM_NAME` |

## 7. The pre-registered predictions

| id | prediction | falsifier |
|---|---|---|
| P1 | the hold-out ranking accuracy (C1) and the margin satisfaction rate (C2) both exceed the frozen TIGER values | the accuracy stays at or below the TIGER's 0.538605 |
| P2 | the SID's prefix discrimination (C5) does not fall below the frozen TIGER's | the LCP(pos) > LCP(neg) rate falls below 0.136889 |
| P3 | the behavioural neighbourhood fidelity (C6) stays within the frozen 5 % relative tolerance | the L1 positive hit rate falls below 0.0793179 |
| P4 | the SID capacity (C3, C4) and the reconstruction (C9) do not regress against the frozen thresholds | the L2 < 10,918 or the rank < 8 or the recon > 8.589273e-05 |
| P5 | the global relation (C7, C8) does not regress | the full J > 0.3383359 or the 1-hop/2-hop J above the 5 % tolerances |
| P6 | the realised gradient share of `L_geo` stays inside the calibrated budget at both probes, and the encoder's representation does not degenerate (the variance ratio and the slope diagnostics do not collapse as in the #269 arm) | the share exceeds the cap, or the variance ratio/slope collapse |

## 8. The evaluation

* the hold-out ranking metrics: `results/research/issue274/measure_ranking.py` (the same script and
  the same caliber for the frozen TIGER and for the candidate; the margin is inherited, never
  re-fitted);
* the SID structure: `results/research/issue271/collect_metrics.py` (the frozen collector);
* the behavioural neighbourhood fidelity: `tools/issue252/layer_metrics.py` (the frozen
  implementation) through `results/research/issue274/write_layer_metrics.py`;
* the global J: the candidate's own `model/global_constraint.py` (the unchanged #269/#271 module),
  exactly as the #272/#273 rounds measured it;
* the curvature fidelity (MAE / Spearman / sign / within-centre ordering) and the prediction
  variance/slope: `results/research/issue273/within_centre_metrics.py`, **as diagnostics only**;
* Stage3 is forbidden in every outcome: no T5 run, no new test, no Recall.

## Round 2 amendment (2026-11-28, after the round-1 result)

Round 1's loss reached the codebooks only (section 4's original design). The measured consequence:
the SID's conditional L2 diversity stayed at 30.97 (the mean L2 per L1) against the #265's 42.65, and
the C3 (the L2 distinct prefixes) failed at 7,929 against 10,918. A short probe established that the
**encoder** channel is the missing lever: routing the same pair term through the reference
implementation's identity path (`q = sum_l c_l + (z - z.detach())`, `TIGER_BEHAVIOR_SAMPLE_RQ-VAE`'s
`quantized_representation` convention) lifts the L2 from 7,929 to 9,058 in **one epoch-equivalent
(100 steps)** at a 2 % encoder share, 9,301 at 5 %, 9,625 at 20 % (the sweep:
`results/research/issue274/probe_encoder_share_sweep.json`).

Round 2 changes exactly one primary thing: the pair term's gradient is routed through the encoder as
well (the reference's STE), and the per-module gradient budget (0.02) now covers the **encoder plus the
three codebooks**, so no module's pair gradient exceeds 2 % of its own reconstruction gradient. The
forward value stays exactly the hard `q = sum_l c_l`; the margin, the frozen pair index, the ramp and
everything else are unchanged.

The C3's and the C9's risk is explicit: the probe shows the L2 raising while the reconstruction ratio
rises with it (1.43-1.47 in the probe's argmin-evaluation metric), so the round-2 run must show
whether the full training keeps the reconstruction at or below 8.589273e-05 while the L2 crosses
10,918.

## Round 3 (the sampling / matching rule) — the frozen amendment

The round-2 evidence bounds the mechanism's frontier (the L2 7,923 the encoder channel included). The
issue's allowed levers include "采样 / 匹配规则"; the round-3 freezes this exact change:

1. The matching support is relaxed from ">= 2 closed and >= 2 open neighbours per centre" to
   ">= 1 closed and >= 1 open". The group formation, the curvature reversal, the 50 %-hold-out rule
   and the extremum selection are unchanged; a 1-vs-1 group's curvature ordering is degenerate, and
   such groups enter with the positive/negative pair the rule produces (the frozen ordering is a
   total order, so the selection is deterministic).
2. The index is `results/research/issue274/pairs_index_cover1.npz` (38,871 groups: 31,095 train,
   7,776 hold-out, against 26,583 / 21,221 / 5,362), built by the same script, with the new manifest
   `pairs_index_cover1.manifest.json` and the sha256 `30b794b5...`.
3. The lambda / the gradient budget / the paired margin are unchanged from the round-2 calibration;
   the capacity budget shrinks with the larger group count per the measured rule.
4. The candidate is `TIGER_pairrank_cover_rq` (the round-2 tree plus the identity save; 19 tests
   passing).
