# HBPE-A1-R2 (#34) — L1 enrichment weight sweep and hard-assignment occupancy safeguard

Stage 2 only. Three 3000-epoch cold starts, all forked read-only from
`TIGER_HBPE_A1` with identical seed 42, KMeans initialisation, 3x256 codebooks, the
original three-level transition contrastive loss, the temperature annealing and the
behaviour ramp. Nothing was warm-started: the only `load_state_dict` from
`rqvae_best.pth` in the trainer is gated on `_ANCHOR_ACTIVE`, which none of these
arms sets, so the seeded checkpoint is a smoke-probe reference and never an
initialisation.

## 1. Setup and verification

Held-out set: `behavior_pairs_valid_s42.npz`, 52,855 transitions whose scored
relations never appear in `train.parquet`, 10 popularity/frequency-matched negatives
each, seed 42. Bootstrap: 1000 replicates, 95% percentile intervals.

The canonical Resample export re-measures **bit-identically** on the same pairs
(L1 ER 16.3895, L1+L2 ER 58.3871), so every comparison below is like-for-like.

**Every comparison is a paired delta.** Both arms' replicate ratios are recomputed on
the *same* resampled rows, so replicate *i* differences the two arms on identical
data. Whether two marginal intervals happen to overlap is not a difference test and is
shown for context only. This matters here: `hbpe_a1`'s L1+L2 improvement is exactly a
case where marginal intervals overlap heavily and the naive reading would mislead.

Negative-sampler audit (unchanged from A1, as step 5 requires):
anchor-bucket match 99.9986%, **positive-bucket match 34.07%**, zero false negatives,
zero duplicate draws, zero self/positive collisions, 24,587 distinct items drawn. The
negatives match the *anchor's* popularity, not the target's. Left untouched to keep
the weight ablation single-factor; flagged below as a separate follow-up.

## 2. Results

| arm | lambda_ER1 | + occ | pos_share L1 | neg_share L1 | **L1 ER** | 95% CI | L1 codes | eff. usage exp(H) | L1+L2 ER | collision |
|---|---:|:---:|---:|---:|---:|:---:|---:|---:|---:|---:|
| Resample | — | — | 0.07944 | 0.00485 | **16.39** | [15.59, 17.22] | 241 | 226.88 | **58.39** | 0.0363 |
| `hbpe_a1` (#31) | 0.020 | no | 0.09218 | 0.00624 | **14.76** | [14.17, 15.41] | 172 | 160.00 | **64.00** | 0.0667 |
| `a1r2_w005` | 0.005 | no | 0.09397 | 0.00591 | **15.92** | [15.27, 16.59] | 185 | 172.02 | **47.19** | 0.0487 |
| `a1r2_w010` | 0.010 | no | 0.09657 | 0.00584 | **16.53** | [15.88, 17.20] | 184 | 169.75 | **74.19** | 0.0545 |
| `a1r2_w010_occ` | 0.010 | yes | 0.09397 | 0.00617 | **15.21** | [14.61, 15.83] | 173 | 161.02 | **70.50** | 0.0612 |

### Paired delta-ER (reference − candidate; CI excluding 0 = significant)

| candidate | reference | level | delta ER | 95% CI | significant |
|---|:---:|:---:|---:|:---:|:---:|
| `hbpe_a1` | Resample | L1 | +1.6259 | [+0.8074, +2.4442] | yes (worse) |
| `a1r2_w005` | Resample | L1 | +0.4713 | [−0.3038, +1.2835] | no |
| **`a1r2_w010`** | **Resample** | **L1** | **−0.1401** | **[−0.9849, +0.6745]** | **no — indistinguishable** |
| `a1r2_w010_occ` | Resample | L1 | +1.1747 | [+0.4302, +1.9393] | yes (worse) |
| `a1r2_w005` | `hbpe_a1` | L1 | −1.1546 | [−1.8696, −0.4468] | yes (better) |
| **`a1r2_w010`** | **`hbpe_a1`** | **L1** | **−1.7660** | **[−2.4634, −1.0521]** | **yes (better)** |
| `a1r2_w010_occ` | `hbpe_a1` | L1 | −0.4512 | [−1.0535, +0.1603] | no |
| any of the four | any | L1+L2 | — | — | no, every interval spans 0 |

## 3. Findings

### 3.1 Halving the weight removes A1's L1 regression, and the regression was real

A1's L1 loss against the baseline is significant (+1.63 in the wrong direction,
CI [+0.81, +2.44]). At `lambda_ER1 = 0.010` that deficit disappears entirely:
`a1r2_w010` sits at −0.14 from the baseline with a CI straddling zero, and it beats A1
by 1.77 with a CI excluding zero. So the mechanism is not what damaged L1; the
coefficient was. A quarter of the coefficient (`w005`) is already past the optimum and
regresses partway back, so the L1 effect is non-monotonic in the weight across the
0.005–0.020 range sampled here.

Codebook health tracks the same story: 241 codes at the baseline, 172 under A1's
0.020, 184 at 0.010, 185 at 0.005. Weaker enrichment pressure leaves more of the
codebook in use.

### 3.2 The L1+L2 numbers are not resolvable at this sample size

Every L1+L2 comparison has a paired interval that spans zero, with a half-width around
±20–30 ER units on a point estimate of 6–15. The reason is the denominator: the
held-out L1+L2 negative count is 31 draws out of 528,550. The ratio's variance is
dominated by that count, so **the L1+L2 enrichment ratio cannot be compared between
arms at 52,855 pairs**, and the apparent ordering `w010` 74.19 > `occ` 70.50 >
`hbpe_a1` 64.00 > `w005` 47.19 carries no signal. Any future claim about L1+L2 needs
either more held-out pairs or a metric with a non-degenerate denominator.

This is also a correction to how #31's result was read. A1's L1+L2 ER of 64.00 against
the baseline's 58.39 is a higher point estimate, but the paired delta is
−5.61 with CI [−29.58, +24.54] — **not significant**. "A1 beats the baseline at L1+L2"
was a point-estimate statement and should not have been presented as a result.

### 3.3 The occupancy guard fires but is two orders of magnitude too weak to matter

`a1r2_w010_occ` differs from `a1r2_w010` by one term: `L_occ = ReLU((H_ref − H(π))/H_ref)^2`
at weight 0.01, measured on a uniform catalogue sample. It is correctly wired — at the
final epoch `H_π = 5.0589` against `H_ref = 5.4243`, so the barrier is genuinely
violated and `L_occ = 0.00454` with an encoder gradient norm of **0.0142**, versus
**9.82** for the enrichment term. The guard contributes about 0.14% of the new-term
gradient.

The result matches: codebook use barely moves (184 → 173 distinct codes, effective
usage 169.75 → 161.02) and held-out L1 ER is 1.32 lower than without it, with a paired
CI [+0.43, +1.94] against the baseline — i.e. the guard made L1 *worse*, not better.
At `lambda_occ = 0.01` and a shortfall of only 0.37 nats out of 5.42, a squared hinge
simply cannot compete with an ER term that is three orders of magnitude larger. Making
this barrier bite would need a weight several orders higher, which is a different
experiment.

### 3.4 The soft ratio is flat where the hard ratio moves

Final-epoch training metrics:

| lambda_ER1 | Soft_ER1 | encoder grad norm | held-out hard L1 ER | L1 codes |
|---:|---:|---:|---:|---:|
| 0.005 | 39.92 | 16.85 | 15.92 | 185 |
| 0.010 | 43.55 | 9.24 | **16.53** | 184 |
| 0.020 (`hbpe_a1`) | 40.33 | 3.51 | 14.76 | 172 |
| 0.010 + occ | 38.21 | 9.82 | 15.21 | 173 |

The gradient norm scales cleanly with the coefficient, but Soft_ER1 moves within
38–44 and is not ordered the way the hard ratio is. The soft quantity is an inner
product between two full softmax distributions, so it measures distribution overlap
rather than discrete hits and saturates early.

## 4. Summary

- `a1r2_w010` is the best condition: held-out L1 ER 16.53, statistically
  indistinguishable from the canonical baseline (paired delta −0.14, CI includes 0)
  and significantly better than A1 (paired delta −1.77, CI excludes 0), with 184 of
  256 L1 codes in use.
- The occupancy guard at `lambda_occ = 0.01` is correctly implemented and verifiably
  violated by the trained model, but is too weak to change the outcome and slightly
  hurts.
- No L1+L2 claim is supportable at this held-out size; the ratio's denominator is too
  small for the sample count.
- A Pareto improvement over the baseline is not achieved on any axis: L1 ER is matched,
  not exceeded, and both L1 codebook spread (184 vs 241) and collision (0.0545 vs
  0.0363) remain worse.

## 5. Follow-ups this round did not do

1. The negatives are matched to the anchor's popularity decile, not the target's
   (34.07% positive-bucket match). Changing it would be a separate single-factor
   experiment; it was deliberately left alone here.
2. The occupancy barrier needs a coefficient that is competitive with the ER term
   before it can be evaluated as a mechanism at all.

## 6. Artefacts

- Trees: `stage2_RQ-VAE/TIGER_HBPE_A1_R2_{W005,W010,W010_OCC}/`
- Exports: `results/stage2_RQ-VAE/TIGER_HBPE_A1_R2_*/<arm>/` — `rqvae_best.pth`,
  `sids_for_hgrec.npy`, `item_sids.json`, 24,587 unique three-level SIDs each
- Metrics: `stage2_RQ-VAE/TIGER_HBPE_A1_R2_*/logs/<arm>/training_metrics.jsonl`
- Report data: `results/stage2_RQ-VAE/hbpe_a1_r2/` (json / csv)
- Scripts: `scripts/analyze_hbpe_r2_enrichment.py`,
  `scripts/hbpe_r2_gradient_selftest.py`, `scripts/hbpe_r2_reference_constants.py`

*No Stage 3 job was started, queued or referenced, and no Recall@10 or NDCG@10 figure
is produced or estimated in this round.*
