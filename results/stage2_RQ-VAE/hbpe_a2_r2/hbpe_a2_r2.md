# HBPE-A2-R2 (#35) — Preventing L1 collapse under two-level soft enrichment

Stage 2 only. Three 3000-epoch cold starts forked read-only from `TIGER_HBPE_A2` with
identical seed 42, KMeans init, 3x256 codebooks, the original three-level transition
contrastive loss, frozen per-level reference distance gaps with `tau` annealed
1.0 -> 0.2, and the same anchor-bucket negative sampler. `lambda_ER1` is pinned at 0.02
in all three; the only differences are `lambda_ER12` and, in one condition, whether
`s12` is built from a stop-gradiented `s1`. No A1/A2/A3 checkpoint is read.

## 1. Setup and verification

Held-out set `behavior_pairs_valid_s42.npz`: 52,855 transitions never observed in
training, 10 matched negatives each, seed 42, 1000-replicate bootstrap. The canonical
Resample export re-measures bit-identically (L1 ER 16.3895, L1+L2 ER 58.3871).

All comparisons below are **paired delta-ER** — both arms recomputed on the same
resampled rows in each replicate. Marginal interval overlap is not used as a
difference test.

Gradient self-tests (`scripts/hbpe_r2_gradient_selftest.py`, all passing) confirm the
isolation is real: under `sgL1`, `d(L_ER12)/dp1` is exactly zero while
`d(L_ER12)/dp2` is non-zero, and `L_ER1`'s own `s1` is never detached. `r2_proxy`
gradients survive the STE, and every new term has an identically zero codebook
gradient at L1/L2/L3.

## 2. Results

| arm | lambda_ER12 | s12 form | pos_share L1 | neg_share L1 | **L1 ER** | 95% CI | L1 codes | eff. usage | L1+L2 ER | collision |
|---|---:|:---:|---:|---:|---:|:---:|---:|---:|---:|---:|
| Resample | — | — | 0.07944 | 0.00485 | **16.39** | [15.59, 17.22] | 241 | 226.88 | **58.39** | 0.0363 |
| `hbpe_a1` (#31) | 0 (no L2) | — | 0.09218 | 0.00624 | **14.76** | [14.17, 15.41] | 172 | 160.00 | **64.00** | 0.0667 |
| `hbpe_a2` (#32) | 0.020 | `s1 * s2` | 0.12686 | 0.01952 | **6.50** | [6.32, 6.70] | 58 | 54.95 | **26.25** | 0.1335 |
| `a2r2_l2w002` | 0.002 | `s1 * s2` | 0.08994 | 0.00637 | **14.12** | [13.52, 14.71] | 174 | 161.02 | **46.51** | 0.0741 |
| `a2r2_l2w005` | 0.005 | `s1 * s2` | 0.09430 | 0.00761 | **12.39** | [11.88, 12.87] | 147 | 135.24 | **38.28** | 0.0876 |
| `a2r2_l2w005_sgL1` | 0.005 | `stopgrad(s1) * s2` | 0.09649 | 0.00765 | **12.62** | [12.13, 13.14] | 143 | 132.56 | **44.27** | 0.0874 |

### Paired delta-ER (reference − candidate; CI excluding 0 = significant)

| candidate | reference | level | delta ER | 95% CI | significant |
|---|:---:|:---:|---:|:---:|:---:|
| `a2r2_l2w002` | `hbpe_a2` | L1 | −7.6259 | [−8.1648, −7.0682] | yes (better) |
| `a2r2_l2w005` | `hbpe_a2` | L1 | −5.8910 | [−6.3270, −5.4491] | yes (better) |
| `a2r2_l2w005_sgL1` | `hbpe_a2` | L1 | −6.1198 | [−6.5847, −5.6621] | yes (better) |
| `a2r2_l2w002` | `hbpe_a2` | L1+L2 | −20.2629 | [−36.7502, −9.7672] | yes (better) |
| `a2r2_l2w005` | `hbpe_a2` | L1+L2 | −12.0308 | [−23.2998, −3.7691] | yes (better) |
| `a2r2_l2w005_sgL1` | `hbpe_a2` | L1+L2 | −18.0232 | [−31.4034, −8.8103] | yes (better) |
| `a2r2_l2w002` | `hbpe_a1` | L1 | +0.6400 | [+0.0774, +1.2484] | yes (worse) |
| `a2r2_l2w005` | `hbpe_a1` | L1 | +2.3749 | [+1.7872, +2.9701] | yes (worse) |
| `a2r2_l2w005_sgL1` | `hbpe_a1` | L1 | +2.1461 | [+1.5642, +2.7251] | yes (worse) |
| `a2r2_l2w005` | `hbpe_a1` | L1+L2 | +25.7241 | [+9.8001, +45.7532] | yes (better) |
| `a2r2_l2w005_sgL1` | `hbpe_a1` | L1+L2 | +19.7317 | [+3.2481, +39.2680] | yes (better) |
| all three | Resample | L1 | +2.27 / +4.00 / +3.77 | — | yes (all still below baseline) |

## 3. Findings

### 3.1 The L2 weight was the whole problem

The #32 hypothesis was that `lambda_ER12 = 0.02` was simply too strong. It was. Cutting
it tenfold recovers almost all of the loss:

| lambda_ER12 | L1 ER | L1 codes | eff. usage | neg_share L1 |
|---:|---:|---:|---:|---:|
| 0.020 (`hbpe_a2`) | 6.50 | 58 | 54.95 | 0.01952 |
| 0.005 | 12.39 | 147 | 135.24 | 0.00761 |
| 0.002 | **14.12** | **174** | **161.02** | **0.00637** |
| 0 (no L2 term, `hbpe_a1`) | 14.76 | 172 | 160.00 | 0.00624 |

The collapse was monotone in the coefficient. At 0.002 the codebook is essentially
back to A1's health (174 vs 172 distinct codes) and the negative sharing rate is within
3% of A1's. The L1+L2 ratio also improves significantly against A2 at every weight
(−20.26, −12.03, −18.02).

This is a clean single-factor result: the codebook collapse in #32 was an artefact of
the coefficient, not of two-level supervision as such.

### 3.2 Gradient isolation of the L1 branch adds nothing on top of the weight cut

`sgl1` versus plain `l2w005` — the only difference is `stopgrad(s1)` inside `s12`:

| | L1 ER | L1 codes | eff. usage | L1+L2 ER | collision |
|---|---:|---:|---:|---:|---:|
| `a2r2_l2w005` | 12.39 | 147 | 135.24 | 38.28 | 0.0876 |
| `a2r2_l2w005_sgL1` | 12.62 | 143 | 132.56 | 44.27 | 0.0874 |

`sgl1` is *worse* on L1 and its codebook is slightly smaller. Cutting the L1 gradient
path out of `s12` bought nothing, which means the L1 damage in #32 was never
transmitted through `s1` in the first place — it was transmitted through the shared
gradient magnitude. At `lambda_ER12 = 0.005` the `L_ER12` gradient norm on the Encoder
is 1056.8 against 10.3 for `L_ER1`, a ratio of 103:1, so `L_ER1`'s contribution to the
total gradient is negligible regardless of whether the two share a subgraph.

This also retires the "L2 drags L1 through `s1`" hypothesis. It was a reasonable
guess, but the gradient geometry does not support it: the cosine between the two ER
gradients is +0.144 at `l2w002`, **−0.033** at `l2w005` and +0.094 at `sgl1`. The two
terms were never strongly aligned or opposed at any weight, so there was no coherent
"shared direction" to isolate.

### 3.3 Soft_ER12 is meaningless at this operating point

Final-epoch training metrics:

| arm | Soft_ER1 | Soft_ER12 | hard L1 ER (train) | hard L1+L2 ER (train) | held-out L1 ER | held-out L1+L2 ER |
|---|---:|---:|---:|---:|---:|---:|
| `hbpe_a2` | 7.83 | 89.32 | — | — | 6.50 | 26.25 |
| `a2r2_l2w002` | 28.14 | **675,634** | 28.30 | 85.67 | 14.12 | 46.51 |
| `a2r2_l2w005` | 20.18 | **171,098** | 20.31 | 97.33 | 12.39 | 38.28 |
| `a2r2_l2w005_sgL1` | 19.02 | **358,127** | 18.97 | 173.83 | 12.62 | 44.27 |

`S12_neg` sits around 1.6e-05 to 2.0e-05, so dividing `S12_pos` by it produces a ratio
of order 1e5. These are not enrichment ratios in any meaningful sense — the soft
two-level quantity is measuring the product of two already-small soft overlaps.

The training-side hard statistics, by contrast, track the held-out ones closely
(l2w002 train hard L1 ER 28.30 vs held-out 14.12 in absolute terms, but the *ordering*
across the three conditions matches exactly: 28.30 > 20.31 > 18.97 against held-out
14.12 > 12.39 > 12.62 in rank order for the two heaviest). If a soft ratio is wanted,
the hard in-batch one should be logged instead.

### 3.4 No configuration recovers the baseline

The best R2 condition, `l2w002`, reaches L1 ER 14.12 — still significantly below the
canonical baseline (paired delta +2.27 in the wrong direction, CI [+1.49, +3.05]) and
significantly below A1 (+0.64, CI [+0.08, +1.25]). Two-level enrichment at any weight
tested here costs L1 selectivity relative to supervising L1 alone. L1+L2 improves over
A1 significantly (paired delta +17.49, CI [−0.64, +36.89] for l2w002 — which actually
straddles zero — while l2w005 and sgl1 do exclude zero), but the L1+L2 denominator
remains too small for these intervals to be tight.

## 4. Summary

- `lambda_ER12` was the cause of the #32 L1 collapse. Cutting it from 0.02 to 0.002
  recovers L1 ER from 6.50 to 14.12 and the L1 codebook from 58 to 174 codes, both
  significantly against A2 on paired deltas.
- Gradient isolation (`stopgrad(s1)`) adds nothing on top of the weight cut and
  slightly hurts L1; the gradient cosine shows the two terms were never co-aligned.
- No condition reaches the canonical baseline on L1. The best is still 2.27 ER units
  below it, significantly.
- `Soft_ER12` reaches 6.8e5 and should not be used as a selection criterion.

## 5. Artefacts

- Trees: `stage2_RQ-VAE/TIGER_HBPE_A2_R2_{L2W002,L2W005,L2W005_SGL1}/`
- Exports: `results/stage2_RQ-VAE/TIGER_HBPE_A2_R2_*/<arm>/` — `rqvae_best.pth`,
  `sids_for_hgrec.npy`, `item_sids.json`, 24,587 unique SIDs each
- Metrics: `stage2_RQ-VAE/TIGER_HBPE_A2_R2_*/logs/<arm>/training_metrics.jsonl`, with
  `L_ER1, L_ER12, S1_pos, S1_neg, S12_pos, S12_neg, Soft_ER1, Soft_ER12, T1, T2`,
  the per-term Encoder gradient norms `g_er1` / `g_er12` and their cosine, and the
  in-batch hard prefix sharing rates and codebook entropy, logged every epoch
- Report data: `results/stage2_RQ-VAE/hbpe_a2_r2/` (json / csv)

*No Stage 3 job was started, queued or referenced, and no Recall@10 or NDCG@10 figure
is produced or estimated in this round.*
