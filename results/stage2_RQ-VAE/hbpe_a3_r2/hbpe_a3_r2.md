# HBPE-A3-R2 (#36) — Negative-sharing cap and hard-occupancy guard

Stage 2 only. Two 3000-epoch cold starts forked read-only from `TIGER_HBPE_A3` with
identical seed 42, KMeans init, 3x256 codebooks, the original VQLayer/STE/last-level
Sinkhorn, the original three-level transition contrastive loss, the same
`history[-1] -> target` positives and the same fixed sampler. A3's two ER terms and its
`0.01*(L_cov1 + L_cov12)` floors are kept unchanged; the only additions are the
negative-sharing cap and, in one condition, the occupancy guard. The frozen canonical
Resample export is loaded read-only as a **threshold measurer**, never warm-started.

## 1. Setup and verification

Held-out set `behavior_pairs_valid_s42.npz`: 52,855 transitions never observed in
training, 10 matched negatives each, seed 42, 1000-replicate bootstrap. Canonical
Resample re-measures bit-identically (L1 ER 16.3895, L1+L2 ER 58.3871).

All comparisons are **paired delta-ER** on the same resampled rows. Marginal interval
overlap is not a difference test.

**The teacher reference.** `B_neg(tau)` is evaluated online, per batch, under
`no_grad` on exactly the items the current model's `S_neg` was measured on, using the
same soft formulas with the teacher's own reference distance gaps scaled by the same
`tau(epoch)`. A soft threshold is therefore compared against a soft quantity and never
against a hard baseline rate. Measured once offline, the teacher's constants are
`D_ref1 = 0.010180`, `D_ref2 = 0.001840`, `B1_neg(1.0) = 0.004531`,
`B12_neg(1.0) = 2.0748e-05`. Both runs independently converged to the same logged
`B1_neg = 0.0048598` / `B12_neg = 2.0918e-05`, so the measurement is reproducible.

Self-tests confirm the cap fires above its reference, is exactly zero at it and below
it, that the 1e-8 floor in `max(B_neg, 1e-8)` keeps the penalty finite, and that both
new terms have identically zero codebook gradients.

## 2. Results

| arm | pos_share L1 | neg_share L1 | **L1 ER** | 95% CI | pos_share L1+L2 | neg_share L1+L2 | **L1+L2 ER** | L1 codes | eff. usage | collision |
|---|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|
| Resample | 0.07944 | 0.00485 | **16.39** | [15.59, 17.22] | 0.00342 | 0.00006 | **58.39** | 241 | 226.88 | 0.0363 |
| `hbpe_a2` (#32) | 0.12686 | 0.01952 | **6.50** | [6.32, 6.70] | 0.01256 | 0.00048 | **26.25** | 58 | 54.95 | 0.1335 |
| `hbpe_a3` (#33) | 0.13669 | 0.02037 | **6.71** | [6.53, 6.91] | 0.01063 | 0.00038 | **27.96** | 55 | 51.45 | 0.1203 |
| `a3r2_negcap` | 0.04223 | 0.01108 | **3.81** | [3.63, 3.99] | 0.00216 | 0.00016 | **13.26** | 145 | 103.57 | 0.0487 |
| `a3r2_negcap_occ` | 0.04698 | 0.01355 | **3.47** | [3.32, 3.63] | 0.00201 | 0.00022 | **9.14** | 101 | 80.69 | 0.0545 |

### Paired delta-ER (reference − candidate; CI excluding 0 = significant)

| candidate | reference | level | delta ER | 95% CI | significant |
|---|:---:|:---:|---:|:---:|:---:|
| `a3r2_negcap` | `hbpe_a3` | L1 | +2.9001 | [+2.6630, +3.1419] | yes (**worse**) |
| `a3r2_negcap` | `hbpe_a3` | L1+L2 | +14.7044 | [+9.4533, +20.4682] | yes (**worse**) |
| `a3r2_negcap` | `hbpe_a2` | L1 | +2.6869 | [+2.4404, +2.9268] | yes (worse) |
| `a3r2_negcap` | `hbpe_a2` | L1+L2 | +12.9892 | [+8.0800, +18.1354] | yes (worse) |
| `a3r2_negcap` | Resample | L1 | +12.5787 | [+11.7723, +13.4109] | yes (worse) |
| `a3r2_negcap` | Resample | L1+L2 | +45.1313 | [+27.5695, +76.1960] | yes (worse) |
| `a3r2_negcap_occ` | `hbpe_a3` | L1 | +3.2450 | [+3.0323, +3.4584] | yes (worse) |
| `a3r2_negcap_occ` | `hbpe_a3` | L1+L2 | +18.8223 | [+14.2530, +24.0740] | yes (worse) |
| `a3r2_negcap_occ` | Resample | L1 | +12.9236 | [+12.1226, +13.7548] | yes (worse) |
| `a3r2_negcap_occ` | Resample | L1+L2 | +49.2492 | [+32.6343, +81.8375] | yes (worse) |

## 3. Findings

### 3.1 The cap did exactly what it was asked to do, and that is what broke it

The diagnosis in #33 was right about the failure mode and wrong about the fix. Negative
sharing was the problem: A3 sat at 0.02037 against the baseline's 0.00485. The cap
works. Held-out L1 negative sharing fell from 0.02037 to **0.01108**, a 46% reduction,
and the L1+L2 negative rate fell from 0.00038 to 0.00016.

But the enrichment ratio fell with it, and much further:

| | pos_share L1 | neg_share L1 | L1 ER |
|---|---:|---:|---:|
| `hbpe_a3` | 0.13669 | 0.02037 | 6.71 |
| `a3r2_negcap` | **0.04223** | 0.01108 | **3.81** |

Positive sharing collapsed by 69% while negative sharing fell by 46%. Since
`ER = pos_share / neg_share`, a constraint applied only to the denominator pushes the
optimiser to satisfy it by shrinking the numerator, and that is what happened. On the
training batch the same pattern is starker: `S1_pos` fell from A3's 0.15782 to
**0.04683** while `S1_neg` was held at 0.01129 against a reference of 0.00486.

The cap was never inactive: `negcap1_active_rate = 1.000`, `negcap12_active_rate =
0.833`, final `L_neg1 = 1.835`, `L_neg12 = 47.84`. Compare the ER terms at
`L_ER1 = −1.42`, `L_ER12 = −4.59`. The **cap's Encoder gradient norm is 78.7** against
A3's ~2.3 for the whole enrichment objective — the new term carries roughly 34x the
gradient of the thing it is supposed to regularise.

### 3.2 The occupancy guard inherits the damage

`a3r2_negcap_occ` is worse than `a3r2_negcap` on every enrichment axis: L1 ER 3.47 vs
3.81, L1+L2 9.14 vs 13.26, and it uses *fewer* distinct L1 codes (101 vs 145) despite
the guard existing to increase them.

The mechanism is visible in the logs. At the final epoch `H_π = 4.3821` against
`H_ref = 5.4243`, so the guard is violated more deeply than in the #34 occupancy run
(`H_π = 5.0589`), and `L_occ = 0.0369` with an Encoder gradient norm of **0.1699** —
against the cap's **103.83** in the same run. The guard is 600x weaker than the cap it
is sitting behind, and the cap has already flattened the distribution, so the guard
simply follows it down instead of pushing back. Adding a weak entropy term on top of an
over-powered negative cap produces a worse version of the same problem.

### 3.3 The cap does repair codebook health, which is worth recording

This is the one thing that improved. Collision drops from 0.1203 to **0.0487** — better
than A1 (0.0667) and close to the baseline's 0.0363 — and distinct L1 codes rise from 55
to 145 with effective usage 51.45 to 103.57. So a negative-sharing constraint really
does control the codebook collapse; it just does so by suppressing behaviour-selective
sharing rather than by spreading the codebook over more behavioural structure, and the
enrichment ratio pays for it.

That also explains why the two constraints fail in different ways. #33's coverage floor
on the *positive* rate never fired, because positive sharing was never the binding
constraint. #36's cap on the *negative* rate fires constantly, because negative
sharing is what actually collapses — but constraining it alone, with nothing holding
positive sharing up, starves the numerator.

### 3.4 Soft and hard ratios diverge again, in the same direction as before

| arm | Soft_ER1 | Soft_ER12 | held-out L1 ER | held-out L1+L2 ER |
|---|---:|---:|---:|---:|
| `hbpe_a3` | 8.19 | 102.98 | 6.71 | 27.96 |
| `a3r2_negcap` | 4.19 | 40,336 | 3.81 | 13.26 |
| `a3r2_negcap_occ` | 3.80 | 39.35 | 3.47 | 9.14 |

Here the soft and hard ratios at least agree in *ordering*, unlike the #31–#33 rounds
where they inverted. But `Soft_ER12 = 40,336` is again a meaningless number: `S12_neg`
is 1.31e-04 and the ratio is dominated by that denominator. Reporting it would be
reporting a division artefact.

## 4. Summary

- The negative-sharing cap reduces negative sharing as designed (0.02037 -> 0.01108
  held out, 46%) but costs far more positive sharing than it saves in the denominator,
  and L1 ER falls from 6.71 to 3.81 — significantly worse than A3, A2 and the baseline
  on paired deltas at both levels.
- It carries about 34x the Encoder gradient of the ER terms it regularises, so the
  objective it shapes is mostly the cap.
- The occupancy guard on top is 600x weaker than the cap and cannot recover it; it
  makes every enrichment number slightly worse while using fewer codes.
- The one genuine gain is collision: 0.1203 -> 0.0487, with distinct L1 codes 55 -> 145.
  A negative-sharing constraint does control the collapse, but by flattening
  behaviour selectivity rather than by spreading the codebook over more structure.
- **Negative result overall.** Neither safeguard moves any arm toward the canonical
  baseline. The finding for a next round is that holding negative sharing down requires
  holding positive sharing up at the same time; the cap alone trades one for the other.

## 5. Artefacts

- Trees: `stage2_RQ-VAE/TIGER_HBPE_A3_R2_{NEGCAP,NEGCAP_OCC}/`
- Exports: `results/stage2_RQ-VAE/TIGER_HBPE_A3_R2_*/<arm>/` — `rqvae_best.pth`,
  `sids_for_hgrec.npy`, `item_sids.json`, 24,587 unique SIDs each
- Metrics: `stage2_RQ-VAE/TIGER_HBPE_A3_R2_*/logs/<arm>/training_metrics.jsonl`, with
  `S1_pos, S1_neg, S12_pos, S12_neg, Soft_ER1, Soft_ER12, L_neg1, L_neg12,
  negcap1_active_rate, negcap12_active_rate, B1_neg, B12_neg, L_occ, H_pi, H_ref_L1,
  effective_codes_exp_H, hard_distinct_L1_codes` logged every epoch, together with the
  in-batch hard prefix sharing rates
- Report data: `results/stage2_RQ-VAE/hbpe_a3_r2/` (json / csv)
- Reference constants and sampler audit: `results/stage2_RQ-VAE/hbpe_r2_reference_constants.json`

*No Stage 3 job was started, queued or referenced, and no Recall@10 or NDCG@10 figure
is produced or estimated in this round.*
