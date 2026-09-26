# Iter26 Gate Decision

## Outcome

NO-GO. Iter26 does not satisfy the adoption criterion. The closed-form mapping
with the corrected raw residual medians did not break the persistent downstream
ceiling.

## Numbers

- Stage3 test_R@10: **0.057017009349048554** (n_eval=57439).
- Stage3 test_R@5: 0.03788366789115409.
- Stage3 test_ndcg@10: 0.03133359761524377.
- Adoption threshold (test_R@10): 0.065.
- Canonical Iter11 test_R@10: 0.0597677536 (delta −0.002751).
- Best completed Iter18 test_R@10: 0.0598896220 (delta −0.002873).
- Completed Iter25 test_R@10: 0.0588276258 (delta −0.001811).
- Completed Iter24 test_R@10: 0.05619875 (delta +0.000818).

## Mechanism executed

- Pre-registered bounded branching-plus-raw-residual closed-form layer-scale
  prior `u_l_prior = [0.6145357379232853, 0.5333020920777128,
  0.3814078098431606]`, computed from:
  `B = [19.324911558712664, 1.4605688962651735, 1.0148104414712726]`,
  `m_raw = [1.0, 0.10941, 0.09331]`,
  `c_base = 0.5`, `alpha = 0.2`,
  `s_l = log1p(B_l) / log1p(m_l_raw / min(m_raw))`,
  `z_l = (s_l - mean(s_l)) / (std(s_l) + 1e-12)`,
  `c_l = clip(c_base * exp(alpha * z_l), 0.05, 1.5)`.
- Fixed closed-form curvature; no learnable `c_layer_scale`; no cyclic `c(t)`;
  no curvature regularization.
- Stage2 ran 100,000 global steps on 4-GPU DDP, 1155.9 s wall-clock,
  ckpt every 10k step. The fixed-c invariance contract fired at step
  0/25000/50000/100000 with all four snapshots reporting
  `[0.614536, 0.533302, 0.381408]` within 1e-6 of the locked targets.
- Stage3 ran 150 epochs on 4-GPU DDP with cosine LR + 10% warmup, ndcg
  callbacks disabled (NO_EVAL=True per current protocol); final test
  inference produced the numbers above.

## Hard contract evidence

`scripts/mvg_check.py` records:

```
MVG PASS
closed_form_c_l=[0.6145357379232853, 0.5333020920777128, 0.3814078098431606]
initial_c_live=[0.6145357489585876, 0.5333020687103271, 0.38140779733657837]
invariance_snapshots={0: [...], 25000: [...], 50000: [...], 100000: [...]}
five_step_relative_updates=[0.024..., 0.275..., 0.357..., 0.008..., 0.007..., ...]
```

The MVG invariance step scanned `step ∈ {0, 25_000, 50_000, 100_000}` and the
runtime `_check_fixed_curvature_invariant` helper logged every step without
raising; the on-disk checkpoint `rqvae_best.pth` matches the locked values.

## Stage2 descriptive comparison (informational)

| Metric | Iter26 | Iter18 | Iter25 |
|---|---:|---:|---:|
| full_3token_gini | 0.0681 | 0.0642 | 0.0637 |
| per_layer_mean_gini | 0.2298 | 0.2028 | 0.2002 |
| collision_rate | 7.16% | 6.65% | 6.61% |
| unique_3token_sids | 22827 | 22951 | 22962 |
| H(L1\|L0) bits | 5.4343 | 5.5907 | 5.5874 |
| 3-token Stage3 test_R@10 | 0.057017 | 0.059890 | 0.058828 |

Iter26 geometry is slightly less balanced than Iter18 and Iter25 on every
reported metric, but the difference is small. The downstream recall regressed
relative to every completed baseline except Iter24, and remains 0.008 below the
0.065 adoption threshold.

## Interpretation

This iteration was the cleanest possible one-factor test of the hypothesis that
the Iter25 regression was caused by feeding the wrong residual vector into the
closed-form mapping. With the corrected raw residual medians `[1.0, 0.10941,
0.09331]`, the derived `c_l = [0.6145, 0.5333, 0.3814]` was locked for the entire
100,000-step run and Stage3 was run on the resulting SIDs. The hard contract was
met. Stage3 recall dropped to 0.057017 — below Iter25 (0.058828), below Iter18
(0.059890), and below Iter11 (0.059768).

The persistent class-3 ceiling hypothesis is now closed for the Stage2 closed-form
fixed-curvature mechanism family. Correcting the residual input did not break the
ceiling, which means the binding lever is downstream of the Stage2 curvature
family. The failure mode is the same one Iter25 documented, just at a slightly
worse downstream level.

## Adoption status

- test_R@10 > 0.065: NO (0.057017).
- Comparison vs canonical Iter11 0.059768: NO (-0.002751).
- Comparison vs best completed Iter18 0.059890: NO (-0.002873).
- Improvement over Iter25: NO (regressed by 0.001811).
- Improvement over Iter24: YES (+0.000818) but does not meet the absolute
  threshold.

Iter26 is not adopted as the new promoted baseline. The previous best completed
baseline remains Iter18 (0.0598896220).

## Notes for next iteration

- Iter26 should be the **last** Iter16-cloned experiment. The two one-factor
  perturbations (residual vector corrected, fixed c) together do not break the
  0.060 ceiling. Stage2 closed-form curvature, in all its variants tried, is
  no longer a candidate lever.
- The failed mechanism ledger does not gain a new entry. Iter26 is a
  one-factor replication of Iter16 with the only documented change being the
  residual input vector; the direct-effects checklist all PASS and Agent D's
  three-state judgement was `PARTIAL` (proxy direction held, but the gap to
  canonical closed over the run). This continues the class-3 ceiling pattern.
- Iter27 must move outside Stage2 mechanics. The two remaining levers that
  have not been seriously explored in this project are (a) the Stage1
  embedding model and (b) the Stage3 trainer/protocol. The bridge hypothesis
  that the Stage3 trainer is the binding constraint is now strongly supported
  by three consecutive Stage2 family attempts (Iter16 / Iter25 / Iter26).
- Iter27 should commit to one of those two levers before re-opening the
  per-layer curvature family. The (c) stopping condition
  (`same mechanism tried with 3 different hyperparameter settings, all
  NO-GO`) is now satisfied for the closed-form family; per the skill rules,
  the family should be retired from the candidate pool.
- Audit commit: `2239336f5e0254de70486441fca3b61e8acf1b61`
