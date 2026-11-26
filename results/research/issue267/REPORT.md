# Issue #267 report

## Root cause

The #265 hard-q curvature representation is weak on the complete target
population: full 28,391,948-triple q Spearman `-0.09342`, z `-0.09208`, q
prediction variance `0.88857` vs target `2.87297`, and a +1.7986 mean offset.
Within centres the median Spearman is only `0.08695` (q) / `0.08758` (z), while
the overall negative sign is dominated by centre offsets. The sample gate probe
shows learned gate improves MAE over equal weight (`2.05557` vs `2.43348`) but is
not a full-table proof. See `DIAGNOSIS.md`, `analysis/CURVATURE_FULL_109_Q.json`,
`analysis/CURVATURE_FULL_109_Z.json`, `analysis/CENTER_ORDER_109.json`,
`analysis/GATE_PROBE.json`.

## Mechanism

Parent #265 is retained: frozen GNN/token cache, Euclidean residual selection,
floor/last-level Sinkhorn, point-value local mixed curvature and its calibration.
The only new term is a same-centre softplus sign-ranking loss on the actual hard
q, with target-derived `tau=1.0516836255159168`, ties excluded at `1e-6`, and
`alpha = 0.2 * ||grad L_point|| / ||grad L_rel||` frozen on the fixed initial
sample. Full formula and gradient route: `CONTRACT_MECHANISM.md`.

Actual Stage2 calibration: alpha `0.3075788718799832`, point/relative gradient
norms `7.125525425767778 / 4.633299668611924`, 378 valid pairs, 0 ties/cross-centre
pairs. The structural budget is recomputed after adding the term; no point term
is removed.

## Stage2

Run `stage2_88816eeca29442c7b3284f4d4a68f610`, 3000 epochs, four ranks,
seed42, final-epoch checkpoint. Final collision `0.04534916825964941`; wall
`811.659 s`; SID `[24587, 4]`, `unique_after_extend=24587`. The initial export
table differed from the checkpoint on 238/24587 items; the existing
`export_check(rewrite=True)` interface rewrote the export from the checkpoint and
the manifest records the original difference; the archive binding is
`f03834cf…`.

The §63 layer record is at
`results/stage2_RQ-VAE/TIGER_relative_curvature_ranking_rq/stage2_metrics/0b25f3b5e0a9e43d2f21c553cf3689467e1d6542cadab4bfc70484689dd0a454.json`
(sha256 `118ff26f…`): L3 collision `0.04547118395900274`, L1 Hard ER
`8.3595751`, L1 positive hit rate `0.06100385`, 57439 valid rows.

## Pre-Stage3 score

Before Stage3, with the frozen formula and no new Recall read:

- `R = 8.386929403059185e-05`
- `G_gate = 0.10466435496425182`
- `G_L3 = 0.0005665696095017952`
- `N_R = 0.07529986029441926`, `N_gate = 1.1096520270125871`,
  `N_L3 = 2.089176670111412`
- **S = 1.2921335084032926**, above all ten historical scores (max `0.7411182932`).

This is a pre-registered out-of-range prediction, not a rank-selection result.

## Independent full comparison

Relative to #265:

| space | metric | #265 | new | delta |
|---|---|---:|---:|---:|
| q | Pearson | -0.03310 | 0.02130 | +0.05440 |
| q | Spearman | -0.09342 | -0.03373 | +0.05968 |
| q | MAE | 2.24595 | 2.20395 | -0.04200 |
| q | prediction var | 0.88857 | 0.88073 | -0.00784 |
| z | Pearson | -0.02838 | 0.02340 | +0.05178 |
| z | Spearman | -0.09208 | -0.03510 | +0.05698 |
| z | MAE | 2.24801 | 2.20852 | -0.03949 |

Within-centre q mean/median Spearman: `0.09581 / 0.09119` (new) vs
`0.09288 / 0.08695` (#265); positive-centre fraction `0.63785` vs `0.64134`.
The relative term moves the ordering in the intended direction, but the full-table
Spearman remains negative and the prediction variance remains compressed.

## Stage3

One frozen 150-epoch run, four-rank DDP, seed42, beam20, no retry:
`results/stage3_T5Train/TIGER_relative_curvature_ranking_rq/logs/Amazon_2023_Instruments/Nov-26-2026_17-58-19`
(PID `1180973`). `n_eval=57439`.

| metric | value |
|---|---:|
| test_recall@5 | 0.03300893121398353 |
| **test_recall@10** | **0.05078431031180905** |
| test_ndcg@5 | 0.021512508774359167 |
| test_ndcg@10 | 0.027244785026023777 |

Parent #265 (`TIGER_euclid_local_curve_supervision_rq`) scored
`test_recall@10 = 0.052594926791900976`; the relative-ranking arm is
`-0.0018106164800919256` (`-3.4426%`) below it.

### Eleven-arm prospective check

`analysis/STAGE3_RECALL_11.json`. Recall-ascending order:
`101, 104, 100, 107, 106, **110**, 105, 108, 109, 103, 102`; the new arm is
6th of 11.

| statistic | historical 10 | incl. 110 |
|---|---:|---:|
| Pearson(S, Recall@10) | 0.928526 | 0.513532 |
| Spearman | 1.000000 | 0.863636 |
| Kendall concordant/discordant | 45 / 0 | 50 / 5 (tau 0.818182) |

Prediction vs outcome: S ranked 110 as the highest score (11th of 11, above the
frozen range). Actual recall rank is 6th; five arms (`101, 104, 100, 107, 106`)
scored lower on S but higher on Recall@10. The interval was again out-of-range
(above), this time on the wrong side of five arms, so the ten-arm monotone fit
did not transfer. S is retained as a diagnostic; it is not an admission gate.

## Performance (C8)

Stage2 inclusive wall `811.659 s` vs parent `813.357 s`; the per-step relative
term is one softplus over same-centre pairs plus one graph backward, with the
alpha calibration paid once before the first optimizer step. No extra forward of
the frozen GNN/token encoder, no additional Stage2/Stage3 run, no skipped check.
Stage3 test inference `135.5 s` for rank 0 (four-shard mean ≈ 131 s/shard,
`n_eval=57439`).

## Conclusion

The relative same-centre ranking term did what it was designed to do locally — it
raised full-table q/z Pearson above zero and improved within-centre ordering and
MAE — but it did not convert into recommendation gain: Stage3 `test_R@10`
decreased from `0.052595` to `0.050784`, and the pre-Stage3 score again failed to
predict the new arm's rank. The mechanism is a valid negative result under the
frozen protocol; the arm is retained with its evidence and is not retried.
