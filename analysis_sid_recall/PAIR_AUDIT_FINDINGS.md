# Are the 182 (SID, recall) pairs real pairs?

`analysis_sid_recall/sid_recall.csv` pairs every recoverable SID table with a
Stage 3 recall. `collect.py` paired them by the run's own `train_start` record,
which names the SID path the run generated from. That is consumption evidence,
not a proof that the two artifacts were published together. This audit applies
two mechanical criteria and then re-computes the curvature statistics on the
survivors.

## The two criteria

1. **Same commit.** `git log --diff-filter=A -1 --format=%H%x09%ct -- <path>`
   gives each file's *introduction* commit. The SID table and the Stage 3 result
   file that carries the recall must be introduced by the same commit. The
   collected copies under `analysis_sid_recall/sids/` are ignored: they only
   prove the content was collected, not that it belonged to the run.
2. **Same directory name.** Under `results/stage2_RQ-VAE/` the mechanism
   directory and under `results/stage3_T5Train/` the variant directory must
   carry the same name. Classes: `EXACT` (identical), `FLATTENED` (identical
   after `/` became `__`), `LEAF` (only the last component), `NONE`.

Tools: `pair_audit.py` (the per-pair verdict, `pair_audit.csv`) and
`audit_correlation.py` (the statistics by group, `AUDIT_CORRELATION.md`). The
join key is the SID's sha256 — **never** the variant name, because 182 rows
carry only 168 distinct variant names (`curvature_RQ-VAE` alone has 11 rows,
each with its own SID content and its own dated Stage 3 run).

## The counts

| verdict | pairs |
|---|---|
| `VERIFIED` (same commit **and** same name) | **73** |
| `NAME_OK_COMMITS_DIFFER` | 20 |
| `COMMIT_OK_NAME_DIFFERS` | 13 |
| `NAME_OK_UNTRACKED` (the recall file never entered git) | 17 |
| `UNVERIFIED` (neither) | 59 |

* SID path tracked in git: 182/182; the recall file tracked: 164/182.
* Same introduction commit: 86 pairs, spread over **73 distinct commits** (at
  most 6 pairs share one commit), so the signal is not an artefact of the
  history rewrite that squashed some trees.
* Same directory name: 110/182 (100 `EXACT`, 10 `FLATTENED`); 72 are `NONE`.
  On disk today only 34 of the 54 `results/stage2_RQ-VAE/*` directories have an
  identically named `results/stage3_T5Train/*` directory.

### Time, as the primary yardstick

164 pairs have both commit times. 86 have gap 0 (same commit). The 78 that
differ have a median gap of **11.9 hours** and a maximum of 25.2 days. Four have
a *negative* gap — the recall was committed before the SID file appeared:
three are the `results/research/sid_generation_forensics/inputs/*` tables (whose
only tracked path is a collected copy, so the negative gap is an artefact of the
path chosen), and one is real: `tiger_sample_MARGIN`, 15 minutes.

### Consumption evidence, as a third column

The Stage 3 run's own `train_start` record names the SID it generated from. It
matches one of the row's SID paths for **178/182** pairs. The four exceptions:

| variant | SID path the run names | note |
|---|---|---|
| `tiger_fork_hyper_l1_s42` | `results/research/sid_generation_forensics/inputs/hyper_l1_s42/item_sids.json` | a forensics input, not a Stage-2 arm |
| `tiger_fork_hyper_l1_br_s42` | `.../hyper_l1_br_s42/item_sids.json` | same |
| `tiger_fork_hyper_l1_br_euclid_s42` | `.../hyper_l1_br_euclid_s42/item_sids.json` | same |
| `TIGER_GDELTA-SUCC-PULL-RQ` | `.../_regime_batch1024/item_sids.json` | path form differs from the row's list |

So the table is not arbitrary: in 178 cases the run's own log says which SID it
consumed. The two criteria answer a stricter question — whether the artifacts
were *published* as one pair — and 73 pairs pass it.

## What the statistics look like on each set

`per_arm_features_32edge.json`, 32 edges and 3 null replicates per (arm, level);
the `ALL` column reproduces `curvature_summary.json` exactly for the four
z-features (0.0000 delta).

| feature | ALL (182) | CODE_PATH_OK (178) | SAME_COMMIT (86) | VERIFIED (73) |
|---|---|---|---|---|
| `z_curv_L1` | −0.091 (p 0.24) | −0.091 (p 0.24) | +0.053 (p 0.63) | +0.150 (p 0.21) |
| `z_curv_L2` | **+0.150 (p 0.035)** | **+0.161 (p 0.036)** | +0.074 (p 0.50) | +0.129 (p 0.28) |
| `z_curv_L3` | **+0.145 (p 0.049)** | +0.140 (p 0.077) | +0.029 (p 0.80) | −0.044 (p 0.71) |
| profile (`L1−L3`) | −0.095 (p 0.23) | −0.095 (p 0.23) | +0.051 (p 0.64) | +0.141 (p 0.24) |
| `sign_change_L1_L3` | −0.087 (p 0.28) | −0.087 (p 0.28) | +0.175 (p 0.11) | +0.106 (p 0.42) |

Recall medians are identical across groups (0.0580), but the tails are not: the
verified set spans [0.0423, 0.0634] while the unverified set spans
[0.0306, 0.0643].

## Verdict

* The only correlations that reach p ≈ 0.05 are the ones computed on the full
  table: `z_curv_L2` +0.150 (p 0.035) and `z_curv_L3` +0.145 (p 0.049).
* Restricting to pairs whose SID and recall were introduced by the same commit
  removes them: +0.074 and +0.029, p 0.50 and 0.80. The strict joint set gives
  +0.129 (p 0.28) and −0.044 (p 0.71).
* The level-1 and profile features change sign between groups, and a change of
  the edge-sampling budget alone moves the same features by up to 0.19 with sign
  reversals (`curvature/stability.md`).
* The verified subset is smaller and lacks the low-recall tail, so "no
  association" cannot be *proven* from it either. The defensible statement is
  narrower and stronger: **the curvature–recall association is not reproducible
  under the pairing criteria or under the sampling budget, so it cannot be
  claimed in either direction.**

## What follows

1. `curvature/report.md`, `curvature/stability.md` and `docs/issue128_status.md`
   carry the caveat; the unaudited numbers are labelled as such.
2. Any future use of `sid_recall.csv` must filter on the audit — a
   `verified` column is available in `pair_audit.csv`.
3. The four Stage 2/Stage 3 arms of Issue #121 (baseline and the three
   mechanism rounds) are unaffected: their recalls come from their own
   `results/stage3_T5Train/<variant>/` runs, and the mechanism name matches the
   Stage 2 directory exactly (`TIGER_EQVOTE-HYP-RQ` in both trees).

## The composition problem (the verified set is one series)

The two criteria are not neutral about *which* arms they keep. Folding the
`_iterNN` suffix into its parent series:

| tier | pairs | series | largest series share |
|---|---|---|---|
| A — same commit and same name (rule 64b) | 73 | 2 | **99%** (`curvature_RQ-VAE`, 72) |
| B — same commit and the run's own `code_path` | 86 | 5 | 94% |
| C — the run's own `code_path` only | 178 | 40 | 58% |

So the strict set is essentially the `curvature_RQ-VAE` iteration sweep: a
single mechanism family, recall range [0.0423, 0.0634]. The binding criterion is
*co-publication* (the same commit), not the directory name — B, which drops the
name rule, is still 94% one series. The diverse set only exists at C, where the
SID and the Stage 3 result were often committed separately (the median gap is
11.9 hours; four pairs have the recall committed before the SID).

What this means for the numbers:

* the null result on A is a **within-series** statement (the curvature sweep
  iterations do not track their own recall); it cannot be read as evidence that
  curvature fails *across* mechanisms;
* the only cross-mechanism estimate is C: `z_curv_L2` +0.161 (p 0.036),
  `z_curv_L3` +0.140 (p 0.077), level 1 and the profile negative and null — and
  it moves by up to 0.29 when the per-arm sample or the null-replicate count
  changes (`CONVERGENCE_FINDINGS.md`), so it is not established either;
* rule 64b (v) now requires every statistic to carry its tier composition, and
  forbids cross-mechanism claims from a set whose largest series is ≥90%.

The design consequence: a campaign must commit each arm's SID and its Stage 3
result together, under the same directory name, or it produces arms that no
later analysis can admit. That is what rule 64b (iii) requires going forward.
