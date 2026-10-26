# Behaviour metrics of a SID table versus its Stage 3 recall

67 SID tables, each paired with exactly one Stage 3 run and its `test_recall@10`
(`analysis_sid_recall/sid_recall.csv`). Every predictor below is defined on the
**held-out behaviour pairs**: 52,855 anchor -> positive transitions taken from
`valid.parquet`, each with ten fixed negatives drawn from items that are not adjacent in
the training histories. A value therefore describes how the table partitions
*behaviour*, never how large a bucket is.

- `p_pos_Ld` / `p_neg_Ld` — the share of pairs whose positive (negative) shares the
  anchor's prefix of depth `d`.
- `er_Ld = p_pos_Ld / p_neg_Ld` — how much more often a behavioural neighbour shares the
  prefix than an unrelated item does (undefined when no negative shares it).
- `gap_Ld = p_pos_Ld - p_neg_Ld` — the same contrast in absolute terms.

The structural counts are listed separately as **controls**: they are not behaviour
metrics, and they are reported only to show that the behavioural signal is not a proxy for
them.

## Rank correlation with `test_recall@10` (Spearman, pair-level bootstrap, 2000 draws)

| group | metric | n | Pearson | **Spearman** | 95% CI |
|---|---|---|---|---|---|
| behaviour | `er_L1` | 67 | 0.704 | **0.782** | [0.659, 0.858] |
| behaviour | `gap_L1` | 67 | 0.398 | 0.555 | [0.299, 0.747] |
| behaviour | `p_pos_L1` | 67 | −0.115 | 0.355 | [0.071, 0.598] |
| behaviour | `p_neg_L1` | 67 | −0.262 | 0.019 | [−0.257, 0.281] |
| behaviour | `er_L2` | 65 | 0.701 | **0.758** | [0.627, 0.843] |
| behaviour | `p_neg_L2` | 67 | −0.199 | **−0.492** | [−0.666, −0.274] |
| behaviour | `p_pos_L2` | 67 | −0.175 | −0.089 | [−0.331, 0.159] |
| behaviour | `gap_L2` | 67 | −0.123 | 0.010 | [−0.235, 0.253] |
| behaviour | `er_L3` | 62 | 0.627 | **0.783** | [0.653, 0.856] |
| behaviour | `p_neg_L3` | 67 | −0.169 | **−0.640** | [−0.758, −0.468] |
| behaviour | `p_pos_L3` | 67 | −0.153 | −0.392 | [−0.593, −0.155] |
| behaviour | `gap_L3` | 67 | −0.133 | −0.359 | [−0.562, −0.132] |
| control | `L1_distinct` | 67 | 0.008 | −0.093 | [−0.417, 0.084] |
| control | `L2_distinct` | 67 | 0.421 | 0.477 | [0.225, 0.679] |
| control | `L3_distinct` | 67 | 0.661 | 0.677 | [0.492, 0.806] |
| control | `ext_distinct` | 67 | −0.198 | −0.470 | [−0.645, −0.247] |

## What the numbers say

1. **The behavioural prefix discrimination is the strongest behavioural predictor, at
   every depth.** `er_L1 = 0.78`, `er_L2 = 0.76`, `er_L3 = 0.78`, all with intervals far
   from zero.
2. **It is the separation of non-neighbours that carries the signal, not the co-location
   of neighbours.** `p_neg` is the term with the large negative correlation at depth 2 and
   3 (`−0.49`, `−0.64`), while `p_pos` is weak and turns negative at depth 3. A table that
   puts behaviourally unrelated items into different prefixes ranks high; one that merely
   groups neighbours together does not.
3. **The effect is not a family label.** Within each family the rank correlation survives:

   | family | n | Spearman(`er_L1`) | Spearman(`er_L3`) | ER range |
   |---|---|---|---|---|
   | `curvature_RQ-VAE*` | 47 | 0.709 | 0.760 | 1.00 – 14.90 |
   | `tiger_sample_*` (BEHAVIOR_SAMPLE) | 11 | 0.727 | 0.588 | 1.63 – 16.52 |
   | newer `TIGER_*` arms | 9 | 0.550 | 0.767 | 1.28 – 15.46 |
4. **The association is not explained by the size of the SID space.** The first-level
   code count has no rank correlation at all (`−0.09`, interval spanning zero) while
   `er_L1` reaches `0.78`. The third-level count does correlate (`0.68`), and the two are
   partly confounded: a table that spreads items over more triples also tends to keep
   negatives apart. The behavioural term is the one with a mechanism behind it.

## Limitations that must be read with the table

- **Selection.** 127 further Stage 3 runs (in `stage3_runs_without_sid.csv`) used tables
  that no longer exist anywhere, so they are absent from these 67. They include exactly
  the arms that made the counterexample interesting: `l1_only` and `l1_br_euclid` reached
  0.0623 / 0.0621 with an `L1 Hard ER` of 1.000. The association reported here holds
  *within the surviving sample* and over the ER range it covers (1.0 – 16.5); it is not a
  claim that ER = 1 caps recall.
- Two tables in the sample sit at the low end with very different recalls
  (`curvature_RQ-VAE_iter59`, ER 1.00, 0.0447; `TIGER_successor_entropy_curvature`,
  ER 1.28, 0.0506), which is the same caution stated from the other side: the trend is
  rank-average, not a per-table law.
- 39 of the 67 pairings resolve through the `earliest_surviving` fallback (the table was
  committed after its run), so those rows pair a run with the only surviving copy of its
  table rather than provably with the exact bytes it read. `sid_resolution` marks them.
- Recall is one Stage 3 run per table (the highest, when several runs shared a table); the
  spread between runs of the same table is up to 0.0063 (see
  `sid_recall_alternatives.csv`), so a single run carries its own noise.

## Files

| file | content |
|---|---|
| `sid_behaviour_metrics.csv` | 67 rows: the behaviour metrics, the structure control, the recall |
| `behaviour_correlation.csv` | the correlations above, with n and the bootstrap interval |
| `sid_recall.csv` | the pairing (SID -> one Stage 3 run and its recall) |

Reproduce: `python analysis_sid_recall/behaviour_correlation.py`

## Second battery: train-only transition-graph metrics

The first battery measures prefix sharing on the held-out pairs. This battery asks a
different question — does the partition agree with the *training* transition graph?
Everything here is train-only (`sid_behaviour_metrics2.csv`).

| metric | what it says | n | Pearson | **Spearman** | 95% CI |
|---|---|---|---|---|---|
| `succ_overlap_within_L1` | item pairs sharing the first code have similar successor rows (mean cosine overlap) | 67 | 0.679 | **0.847** | [0.748, 0.906] |
| `adjacency_within_L1` | item pairs sharing the first code really do follow one another in training | 67 | 0.687 | **0.837** | [0.728, 0.903] |
| `succ_prefix_purity` | an item's successors carry its own first code (count-weighted) | 67 | −0.045 | 0.470 | [0.199, 0.691] |
| `mi_prefix_successor` | normalised mutual information between the first code and the successor | 67 | 0.286 | 0.388 | [0.172, 0.579] |
| `succ_mass_top_decile` | share of successor mass in the heaviest decile of codes | 67 | −0.356 | −0.294 | [−0.513, −0.052] |
| `L1_code_entropy_bits` | usage entropy of the first level (not a behaviour metric) | 67 | 0.213 | −0.035 | [−0.303, 0.236] |

**Two of these beat the held-out ER**: the *cohort coherence* of the first level —
do the items I group together have similar successors (`0.85`) and do they actually
follow one another (`0.84`).

### The trap in `succ_prefix_purity`

Purity alone is gameable by collapsing the level:

| arm | recall@10 | purity | MI | overlap | adjacency |
|---|---|---|---|---|---|
| `tiger_sample_resample` | 0.0637 | 0.116 | **0.268** | 0.009 | **0.022** |
| `tiger_sample_crowd_posattract_w3` | 0.0643 | 0.126 | 0.256 | 0.009 | 0.020 |
| `TIGER_behaviour_code_cohesion` (this session) | 0.0546 | **0.494** | **0.036** | 0.004 | **0.000** |
| `TIGER_successor_entropy_curvature` | 0.0506 | 0.123 | 0.139 | 0.002 | 0.003 |
| `curvature_RQ-VAE` | 0.0315 | 0.027 | 0.130 | 0.003 | 0.003 |

The cohesion arm has the **highest purity in the whole sample** and one of the lowest
informations: it made an item's successors share its own code by shrinking the first
level to 16 codes (one of them holds 40% of the catalogue), so the successors are in the
same code because almost everything is. Purity went up, the code stopped telling you
*which* successor follows (MI 0.036 against 0.268), no two co-located items actually
follow one another (adjacency 0.000), and recall stayed below the baseline. The metric
worth optimising is **coherence with information** (`adjacency`, `overlap`, `MI`), never
purity on its own.

## Non-linear shape (out-of-sample, 5-fold × 20 repeats)

Rank correlation only measures a monotone average. Each candidate shape was therefore
fitted on four folds and scored on the fifth; a flexible shape wins only if it really
generalises (`nonlinear_shapes.csv`).

**The merge is on the SID's identity.** The three metric batteries are joined on
`sid_sha256`, not on the arm name: the 67 SIDs carry only 53 distinct names
(`curvature_RQ-VAE` alone names 15 different SIDs), so a merge on the name silently
drops 14 rows and attaches one SID's metrics to another SID's recall. Every battery now
records `sid_sha256` and the three sets of identifiers are verified to be identical
before the analysis runs.

| metric | linear | quadratic | log | kNN(7) | isotonic | best |
|---|---|---|---|---|---|---|
| **`cooccur_overlap_L3`** | 0.554 | 0.655 | 0.663 | 0.648 | **0.705** | isotonic |
| `cooccur_lift_L3` | 0.554 | 0.655 | 0.663 | 0.648 | **0.705** | isotonic |
| `cooccur_overlap_L1` | 0.428 | 0.395 | 0.450 | **0.578** | 0.374 | kNN |
| `cooccur_overlap_L2` | 0.445 | 0.553 | 0.538 | 0.521 | 0.571 | isotonic |
| `adjacency_L3` | 0.324 | 0.425 | 0.521 | **0.526** | 0.406 | kNN |
| `adjacency_L1` | 0.243 | 0.331 | — | **0.478** | 0.279 | kNN |
| `succ_overlap_within_L1` | 0.124 | 0.486 | 0.340 | **0.498** | 0.409 | kNN |

(`cooccur_lift_L3` is `cooccur_overlap_L3` divided by the random-pair baseline
0.026873, a constant, so it carries exactly the same information and the same fit; the
lift is only useful for reading the scale.)

Three things follow:

1. **At the fine level the relation is non-linear, and the non-linear shape generalises.**
   For `cooccur_overlap_L3` the monotone isotonic fit reaches an out-of-sample R² of
   **0.705** against the straight line's 0.554. The two shapes are fitted on the same
   folds, so the difference is paired: **+0.151, 95% CI [+0.126, +0.175], and the
   isotonic fit wins 20 of 20 repeats.** A third of the variance a linear model misses
   is real and predictable.
2. **The first level is non-linear too, but not in the same way.** `cooccur_overlap_L1` is
   best served by the local kNN fit (0.578 against the line's 0.428) and the smooth
   monotone fit is the *worst* of the five (0.374). What separates the levels is not
   "linear versus non-linear" but *which* non-linearity generalises: at L2/L3 a smooth
   monotone curve transfers, at L1 only a piecewise-local one does. An earlier draft of
   this file called the L1 relation linear; with the full 67 correctly paired that
   claim does not hold.
3. **The curve saturates.** Quintile means of `cooccur_overlap_L3` against mean recall
   (all 67 SIDs):

   | quintile metric mean | 0.0503 | 0.0721 | 0.0907 | 0.1125 | 0.1527 |
   |---|---|---|---|---|---|
   | mean recall@10 | 0.0472 | 0.0552 | 0.0572 | 0.0601 | 0.0623 |
   | gain from the previous quintile | — | +0.0080 | +0.0020 | +0.0029 | +0.0022 |

   The first step buys 0.0080 of recall for 0.022 of metric; the last buys 0.0022 for
   0.040. Returns fall as the metric grows — the practical consequence is that a
   mechanism should aim at the middle of the range, not at a perfectly coherent
   partition.

A threshold scan on the same metric finds a split at **0.0585**: tables above it average
**+0.0120** recall (permutation p = 0.0005, 2000 shuffles), so the "jump" is a property of
the data and not of the search over splits. (The 53-row draft put the split at 0.062; the
correctly paired full set puts it at 0.0585.)
