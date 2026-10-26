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

## Which metrics are correlated with recall (survey over all three batteries)

`correlation_summary.csv` merges the three batteries on `sid_sha256` and tests every
metric against `test_recall@10` on the same 67 SIDs: Spearman with a bootstrap interval,
a permutation p-value, the Benjamini-Hochberg adjusted p-value over the 31 metrics that
survived the duplicate collapse (a lift is its metric divided by a constant, so it is the
same test), and the best out-of-sample R2 of the shape analysis.

| metric | rho | CI95 | p (BH) | best shape | out-of-sample R2 |
|---|---|---|---|---|---|
| `cooccur_overlap_L3` | **0.901** | [0.832, 0.940] | 0.0008 | isotonic | **0.705** |
| `cooccur_overlap_L2` | 0.864 | [0.774, 0.916] | 0.0008 | isotonic | 0.571 |
| `cooccur_overlap_L1` | 0.855 | [0.749, 0.917] | 0.0008 | kNN | 0.578 |
| `succ_overlap_within_L1` | 0.847 | [0.749, 0.907] | 0.0008 | kNN | 0.498 |
| `adjacency_within_L1` | 0.838 | [0.730, 0.903] | 0.0008 | kNN | 0.470 |
| `adjacency_L1` | 0.825 | [0.706, 0.901] | 0.0008 | kNN | 0.478 |
| `adjacency_L3` | 0.822 | [0.693, 0.897] | 0.0008 | kNN | 0.526 |
| `adjacency_L2` | 0.820 | [0.704, 0.889] | 0.0008 | log | 0.449 |
| `er_l1` | 0.782 | [0.659, 0.859] | 0.0008 | kNN | 0.375 |
| `er_l3` | 0.781 | [0.656, 0.857] | 0.0008 | log | 0.439 |
| `er_l2` | 0.759 | [0.629, 0.844] | 0.0008 | isotonic | 0.426 |
| `L3_distinct` | 0.676 | [0.494, 0.806] | 0.0008 | kNN | 0.176 |
| `mi_prefix_successor_L3` | 0.668 | [0.491, 0.795] | 0.0008 | kNN | 0.209 |
| `p_neg_l3` | −0.639 | [−0.761, −0.469] | 0.0008 | kNN | 0.079 |
| `gap_l1` | 0.556 | [0.299, 0.748] | 0.0013 | kNN | 0.488 |
| `mi_prefix_successor_L2` | 0.493 | [0.256, 0.681] | 0.0008 | kNN | −0.007 |
| `p_neg_l2` | −0.492 | [−0.667, −0.275] | 0.0008 | kNN | −0.051 |
| `L2_distinct` | 0.476 | [0.226, 0.679] | 0.0008 | kNN | 0.103 |
| `ext_distinct` | −0.471 | [−0.648, −0.245] | 0.0008 | log | −0.101 |
| `succ_prefix_purity` | 0.470 | [0.200, 0.693] | 0.0013 | kNN | 0.391 |
| `mi_prefix_successor_L1` | 0.453 | [0.242, 0.630] | 0.0008 | kNN | 0.015 |
| `p_pos_l3` | −0.390 | [−0.592, −0.155] | 0.0013 | log | −0.129 |
| `mi_prefix_successor` | 0.387 | [0.173, 0.580] | 0.0013 | kNN | −0.030 |
| `gap_l3` | −0.358 | [−0.562, −0.131] | 0.0032 | log | −0.131 |
| `p_pos_l1` | 0.355 | [0.071, 0.599] | 0.0050 | kNN | 0.363 |
| `succ_mass_top_decile` | −0.294 | [−0.513, −0.052] | 0.0203 | kNN | 0.040 |
| `L1_distinct` | −0.199 | [−0.431, 0.070] | 0.117 | quadratic | −0.157 |
| `p_pos_l2` | −0.090 | [−0.332, 0.160] | 0.527 | log | −0.192 |
| `L1_code_entropy_bits` | −0.035 | [−0.303, 0.236] | 0.832 | kNN | 0.002 |
| `p_neg_l1` | 0.019 | [−0.259, 0.281] | 0.904 | kNN | −0.027 |
| `gap_l2` | 0.010 | [−0.235, 0.253] | 0.946 | linear | −0.178 |

Three readings, and the third is the one that changes how a mechanism should be designed:

1. **Three correlated families.** Co-occurrence overlap (L3/L2/L1, rho 0.86–0.90),
   successor-side coherence (`succ_overlap_within_L1`, `adjacency_within_L1`,
   `adjacency_L1/L2/L3`, rho 0.82–0.85) and held-out prefix discrimination
   (`er_l1/l2/l3`, rho 0.76–0.78) all survive the multiplicity correction.
   `L1_distinct`, `L1_code_entropy_bits`, `p_pos_l2`, `p_neg_l1` and `gap_l2` do not:
   the number of codes at the first level, the entropy of that level, and the
   positive-side hit rates carry no information about the recall.
2. **Ranking is not the same as predicting.** `L3_distinct` (rho 0.676) and `p_neg_l3`
   (rho −0.639) rank the tables well but their best out-of-sample R2 is 0.18 and 0.08:
   they are monotone but noisy. `gap_l1` is the other way round (rho 0.556, R2 0.488).
   Only `cooccur_overlap_L3` is both strongly ranked and strongly predictive.
3. **The families are not independent levers.** Their mutual rank correlations run from
   0.58 to 0.97, and `mi_prefix_successor_L3`, `L3_distinct` and `p_neg_l3` correlate
   with each other at ±1.00, i.e. they are the same quantity read three ways. Moving one
   of these metrics moves the others, so the choice of metric is a choice of *which
   behaviour to enforce*, not of which knob to turn.

## Four questions the overlap metric does not ask

`sequence_and_cut_metrics.csv` answers four further questions about the same SID tables. None
of them compares a positive item with a negative over prefix length, which is the comparison
the reference work makes.

### A. Multi-step prediction inside the SID's own code space

A first-, second- and third-order Markov predictor over each level's code space is estimated
from the training transitions only (339,519 of them, one per consecutive pair inside a user's
longest training sequence) and scored on the held-out test transitions. `nll_norm` is the
per-transition negative log-likelihood divided by `log(codes_used)`, so 1 means "as bad as the
marginal" and 0 means "perfectly predictable"; `hit1_excess` and `hit10_excess` subtract the
rate a guesser gets with the same number of codes.

| level | context | nll_norm (median) | rho(recall) | rho(`cooccur_overlap_L3`) |
|---|---|---|---|---|
| L1 | 1 or 3 behaviours | 0.8693 | −0.134 | −0.046 |
| L2 | 1 or 3 behaviours | 0.9582 | +0.545 | +0.619 |
| L3 | 1 or 3 behaviours | 0.9728 | +0.572 | **+0.739** |

Three things to read out of it:

1. **The sign is the opposite of the naive expectation.** A *less* predictable code space
   (higher `nll_norm`) goes with *higher* recall. A coarse level whose next code is trivially
   guessable has been drained of information; the fine levels are informative exactly because
   they are hard to guess.
2. **It is largely redundant with the overlap metric.** The strongest version correlates 0.739
   with `cooccur_overlap_L3` and only 0.572 with the recall, so as a predictor of a new SID's
   recall it adds nothing over what is already measured.
3. **The third-order context is not estimable at this scale, and the first version of this
   script hid that.** The first run fed the second-order path for both the two- and the
   three-behaviour context, so `ctx2` and `ctx3` came out identical for 9 of 12 field pairs;
   the third-order table was then added properly and the columns are still identical
   (rho = 1.000) — the triple space at L3 has 256^3 = 16.7M cells for 340k training
   transitions, so every test triple falls back to the first-order table. The honest statement
   is therefore "the code space is first-order predictable at this data scale; a 2-3 behaviour
   context adds nothing measurable", not "the SID preserves multi-step history".

### B. Recall sliced by target frequency

Taken from the per-target rank dump of the Stage 3 evaluation (`_test_shards/rank*.json`,
`target_rank` in 0..9 or null). **15 of the 67 runs wrote that dump**, so this and the next two
sections cover those 15 and not the full sample.

| target frequency in training | test rows | mean Recall@10 | spread over the 15 SIDs | rho with the overall recall |
|---|---|---|---|---|
| never a training target | 7,248 | **0.0015** | 0.0005 | +0.575 |
| 1–2 | 14,947 | 0.0039 | 0.0010 | +0.696 |
| 3–9 | 17,191 | 0.0174 | 0.0054 | +0.904 |
| 10 or more | 18,053 | **0.1626** | 0.0083 | **+0.964** |

**89.1% of every run's hits come from the frequently-targeted items**, and the gap between the
common and the never-seen bucket is 0.161 (range 0.145-0.174 across the 15). All 15 SIDs are
equally blind on unseen and rare targets: their spread there is 5e-4, i.e. five in ten
thousand. The 67-table comparison is therefore a comparison of *popular* next items; nothing in
it says a SID helps rare or genuinely new transitions.

### C. Recall sliced by interest diversity

Two splits, both computed per test row and identical for every SID:

| split | group | rows | mean Recall@10 | rho with the overall recall |
|---|---|---|---|---|
| history length | ≤6 behaviours | 34,246 | 0.0651 | +0.900 |
| | 7–15 | 18,536 | 0.0494 | +0.857 |
| | ≥16 | 4,657 | 0.0344 | +0.886 |
| semantic concentration (Stage 1 embedding) | dispersed tertile | 19,147 | 0.0458 | +0.868 |
| | middle | 19,146 | 0.0568 | +0.854 |
| | concentrated | 19,146 | 0.0700 | +0.900 |

Longer histories are harder and semantically concentrated users are easier, by 1.5-2x, and all
six groups correlate 0.85-0.90 with the overall recall: the SID differences are not confined to
one kind of user, they shift the level of every group.

### D. Behaviour drift instead of time

**The dataset has no timestamps**, so the requested stable-versus-recent split cannot be made.
The drift of the last three behaviours away from the user's earlier behaviour, measured as the
cosine distance between the two mean embeddings, is used instead and is only a proxy:

| drift tertile | rows | mean Recall@10 | rho with the overall recall |
|---|---|---|---|
| stable | 19,147 | 0.0678 | +0.857 |
| drifting | 19,146 | 0.0547 | +0.871 |
| shifted | 19,146 | 0.0500 | +0.907 |

Recall falls monotonically as the recent behaviour drifts, which is what a long-term-rules
reading predicts, but without timestamps this cannot distinguish "the SID learns long-term
regularity" from "users whose recent items are unusual are harder to predict for any model".

### Method notes and dead ends

* Two natural diversity measures are degenerate on this data: the history's distinct-item ratio
  and the novelty of the last three behaviours are **1.0 for every one of the 57,439 rows**,
  because no user repeats an item inside a history. The semantic measures above replaced them.
* The five runs without the per-target dump are named in the output file
  (`has_target_dump = 0`) rather than dropped silently.
* Metrics B/C/D are computed on 15 SIDs, so their correlations carry much wider intervals than
  the 67-SID analysis; the level differences across buckets are consistent in all 15 runs and
  are the part that can be read.

## Incoming predecessor-context diversity (directed adjacency)

`predecessor_context_metrics.csv`, produced by `predecessor_context_analysis.py`. For every
adjacent **training** transition ``A -> B`` the predecessor ``A`` is mapped to the region its
SID assigns, and for each target item ``B`` the distinct regions among its predecessors are
counted:

* ``region_ratio(B) = distinct regions of the predecessors / distinct predecessors`` is 1 when
  every predecessor sits in its own region and small when they all share one, so it does not
  grow with the item's in-degree;
* ``region_entropy(B)`` is the predecessor-region entropy divided by ``log`` of the regions the
  item could use.

Both are averaged over the target items, overall and inside the target-frequency buckets. This
is a property of the **directed** adjacency graph read through the SID: it uses neither a
positive/negative ratio nor a co-occurrence vector.

### The region depth decides the sign, and that is the finding

| region depth | median ratio | raw rho(recall) | rho with `L3_distinct` | partial rho(recall \| `L3_distinct`) |
|---|---|---|---|---|
| L1 | 0.8865 | **−0.292** | +0.182 | **−0.574** |
| L2 | 0.9920 | +0.056 | +0.581 | **−0.563** |
| L3 | 0.9989 | **+0.333** | +0.859 | **−0.658** |

The L3 reading is saturated by construction: about 20k distinct triples for 24.6k items means
two distinct predecessors almost always sit in different triples, so at L3 the ratio equals the
in-degree and only collapsed tables differ from 1 (8 of 67 tables are below 0.99). Its raw
positive correlation is therefore a re-expression of the L3 granularity, which correlates 0.86
with it and 0.68 with the recall.

Once the granularity is controlled the relation is **negative at every depth** (−0.56 to −0.66,
permutation p = 0.0002, the smallest value 4,000 permutations can produce). Holding a table's
granularity fixed, a target whose predecessors arrive from *more* distinct behavioural regions
has *lower* recall: a concentrated incoming context is the positive property, a diffuse one is
not. This is the opposite of the naive "more diversity is better" reading, and the raw L3
number alone would have said the naive thing.

### By target frequency

| bucket | items | median ratio (L1) | raw rho(recall) | p (BH over 8 tests) | partial rho(\| `L3_distinct`) |
|---|---|---|---|---|---|
| never a training target | 7,470 | 0.9324 | −0.359 | 0.012 | −0.610 |
| 1–2 | 11,218 | 0.9066 | −0.323 | 0.021 | −0.589 |
| 3–9 | 4,906 | 0.8233 | −0.254 | 0.053 | −0.549 |
| 10 or more | 930 | 0.6346 | −0.202 | 0.126 | −0.496 |

Frequently targeted items are reached from *fewer* coarse regions (median ratio 0.63 against
0.93 for never-targeted items), and the negative relation holds in **every** bucket, strongest
for rare and unseen targets. So the effect is not confined to popular items; if anything the
popular items are where the raw signal is weakest.

### Redundancy and multiplicity

* This metric is **not** the co-occurrence one: at L1 it correlates −0.15 with
  `cooccur_overlap_L3`. At L3 the apparent +0.54 goes through the granularity.
* Eight variants were specified before the run (four depths/entropies plus four frequency
  buckets); the Benjamini-Hochberg column above is over those eight, and the two that do not
  survive are the ones with the weakest raw effect (`d1_common`, `d2_mean`).
