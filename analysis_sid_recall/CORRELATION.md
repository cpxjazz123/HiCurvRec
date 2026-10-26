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
