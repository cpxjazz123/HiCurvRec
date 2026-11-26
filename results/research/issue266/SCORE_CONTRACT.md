# Frozen score contract — Issue #266

Written before computation on 2026-11-26. The formula, bounds and direction below
are copied from the issue and are not fitted to 108/109.

## C and G1

For one final Stage2 checkpoint, use the real candidate model and the same fixed
diagnostic forward path as #262:

- Fixed catalogue batch: unique `target` item IDs in
  `results/stage0_build_parquet/train.parquet`, then
  `np.random.default_rng(42).choice(train_ids, size=4096, replace=False)`, sorted.
  Expected SHA-256 of the int64 array: `251dced08f5c42703e2ce66b3c56924e91d14200a15a3771f45c8d8c66f1c457`.
- Structural sample: `model.structural_sampler().draw(0, 0, 1, track=False)`.
  The sampler is deterministic from its frozen table plus `(seed=42, global_step=0, rank=0)`.
  Record SHA-256 of `centre`, `ids`, `triples_global`, `references_global`, `target`, `row_ids`.
- Forward: `model(batch_embeddings, batch_item_ids, catalogue_embeddings=catalogue, structural_sample=sample)`.
  The returned fifth value is the native curvature loss.
- `C = float(curvature.detach())` — the raw **unweighted** curvature training loss.
- `G1 = sqrt(sum_p ||grad_p(curvature)||_2^2)` over
  `tuple(model.rq.vq_layers[0].parameters())` — the raw **unweighted** L2 norm of the
  curvature gradient on all L1 codebook parameters.
- No gradient clipping, no `combined_loss`, no reconstruction loss, no GNN-pretraining loss,
  no fixed-evaluation loss and no epoch history substitutes for C or G1.
- Parameters are frozen; `parameter_checksum_before == parameter_checksum_after`;
  optimizer steps = 0; no SID write.

The same script is run once per arm, in a fresh process, so candidate imports cannot mix.

## Frozen eight-arm score

With

```
loC = 0.38207152485847473
hiC = 3.225719451904297
loG = 1.6841177967230931
hiG = 32.588647951083324

S = 0.8 * (C - loC) / (hiC - loC)
  + 0.2 * (G1 - loG) / (hiG - loG)
```

No clipping to [0,1], no refit, no new search. The historical eight values are read
from `results/research/issue262/MATRIX.json` and recomputed from the formula.

## Ranking definitions

For any arm set, sort by `Recall@10` ascending. A **strict adjacent monotonicity** is
`S_i < S_{i+1}` for every adjacent pair in that order; equality is a tie and fails
strictness. The **inversion count** is the number of pairs `(i,j)` with
`Recall_i < Recall_j` but `S_i > S_j` (ties excluded); tied S is reported separately
and is not counted as an inversion. The **violating arms** are the arms in the
inverted pairs. Exact arm names and pair tuples are retained.

Reported statistics for `S` vs `Recall@10`:

- Pearson (ordinary product-moment);
- Spearman (average ranks for ties);
- Kendall tau-b (tie-corrected; explicit tie counts);
- strict adjacent monotonicity, inversion count, and violating pairs;
- original eight arms and extended ten arms separately;
- for the two new arms: whether each lies within the frozen historical score
  interval and whether `S_108 < S_109`.

The issue's local bracket is the interval between the adjacent historical arms
105 (`S=0.48640629657478884`) and 103 (`S=0.5096899206435366`). Full ten-arm
strict monotonicity is a separate, stronger check.

## Edge cases

- A missing C or G1 is `null` and makes its S `null`; never 0.
- A score outside the historical [0,1] range is retained and explicitly reported.
- A tied Recall is handled by average ranks for Spearman and Kendall tau-b;
  strict adjacent monotonicity treats equal S as a violation.
- A tied S is not silently counted as an inversion; tied pair count is separate.
- A missing arm or missing recall is reported as missing, not dropped silently.

## Cross-mechanism comparability

Arm 108 uses the #263 frozen-token structural objective. Arm 109 changes the
selection rule to Euclidean and replaces the parent's structural target with a
center-local mixed-curvature objective on the final hard `q`. Therefore the native
C/G1 values for 109 are an external-transferability test of the frozen formula,
not evidence that the same structural objective was optimized. Any auxiliary
common-geometry diagnostic (if computed) is reported separately and cannot replace
the native formula test.

## Outputs

`ENDPOINT_108.json`, `ENDPOINT_109.json`, `SCORES.json`, `SCORES.csv`, `REPORT.md`,
`VERDICT.md`, `tests/test_reference.py`, `reference.py`.
