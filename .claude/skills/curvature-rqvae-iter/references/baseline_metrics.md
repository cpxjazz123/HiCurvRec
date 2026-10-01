# Baseline Registry — Current Protocol Only

This file is descriptive and subordinate to the Protocol Lock in `SKILL.md`.

## Current comparable protocol

Directly comparable runs use:

- Amazon-2023 Instruments;
- Stage3 `n_eval = 57439`;
- the same Stage1 embedding source and current Stage3 trainer/protocol;
- exact values read from each run's `test_final.json`.

## Rolling baseline

There is no fixed iteration registry. The baseline is **the current accepted
condition in `stage2_RQ-VAE/curvature_RQ-VAE`**, i.e. the tree state that the
last non-reverted commit leaves behind. A rejected condition is reverted, so it
never becomes the baseline for the next one.

Main reads the parent value from the current `test_final.json` under
`results/stage3_T5Train/curvature_RQ-VAE/logs/Amazon_2023_Instruments/`.

Current state as of the in-place migration:

- mechanism: fixed-curvature Poincare RQ-VAE, `LAYER_CURVATURES = (1.0, 1.0, 1.0)`
- Stage2 budget: 72000 global steps = 18000 local optimizer updates, matched to
  the TIGER baseline (3000 epochs x 6 steps x 4 ranks)
- Stage2 endpoint collision: 0.316
- `test_R@10 = 0.04806838559167116`
- `test_NDCG@10 = 0.02634937592483783`
- `n_eval = 57439`

Project target remains `test_R@10 > 0.065`.

## Reference: TIGER baseline

Same budget, same dataset, Euclidean geometry, final-epoch checkpoint only:

- `test_R@10 = 0.05367433277041731`
- `test_NDCG@10 = 0.029444663223495537`
- Stage2 endpoint collision: 0.177

The curvature model currently trails this baseline, so the ball geometry is
not yet earning its cost.

## Historical non-comparable results

Older results around `test_R@10 ≈ 0.1179` used a different evaluation
population/protocol (for example `n_eval = 24832`) and are:

`HISTORICAL_NONCOMPARABLE`

They must not be called the current best or used in current promotion deltas.

## Stage2 metrics

Gini, collision, unique SID count, layer utilization, and conditional entropy
are descriptive only. They are not promotion gates and must not be used as
substitutes for Stage3 evidence.

## Baseline disagreement rule

If any other repository file disagrees with the values above:

1. read the exact run's `test_final.json`;
2. verify protocol compatibility;
3. use the exact file as source of truth;
4. document the mismatch before comparing methods.
