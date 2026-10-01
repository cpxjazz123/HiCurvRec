# Baseline — Current Condition

This file is descriptive and subordinate to `SKILL.md`.

## Protocol

Runs are directly comparable only when they share:

- Amazon-2023 Instruments;
- Stage3 `n_eval = 57439`;
- the same Stage1 embedding source and Stage3 trainer/protocol.

Values are read from each run's own `test_final.json`. Never compare against a
quoted or summarized number.

## Where the baseline lives

The baseline is **the current accepted condition in the working tree**, i.e.
the state the last non-reverted commit leaves behind. A rejected condition is
reverted, so it never becomes the baseline for the next one.

Main reads the parent value from the current `test_final.json` under:

```
results/stage3_T5Train/curvature_RQ-VAE/logs/Amazon_2023_Instruments/
```

## Current state

Mechanism: fixed-curvature Poincare RQ-VAE,
`LAYER_CURVATURES = (1.0, 1.0, 1.0)`.

- Stage2 budget: 72000 global steps = 18000 local optimizer updates, matched
  to the TIGER baseline (3000 epochs x 6 steps x 4 ranks)
- Stage2 endpoint collision: 0.316
- `test_R@10 = 0.04806838559167116`
- `test_NDCG@10 = 0.02634937592483783`
- `n_eval = 57439`

Project target: `test_R@10 > 0.065`.

## Comparison point: TIGER baseline

Same budget, same dataset, Euclidean geometry, final-epoch checkpoint only:

- `test_R@10 = 0.05367433277041731`
- `test_NDCG@10 = 0.029444663223495537`
- Stage2 endpoint collision: 0.177

The curvature model currently trails this, so the ball geometry is not yet
earning its cost.

## Stage2 metrics

Gini, collision, unique SID count, layer utilization, and conditional entropy
are descriptive only. They are not promotion gates and must not substitute for
Stage3 evidence.

## Disagreement rule

If any other repository file disagrees with the values above:

1. read the exact run's `test_final.json`;
2. verify protocol compatibility;
3. use the exact file as source of truth;
4. note the mismatch before comparing.
