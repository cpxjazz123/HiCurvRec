# Baseline Registry — Current Protocol Only

This file is descriptive and subordinate to the Protocol Lock in `SKILL.md`.

## Current comparable protocol

For the current 2026-09-25/26 evaluation protocol, directly comparable runs use:

- Amazon-2023 Instruments;
- Stage3 `n_eval = 57439`;
- the same Stage1 embedding source and current Stage3 trainer/protocol;
- exact values read from each run's `test_final.json`.

## Canonical reference

Canonical Iter11 formal run:

- run: `results/stage3_T5Train/curvature_RQ-VAE_iter11/logs/Amazon_2023_Instruments/Sep-25-2026_05-25-18/test_final.json`
- `test_R@5 = 0.0388934348`
- `test_R@10 = 0.0597677536`
- `test_NDCG@5 = 0.0256396540`
- `test_NDCG@10 = 0.0323597951`
- `n_eval = 57439`

Best completed Iter17–25 observed so far:

- Iter18 `test_R@10 = 0.0598896220`, `n_eval = 57439`

The Iter18-vs-Iter11 difference is tiny and should be treated as near-parity unless replicated.

## Historical non-comparable results

Older results around `test_R@10 ≈ 0.1179` used a different evaluation population/protocol (for example `n_eval = 24832`) and are:

`HISTORICAL_NONCOMPARABLE`

They must not be called the current best or used in current promotion deltas.

## Stage2 metrics

Gini, collision, unique SID count, layer utilization, and conditional entropy are descriptive only.

They are not promotion gates and must not be used as substitutes for Stage3 evidence.

## Baseline disagreement rule

If any other repository file disagrees with the values above:

1. read the exact run's `test_final.json`;
2. verify protocol compatibility;
3. use the exact file as source of truth;
4. document the mismatch before comparing methods.
