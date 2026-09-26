# stage3_outcome_iter28

## Final Stage3 result

```
{
  "best_checkpoint": "/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter28/ckpt/Amazon_2023_Instruments/Sep-27-2026_00-15-26/HG_Rec_best.pth",
  "n_eval": 57439,
  "test_recall@5": 0.03706540852034332,
  "test_recall@10": 0.05649471613363742,
  "test_ndcg@5": 0.024799025279798593,
  "test_ndcg@10": 0.031058744990046853
}
```

`test_recall@10 = 0.05649471613363742` (n_eval=57439, beam=20).

## Comparison vs canonical iter18 (R@10=0.05988962203380978)

| Metric | iter28 (SREMA) | iter18 (baseline) | iter11 (β2) | iter25 (prior) | iter26 (FCCR-1) | iter27 (ABORT) |
|---|---:|---:|---:|---:|---:|---:|
| R@10 | **0.0565** | **0.0599** | 0.0598 | 0.0588 | 0.0570 | N/A |
| Δ vs iter18 | **−0.0034** | — | −0.0001 | −0.0011 | −0.0029 | — |
| R@5 | 0.0371 | 0.0405 | — | — | — | — |
| NDCG@5 | 0.0248 | 0.0270 | — | — | — | — |
| NDCG@10 | 0.0311 | 0.0332 | — | — | — | — |

## Mechanism status

- **Mechanism: ACTIVE_NEGATIVE**
  - SREMA was verifiably active (MVG: `frac_updated = 1.000`, `median_update ≈ 1e-3`, `saturation_max < 0.22`, 200-step ON/OFF `loss_delta = 0.30`, codebook max-abs-diff per layer = `[0.37, 0.18, 0.13]`).
  - The mechanism measurably altered the codebook trajectory.
  - But the downstream R@10 dropped by `0.0034` (≈6% relative) vs iter18.
  - This is **below** iter18's noise band (~0.003), so the regression is real, not random.

- **Promotion: PROMOTION_FAIL** (R@10 = 0.0565 < iter18's 0.0599, far below target 0.065).

## What the iter28 result tells us

The Sinkhorn-weighted Riemannian EMA made the codebook move more often (`frac_updated = 1.000` vs AdamW's near-zero fraction at this displacement scale) and changed the codebook trajectory by **0.37 / 0.18 / 0.13** Euclidean per layer over 200 steps (≈10× iter27's effect).  However the downstream R@10 dropped:

- iter28 R@10 = 0.0565 < iter18 R@10 = 0.0599 (−0.0034, below noise band).
- iter28 R@10 = 0.0565 < iter26 R@10 = 0.0570 (−0.0005).
- iter28 R@10 ≈ iter25 R@10 = 0.0588, but worse.

The mechanism hypothesis "codebook lives on the Poincaré ball" does **not** translate to R@10 gains in this single seed.  The geometric-correct codebook update is *active* (verifiably changes the trajectory), but the trajectory it produces is *worse* than AdamW's for HG-Rec downstream recall.

## Falsification re-check

- F-1 (mechanism inactive): **NOT TRIGGERED** (mechanism is active).
- F-2 (escape / saturation): **NOT TRIGGERED** (`saturation_max < 1` throughout).
- F-3 (downstream regression > noise): **TRIGGERED** — R@10 dropped by `0.0034 > iter18 noise band (~0.003)`.

Per `hypothesis_iter28.md` F-3: **classified as `ACTIVE_NEGATIVE + PROMOTION_FAIL`**.

## Carryover to iter29

- iter28 is a clean negative result under CAO-1, NOT a failed experiment.  The SREMA mechanism runs and is geometrically valid; the data just doesn't reward the additional manifold constraint at the iter18 hyperparameters.
- A future iteration may try:
  - tighter EMA momentum (β = 0.995 or 0.9999) to reduce SREMA's per-step contribution;
  - a Riemannian AdamW (parallel-transported momentum + exp-map retraction + native Riemannian Adam) instead of SREMA;
  - running iter28 with a different seed (single-seed = weak evidence);
  - **reverting to iter18** as the next parent and pivoting away from codebook-side optimization.
- The iter27 (trust-region) and iter28 (SREMA) results jointly suggest: in this Stage2 protocol, **constraining or replacing the codebook's optimizer is not a productive axis** under CAO-1.  The bottleneck may live elsewhere (encoder capacity, behavior contrastive loss weight, Sinkhorn ε schedule, or Stage3's own hyperparams).

## Hard target status

`test_R@10 > 0.065`: **NOT MET** (0.0565 ≪ 0.065).
`test_R@10 > iter18 baseline 0.0599`: **NOT MET** (0.0565 < 0.0599).