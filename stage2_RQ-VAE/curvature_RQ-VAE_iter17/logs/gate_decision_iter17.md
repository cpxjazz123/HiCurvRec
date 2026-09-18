# iter17 (L0 sk_eps=0.7) — Gate Decision: NO-GO

## Decision
**NO-GO** — SID gate auto-triggered early stop at step 10000 because `l01_unique_pairs=6181 < baseline*ratio=6670`.

## Mechanism
- Cloned from iter11, single variable change: `PER_LAYER_SK_EPS = [0.7, 0.05, 0.05]` (was [0.5, 0.05, 0.05])
- Goal: push L0 collapse trend harder than iter11 to test if more collapse → higher test_R@10

## SID Gate Result (step 10000)

| metric | iter17 (step 10k) | iter11 (final 100k) | TIGER baseline | threshold |
|---|---|---|---|---|
| hitrate@50 | 0.7165 | 0.7165 | 0.7165 | ≥ 0.7115 ✓ |
| full_gini | 0.1128 | 0.1143 | 0.0672 | ≤ 0.0722 ✗ |
| per_layer_gini | [0.2495/0.0943/0.0601] | [0.2034/0.0564/0.0789] | [0.3598/0.2467/0.1656] | — |
| l01_unique_pairs | **6181** | 7509 | 13340 | **> 6670** ✗ |
| h_l1_given_l0 | 6.4293 | 6.3328 | 5.6116 | — |

### Failure analysis
- iter17's l01_pairs (6181) at step 10k is even below iter11's final l01_pairs (7509).
- L0 codes count similar (40 vs 57), but (L0, L1) joint distribution is **more concentrated** under sk_eps=0.7.
- Why: sk_eps=0.7 makes L0 assignment super peaky — items in the same L0 cluster tend to also fall in similar L1 codes.
- sk_eps=0.5 (iter11) was already aggressive enough; pushing to 0.7 destroys (L0, L1) pair diversity.

## Artifacts saved
- `results/stage2_RQ-VAE/curvature_RQ-VAE_iter17/out/rqvae/instruments/`
  - `rqvae_step10000.pt` (13 MB)
  - `sids_step10000.npy` (295 KB)
  - `quality_step10000.json`
- Total wall-clock: 93.9s (auto-stopped at first SID gate check)
- GPU 0-3 released after auto-stop (no leaked processes)

## Next iteration proposal: iter18 = sk_eps=0.6
- Reason: between iter11 (0.5, PASS) and iter17 (0.7, FAIL)
- Expected: pass SID gate with l01_pairs between 6670 and 7509
- If iter18 PASS + test_R@10 > iter11 (0.06017) → collapse trend confirmed
- If iter18 PASS but test_R@10 ≤ iter11 → collapse optimum is at or below iter11's sk_eps=0.5
- If iter18 FAIL → iter11 is the collapse optimum, switch to other mechanism axis

## What NOT to do this iteration
- Don't add Plan A (midpoint auxiliary) — single-variable isolation requires single-variable experiment
- Don't override SID gate ratio — hard rule per Project Rule §2
- Don't extend iter17 training — early-stop already saved the right ckpt