# Stage2 stability probe

- tree: `/fs04/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_r6_04`
- declaration sha256: `ba72757be6c89de715a3a85f2c12fc473f99f6dfaff3173e1b1fa60a6ca63686`
- probe config sha256: `0586a65d266e01679f8ad9e8a950573d0db1d5a5b0248219f76ca1f89e1a7f2e`
- executed steps: 160 (behaviour-active optimizer steps 128, mechanism calls 384)
- elapsed: 12.48s

## status = `valid`

## risk = `high`

risk is a HEURISTIC early-warning indicator, not a validated predictor, and never a performance gate: the frozen Hard-L1 ER remains the only Stage2 acceptance criterion

Fired: `deep_gradient_domination`, `l1_entropy_drop`, `l1_effective_codes_drop`, `negative_train_pair_match_rise`, `max_code_share_rise`

## Snapshots (RAW hard SIDs; L1-only and 3-level tuple kept separate)

| step | active steps | mechanism calls | L1 used/256 | L1 H/log256 | L1 eff.codes | L1 max share | L1 collision | 3-level collision |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 0 | 0 | 0 | 256/256 | 0.8766 | 129.1 | 0.0181 | 0.9896 | 0.0496 |
| 8 | 0 | 0 | 256/256 | 0.8680 | 123.1 | 0.0237 | 0.9896 | 0.0488 |
| 16 | 0 | 0 | 256/256 | 0.8026 | 85.7 | 0.0803 | 0.9896 | 0.1048 |
| 32 | 0 | 0 | 246/256 | 0.6462 | 36.0 | 0.1486 | 0.9900 | 0.3684 |
| 64 | 32 | 96 | 196/256 | 0.4696 | 13.5 | 0.3902 | 0.9920 | 0.7591 |
| 96 | 64 | 192 | 128/256 | 0.2549 | 4.1 | 0.4444 | 0.9948 | 0.9161 |
| 128 | 96 | 288 | 76/256 | 0.2301 | 3.6 | 0.4512 | 0.9969 | 0.9251 |
| 160 | 128 | 384 | 54/256 | 0.2276 | 3.5 | 0.4575 | 0.9978 | 0.9165 |

L1-only collision and the three-level tuple collision have very different natural scales and are never mixed.

## Isolation

- production artifacts unchanged: **True**
  - `frozen_heldout_pairs`: `04d779208343a9e5` -> `04d779208343a9e5`
  - `log:train_migrated.log`: `965903fd7dbf69ba` -> `965903fd7dbf69ba`
  - `log:training_metrics.jsonl`: `307a3fc2c8a84c0e` -> `307a3fc2c8a84c0e`
  - `result:item_sids.json`: `fc7f5ed4b0747678` -> `fc7f5ed4b0747678`
  - `result:rqvae_best.pth`: `0fec712c8eaf5dc8` -> `0fec712c8eaf5dc8`
  - `result:sids_for_hgrec.npy`: `8d39e8a7b4484a65` -> `8d39e8a7b4484a65`
  - `trainer`: `d2529222cb86980f` -> `d2529222cb86980f`

The frozen held-out pair artifact is never read by this probe. A risk verdict never unlocks Stage3 and never retunes training.
