# Stage2 stability probe

- tree: `/fs04/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_r6_04`
- declaration sha256: `ba72757be6c89de715a3a85f2c12fc473f99f6dfaff3173e1b1fa60a6ca63686`
- probe config sha256: `0586a65d266e01679f8ad9e8a950573d0db1d5a5b0248219f76ca1f89e1a7f2e`
- executed steps: 160 (behaviour-active 112)
- elapsed: 41.52s

## status = `valid`

## risk = `high`

risk is a HEURISTIC early-warning indicator, not a validated predictor, and never a performance gate: the frozen Hard-L1 ER remains the only Stage2 acceptance criterion

Fired: `deep_gradient_domination`, `l1_entropy_drop`, `l1_effective_codes_drop`, `negative_train_pair_match_rise`, `max_code_share_rise`

## Snapshots (RAW hard SIDs; L1-only and 3-level tuple kept separate)

| step | active | L1 used/256 | L1 H/log256 | L1 eff.codes | L1 max share | L1 collision | 3-level collision |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 8 | 0 | 256/256 | 0.8680 | 123.2 | 0.0236 | 0.9896 | 0.0496 |
| 16 | 0 | 256/256 | 0.8026 | 85.7 | 0.0818 | 0.9896 | 0.0982 |
| 32 | 0 | 247/256 | 0.6474 | 36.2 | 0.1486 | 0.9900 | 0.3571 |
| 64 | 16 | 175/256 | 0.4459 | 11.9 | 0.3793 | 0.9929 | 0.6978 |
| 96 | 48 | 117/256 | 0.3125 | 5.7 | 0.3525 | 0.9952 | 0.9301 |
| 128 | 80 | 72/256 | 0.2693 | 4.5 | 0.3145 | 0.9971 | 0.9372 |
| 160 | 112 | 51/256 | 0.2646 | 4.3 | 0.3034 | 0.9979 | 0.9167 |

L1-only collision and the three-level tuple collision have very different natural scales and are never mixed.

## Isolation

- production artifacts unchanged: **True**
  - `frozen_heldout_pairs`: `04d779208343a9e5` -> `04d779208343a9e5`
  - `log:train_migrated.log`: `965903fd7dbf69ba` -> `965903fd7dbf69ba`
  - `log:training_metrics.jsonl`: `307a3fc2c8a84c0e` -> `307a3fc2c8a84c0e`
  - `result:item_sids.json`: `fc7f5ed4b0747678` -> `fc7f5ed4b0747678`
  - `result:rqvae_best.pth`: `0fec712c8eaf5dc8` -> `0fec712c8eaf5dc8`
  - `result:sids_for_hgrec.npy`: `8d39e8a7b4484a65` -> `8d39e8a7b4484a65`
  - `trainer`: `82b2962c46494b97` -> `82b2962c46494b97`

The frozen held-out pair artifact is never read by this probe. A risk verdict never unlocks Stage3 and never retunes training.
