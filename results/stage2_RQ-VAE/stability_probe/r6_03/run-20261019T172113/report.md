# Stage2 stability probe

- tree: `/fs04/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_r6_03`
- declaration sha256: `f166ba02d373568cdb9cb2b6bfc839f7aadc3fd787957d901635f2069bfe4fe7`
- probe config sha256: `0586a65d266e01679f8ad9e8a950573d0db1d5a5b0248219f76ca1f89e1a7f2e`
- executed steps: 160 (behaviour-active 112)
- elapsed: 39.53s

## status = `valid`

## risk = `high`

risk is a HEURISTIC early-warning indicator, not a validated predictor, and never a performance gate: the frozen Hard-L1 ER remains the only Stage2 acceptance criterion

Fired: `deep_gradient_domination`, `l1_effective_codes_drop`, `negative_train_pair_match_rise`, `max_code_share_rise`

## Snapshots (RAW hard SIDs; L1-only and 3-level tuple kept separate)

| step | active | L1 used/256 | L1 H/log256 | L1 eff.codes | L1 max share | L1 collision | 3-level collision |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 8 | 0 | 256/256 | 0.8678 | 123.0 | 0.0239 | 0.9896 | 0.0482 |
| 16 | 0 | 256/256 | 0.8015 | 85.2 | 0.0826 | 0.9896 | 0.1005 |
| 32 | 0 | 246/256 | 0.6426 | 35.3 | 0.1484 | 0.9900 | 0.3655 |
| 64 | 16 | 161/256 | 0.5081 | 16.7 | 0.1758 | 0.9935 | 0.6565 |
| 96 | 48 | 73/256 | 0.4411 | 11.5 | 0.1799 | 0.9970 | 0.8605 |
| 128 | 80 | 71/256 | 0.4589 | 12.7 | 0.2518 | 0.9971 | 0.8321 |
| 160 | 112 | 77/256 | 0.4340 | 11.1 | 0.2715 | 0.9969 | 0.8160 |

L1-only collision and the three-level tuple collision have very different natural scales and are never mixed.

## Isolation

- production artifacts unchanged: **True**
  - `frozen_heldout_pairs`: `04d779208343a9e5` -> `04d779208343a9e5`
  - `log:train_migrated.log`: `ceed129ab460e7ce` -> `ceed129ab460e7ce`
  - `log:training_metrics.jsonl`: `980342779347fc61` -> `980342779347fc61`
  - `result:item_sids.json`: `31d5a937d30a7bef` -> `31d5a937d30a7bef`
  - `result:rqvae_best.pth`: `076909b200678f6c` -> `076909b200678f6c`
  - `result:sids_for_hgrec.npy`: `21730f48e316d06c` -> `21730f48e316d06c`
  - `trainer`: `ae978e330354e414` -> `ae978e330354e414`

The frozen held-out pair artifact is never read by this probe. A risk verdict never unlocks Stage3 and never retunes training.
