# Stage2 stability probe

- tree: `/fs04/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_r6_03`
- declaration sha256: `f166ba02d373568cdb9cb2b6bfc839f7aadc3fd787957d901635f2069bfe4fe7`
- probe config sha256: `0586a65d266e01679f8ad9e8a950573d0db1d5a5b0248219f76ca1f89e1a7f2e`
- executed steps: 160 (behaviour-active 112)
- elapsed: 11.17s

## status = `valid`

## risk = `high`

risk is a HEURISTIC early-warning indicator, not a validated predictor, and never a performance gate: the frozen Hard-L1 ER remains the only Stage2 acceptance criterion

Fired: `deep_gradient_domination`, `l1_effective_codes_drop`, `negative_train_pair_match_rise`, `max_code_share_rise`

## Snapshots (RAW hard SIDs; L1-only and 3-level tuple kept separate)

| step | active | L1 used/256 | L1 H/log256 | L1 eff.codes | L1 max share | L1 collision | 3-level collision |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 8 | 0 | 256/256 | 0.8679 | 123.0 | 0.0239 | 0.9896 | 0.0513 |
| 16 | 0 | 256/256 | 0.8025 | 85.6 | 0.0810 | 0.9896 | 0.1068 |
| 32 | 0 | 248/256 | 0.6442 | 35.6 | 0.1542 | 0.9899 | 0.3801 |
| 64 | 16 | 163/256 | 0.5092 | 16.8 | 0.1700 | 0.9934 | 0.6394 |
| 96 | 48 | 82/256 | 0.4255 | 10.6 | 0.2588 | 0.9967 | 0.8568 |
| 128 | 80 | 76/256 | 0.4337 | 11.1 | 0.2530 | 0.9969 | 0.8389 |
| 160 | 112 | 79/256 | 0.4340 | 11.1 | 0.2685 | 0.9968 | 0.8158 |

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
