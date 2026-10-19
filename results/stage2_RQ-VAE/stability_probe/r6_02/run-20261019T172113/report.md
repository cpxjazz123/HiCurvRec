# Stage2 stability probe

- tree: `/fs04/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_r6_02`
- declaration sha256: `1a79ab1d73e98e7498d029d119690b817c113e4ea3b305e35a5b950f67fceb59`
- probe config sha256: `0586a65d266e01679f8ad9e8a950573d0db1d5a5b0248219f76ca1f89e1a7f2e`
- executed steps: 160 (behaviour-active 112)
- elapsed: 39.11s

## status = `valid`

## risk = `high`

risk is a HEURISTIC early-warning indicator, not a validated predictor, and never a performance gate: the frozen Hard-L1 ER remains the only Stage2 acceptance criterion

Fired: `deep_gradient_domination`, `l1_effective_codes_drop`, `negative_train_pair_match_rise`

## Snapshots (RAW hard SIDs; L1-only and 3-level tuple kept separate)

| step | active | L1 used/256 | L1 H/log256 | L1 eff.codes | L1 max share | L1 collision | 3-level collision |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 8 | 0 | 256/256 | 0.8679 | 123.1 | 0.0238 | 0.9896 | 0.0469 |
| 16 | 0 | 256/256 | 0.8019 | 85.4 | 0.0823 | 0.9896 | 0.0990 |
| 32 | 0 | 246/256 | 0.6450 | 35.8 | 0.1463 | 0.9900 | 0.3613 |
| 64 | 16 | 160/256 | 0.5138 | 17.3 | 0.1779 | 0.9935 | 0.6458 |
| 96 | 48 | 81/256 | 0.4554 | 12.5 | 0.1814 | 0.9967 | 0.8500 |
| 128 | 80 | 70/256 | 0.4550 | 12.5 | 0.2224 | 0.9972 | 0.8092 |
| 160 | 112 | 63/256 | 0.4726 | 13.7 | 0.1470 | 0.9974 | 0.7799 |

L1-only collision and the three-level tuple collision have very different natural scales and are never mixed.

## Isolation

- production artifacts unchanged: **True**
  - `frozen_heldout_pairs`: `04d779208343a9e5` -> `04d779208343a9e5`
  - `log:train_migrated.log`: `82cb004290302d7f` -> `82cb004290302d7f`
  - `log:training_metrics.jsonl`: `e3c60539c8cc881c` -> `e3c60539c8cc881c`
  - `result:item_sids.json`: `ef09df22e66a2742` -> `ef09df22e66a2742`
  - `result:rqvae_best.pth`: `8427306b9a829d6a` -> `8427306b9a829d6a`
  - `result:sids_for_hgrec.npy`: `25b4dbba2a8f9650` -> `25b4dbba2a8f9650`
  - `trainer`: `02f31155c173c39d` -> `02f31155c173c39d`

The frozen held-out pair artifact is never read by this probe. A risk verdict never unlocks Stage3 and never retunes training.
