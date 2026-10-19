# Stage2 stability probe

- tree: `/fs04/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_r6_02`
- declaration sha256: `1a79ab1d73e98e7498d029d119690b817c113e4ea3b305e35a5b950f67fceb59`
- probe config sha256: `0586a65d266e01679f8ad9e8a950573d0db1d5a5b0248219f76ca1f89e1a7f2e`
- executed steps: 160 (behaviour-active optimizer steps 128, mechanism calls 384)
- elapsed: 13.11s

## status = `valid`

## risk = `high`

risk is a HEURISTIC early-warning indicator, not a validated predictor, and never a performance gate: the frozen Hard-L1 ER remains the only Stage2 acceptance criterion

Fired: `deep_gradient_domination`, `l1_effective_codes_drop`, `negative_train_pair_match_rise`

## Snapshots (RAW hard SIDs; L1-only and 3-level tuple kept separate)

| step | active steps | mechanism calls | L1 used/256 | L1 H/log256 | L1 eff.codes | L1 max share | L1 collision | 3-level collision |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 0 | 0 | 0 | 256/256 | 0.8766 | 129.1 | 0.0181 | 0.9896 | 0.0482 |
| 8 | 0 | 0 | 256/256 | 0.8680 | 123.2 | 0.0237 | 0.9896 | 0.0499 |
| 16 | 0 | 0 | 256/256 | 0.8025 | 85.6 | 0.0817 | 0.9896 | 0.0985 |
| 32 | 0 | 0 | 247/256 | 0.6469 | 36.1 | 0.1498 | 0.9900 | 0.3606 |
| 64 | 32 | 96 | 184/256 | 0.5796 | 24.9 | 0.1920 | 0.9925 | 0.6591 |
| 96 | 64 | 192 | 110/256 | 0.5200 | 17.9 | 0.1545 | 0.9955 | 0.7914 |
| 128 | 96 | 288 | 83/256 | 0.4718 | 13.7 | 0.1706 | 0.9966 | 0.7907 |
| 160 | 128 | 384 | 65/256 | 0.4936 | 15.4 | 0.1900 | 0.9974 | 0.7442 |

L1-only collision and the three-level tuple collision have very different natural scales and are never mixed.

## Isolation

- production artifacts unchanged: **True**
  - `frozen_heldout_pairs`: `04d779208343a9e5` -> `04d779208343a9e5`
  - `log:train_migrated.log`: `82cb004290302d7f` -> `82cb004290302d7f`
  - `log:training_metrics.jsonl`: `e3c60539c8cc881c` -> `e3c60539c8cc881c`
  - `result:item_sids.json`: `ef09df22e66a2742` -> `ef09df22e66a2742`
  - `result:rqvae_best.pth`: `8427306b9a829d6a` -> `8427306b9a829d6a`
  - `result:sids_for_hgrec.npy`: `25b4dbba2a8f9650` -> `25b4dbba2a8f9650`
  - `trainer`: `b812b5be01dae5a3` -> `b812b5be01dae5a3`

The frozen held-out pair artifact is never read by this probe. A risk verdict never unlocks Stage3 and never retunes training.
