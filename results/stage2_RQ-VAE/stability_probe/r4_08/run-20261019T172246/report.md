# Stage2 stability probe

- tree: `/fs04/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_r4_08`
- declaration sha256: `90b59345a00056101391bab12685ee759a67f443fc9b20cb382e3cb34abece8d`
- probe config sha256: `0586a65d266e01679f8ad9e8a950573d0db1d5a5b0248219f76ca1f89e1a7f2e`
- executed steps: 160 (behaviour-active 112)
- elapsed: 12.04s

## status = `valid`

## risk = `high`

risk is a HEURISTIC early-warning indicator, not a validated predictor, and never a performance gate: the frozen Hard-L1 ER remains the only Stage2 acceptance criterion

Fired: `deep_gradient_domination`, `l1_effective_codes_drop`, `negative_train_pair_match_rise`

## Snapshots (RAW hard SIDs; L1-only and 3-level tuple kept separate)

| step | active | L1 used/256 | L1 H/log256 | L1 eff.codes | L1 max share | L1 collision | 3-level collision |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 8 | 0 | 256/256 | 0.8678 | 123.0 | 0.0239 | 0.9896 | 0.0482 |
| 16 | 0 | 256/256 | 0.8016 | 85.2 | 0.0824 | 0.9896 | 0.1005 |
| 32 | 0 | 247/256 | 0.6429 | 35.3 | 0.1480 | 0.9900 | 0.3650 |
| 64 | 16 | 143/256 | 0.5103 | 16.9 | 0.1560 | 0.9942 | 0.6453 |
| 96 | 48 | 88/256 | 0.4861 | 14.8 | 0.2803 | 0.9964 | 0.7753 |
| 128 | 80 | 89/256 | 0.5347 | 19.4 | 0.1565 | 0.9964 | 0.6348 |
| 160 | 112 | 88/256 | 0.5231 | 18.2 | 0.1553 | 0.9964 | 0.4769 |

L1-only collision and the three-level tuple collision have very different natural scales and are never mixed.

## Isolation

- production artifacts unchanged: **True**
  - `frozen_heldout_pairs`: `04d779208343a9e5` -> `04d779208343a9e5`
  - `log:train_migrated.log`: `None` -> `None`
  - `log:training_metrics.jsonl`: `None` -> `None`
  - `result:item_sids.json`: `None` -> `None`
  - `result:rqvae_best.pth`: `None` -> `None`
  - `result:sids_for_hgrec.npy`: `None` -> `None`
  - `trainer`: `959d987918d25202` -> `959d987918d25202`

The frozen held-out pair artifact is never read by this probe. A risk verdict never unlocks Stage3 and never retunes training.
