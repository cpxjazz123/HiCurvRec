# Stage2 stability probe

- tree: `/fs04/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_r4_08`
- declaration sha256: `90b59345a00056101391bab12685ee759a67f443fc9b20cb382e3cb34abece8d`
- probe config sha256: `0586a65d266e01679f8ad9e8a950573d0db1d5a5b0248219f76ca1f89e1a7f2e`
- executed steps: 160 (behaviour-active optimizer steps 128, mechanism calls 384)
- elapsed: 12.7s

## status = `valid`

## risk = `high`

risk is a HEURISTIC early-warning indicator, not a validated predictor, and never a performance gate: the frozen Hard-L1 ER remains the only Stage2 acceptance criterion

Fired: `deep_gradient_domination`, `l1_effective_codes_drop`, `negative_train_pair_match_rise`

## Snapshots (RAW hard SIDs; L1-only and 3-level tuple kept separate)

| step | active steps | mechanism calls | L1 used/256 | L1 H/log256 | L1 eff.codes | L1 max share | L1 collision | 3-level collision |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 0 | 0 | 0 | 256/256 | 0.8766 | 129.1 | 0.0181 | 0.9896 | 0.0468 |
| 8 | 0 | 0 | 256/256 | 0.8679 | 123.0 | 0.0239 | 0.9896 | 0.0482 |
| 16 | 0 | 0 | 256/256 | 0.8015 | 85.1 | 0.0826 | 0.9896 | 0.1012 |
| 32 | 0 | 0 | 247/256 | 0.6424 | 35.2 | 0.1498 | 0.9900 | 0.3621 |
| 64 | 32 | 96 | 175/256 | 0.5217 | 18.1 | 0.2780 | 0.9929 | 0.6464 |
| 96 | 64 | 192 | 131/256 | 0.4894 | 15.1 | 0.2076 | 0.9947 | 0.6985 |
| 128 | 96 | 288 | 122/256 | 0.5309 | 19.0 | 0.1260 | 0.9950 | 0.5690 |
| 160 | 128 | 384 | 116/256 | 0.5447 | 20.5 | 0.1469 | 0.9953 | 0.4391 |

L1-only collision and the three-level tuple collision have very different natural scales and are never mixed.

## Isolation

- production artifacts unchanged: **True**
  - `frozen_heldout_pairs`: `04d779208343a9e5` -> `04d779208343a9e5`
  - `log:train_migrated.log`: `None` -> `None`
  - `log:training_metrics.jsonl`: `None` -> `None`
  - `result:item_sids.json`: `None` -> `None`
  - `result:rqvae_best.pth`: `None` -> `None`
  - `result:sids_for_hgrec.npy`: `None` -> `None`
  - `trainer`: `18231f691736d2ed` -> `18231f691736d2ed`

The frozen held-out pair artifact is never read by this probe. A risk verdict never unlocks Stage3 and never retunes training.
