# Stage2 stability probe

- tree: `/fs04/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_r4_08`
- declaration sha256: `90b59345a00056101391bab12685ee759a67f443fc9b20cb382e3cb34abece8d`
- probe config sha256: `0586a65d266e01679f8ad9e8a950573d0db1d5a5b0248219f76ca1f89e1a7f2e`
- executed steps: 160 (behaviour-active optimizer steps 128, mechanism calls 0)
- elapsed: 10.64s

## status = `invalid`

- protocol failure: schedule_mismatch: configured 32 baseline + 128 active, observed 0 baseline + 0 active over 0 step(s)

## risk = `unknown`

risk is a HEURISTIC early-warning indicator, not a validated predictor, and never a performance gate: the frozen Hard-L1 ER remains the only Stage2 acceptance criterion

## Snapshots (RAW hard SIDs; L1-only and 3-level tuple kept separate)

| step | active steps | mechanism calls | L1 used/256 | L1 H/log256 | L1 eff.codes | L1 max share | L1 collision | 3-level collision |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 0 | 0 | 0 | 256/256 | 0.8766 | 129.1 | 0.0181 | 0.9896 | 0.0479 |
| 8 | 0 | 0 | 256/256 | 0.8680 | 123.2 | 0.0236 | 0.9896 | 0.0497 |
| 16 | 0 | 0 | 256/256 | 0.8025 | 85.6 | 0.0819 | 0.9896 | 0.0985 |
| 32 | 0 | 0 | 247/256 | 0.6477 | 36.3 | 0.1479 | 0.9900 | 0.3565 |
| 64 | 32 | 0 | 35/256 | 0.1036 | 1.8 | 0.8643 | 0.9986 | 0.9096 |
| 96 | 64 | 0 | 14/256 | 0.0856 | 1.6 | 0.8738 | 0.9994 | 0.8242 |
| 128 | 96 | 0 | 19/256 | 0.0814 | 1.6 | 0.8967 | 0.9992 | 0.7627 |
| 160 | 128 | 0 | 18/256 | 0.0978 | 1.7 | 0.8602 | 0.9993 | 0.6550 |

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
