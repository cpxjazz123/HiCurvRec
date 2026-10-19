# Stage2 stability probe

- tree: `/fs04/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_r6_03`
- declaration sha256: `f166ba02d373568cdb9cb2b6bfc839f7aadc3fd787957d901635f2069bfe4fe7`
- probe config sha256: `0586a65d266e01679f8ad9e8a950573d0db1d5a5b0248219f76ca1f89e1a7f2e`
- executed steps: 160 (behaviour-active optimizer steps 128, mechanism calls 384)
- elapsed: 12.02s

## status = `valid`

## risk = `high`

risk is a HEURISTIC early-warning indicator, not a validated predictor, and never a performance gate: the frozen Hard-L1 ER remains the only Stage2 acceptance criterion

Fired: `deep_gradient_domination`, `l1_effective_codes_drop`, `negative_train_pair_match_rise`

## Snapshots (RAW hard SIDs; L1-only and 3-level tuple kept separate)

| step | active steps | mechanism calls | L1 used/256 | L1 H/log256 | L1 eff.codes | L1 max share | L1 collision | 3-level collision |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 0 | 0 | 0 | 256/256 | 0.8768 | 129.3 | 0.0181 | 0.9896 | 0.0496 |
| 8 | 0 | 0 | 256/256 | 0.8680 | 123.2 | 0.0239 | 0.9896 | 0.0493 |
| 16 | 0 | 0 | 256/256 | 0.8025 | 85.6 | 0.0805 | 0.9896 | 0.0997 |
| 32 | 0 | 0 | 245/256 | 0.6377 | 34.3 | 0.1701 | 0.9900 | 0.3547 |
| 64 | 32 | 96 | 191/256 | 0.5914 | 26.6 | 0.1984 | 0.9922 | 0.6502 |
| 96 | 64 | 192 | 112/256 | 0.4849 | 14.7 | 0.1777 | 0.9954 | 0.8164 |
| 128 | 96 | 288 | 95/256 | 0.4756 | 14.0 | 0.2114 | 0.9961 | 0.7986 |
| 160 | 128 | 384 | 72/256 | 0.4672 | 13.3 | 0.2313 | 0.9971 | 0.7666 |

L1-only collision and the three-level tuple collision have very different natural scales and are never mixed.

## Isolation

- production artifacts unchanged: **True**
  - `frozen_heldout_pairs`: `04d779208343a9e5` -> `04d779208343a9e5`
  - `log:train_migrated.log`: `ceed129ab460e7ce` -> `ceed129ab460e7ce`
  - `log:training_metrics.jsonl`: `980342779347fc61` -> `980342779347fc61`
  - `result:item_sids.json`: `31d5a937d30a7bef` -> `31d5a937d30a7bef`
  - `result:rqvae_best.pth`: `076909b200678f6c` -> `076909b200678f6c`
  - `result:sids_for_hgrec.npy`: `21730f48e316d06c` -> `21730f48e316d06c`
  - `trainer`: `13e98e644579c16f` -> `13e98e644579c16f`

The frozen held-out pair artifact is never read by this probe. A risk verdict never unlocks Stage3 and never retunes training.
