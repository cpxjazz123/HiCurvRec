## Inventory and frozen analysis contract (#270)

**INFRA_MODE = FROZEN.** The issue explicitly forbids Stage2/Stage3, retraining, mechanism and
public-infra changes; no training is required (analysis-only). Body sha256 `04c90c24…`.

**Inventory.** 204 Stage3 runs with `test_final.json`; the R2 ledger has 111 arms with a four-piece
entry. Resolving exact SID + checkpoint + the 57,439 saved predictions gives a usable geometry
cohort of **42 arms**: **12** are record-verified against the immutable-run hashes, **30** are
legacy arm-directory bindings (the pre-`_runs` Stage3 records carry no SID/checkpoint path; their
files and hashes are recorded in `COHORT.json`). No duplicate SIDs. Protocol proxy: 1 single-card
arm, 41 four-card.

**Gaps (recorded, not imputed).** The #259–#264 geometry arms (`role_geom*`, `mixed_rq_F0`) have
SIDs/checkpoints but no saved Stage3 prediction shards locally, so they cannot enter the
recommendation linkage. No validation predictions exist (Stage3 `NO_EVAL=True`), so the
recommendation linkage is explanatory error analysis on the frozen test set, not an independent
validation. Target-sequence length is not stored in the predictions and stays uncontrolled.

**Frozen ruler and quantities.** The common ruler is the #265 parent `tangent_scale`
`0.08433422446250916` (TIGER has no `tangent_scale`; #269's `0.0825626328587532` is a sensitivity).
The A.1 target is unchanged (28,391,948 rows); hard q is reconstructed exactly from the checkpoint
codebooks + SID; continuous `z` is reported only for the three primary arms. `J` (distance
distortion) and the #269 `kappa_t` ruler are computed on the fixed 20,000-row sample only and are
labelled as such.

**Pre-registered tests.** P1 arm-level curvature improvement vs ΔR@10; P2 conditional on SID
preservation (L2 ratio ≥ 0.9); P3 #269 vs #265 SID compression; P4 family/protocol dominance.
Family-cluster bootstrap (2,000), item-cluster bootstrap, leave-one-family-out, BH FDR q=0.10,
counterexample audit. `CONTRACT.md` freezes these before any correlation is computed.

**Assets reused (read-only).** `assets/train/visible_undirected.npz:097a719a…`,
`assets/v8_runtime_layout/targets.npy:8c0ff9f8…`, `target_offsets.npy:33f30b29…`,
`assets/node_moments.npz`, `assets/index.npz`, `assets/shards/*`, the frozen TIGER/#265/#267/#269
checkpoints and SIDs, and the canonical `results/research/behavior_graph_directed/edges.parquet`.

Evidence: `results/research/issue270/{RUN_INVENTORY.json,COHORT.json,INVENTORY.md,CONTRACT.md,
DIAGNOSIS.md,RUN_LOG.md}`.
