# S03_PROVENANCE — canonical source packet

STAGE_ID=S03_PROVENANCE
ROUND=1
ITERATION=30

## Objective
Independently adjudicate whether each historical input to the S02 mappings has a supported semantic definition, source/method lineage, layer ordering, and exact recorded value under FCCR-1. Specifically audit `behavior_branching` and the explicit `raw_residual_medians` vector. Preserve the distinction between historical method/value provenance and checkpoint-level reproduction / historic input-byte identity. This is a hard gate: failure blocks downstream mapping approval/MVG/Stage2. Do not recalibrate, rerun historical writers, or substitute inputs.

## Governing canonical sources
1. Root `CLAUDE.md`, active `skill://curvature-rqvae-iter` (especially §§3, 6–8, 17–19), and S00/S01/S02 canonical files.
2. S00: `stage2_RQ-VAE/curvature_RQ-VAE_iter30/logs/source_snapshot_iter30.md`.
3. S01: `stage2_RQ-VAE/curvature_RQ-VAE_iter30/logs/protocol_manifest_iter30.md` and `logs/deliberation/S01_PROTOCOL_LOCK/round_1/judge.md`.
4. S02: `stage2_RQ-VAE/curvature_RQ-VAE_iter30/logs/hypothesis_iter30.md` and `logs/deliberation/S02_HYPOTHESIS/round_1/judge.md`. S02 accepted two equations but explicitly leaves residual provenance `PROVISIONAL_PENDING_S03`; no S03 pass exists yet.
5. Primary historical evidence to inspect directly:
   - behavior producer: `stage2_RQ-VAE/curvature_RQ-VAE_iter12/scripts/compute_behavior_branching.py`;
   - preserved value artifact: `stage2_RQ-VAE/curvature_RQ-VAE_iter26/scripts/computed_behavior_branching.json`;
   - residual measurement implementation/log: `stage2_RQ-VAE/curvature_RQ-VAE_iter10/scripts/calibrate_residual_scales.py` and `stage2_RQ-VAE/curvature_RQ-VAE_iter10/logs/calibrate_residual_scales.log`;
   - older semantic record: `stage2_RQ-VAE/curvature_RQ-VAE_iter10/logs/hypothesis_iter1.md`;
   - consumer/key history: iter26 `scripts/compute_closed_form_curvature.py`, `logs/hypothesis_iter26.md`, `logs/mechanism_manifest_iter26.md`; iter29 `scripts/compute_closed_form_curvature.py`, `logs/mechanism_manifest_iter29.md` and prior iter29 S03 Judge artifact, as subordinate historical references only.

## Exact S02 input values and mapped outputs
All entries are ordered `[L0,L1,L2]`:
```text
behavior_branching = [19.324911558712664, 1.4605688962651735, 1.0148104414712726]
raw_residual_medians = [1.0, 0.10941, 0.09331]  # currently PROVISIONAL_PENDING_S03
normalized_layer_scales = [0.001, 0.932889, 1.0]  # distinct; forbidden as substitute
```
The paired S02 mappings and outputs are already fixed; S03 audits input semantics/provenance only and must not reselect/recompute/retune either mapping. Iter26 historical closed-form output is `[0.6145357379232853,0.5333020920777128,0.3814078098431606]`; iter29 output is `[1.3660953164241916,0.7347829661951981,0.6439958072706683]`.

## Direct historical source observations to check independently
- The iter26 JSON records entropy `[2.9613950179410296,0.37882601480060585,0.014701837881298745]`, the exact branching vector above, an explicit `raw_residual_medians` field with the exact residual vector, and a legacy `residual_norm` field numerically equal to it. Its `source.script` metadata identifies iter12 `compute_behavior_branching.py`, and source metadata points to iter8 raw SIDs and Stage0 train parquet. This is an aggregated historical artifact; field equality does not alone establish the alias's semantics.
- Iter12 behavior code loads `ITER8_RAW_SIDS` and Stage0 `train.parquet`; it computes ordered conditional entropies `H(T0|source)`, `H(T1|source,T0)`, `H(T2|source,T0,T1)` using observation-count-weighted natural-log entropies and sets `branching=exp(entropy)`. `RESIDUAL_NORMS=[0.001,0.932889,1.0]` feeds the separate `raw_capacity`/`branch_mid` side fields, not the entropy or branching calculations. The raw-SID path and train path are source code constants.
- Iter10 residual calibration code loads the named baseline checkpoint and Stage1 item embeddings, runs `get_semantic_ids` in chunks, takes `result.residuals.norm(dim=1)` over embedding dimension for each layer/item, concatenates across item chunks, and takes the median over items (`median(dim=1)`). It separately transforms those medians by `max(log(s_max / m_l) / log(s_max / s_min), 1e-3)` to form normalized layer scales.
- Iter10 contemporaneous calibration log reports `step=100000`, raw `medians=[1.0,0.10941,0.09331]`, and separate `layer_norms=[0.001,0.932889,1.0]`. The same log identifies the historical baseline checkpoint path `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE/rqvae_best.pth` and Stage1 embedding path. Iter1 historical hypothesis repeats the raw/normalized distinction.
- Current inventory evidence in S01 says the referenced calibration checkpoint and iter8 `sids_raw.npy` are absent at their historical paths. Their historic SHA256 identities were not recorded. The present iter8 checkpoint and present Stage1/Stage0 files were hash-checked at S01, but these current identities do not prove historic bytes or historical runtime use.
- Iter26's `compute_closed_form_curvature.py` consumes ambiguous JSON key `residual_norm`; its preserved JSON gives that alias values equal to `raw_residual_medians`, while iter12's source uses the same key name for normalized scales. Iter26's JSON metadata names iter12 as branching producer although it also contains a raw-residual field sourced elsewhere. S02 therefore requires every new iter30 consumer to use the explicit `raw_residual_medians` field and fail closed if missing; never rely on `residual_norm`.

## Required audit output
Independently produce a complete proposed `mechanism_manifest_iter30.md` with a per-input table containing: symbol/canonical key; exact semantic meaning; source script and producing run; raw recorded value in layer order; every transformation; actual S02 formula input; expected range/sanity bound; evidence strength and limitations. Explicitly distinguish:
```text
raw_residual_median != normalized_layer_scale != learnable_c_layer_scale
```
Classify branching and residual confidence separately. State whether hard gate passes only for historical method/value provenance or fails under active skill §7, and explain why. Do not claim current recalculation, checkpoint replay, historic byte identity, or S03 approval solely from S00/S01/S02 or later declarations. Record the legacy-key hazard and explicit `raw_residual_medians` no-fallback consumer requirement. If a historical producer/output attribution is ambiguous, preserve the ambiguity rather than asserting a false lineage.

## Candidate independence and execution boundary
Agent A and Agent B receive this identical packet and independently inspect primary source files. Each writes only its candidate artifact under `logs/deliberation/S03_PROVENANCE/round_1/`; no candidate may read the other's draft or overwrite the canonical manifest. Judge C adjudicates with direct primary evidence. No source code edits, recalibration, GPU work, or Stage2/Stage3 execution is authorized by S03. If Judge C finds provenance insufficient, stop/block before subsequent approval/MVG/Stage2; do not use normalized scales or infer a replacement.