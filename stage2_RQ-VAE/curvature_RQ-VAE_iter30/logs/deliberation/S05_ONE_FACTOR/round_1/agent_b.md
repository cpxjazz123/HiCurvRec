ROLE=AGENT_B
INDEPENDENCE_DECLARATION=I did not read the other candidate before completing this artifact.
SOURCE_PACKET=stage2_RQ-VAE/curvature_RQ-VAE_iter30/logs/deliberation/S05_ONE_FACTOR/round_1/source_packet.md
STAGE_ID=S05_ONE_FACTOR

# Candidate One-Factor Diff — iter30 (Agent B, independent)

```text
PARENT_ITER=iter29 (implementation/source-lineage parent; not paired control)
PARENT_COMMIT=ecd01e4712a1badd38a0338255f4b2ec7b030aff
PARENT_DELIVERABLE_COMMIT=57b4a594d92435fd74fa4bba7c7c1439a33eb102
CANONICAL_BASELINE_ITER=iter26 (historical baseline and fresh paired-control mapping)
EXPERIMENT_TYPE=single_factor; prospective matched-seed replication of two already-existing FCCR-1 mapping settings
ACTIVE_MECHANISMS_BEFORE=FCCR-1 fixed closed-form curvature plus unchanged inherited RQ-VAE behavior/reconstruction/quantization mechanisms
NEW_MECHANISM=No new conceptual mechanism. Compare only the S14-approved iter26 and iter29 fixed FCCR-1 curvature mappings using common registered inputs.
ACTIVE_MECHANISMS_AFTER=FCCR-1 fixed closed-form curvature under exactly one of the two registered mappings; no stacking or third map
STAGE2_EXECUTION_AUTHORIZATION=NO
STAGE3_EXECUTION_AUTHORIZATION=NO
```

## Decision and causal boundary

**Candidate verdict: prospective within-seed pair is single-factor, conditionally**, if S06 implements one common iter30 training/consumer source and policy path for both arms, changing only the selected registered mapping and resulting fixed vector. Both arms must explicitly consume the same `branching` vector and the same `raw_residual_medians` values in `[L0,L1,L2]` order. The plural `raw_residual_medians` JSON key is mandatory and missing-key behavior must fail closed; `residual_norm`, normalized scales, fallback, or inferred/recomputed values are prohibited. The legacy iter26 `residual_norm` alias is a historical consumer hazard, not an allowed prospective data route.

This is not an experiment comparing all of iter26's historical source tree with all of iter29's. Iter29 is the source-lineage parent; the direct prospective control is a fresh iter30 run of the iter26 map at the same seed as its iter29-map candidate. The historical seed-42 iter26 test result is the canonical baseline record, not a replacement for a paired observation. Likewise, the historical seed-42 iter29 result is context only, not a prospective observation or a fourth pair. S01 excludes historical seed 42 from the primary paired estimator.

## Exact fixed mapping-only contrast

Registered common values, order `[L0,L1,L2]`:

```text
B = [19.324911558712664, 1.4605688962651735, 1.0148104414712726]
raw_residual_medians = [1.0, 0.10941, 0.09331]
normalized_layer_scales = [0.001, 0.932889, 1.0]  # distinct; never substitute
```

S03 approves historical method/value provenance only: branching confidence HIGH; raw-residual confidence MEDIUM, without checkpoint replay, present-day recalculation, or historical byte-identity proof. These limits travel with both arms. The future common implementation reads the explicit raw key and checks the registered vector; it must not rerun an old writer that overwrites or fails to provide it.

| Arm | Mapping and exact constants | Exact intermediates `[L0,L1,L2]` | Fixed `c=[c0,c1,c2]` |
|---|---|---|---|
| iter26 mapping (control) | `m_min=min(m_raw)`; `s=log1p(B)/log1p(m_raw/m_min)`; `z=(s-mean(s))/(std_population(s)+1e-12)`; `c=clip(0.5*exp(0.2*z),0.05,1.5)` | `m_min=0.09331`; `s=[1.2238118890466054,1.1604516044603779,1.010644112691917]`; `z=[1.03129493309937,0.3223997103378134,-1.3536946434371857]` | `[0.6145357379232853,0.5333020920777128,0.3814078098431606]` |
| iter29 mapping (candidate) | `x=B/(B+2.0)`; `y=m_raw/(m_raw+0.1)`; `u=(x+y)/2`; `c=0.05+1.45*u` | `x=[0.9062129756321139,0.42206034326942476,0.3366083742817442]`; `y=[0.9090909090909091,0.5224678859653312,0.48269618747090165]`; `u=[0.9076519423615115,0.4722641146173780,0.40965228087632294]` | `[1.3660953164241916,0.7347829661951981,0.6439958072706683]` |
| candidate minus control | mapping output only | — | `[0.7515595785009063,0.2014808741174853,0.2625879974275077]` |

The output vectors are already-fixed FCCR-1 curvatures. Within a seed pair, no other parameter, equation, mechanism, or data input changes. Curvature consumers can therefore realize different distances/assignments/effective epsilon as downstream consequences of `c`; those are mediation through the changed mapping output, not independent treatments.

## Inherited mechanisms and implementation constraints

Direct primary inspection of iter26/iter29 `modules/rqvae.py` showed the same module snapshot/content for the RQ-VAE loss/architecture path. The configs and trainer profiles corroborate the shared values below; these are not a claim of byte-identical full historical trainer trees.

- **Architecture/data path:** input dimension 768; hidden dimensions `[512,256,128]`; embedding dimension 32; 3 quantizer layers × 256 codebook entries; normalized MLP encoder, decoder/reconstruction path, K-means codebook initialization; all-item reconstruction plus training-only next-item transitions. Stage2 reads the same locked Stage1 embedding, item-ID sidecar, and Stage0 `train.parquet`; the same immutable iter8 checkpoint is the common warm start. Preserve data order and sampling semantics. No Stage1 retraining or embedding change.
- **Behavior/reconstruction/quantization losses:** existing reconstruction and quantization/commitment losses remain; `COMMITMENT_WEIGHT=1.0`; shared `BEHAVIOR_LOSS_WEIGHT=0.20`, `BEHAVIOR_TEMPERATURE=0.07`, `CURVATURE_REG_WEIGHT=0.0`. No new auxiliary loss, changed weight, loss term, or curvature regularizer. Fixed curvature does not receive gradients.
- **Optimizer/training:** AdamW `lr=1e-3`, `weight_decay=1e-4`; global batch 2560 (640/GPU × 4 ranks); 100,000 global steps; checkpoint cadence 10,000; gradient clipping norm 1.0; identical 4-rank DDP launch/runtime settings and serial execution. The historical trainers hardcode seed 42; future profiles must instead use exactly paired seeds 43, 44, 45, with the same seed on both Stage2 arms of each pair. Seed-specific profiles do not create a treatment difference within pair.
- **Sinkhorn and geometry consumers:** configured `sk_eps=0.05`, `sk_iters=3`; same centering/assignment implementation and Poincaré distance, exp/log maps, Möbius residual/M2 update and M3 transport paths. The quantizer's existing `effective_eps = sk_eps * (c/c_cyclic_max)` rule is held fixed. Its realized value changes as a deterministic, inherited consumer of the changed fixed `c`; it is not an added or retuned Sinkhorn mechanism. Same midpoint mask, residual configuration, and curvature-supported geometry outside the registered vector.
- **Fixed-curvature invariants:** both mappings are computed before Stage2, stored/consumed as fixed, non-trainable buffers, and held invariant to step, optimizer updates, and train/eval state; no cyclic schedule or trainable curvature. Verify each arm against its own vector. Do not interpret the historical iter26 path's legacy key ambiguity as a permission to bypass iter30 explicit-key validation.
- **Stage2 reporting/policy:** root policy is descriptive-only metrics and no quality-proxy admission/early-stop gate. Iter29's existing policy migration removed the SID-independent HitRate@50 subprocess and leaves SID-derived descriptions; iter30 must use one common implementation/reporting path for both arms. This is policy/reporting hygiene, not an arm difference. It must not alter sampling, gradients, update count, checkpoint/SID generation, or export.

## Optimizer, loss, Stage1, Stage3 comparison declarations

```text
OPTIMIZER_DIFFERENCES=None within each prospective pair; same AdamW implementation and lr=1e-3 / weight_decay=1e-4. Stage2 seed is matched within pair (43, 44, or 45), not seed 42.
LOSS_DIFFERENCES=None within each prospective pair; same reconstruction, commitment/quantization, and behavior terms and weights; no added curvature regularization or curvature-conditioned auxiliary loss.
STAGE1_DIFFERENCES=None; Stage1 embedding and item-ID/order sidecar are fixed shared protocol inputs, not re-generated per arm.
STAGE3_DIFFERENCES=None for trainer, model/configuration, runtime, data split, and metric computation. Use the unchanged Stage3 trainer SHA256 locked by S01 (9861eb6b2a5b0bfac546524c50d402aba5827e5079a228e2ae5501909bc937cb), matching Stage3 seeds, unchanged evaluation settings, and final test for each valid run.
```

Stage3 remains 150 configured maximum epochs with existing `NO_EVAL=True` and train-loss patience behavior, `SKIP_TEST=False`, beam size 20, top K `[5,10]`, expected `n_eval=57439`, batch 4096, inference batch 1024, four ranks and port 50201. Preserve trainer model/optimizer/scheduler/precision/determinism/NCCL settings. Do not modify the Stage3 trainer to correct metadata. S01 records a nonfatal historical iter29 wrapper caveat: its intended variant label did not appear as intended in training metrics (`unknown_variant`); disclose it and use the same common wrapper behavior in both prospective arms, without making label/reporting behavior a treatment.

## Historical source diff versus prospective treatment

Direct source/config inspection distinguishes these historic differences from the future treatment:

1. **Mapping loader evolution:** iter26's closed-form script computes the older standardized/log-ratio map from `payload["residual_norm"]`; its trainer loads the stored `closed_form_c_l` and range-checks it, but does not itself recompute/validate the registered inputs and mapping there. Iter29's calculator/trainer uses the bounded rational/additive map and validates explicit branching/raw-residual inputs, formula intermediates, mapping constants, and contract. Iter30 requires a common audited implementation for both arms: explicit `raw_residual_medians` only, fail closed, recompute/check the selected registered map and vector. The explicit-key fix applies identically to control and candidate; it is not a control-only conceptual change.
2. **Reporting-policy migration:** iter29 removed the project-local HitRate@50 subprocess/report field and made `should_early_stop()` unconditionally `(False, "")`, consistent with root policy; the old iter26 source tree still contains the historical path. Iter30 must carry the same iter29 policy/hygiene implementation to both arms. It may change reporting/policy code relative to historic iter26 but cannot vary by arm or change training behavior.
3. **Checks/reporting additions:** iter29 has stricter FCCR-1 input/vector/contract checks and may have additional logging/reporting versus iter26. These checks are verification/hygiene and must be shared identically across both arms. No separate check may alter one arm's optimization, data, or execution duration.
4. **Historical configuration/output identity:** iter26 and iter29 configs embed different mechanism names and historical result roots; their old Stage3 wrappers point to their respective old SID/results paths and labels. Those are historical routing differences, not training treatments. Iter30 uses fresh destinations for all six new runs.

No historical claim of complete code identity is made. The iter26 and iter29 source/config S01 hashes are historical evidence anchors only (iter29 `2469b193bee1c4352ba180540c305c9ea837043afafd1b1585ea7b8aff805f4d` / `b41bf1fff2b481abf846d71a43c30c7ba34d0d25bd93bc04f70df486e3163f7a`; iter26 `ed629824d5a0308f4018b6dbaaa24303690a3597def7d1bcbf9abd38c3aca10f` / `4d6b3e06aa2f408a00664d971d64ed83d0e88ef5f88b852be97302a0e2786086`). They do not attest future run-profile identity or actual runtime use.

## Future changed files and protocol-route-only differences

There is no source/config/wrapper code change authorized in this S05 candidate. For a later S06 plan, the only arm-varying values should be the map selector/vector, explicit full run label and the route-specific paths below. All six profiles use a common implementation, same locked upstream inputs, same settings and no CLI/environment run selection. Each pair runs serially; Stage2 seed and Stage3 seed both match across its arms (43, 44, then 45). Within each pair, the following differences are routing/identity metadata only:

- **Stage2 profile/config:** hardcoded seed (pair-specific), `MECHANISM_NAME=iter30_<arm>_seed<seed>`, registered mapping selection/vector, and unique output destinations. Shared upstream input identities/path use, optimizer/model/data/runtime settings. For `<arm>_<seed>` use `results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/<arm>_seed<seed>/` as run root; its `RQVAE_OUT_DIR=<root>/out/rqvae/instruments/`, `RQVAE_CKPT_PATH=<...>/rqvae_best.pth`, `RAW_SIDS_NPY=<...>/sids_raw.npy`, `SIDS_NPY=<root>/dataset/Instruments/sids_for_hgrec.npy`, and `ITEM_SIDS_JSON=<root>/item_sids.json`. Distinct per-run logs/profiles/hashes prevent collision. Keep produced artifacts outside the source tree.
- **Stage3 wrapper:** hardcode matching Stage2 `CODE_PATH=<root>/item_sids.json`, `RQVAE_VARIANT=iter30_<arm>_seed<seed>`, unique Stage3 root `results/stage3_T5Train/curvature_RQ-VAE_iter30/<arm>_seed<seed>/`, `LOG_PATH=<root>/logs/`, `SAVE_PATH=<root>/ckpt/`, and launcher/wrapper log path under that root. The Stage3 trainer itself, all trainer settings, input split and evaluation stay the same. These arm labels and path substitutions route outputs and do not change model behavior.
- The six planned labels are exactly `iter26_mapping_seed43`, `iter29_mapping_seed43`, `iter26_mapping_seed44`, `iter29_mapping_seed44`, `iter26_mapping_seed45`, and `iter29_mapping_seed45`. No seed-42 run enters the primary estimator; no path collision, overwriting, omitted, substituted, or pooled historical observation is permitted.

## Risks, self-rejection and next action

**Confound risks / blocking facts:** (a) if a control consumes `residual_norm`, normalized `[0.001,0.932889,1.0]`, or any key/fallback other than explicit `raw_residual_medians`, it is not the registered contrast; missing key blocks, not triggers a repair by substitution; (b) any mapping/vector mismatch, dynamic/trainable curvature, changed optimizer/loss/Sinkhorn/behavior rule, differing warm-start or Stage1/Stage0 consumer, seed mismatch within pair, Stage1 or Stage3 drift, or arm-specific report/policy path invalidates one-factor claims; (c) changing the curvature vector legitimately changes Poincaré geometry and the inherited curvature-scaled epsilon, so do not mislabel mediated consumer outputs as unchanged, nor count their changed values as a second mechanism; (d) historic iter26/iter29 config/trainer/reporting/path differences cannot simply be carried forward arm-by-arm; future common-source equivalence must be established by S06–S08 and per-run checks; (e) raw residual values remain medium-confidence historical provenance, not replayed or checkpoint-level evidence; (f) the planned runs and hashes do not yet exist, and protocol declarations alone do not prove runtime input use.

**Self-rejection condition:** I reject this candidate's conditional single-factor conclusion and recommend blocking before implementation if primary later evidence cannot establish one shared future Stage2/Stage3 implementation with only the two registered mapping outputs varying, or if the explicit raw field/common input values, common seed/data/warm-start and runtime settings cannot be enforced. Do not proceed by silently narrowing to one seed, dropping an arm, using a historical seed-42 score as control, or changing a map.

**Next action:** Judge C adjudicates this audit against primary evidence and materializes only canonical `logs/one_factor_diff_iter30.md` plus its Judge record. If accepted, proceed to S06 only through its separately adjudicated implementation design; retain all S03 confidence limits and S01 route constraints. This candidate authorizes no source edits, preflight/MVG, Stage2 or Stage3 execution, GPU work, or training launch.

## Primary evidence directly inspected

- Iter30 canonical `logs/deliberation/S05_ONE_FACTOR/round_1/source_packet.md`; canonical `logs/source_snapshot_iter30.md`, `logs/protocol_manifest_iter30.md`, `logs/hypothesis_iter30.md`, `logs/mechanism_manifest_iter30.md`, `logs/mechanism_contract_iter30.json`, and S04 Judge record.
- Iter26 `curvature_config.py`, `curvature_RQ-VAE.py` mapping loader/training profile, `modules/rqvae.py`, `modules/quantize.py`, `scripts/compute_closed_form_curvature.py`, and `scripts/run_stage3_iter26.py`.
- Iter29 `curvature_config.py`, `curvature_RQ-VAE.py` mapping loader/training profile, `modules/rqvae.py`, `scripts/run_stage3_iter29.py`, and canonical historical `logs/one_factor_diff_iter29.md` for reporting-hygiene migration context; historical audit claims were not treated as substitute for inspected primary source.
- S01 protocol locks source/config reference SHA256s and current Stage3 trainer SHA256; these identify reviewed reference states only and are not hashes for future iter30 profiles.
