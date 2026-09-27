ROLE=AGENT_A
INDEPENDENCE_DECLARATION=I did not read the other candidate before completing this artifact.
SOURCE_PACKET=stage2_RQ-VAE/curvature_RQ-VAE_iter30/logs/deliberation/S05_ONE_FACTOR/round_1/source_packet.md
STAGE_ID=S05_ONE_FACTOR

# Proposed `one_factor_diff_iter30.md` — Agent A candidate

```text
PARENT_ITER=iter29 (implementation/source-lineage parent; not the prospective paired control)
PARENT_COMMIT=ecd01e4712a1badd38a0338255f4b2ec7b030aff
CANONICAL_BASELINE_ITER=iter26 (historical baseline and fresh prospective control mapping)
EXPERIMENT_TYPE=single_factor; fresh matched-seed comparison of two already-registered FCCR-1 mappings
ACTIVE_MECHANISMS_BEFORE=FCCR-1 fixed closed-form curvature plus the unchanged inherited RQ-VAE behavior, Poincare quantization, Sinkhorn, M2/M3 residual geometry, and loss consumers
NEW_MECHANISM=No new conceptual mechanism; the sole treatment contrast is choosing the registered iter26 versus iter29 fixed-curvature mapping from identical explicit inputs
ACTIVE_MECHANISMS_AFTER=The same inherited mechanisms with either registered fixed mapping; no stack, third map, optimizer/loss/Sinkhorn rule, Stage1, or Stage3 change
EXECUTION_AUTHORIZATION=NO
```

## Parent, paired control, and historical baseline are distinct roles

- **Source lineage:** iter29 is the implementation parent at S01's current source revision `ecd01e4712a1badd38a0338255f4b2ec7b030aff`. The separate iter29 delivery and S14 synchronization commits recorded by S01 (`57b4a594d92435fd74fa4bba7c7c1439a33eb102` and `fa7801c4d0f9a7e1d96da968cc7e0be891b84bd6`) are history, not the prospective Stage3 source revision.
- **Fresh paired control:** at each new seed, rerun the iter26 mapping in an iter30 profile. It is paired against the iter29 mapping at the same seed. It is not the historical seed-42 model/run.
- **Canonical historical baseline:** iter26's seed-42 result remains the registered baseline context: `test_final.json` at `results/stage3_T5Train/curvature_RQ-VAE_iter26/logs/Amazon_2023_Instruments/Sep-26-2026_18-11-20/test_final.json`, `n_eval=57439`, `R@10=0.057017009349048554`. Iter29 seed 42 is historical candidate context only (`R@10=0.05921064085377531`; `n_eval=57439`). Their historical point delta is `+0.002193631504726755`, not the prospective estimate. Seed-42 runs are excluded from primary pooling; neither replaces a missing matched pair.

## Exact mapping-only treatment contrast

Inputs are shared, explicitly keyed, and ordered `[L0,L1,L2]`:

```text
B = [19.324911558712664, 1.4605688962651735, 1.0148104414712726]
raw_residual_medians = [1.0, 0.10941, 0.09331]
```

The only conceptual contrast within each seed pair is the deterministic map from those inputs to a fixed, non-trainable curvature vector.

**Fresh control — iter26 map:** `m_min=min(m_raw)=0.09331`; NumPy population standard deviation (`ddof=0`):

```text
s_l = log1p(B_l) / log1p(m_l_raw / m_min)
z_l = (s_l - mean(s)) / (std_population(s) + 1e-12)
c26_l = clip(0.5 * exp(0.2 * z_l), 0.05, 1.5)
```

**Candidate — iter29 map:**

```text
x_l = B_l / (B_l + 2.0)
y_l = m_l_raw / (m_l_raw + 0.1)
u_l = (x_l + y_l) / 2
c29_l = 0.05 + 1.45 * u_l
```

| Layer | `B_l` | explicit `m_l_raw` | iter26 `s_l` | iter26 `z_l` | `c26_l` | iter29 `x_l` | iter29 `y_l` | iter29 `u_l` | `c29_l` | `c29-c26` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L0 | 19.324911558712664 | 1.0 | 1.2238118890466054 | 1.03129493309937 | 0.6145357379232853 | 0.9062129756321139 | 0.9090909090909091 | 0.9076519423615115 | 1.3660953164241916 | 0.7515595785009063 |
| L1 | 1.4605688962651735 | 0.10941 | 1.1604516044603779 | 0.3223997103378134 | 0.5333020920777128 | 0.42206034326942476 | 0.5224678859653312 | 0.4722641146173780 | 0.7347829661951981 | 0.2014808741174853 |
| L2 | 1.0148104414712726 | 0.09331 | 1.010644112691917 | -1.3536946434371857 | 0.3814078098431606 | 0.3366083742817442 | 0.48269618747090165 | 0.40965228087632294 | 0.6439958072706683 | 0.2625879974275077 |

Thus `c26=[0.6145357379232853, 0.5333020920777128, 0.3814078098431606]`, `c29=[1.3660953164241916, 0.7347829661951981, 0.6439958072706683]`, and `c29-c26=[0.7515595785009063, 0.2014808741174853, 0.2625879974275077]`. Both are fixed before training, finite, and within `[0.05,1.5]`. All values above agree with iter30 S02/S03 and the respective preserved historical records and formula implementations.

**Input-key rule:** both future arms MUST consume the same explicit `raw_residual_medians` field, fail closed if absent, and use these exact values and layer order. Never use `residual_norm`, normalized scales `[0.001,0.932889,1.0]`, another checkpoint, an inferred/recomputed replacement, or a fallback. Iter26's historical JSON happens to have `residual_norm` numerically equal to the raw vector, but the key is semantically ambiguous: its calculator reads `payload["residual_norm"]`, while iter12 used that key for normalized scales. Equality in that JSON is not evidence for future alias semantics. S03 approves historical method/value provenance only (branching confidence HIGH, residual confidence MEDIUM); it does not establish replay, calibration-checkpoint reproduction, historical byte identity, or future runtime use.

## Direct primary-source comparison and historical-only differences

Reviewed iter26 and iter29 `curvature_RQ-VAE.py`, `curvature_config.py`, mapping scripts and JSON, wrappers, `modules/quantize.py`, `modules/rqvae.py`, `modules/hyperbolic.py`, `modules/loss.py`, `modules/encoder.py`, and supporting modules. The common Stage2 model/support module SHA256 values match pairwise for `encoder.py`, `rqvae.py`, `quantize.py`, `hyperbolic.py`, `loss.py`, `normalize.py`, `step_checks.py`, `utils.py`, and `hab.py`; examples: `rqvae.py=123e7ad2ba146842988e91820cc7a138577cecbe11d479423e15368831ee0944`, `quantize.py=2970ad3e61de79c6a2fd17194258d7d5ed4b9cbbf836e5dde506a8635c2553f8`, `hyperbolic.py=ca7d4fdb5b7f6d4f91cc6e5083462fa9d610386ce4bda65605d3672907e22ae6`, and `loss.py=c5d0106b731ee54b9e96ef40db745f7ad74725e2f76ed9c6b763e77e11108cd1`. The differences below are historical source/reporting/wiring differences; they must not become arm-specific treatment differences in the common prospective implementation.

| Area | Direct historical difference | S05 consequence for the prospective pair |
|---|---|---|
| Mapping and fixed-vector loading | Iter26 `scripts/compute_closed_form_curvature.py::closed_form_curvatures` uses `branching` plus `residual_norm`; its trainer loader accepts stored `closed_form_c_l` after length/range checks. Iter29's calculator requires `branching` and explicit `raw_residual_medians`, checks three finite ordered values, computes bounded rational/additive terms, and its trainer recomputes/compares registered values and contract. | Reuse one reviewed mapping/data path for both arms; only the registered map/vector varies. Require the explicit plural raw key for **both**, and apply equivalent validation/fail-closed checks to both. S04 JSON currently describes the iter29 candidate only; S07/S08 must also verify the iter26 control vector and fixed behavior. Do not let different validation/runtime branches select or execute an arm. |
| Trainer/map and fixed-curvature reporting | Iter26 contains its former mapping loader and iter26-specific log labels; iter29 imports the mapping helper and validates registered inputs/vector/contract, with iter29-specific labels. Both pass three fixed curvatures to the same `RqVae` and check buffer invariance. | Source identity is not the treatment: both future profiles use the same implementation and validation/immutability policy, selected by the hardcoded arm profile. Map-specific numeric expectations are the only scientific difference. |
| Stage2 quality reporting/policy | Iter26's active checkpoint caller invokes `evaluate_sid_quality_full(sids, EMB_NPY)`, which calls the project-local `sid_metrics_any.py` subprocess for HitRate@50. Iter29 calls in-process `evaluate_sid_quality(sids)` and removes the HitRate field/parser/subprocess; its `should_early_stop` is `(False, "")`. Iter26's `should_early_stop` validates metrics and then returns false, while old comments still refer to the removed outer HitRate gate. | Under current root `CLAUDE.md` §2, the common iter30 path must omit the non-SID-dependent HitRate subprocess and report only SID-derived descriptive metrics. Apply this hygiene identically to both arms; keep metrics non-gating and preserve checkpoint/report cadence, SID generation/export, and model/training behavior. Reporting-only migration is not a mechanism or performance gate. |
| Per-iteration config | Iter26/iter29 configs share upstream Stage0/Stage1 paths, determinism/cache/GPU runtime values, but differ in iteration mechanism name and Stage2 result root. | Per-run labels, seeds, and isolated output/log destinations are route metadata; no config/runtime setting besides the registered map may differ within a pair. |
| Stage3 wrapper | Both import the shared trainer and assign `CODE_PATH`, `RQVAE_VARIANT`, `LOG_PATH`, `SAVE_PATH`, and launcher log for their own iteration. Wrapper module names, labels and paths differ. Stage3 trainer is byte-identical to S01's locked file (`9861eb6b2a5b0bfac546524c50d402aba5827e5079a228e2ae5501909bc937cb`). | Future wrappers must use the same wrapper logic and locked trainer, changing only matching seed, arm label/SID path and unique output/log roots for routing. Keep S01's historical metadata caveat: iter29 intended to set `RQVAE_VARIANT`, but its training metrics recorded `unknown_variant`; preserve/disclose it and do not edit Stage3 trainer to repair it. |

No historical code-diff assertion is being made beyond the inspected source areas and the explicit module hash comparisons above. Iter29 source ancestry is not proof that the prospective iter26 control is byte-identical to a historical iter26 execution.

## Prospective changed files and routing (protocol-only except mapping)

There is one common future iter30 implementation; this S05 artifact changes no source/configuration and creates no run profile. For S06, review the common Stage2 trainer, its map-loading/calculation path and reporting helper, hardcoded Stage2 run profiles, and no-argument Stage3 wrappers as one shared implementation. The profiles/wrappers must bake in the S01 selectors without CLI/environment selection. Within each same-seed pair, the only conceptual/source parameter treatment is the iter26 versus iter29 mapping and corresponding fixed vector. No separate policy, optimizer, loss, Sinkhorn, Stage1, or Stage3 code path may depend on arm.

S01's six distinct short-root routes are binding (outputs do not yet exist):

| Seed pair | Arm / run label | Stage2 result root | Stage3 result root |
|---:|---|---|---|
| 43 | iter26 control — `iter26_mapping_seed43` | `results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/iter26_mapping_seed43/` | `results/stage3_T5Train/curvature_RQ-VAE_iter30/iter26_mapping_seed43/` |
| 43 | iter29 candidate — `iter29_mapping_seed43` | `results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/iter29_mapping_seed43/` | `results/stage3_T5Train/curvature_RQ-VAE_iter30/iter29_mapping_seed43/` |
| 44 | iter26 control — `iter26_mapping_seed44` | `results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/iter26_mapping_seed44/` | `results/stage3_T5Train/curvature_RQ-VAE_iter30/iter26_mapping_seed44/` |
| 44 | iter29 candidate — `iter29_mapping_seed44` | `results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/iter29_mapping_seed44/` | `results/stage3_T5Train/curvature_RQ-VAE_iter30/iter29_mapping_seed44/` |
| 45 | iter26 control — `iter26_mapping_seed45` | `results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/iter26_mapping_seed45/` | `results/stage3_T5Train/curvature_RQ-VAE_iter30/iter26_mapping_seed45/` |
| 45 | iter29 candidate — `iter29_mapping_seed45` | `results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/iter29_mapping_seed45/` | `results/stage3_T5Train/curvature_RQ-VAE_iter30/iter29_mapping_seed45/` |

For each Stage2 profile, the locked route is `RQVAE_OUT_DIR=<Stage2 root>/out/rqvae/instruments/`, with checkpoint/raw SID under that directory, `SIDS_NPY=<Stage2 root>/dataset/Instruments/sids_for_hgrec.npy`, `ITEM_SIDS_JSON=<Stage2 root>/item_sids.json`, and `MECHANISM_NAME=iter30_<arm>_seed<seed>`. For the matching Stage3 wrapper, `CODE_PATH` must be exactly that run's `item_sids.json`, `RQVAE_VARIANT=iter30_<arm>_seed<seed>`, and `LOG_PATH`, `SAVE_PATH`, launcher/stdout logs must be under that label's Stage3 root. Keep all generated `.pth`, `.npy`, and JSON outputs outside the source iter30 subtree; give every run distinct profile/log/output paths, inventory destinations before launch, verify the resolved SID file and profile hashes, and run the six pipelines serially. Do not trust wrapper labels in place of checking actual path resolution/use.

**Protocol-only seed/runtime routes:** Stage2/Stage3 seeds are 43, 44, and 45, matched arm-to-arm within each pair. Stage2's trainer must hardcode the pair seed; it seeds Torch/NumPy with `SEED+rank`, uses `DistributedSampler(seed=SEED)`, and uses NumPy for transition-target selection. Stage3's wrapper must set the matching seed consumed by the unchanged `set_seed` (Python, NumPy, Torch, all CUDA) and sampler. Seeds vary across pairs, not within an arm pair; no bitwise-reproducibility claim. Run each no-argument, four-rank Stage2 launch serially on existing port 50200 and Stage3 on port 50201 to avoid collisions. The full mechanism label, arm label, output roots, wrapper import-module alias, and run-specific logs are path/report routing, not mechanisms.

## Inherited training, optimizer, loss, data, and runtime (held common)

- **Model/initialization:** 768 input; hidden `[512,256,128]`; embedding 32; three quantizer layers × 256; codebook k-means initialization enabled with existing rank-0 initialization/broadcast behavior; `midpoint_layer_mask=[False,False,False]`; same encoder, decoder, data transforms and checks. The same immutable iter8 warm-start path is registered (`results/stage2_RQ-VAE/curvature_RQ-VAE_iter8/out/rqvae/instruments/rqvae_best.pth`; S01 check-time SHA256 `189e0affb156b9cdbd11f38a6fdb1127804e3f3095edd23c55f84825beb56c3b`). S01 requires each run to verify digest/path and positive compatible transfer, fail closed on missing/malformed/incompatible state or unexplained missing/unexpected tensors, and skip only the specified legacy curvature entries. Historical iter26/29 code only warns and falls back to random initialization if the checkpoint is absent; that fallback is **not** allowed by the prospective lock. This future guard is shared protocol hardening, not an arm treatment. Do not claim bitwise identity from seed/warm-start alone.
- **Data/order:** same Stage1 `sentence_t5.npy` and `item_ids.json`; Stage2 loads them and Stage0 `train.parquet` histories/targets. `TransitionDataset` builds the same train-only source→next-item lists, samples a target with NumPy, and the same shuffled/drop-last `DistributedSampler`/DataLoader behavior applies. Stage2 SID corpus generation uses the same Stage1 embeddings. S01 locks check-time SHA256 for embedding `6490c71753952b2d8788bb7e4e52cff6a5b0b59819a2cfbd737bee7bc07a77fb`, item sidecar `3df1f3ce6a468ae541c929a149f4dbd5e806f545b89d786b9c2d6801563b7c30`, Stage0 train `80597e38d081029e57e7434c92ff7be2c872f82aebc2e5f072ef973d73e88815`, and warm-start above; rehash/recheck resolved consumers immediately before each actual invocation. Stage2 does not consume Stage0 valid/test/items on its training/export path; their declaration in config is not evidence of use.
- **Optimizer — no difference:** AdamW over the same trainable parameters, `lr=1e-3`, `weight_decay=1e-4`; same gradient clip norm `1.0`, optimizer checks and update loop. Fixed curvature is a buffer, not an optimizer parameter.
- **Loss — no component/weight/formula difference:** same hyperbolic reconstruction loss, quantization codebook/commitment loss with base commitment weight `1.0`, behavior contrastive loss with `BEHAVIOR_LOSS_WEIGHT=0.20` and `BEHAVIOR_TEMPERATURE=0.07`, and `CURVATURE_REG_WEIGHT=0` (curvature regularization stays zero). Existing `QuantizeLoss`'s curvature-dependent commitment modulation is unchanged (`commitment_weight * clamp(c/1.5, 1e-6, 1)^0.25`). Numerical loss values/gradients can differ because the registered `c` differs; this is a mediated consequence of the sole mapping treatment, not an altered loss rule or added loss.
- **Quantization/Sinkhorn — no rule/setting difference:** Poincare distance, distance centering/fallback, STE, argmax IDs, and balanced Sinkhorn are inherited. Configuration remains `sk_eps=0.05`, 3 iterations. Existing `effective_eps=sk_eps*(c/c_cyclic_max)` with `c_cyclic_max=1.5` is identical code/setting in both arms; its realized value changes with mapped `c`, as an intended mapping-mediated effect, not a new Sinkhorn factor. Do not call this a newly curvature-conditioned Sinkhorn mechanism.
- **Other curvature consumers — same inherited paths:** fixed `c` is used by Poincare distance/quantization and existing reconstruction, commitment, and behavior-loss geometry; M2 residual update uses the existing exponential-map/Mobius subtraction/log-map branch under the common all-false midpoint mask; M3 transports residuals through the origin between adjacent layer curvatures. These outputs may differ as a direct/mediated result of `c26` vs `c29`; their equations/flags are not changed. Both curvatures are computed before training, stored as fixed non-trainable buffers, and must be invariant across steps/optimizer updates/train-eval mode; no trainable/scheduled/cyclic curvature or regularization is introduced.
- **Step/runtime/export — same:** batch 640/GPU (global 2560), 4 DDP ranks, 100,000 global steps, checkpoint every 10,000; 4 workers, pin memory and persistent workers enabled, prefetch 4, compile false; CUDA devices `0,1,2,3`, existing cache/determinism settings, no-argument internal launcher/port 50200. SID metrics are descriptive-only and no SID quality gate/early stop is allowed. Preserve raw 3-token SID generation/report cadence and final 4-token Stage3 export/wiring checks; only isolated output paths vary. Root `CLAUDE.md` requires a separate one-checkpoint/one-batch gradient-path check for each profile before each eventual Stage2 run.

## Optimizer differences

`OPTIMIZER_DIFFERENCES=None` within every pair. AdamW hyperparameters, parameter set, clipping, and update schedule stay common. `fixed_curvature` is excluded from trainable optimizer parameters in both arms. Per-run seed/profile/path selectors are protocol routing only.

## Loss differences

`LOSS_DIFFERENCES=None` in formula, components, and weights. Keep reconstruction, quantization/commitment, behavior loss, curvature regularization (zero), and all existing curvature-conditioned terms unchanged. Only their actual inputs/values can respond to the mapped fixed curvature. No new optimizer-side or auxiliary-loss mechanism is allowed.

## Stage1 differences

`STAGE1_DIFFERENCES=None`. Do not retrain, recalculate, or select Stage1 separately by arm. Both use the locked embedding and item-ID order/sidecar, with rehash and actual-consumer checks before invocation. Stage0 training interactions/order are shared as locked.

## Stage3 differences

`STAGE3_DIFFERENCES=None` for trainer/model, protocol, optimizer, evaluation, and runtime. The actual shared trainer remains the locked byte-identical `stage3_T5Train/train_HG-Rec.py` at SHA256 `9861eb6b2a5b0bfac546524c50d402aba5827e5079a228e2ae5501909bc937cb`. Preserve matching wrapper-set seed and the run's own Stage2 SID path; only seed, descriptive arm/run label, SID input route, and unique result/log/checkpoint roots vary as protocol selectors.

Keep the current Stage3 settings: configured max 150 epochs; `NO_EVAL=True`, existing train-loss patience `EARLY_STOP=10` unchanged, `SKIP_TEST=False`; `TOPK_LIST=[5,10]`, beam 20; expected `n_eval=57439`; batch 4096, inference batch 1024, four ranks, serial port 50201; 4 encoder/decoder layers, `d_model=128`, `d_ff=1024`, 6 heads, `d_kv=64`, dropout 0.1, ReLU; AdamW `lr=0.003`, `weight_decay=0.05`, cosine schedule with 10% warmup and min factor 0.20; `FAST=True`, `BF16=True`, `COMPILE=False`; deterministic settings and root-required NCCL settings unchanged. Keep `SCREEN_BASELINE_LOG=""` and the currently inactive `SCREEN_SKIP_TEST_ON_FAIL=True`; final test remains required. The configured maximum and current train-loss patience are existing trainer behavior; never stop/drop a run based on the user's target, arm scores/trends, or Stage2 metrics. Audit every valid run's final test record and actual SID path/use.

The paired primary estimand is `d_s=R29_s-R26_s` for seeds `{43,44,45}`; report all three pairs and six pipelines. Use the user's inclusive per-result criterion `test_recall@10 >= 0.065`; target classification is distinct from the mapping effect and does not permit early stopping. Do not pool historical seed 42.

## Explicit single-factor determination

**Conditional PASS for the registered prospective design:** the pair is single-factor if—and only if—the six profiles use the same common Stage2/Stage3 implementation, exact shared explicit input key/values, warm-start, all non-map settings and reporting policy; each same-seed pair has only its registered mapping/fixed vector as the scientific treatment; and all S01 paths/labels/seeds are routing-only and verified. A changed realized distance, effective epsilon, assignment, residual path, or loss from the resulting `c` is a mapped-curvature effect and is not a second treatment. Historical source changes between iter26 and iter29 are not inherited as arm-specific behavior: in particular, apply the iter29/root-compliant SID reporting hygiene to both arms identically.

**Confound risks / blocking conditions:**

1. Missing or substituted raw residual, legacy `residual_norm` fallback, normalized vector substitution, wrong layer order, different branching vector, map constants/intermediates/vector mismatch, or using different input files/key behavior by arm invalidates the contrast. S03's medium-confidence method/value limitation remains explicit; no replay or recalibration is claimed.
2. Different common-code revision, optimizer/loss/Sinkhorn/M2/M3 behavior, fixed-curvature consumer, model initialization/warm-start transfer, data order, Stage1/Stage0 input, runtime settings, SID report cadence, or metrics gate between arms is a second factor. If any cannot be made common, recommend blocking before Stage2 rather than rationalizing it away.
3. Do not retain iter26's HR@50 subprocess or old-gate assumptions in one arm; do not let metrics trigger stopping. Common reporting hygiene is required under current root policy and must not change model training.
4. Wrong Stage2 seed, Stage3 seed, SID file, `RQVAE_VARIANT`, result path or hash; fallback to a stale SID file; output collision/overwrite; unreviewed per-run profile drift; concurrent use of fixed launcher ports; or unverified warm-start transfer breaks the locked protocol/pairing. The iter29 wrapper's `unknown_variant` metric metadata caveat remains reporting-only and is not permission to modify Stage3 trainer.
5. S04 candidate contract encodes iter29 values only. Control values/fixed behavior must be independently checked by later gates; this S05 candidate does not claim either arm passed preflight/MVG or the counterfactual.
6. Historical seed-42 score difference is one contextual pair, not an effect estimate, and does not justify dropping, pooling, or replacing a new pair.

## Evidence reviewed

- Canonical iter30 S00 `logs/source_snapshot_iter30.md` and S00 Judge; S01 `logs/protocol_manifest_iter30.md` and Judge; S02 `logs/hypothesis_iter30.md` and Judge; S03 `logs/mechanism_manifest_iter30.md` and Judge; S04 `logs/mechanism_contract_iter30.json` and Judge; the canonical S05 source packet.
- Primary iter26: `curvature_RQ-VAE.py` (data loader, map loader, model/optimizer/loop/report/export), `curvature_config.py`, `scripts/compute_closed_form_curvature.py`, `scripts/computed_behavior_branching.json`, `modules/{encoder,rqvae,quantize,hyperbolic,loss,normalize,step_checks,utils,hab,sid_quality}.py`, and `scripts/run_stage3_iter26.py`; iter26 `logs/one_factor_diff_iter26.md` as historical cross-check only.
- Primary iter29: corresponding trainer/config/map calculator/JSON/module/wrapper files plus `logs/one_factor_diff_iter29.md` as historical cross-check only. Direct hash comparison establishes that the nine listed shared model/support modules (excluding the intentionally different `sid_quality.py`) match pairwise; direct source review establishes the mapper and reporting differences above.
- Current Stage3 `stage3_T5Train/train_HG-Rec.py`, directly SHA256-checked as `9861eb6b2a5b0bfac546524c50d402aba5827e5079a228e2ae5501909bc937cb`; direct source confirms seed function, model/evaluation constants and 4-rank port 50201 launcher. S01 is the authority for locked input identities and actual consumers; recheck runtime use before execution.

## Self-rejection and next action

`SELF_REJECTION_CONDITION=Reject this candidate or block the prospective comparison before implementation if Judge C's primary-source check finds an unreported non-map training difference that cannot be held common, if the explicit shared raw input cannot be consumed as registered, or if any paired route necessarily changes a scientific setting beyond the two fixed maps. The historical seed-42 delta and S03 provenance caveat cannot cure such a failure.`

`AUTONOMOUS_NEXT_ACTION=Judge C adjudicates this candidate against Agent B and direct evidence; if accepted, advance only to S06 independent implementation design. S06 must implement the common two-map/per-run routing design once and preserve every constraint above. S05 grants no implementation/preflight/MVG/Stage2/Stage3/GPU authorization; later gates and explicit Stage2/Stage3 authorizations remain mandatory.`