# Iter30 One-Factor Diff — S05 canonical

```text
STAGE_ID=S05_ONE_FACTOR
ROUND=1
STATUS=CANONICAL_CONDITIONAL_PASS_PROSPECTIVE_DESIGN_ONLY
PROTOCOL_ID=FCCR-1_iter30_iter26_vs_iter29_matched_seed43-45
PARENT_ITER=iter29
PARENT_COMMIT=ecd01e4712a1badd38a0338255f4b2ec7b030aff
PARENT_ROLE=implementation/source-lineage parent; not the prospective paired control
PARENT_DELIVERABLE_COMMIT=57b4a594d92435fd74fa4bba7c7c1439a33eb102
PARENT_S14_SYNC_COMMIT=fa7801c4d0f9a7e1d96da968cc7e0be891b84bd6
CANONICAL_BASELINE_ITER=iter26
CANONICAL_BASELINE_ROLE=historical seed-42 baseline result and fresh iter26-mapping prospective control arm
EXPERIMENT_TYPE=single_factor; prospective matched-seed comparison of two pre-existing FCCR-1 mappings
ACTIVE_MECHANISMS_BEFORE=FCCR-1 fixed closed-form curvature plus the unchanged inherited RQ-VAE reconstruction, behavior, quantization, Poincare geometry, Sinkhorn, M2 residual, and M3 transport mechanisms
NEW_MECHANISM=No new conceptual mechanism; the sole scientific treatment is selecting the registered iter26 or iter29 fixed-curvature mapping from the same explicit inputs
ACTIVE_MECHANISMS_AFTER=The same inherited mechanisms with exactly one of the two registered fixed-curvature mappings; no stack, third mapping, or new mechanism
EXECUTION_AUTHORIZATION=NO
```

## Adjudication and causal scope

**Conditional PASS for the prospective design, not a finding that future profiles exist or execute correctly.** The full six-run comparison can be single-factor only if S06 implements and S07/S08 verify one common future Stage2 implementation, data/input path, warm-start policy, metric-reporting policy, and Stage3 trainer/wrapper behavior for both arms. Within each same-seed pair, only the pre-registered map selection and its resulting fixed curvature vector may differ scientifically. The explicit input key and values, seed, warm start, model, optimizer, loss, Sinkhorn configuration/rules, Stage1/Stage0 inputs, runtime, training budget, SID/export behavior, Stage3 trainer/evaluation, and all other settings must be common. Arm labels and unique paths may vary only as routing/identity metadata.

This is not a comparison of the entire historical iter26 tree against the entire historical iter29 tree. Historical trainer/config/mapping-consumer/reporting changes are audited below and are not allowed to become arm-specific behavior. This adjudication authorizes no source edits, preflight, MVG, Stage2/Stage3 training, or GPU work.

## Roles and historical results

- **Source-lineage parent:** iter29 at current S01 source revision `ecd01e4712a1badd38a0338255f4b2ec7b030aff`. The iter29 delivery commit `57b4a594d92435fd74fa4bba7c7c1439a33eb102` and S14 synchronization commit `fa7801c4d0f9a7e1d96da968cc7e0be891b84bd6` are separately recorded historical commits; neither makes iter29 the direct prospective control.
- **Fresh prospective paired control:** the iter26 mapping freshly executed at each of seeds 43, 44, and 45 and paired to the iter29 mapping at the identical seed.
- **Canonical historical baseline:** iter26 seed 42, `results/stage3_T5Train/curvature_RQ-VAE_iter26/logs/Amazon_2023_Instruments/Sep-26-2026_18-11-20/test_final.json`, `n_eval=57439`, R@10 `0.057017009349048554` (R@5 `0.03788366789115409`, NDCG@5 `0.025169911931406087`, NDCG@10 `0.03133359761524377`). This is context only, not a substitute for a new paired run.
- **Historical iter29 seed-42 context:** `results/stage3_T5Train/curvature_RQ-VAE_iter29/logs/Amazon_2023_Instruments/Sep-27-2026_06-11-34/test_final.json`, `n_eval=57439`, R@10 `0.05921064085377531` (R@5 `0.03953759640662268`, NDCG@5 `0.026252776900288842`, NDCG@10 `0.03257647953179143`). Historical iter29-minus-iter26 R@10 is `+0.002193631504726755`, a one-pair historical difference only; it is not the prospective estimate or a run-noise estimate.
- S01 excludes historical seed 42 from the primary estimator. Iter18 remains `HISTORICAL_NONCOMPARABLE`; no historical pair may replace a missing prospective pair or be pooled as a fourth pair.

## Exact registered mapping contrast

Both maps must receive precisely the same `[L0,L1,L2]` values via the explicit `branching` and **`raw_residual_medians`** keys:

```text
B = [19.324911558712664, 1.4605688962651735, 1.0148104414712726]
raw_residual_medians = [1.0, 0.10941, 0.09331]
normalized_layer_scales = [0.001, 0.932889, 1.0]  # distinct; prohibited substitute
```

The future common implementation MUST require `raw_residual_medians` and fail closed if missing or mismatched. It MUST NOT read/fallback to `residual_norm`, use normalized scales, another checkpoint, or infer/recompute a replacement. Iter26's historical calculator read `payload["residual_norm"]`; the equal numeric legacy alias in that historical JSON does not establish its semantics. S03 approves only historical method/value provenance: branching confidence HIGH and raw-residual confidence MEDIUM; historical calibration checkpoint/raw-SID replay, current recalculation, historical input-byte identity, and checkpoint-level reproduction are not established. This S05 result preserves those limitations; it does not upgrade S03 evidence.

### Iter26-map control

With `m_min=min(m_raw)=0.09331` and population standard deviation (`ddof=0`):

```text
s_l = log1p(B_l) / log1p(m_l_raw / m_min)
z_l = (s_l - mean(s)) / (std_population(s) + 1e-12)
c26_l = clip(0.5 * exp(0.2 * z_l), 0.05, 1.5)
```

Constants: base `0.5`, exponent coefficient `0.2`, standardization epsilon `1e-12`, clipping bounds `[0.05,1.5]`.

### Iter29-map candidate

```text
x_l = B_l / (B_l + 2.0)
y_l = m_l_raw / (m_l_raw + 0.1)
u_l = (x_l + y_l) / 2
c29_l = 0.05 + 1.45 * u_l
```

Constants: `B_ref=2.0`, `m_ref=0.1`, `c_min=0.05`, `c_max=1.50`; no cross-layer normalization or additional clipping.

| Layer | B | explicit raw residual median | iter26 s | iter26 z | c26 | iter29 x | iter29 y | iter29 u | c29 | c29-c26 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L0 | 19.324911558712664 | 1.0 | 1.2238118890466054 | 1.03129493309937 | 0.6145357379232853 | 0.9062129756321139 | 0.9090909090909091 | 0.9076519423615115 | 1.3660953164241916 | 0.7515595785009063 |
| L1 | 1.4605688962651735 | 0.10941 | 1.1604516044603779 | 0.3223997103378134 | 0.5333020920777128 | 0.42206034326942476 | 0.5224678859653312 | 0.4722641146173780 | 0.7347829661951981 | 0.2014808741174853 |
| L2 | 1.0148104414712726 | 0.09331 | 1.010644112691917 | -1.3536946434371857 | 0.3814078098431606 | 0.3366083742817442 | 0.48269618747090165 | 0.40965228087632294 | 0.6439958072706683 | 0.2625879974275077 |

```text
c26 = [0.6145357379232853, 0.5333020920777128, 0.3814078098431606]
c29 = [1.3660953164241916, 0.7347829661951981, 0.6439958072706683]
delta_c = c29 - c26 = [0.7515595785009063, 0.2014808741174853, 0.2625879974275077]
```

These are exact registered decimal substitutions. Both vectors are finite, within `[0.05,1.50]`, computed before training, and required to be fixed/non-trainable/time-invariant. S05 does not claim any future implementation has computed or verified them. Distances, assignments, realized curvature-scaled epsilon, residuals, losses, or SIDs may change downstream because `c` changes; these are mapped-curvature consequences through inherited consumers, not additional arm-specific treatments.

## Direct historical source/config comparison and future common cutover

### Mapping input and validation

Iter26 `scripts/compute_closed_form_curvature.py::closed_form_curvatures` computes the standardized/log-ratio map, but its writer passes `payload["residual_norm"]`; the iter26 trainer `_load_closed_form_curvatures()` merely loads the saved `closed_form_c_l`, checks length/range, and does not recompute/validate the selected map and exact inputs there. Iter29 `scripts/compute_closed_form_curvature.py::compute_closed_form_curvature` implements the bounded rational/additive map, requires explicit `raw_residual_medians`, validates finite/positive ordered inputs and mapped terms, while the iter29 trainer checks registered inputs, mapping metadata, recomputed intermediates/vector, and the FCCR-1 contract. These are historical implementation differences, not permission for different future arm validation. S06 must provide one common map-input/validation implementation that chooses only the exact registered map, validates its matching vector, and uses the explicit shared key for both arms. S04's current JSON contract covers the iter29 candidate vector only; later applicable gates must independently verify the iter26 control vector/behavior as required by S01.

### Trainer and configuration

Historical iter26 and iter29 trainers share the same broad training/data flow and settings evidenced in their source; each has historical map-specific imports/log labels/validation and mechanism/result paths. Both source files still hardcode `SEED=42`; prospective profiles must instead bake in the S01 pair seeds 43/44/45. Both trainers' warm-start blocks only test whether the iter8 checkpoint exists, load non-strictly, and warn/continue from random initialization if missing; neither verifies the S01 digest and complete compatible transfer. This is a blocking future implementation defect, not an acceptable shared behavior. S06 must implement one identical fail-closed warm-start policy for all six runs, require the locked digest/path and positive compatible transfer, reject malformed/incompatible state and unexplained missing/unexpected tensors, and skip only the specified legacy curvature entries as applicable.

Historical configs differ in embedded mechanism names and Stage2 output/SID paths. Shared upstream Stage0/Stage1 paths and runtime values are present, but those old configs are not run-profile identity for iter30. Every future arm needs the same locked inputs/settings; only per-pair seed, mapping/vector, and unique route metadata differ. Rehash check-time identities and prove resolved consumer paths/use immediately before each applicable invocation. S01 distinguishes actual consumers: Stage2 reads Stage1 embedding, Stage1 item-ID sidecar, and Stage0 `train.parquet`; its configured path does not consume Stage0 valid/test/items or declared `ITEM_JSON_PATH`. Stage3 consumes Stage0 train and test with `NO_EVAL=True`; the configured path does not consume valid, Stage0 items, or Stage1 files. These declarations and check-time hashes do not by themselves prove later runtime consumption.

### Shared module identity evidence

Direct SHA256 comparison shows exact equal byte hashes between iter26 and iter29 for these nine inspected modules (not a claim that the full trainer trees are byte-identical):

- `encoder.py` `139a2cfa9fad4ecf9b795d435a640a927dd01a2d52911cea7c58a8de53c0a475`
- `rqvae.py` `123e7ad2ba146842988e91820cc7a138577cecbe11d479423e15368831ee0944`
- `quantize.py` `2970ad3e61de79c6a2fd17194258d7d5ed4b9cbbf836e5dde506a8635c2553f8`
- `hyperbolic.py` `ca7d4fdb5b7f6d4f91cc6e5083462fa9d610386ce4bda65605d3672907e22ae6`
- `loss.py` `c5d0106b731ee54b9e96ef40db745f7ad74725e2f76ed9c6b763e77e11108cd1`
- `normalize.py` `e063b3cc49e1d9571c6388251212054e269de43e2794e28238021fe9ccca67f6`
- `step_checks.py` `03afd7f176000043c9e577e214659890571847e4de389add70eb75a3ee16caf40`
- `utils.py` `a0b4b86904592a79373edcd198c608bb5bf140e7f5a475726a576ff1e36e59d7`
- `hab.py` `073825f162aad4d42bd740ce62d2a6d423e0ae44460cfd54ff15bba3eac59792`

`modules/sid_quality.py` differs: iter26 `5dce2f7c2b2d47aaaa4866d54cfb5d8f5a22fc602478cb43abded8df37acbd80`; iter29 `3387d7ad2211a923b5376b5606b66411d8fbc89469baa1274f906244a5923511`. The hashes describe only those two present historical module files; they do not establish byte identity for trainers, configs, map inputs, wrappers, future sources, nor historical invocation/runtime identity.

### Stage2 reporting policy and quality metrics

The current root `CLAUDE.md` §2 requires no Stage2 outer/inner hard gates, removal of HitRate@50 (the metric does not read SID and is invariant across variants), descriptive-only SID quality reporting, and `should_early_stop()` that never stops. Direct source differs from stale comments: iter26's training checkpoint caller invokes `evaluate_sid_quality_full(sids, EMB_NPY)`, whose active helper writes a temporary SID file and launches `scripts/sid_metrics_any.py` to compute/report HitRate@50; the module's comments and `should_early_stop` comment are stale, although its implementation validates descriptive fields then returns `(False, "")`. Iter29's caller invokes in-process SID-derived `evaluate_sid_quality(sids)`; its `should_early_stop` returns `(False, "")` directly and it has no HR@50 subprocess/parser path. Thus historic policy/reporting is not literally the same. S06 must apply root-compliant reporting hygiene once and identically to both arms: no HR@50 subprocess/field, only SID-derived descriptive values; no gate or early stop. Keep checkpoint/SID generation/export cadence and training unaffected. No metrics-based admission/stopping is permitted. Required source changes and their cross-arm equality must be reviewed in S06/S07; S05 does not claim this is already implemented.

## Inherited mechanisms and settings that must remain common

- **Model/quantization and data:** input 768, hidden `[512,256,128]`, embedding 32, 3 quantizer layers × 256 entries; same encoder/decoder, reconstruction path, K-means codebook initialization/rank-0 broadcast, all-false midpoint mask, residual configuration, transforms, data order and sampler semantics. Same registered Stage1 embedding and item-ID sidecar; Stage2 train transition paths consume Stage0 `train.parquet`. Same immutable iter8 warm-start path `results/stage2_RQ-VAE/curvature_RQ-VAE_iter8/out/rqvae/instruments/rqvae_best.pth`, S01 check-time SHA256 `189e0affb156b9cdbd11f38a6fdb1127804e3f3095edd23c55f84825beb56c3b`; it must be verified anew and used identically in all six runs.
- **Optimizer/training:** AdamW on the same trainable parameters, `lr=1e-3`, `weight_decay=1e-4`; gradient clipping norm `1.0`; batch `640/GPU`, 4 DDP ranks/global batch `2560`; 100,000 global steps; checkpoint every 10,000; 4 data workers, pin memory, persistent workers, prefetch 4, compile false; same rank seeding and DistributedSampler semantics except required matched seeds 43/44/45 (within each pair identical; no bitwise reproducibility claim). Both historical trainers currently use seed 42, so per-run source profiles must explicitly set the locked future seed.
- **Loss:** no formula/component/weight difference is registered. Preserve reconstruction, quantization/commitment (base commitment weight `1.0`), behavior contrastive loss (`BEHAVIOR_LOSS_WEIGHT=0.20`, temperature `0.07`), and zero curvature regularization. The `0.005` curvature-reg number found in historical trainer comments is a stale comment, not the active iter29 mechanism or a setting to activate; modules `rqvae.py` and `loss.py` have equal hashes. Curvature-dependent numerical losses/gradients may respond to the mapped `c`; no loss formula or weight may vary by arm.
- **Sinkhorn and geometry:** configured `sk_eps=0.05`, 3 iterations and same centering/fallback, Poincare distance, STE, assignment/ID path. Existing `effective_eps=sk_eps*(c/c_cyclic_max)`, with `c_cyclic_max=1.5`, stays unchanged: its realized value can differ as a mediated consequence of `c`, not a newly introduced or arm-specific Sinkhorn rule. Preserve existing M2 Mobius/exponential/log-map residual update and M3 inter-layer curvature transport under the common all-false midpoint mask. These mechanisms consume fixed curvature; no equation/flag changes.
- **Fixed-curvature contract:** both mappings are closed-form, pre-training, fixed/non-trainable and time-invariant; no trainable curvature, cyclic/time-varying schedule, curvature regularization, new curvature-conditioned optimizer, or new curvature-conditioned auxiliary loss. Later preflight/MVG must test both vectors; fixed curvature itself does not receive gradients.
- **Stage2 reporting/export:** shared descriptive-only SID report policy, same raw 3-token SID generation cadence, final 4-token Stage3 export/wiring behavior, checkpoint cadence, and no quality gate/stop. Training behavior must not be changed by reporting hygiene.
- **Stage3:** trainer/model/protocol/evaluation are common and unchanged. S01 locks `stage3_T5Train/train_HG-Rec.py` SHA256 `9861eb6b2a5b0bfac546524c50d402aba5827e5079a228e2ae5501909bc937cb`. Use matching Stage3 seed per pair; 150 configured maximum epochs, `NO_EVAL=True`, existing train-loss patience `EARLY_STOP=10` retained, `SKIP_TEST=False`, top K `[5,10]`, beam 20, expected `n_eval=57439`, batch 4096, inference batch 1024, four ranks and serial port 50201. Keep the trainer's architecture, AdamW `lr=0.003`/weight decay `0.05`, cosine scheduler (10% warmup, min factor 0.20), precision/determinism/NCCL/runtime and screen settings unchanged. The six runs require their own final test records. Existing `NO_EVAL` train-loss patience is not permission to omit a run based on test target/trend or Stage2 metrics. Historical iter29 wrapper caveat: it assigned intended `RQVAE_VARIANT`, while training metrics recorded `unknown_variant`; disclose/preserve this caveat, use common wrapper behavior, and do not edit Stage3 trainer to repair metadata.

## Future source/profile/routing scope (not yet implemented)

No source/config/profile is changed by this S05 artifact, and no future iter30 run-profile hash or output is claimed to exist. S06 must define one common reviewed source path and, where hardcoded arm/seed selection requires it, six no-argument run profiles/wrappers. The future review scope is:

1. Shared Stage2 trainer and common mapping loader/calculator/input record(s): implement both exact registered equations/vectors under one code path; consume the identical explicit keys/values; validate; keep one common warm-start, model, loss, optimizer, runtime, SID/export behavior. If arm selection is split into source profiles, only the exact map/vector selector may represent the scientific difference; training/science logic cannot fork by arm.
2. Shared Stage2 quality-report module/callsite: remove the active HR@50 subprocess/report path for both arms in accordance with root §2, retain only descriptive SID-derived metrics, and do not add an admission/early-stop policy.
3. Stage2 config and six hardcoded no-argument profiles/runs (`iter26_mapping_seed43`, `iter29_mapping_seed43`, `iter26_mapping_seed44`, `iter29_mapping_seed44`, `iter26_mapping_seed45`, `iter29_mapping_seed45`) or an equivalently explicit no-CLI serial source arrangement. Profiles must set matching pair seeds, map selection/vector, full `MECHANISM_NAME`, immutable inputs, unique output and log destinations. No CLI flags, environment selector, or shell injection.
4. Six corresponding Stage3 no-argument wrappers/profiles or equivalent hardcoded serial wrappers: use only the matching run's SID JSON, matching seed and descriptive variant, and unique Stage3 paths. Do not modify the locked Stage3 trainer. Profile/wrapper/code hash identities must be recorded before invocation; they are not available yet.
5. Output routing, not a scientific mechanism: each route must be checked for pre-existing contents/collision and have distinct Stage2, Stage3, launcher, and stdout destinations. Keep generated `.pth`, `.npy`, and `item_sids.json` outside the iter30 source subtree and use the required short iter30 result roots.

This defines required future scope and invariants, not a pre-authorized patch or a claim that any such source/profile currently exists or runs successfully.

## Six unique planned run routes

All are prospective paths; outputs do not yet exist. Run serially, pairwise, with the same seed in both arms of each pair. Within a Stage2 root, set `RQVAE_OUT_DIR=<Stage2 root>/out/rqvae/instruments/`, `RQVAE_CKPT_PATH=<RQVAE_OUT_DIR>/rqvae_best.pth`, `RAW_SIDS_NPY=<RQVAE_OUT_DIR>/sids_raw.npy`, `SIDS_NPY=<Stage2 root>/dataset/Instruments/sids_for_hgrec.npy`, `ITEM_SIDS_JSON=<Stage2 root>/item_sids.json`, and `MECHANISM_NAME=iter30_<arm>_seed<seed>`. The matching Stage3 wrapper uses that `item_sids.json` as `CODE_PATH`, label as `RQVAE_VARIANT`, and `<Stage3 root>/logs/`, `<Stage3 root>/ckpt/`, plus a unique launcher/stdout log. Verify the wrapper resolves exactly its own SID path and cannot fall back to stale data.

| Pair | Arm and run label | Stage2 seed | Stage3 seed | Planned Stage2 root | Planned Stage3 root |
|---:|---|---:|---:|---|---|
| 43 | iter26 control — `iter26_mapping_seed43` | 43 | 43 | `results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/iter26_mapping_seed43/` | `results/stage3_T5Train/curvature_RQ-VAE_iter30/iter26_mapping_seed43/` |
| 43 | iter29 candidate — `iter29_mapping_seed43` | 43 | 43 | `results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/iter29_mapping_seed43/` | `results/stage3_T5Train/curvature_RQ-VAE_iter30/iter29_mapping_seed43/` |
| 44 | iter26 control — `iter26_mapping_seed44` | 44 | 44 | `results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/iter26_mapping_seed44/` | `results/stage3_T5Train/curvature_RQ-VAE_iter30/iter26_mapping_seed44/` |
| 44 | iter29 candidate — `iter29_mapping_seed44` | 44 | 44 | `results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/iter29_mapping_seed44/` | `results/stage3_T5Train/curvature_RQ-VAE_iter30/iter29_mapping_seed44/` |
| 45 | iter26 control — `iter26_mapping_seed45` | 45 | 45 | `results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/iter26_mapping_seed45/` | `results/stage3_T5Train/curvature_RQ-VAE_iter30/iter26_mapping_seed45/` |
| 45 | iter29 candidate — `iter29_mapping_seed45` | 45 | 45 | `results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/iter29_mapping_seed45/` | `results/stage3_T5Train/curvature_RQ-VAE_iter30/iter29_mapping_seed45/` |

All six routes are distinct. For each profile verify no overwrite, the resolved route, exact source/config/profile hash, locked Stage1/Stage0/warm-start digests and actual consumption immediately before use. Stage3 must be serial on port 50201; Stage2 serial on 50200. S01's source/config/wrapper identities are historical references, not future profile hashes.

## Optimizer, loss, Stage1, and Stage3 differences

```text
OPTIMIZER_DIFFERENCES=None within every same-seed pair. Use common Stage2 AdamW lr=1e-3, weight_decay=1e-4, same trainable parameter set/clipping/update budget; fixed c excluded from optimizer. Stage3 optimizer also unchanged.
LOSS_DIFFERENCES=None in components, equations, or weights. Preserve the existing reconstruction, quantization/commitment, behavior loss, zero curvature-reg weight, and curvature consumers. Only realized values can respond to c.
STAGE1_DIFFERENCES=None. No Stage1 retraining/recalculation; use the locked embedding and item-ID sidecar as shared Stage2 inputs.
STAGE3_DIFFERENCES=None in trainer, model, protocol, data split, evaluation, and metrics. Keep the same locked trainer bytes and all six final tests; only paired seed, arm/run label, matching SID input route, and unique result/log paths are allowed routing selectors.
```

## One-factor blockers, risks, and limits

The conditional PASS is invalidated and Stage2 remains blocked if any of these arise: missing/substituted `raw_residual_medians`, use of `residual_norm` or normalized scales, wrong layer order/input/vector/map constants, future arm-specific mapping/validation/reporting policy, unverified/mismatched warm-start or Stage1/Stage0 inputs, unexplained training/source/optimizer/loss/Sinkhorn/M2/M3/runtime divergence, seed mismatch within pair, changed Stage1/Stage3, SID-path fallback/mismatch, path collision/overwrite, omitted/substituted seed/arm, or use of historical seed 42 as a replacement. Any policy/report hygiene must be identical and must not alter model training. The iter26 Stage3 historical wrapper metadata caveat is disclosure-only, not permission to edit trainer or add arm-specific behavior.

The required later S06/S07/S08 adjudications must establish the common implementation and verify both map vectors and invariants. S03's medium-confidence provenance limitation persists. This document establishes prospective design integrity only: it is not proof that future profiles, artifacts, results, hashes, or execution exist or are correct. No implementation, preflight, MVG, Stage2, Stage3, or GPU execution is authorized.

`AUTONOMOUS_NEXT_ACTION=Advance to S06_IMPLEMENTATION for independent implementation-design deliberation only. Require one common future implementation/reporting policy with explicit raw_residual_medians for both arms, fail-closed common warm-start, and the six exact unique routes; if that common design cannot be concretely specified and later verified without another treatment difference, block before any launch. S06 must not launch Stage2/Stage3 or perform implementation as part of this S05 authorization.`
