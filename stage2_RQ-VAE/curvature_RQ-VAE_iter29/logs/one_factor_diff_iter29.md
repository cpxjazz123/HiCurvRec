# Iter29 One-Factor Diff — S05 Canonical (Round 2)

```text
PARENT_ITER=iter26
PARENT_COMMIT=612a5a41dfe524205b6afa46370ad0dce377882
CANONICAL_BASELINE_ITER=iter26
EXPERIMENT_TYPE=single_factor
ACTIVE_MECHANISMS_BEFORE=FCCR-1 fixed closed-form per-layer curvature using iter26's mapping over behavior_branching and raw_residual_medians; inherited behavior loss and curvature-consuming quantization/Sinkhorn path
NEW_MECHANISM=Replace only the per-layer closed-form curvature mapping with the canonical S02/S04 bounded rational/additive mapping of the same behavior_branching and explicit raw_residual_medians inputs
ACTIVE_MECHANISMS_AFTER=FCCR-1 fixed closed-form per-layer curvature using the canonical iter29 mapping and its fixed output vector; inherited behavior loss and curvature-consuming quantization/Sinkhorn path unchanged
```

## Parent iter26 mapping and output

Iter26 computes its fixed per-layer curvature from the same registered input vectors using:

```text
m_min = min_l(m_l_raw) = 0.09331
s_l = log1p(B_l) / log1p(m_l_raw / m_min)
z_l = (s_l - mean(s)) / (std(s) + 1e-12)
c_l = clip(c_base * exp(alpha * z_l), 0.05, 1.5)
c_base = 0.5
alpha = 0.2
```

In `[L0,L1,L2]` order, the parent transformation and output recorded in its primary mechanism manifest are:

```text
behavior_branching B = [19.324911558712664, 1.4605688962651735, 1.0148104414712726]
raw_residual_medians m_raw = [1.0, 0.10941, 0.09331]
s = [1.2238118890466054, 1.1604516044603779, 1.010644112691917]
z = [1.03129493309937, 0.3223997103378134, -1.3536946434371857]
fixed_curvature = [0.6145357379232853, 0.5333020920777128, 0.3814078098431606]
```

The parent registers each fixed curvature as a non-trainable buffer; `get_c()` returns the fixed value. Cyclic and learnable curvature paths are inactive under the fixed-curvature contract, and curvature regularization is zero.

## Canonical iter29 mapping, inputs, and output

Canonical S02/S04 replace that mapping only. For each layer `l`:

```text
x_l = B_l / (B_l + B_ref)
y_l = m_l_raw / (m_l_raw + m_ref)
u_l = (x_l + y_l) / 2
c_l = c_min + (c_max - c_min) * u_l

B_ref = 2.0
m_ref = 0.1
c_min = 0.05
c_max = 1.50
```

The approved historical-method/value inputs remain identical to the parent and are ordered `[L0,L1,L2]`:

```text
behavior_branching = [19.324911558712664, 1.4605688962651735, 1.0148104414712726]
raw_residual_medians = [1.0, 0.10941, 0.09331]
```

Substitution gives the canonical intermediates and output:

```text
x = [0.9062129756321139, 0.42206034326942476, 0.3366083742817442]
y = [0.9090909090909091, 0.5224678859653312, 0.48269618747090165]
u = [0.9076519423615115, 0.4722641146173780, 0.40965228087632294]
fixed_curvature = [1.3660953164241916, 0.7347829661951981, 0.6439958072706683]
```

**Sole scientific/conceptual delta:** replacing only iter26's per-layer mapping with the S02/S04 mapping, using the same `behavior_branching` values and the explicitly keyed `raw_residual_medians` values, then consuming the resulting fixed vector through the inherited quantization path. The implementation MUST read the plural data key `raw_residual_medians` and fail closed if absent. It MUST NOT substitute `residual_norm`, normalized layer scales, or any fallback/alias, even if numerically equal. S03 approves historical method/value provenance only; this is not checkpoint-level replay, independent historical recomputation, or proof of historical byte identity. Do not rerun the old behavior writer: it overwrites the JSON without this plural key and depends on the absent iter8 raw SID file. The mapping is the only scientific change; the approved policy/hygiene migration below is not a mechanism, treatment, outcome, gate, or protocol change.

## Inherited mechanisms and locked protocol

- FCCR-1 remains closed-form, precomputed before Stage2, fixed in non-trainable buffers, and invariant to step, optimizer updates, and train/eval mode. No learnable curvature, cyclic/scheduled curvature, curvature regularization, or new curvature-conditioned optimizer/loss is introduced. Curvature regularization remains zero.
- Existing behavior loss remains weight `0.20`, temperature `0.07`.
- Sinkhorn remains configured with `sk_eps=0.05`, `iters=3`, and the existing parent rule is unchanged: `effective_eps = sk_eps * (c / c_cyclic_max)`. A changed realized epsilon follows deterministically from the new `c` through this inherited rule; it is a mediated effect of the single mapping change, not a second mechanism or rule/parameter change.
- Architecture remains the iter26 RQ-VAE (`INPUT_DIM=768`, hidden dimensions `[512,256,128]`, `EMBED_DIM=32`), three quantizer layers and codebook size 256 per layer. Seed remains 42; Stage2 remains 100,000 global steps; warm-start remains the iter8 checkpoint with curvature state excluded; optimizer remains AdamW `lr=1e-3`, `weight_decay=1e-4`.
- Stage1/Stage0 inputs, paths, identities and hashes remain those locked in S01. Stage3 trainer/source, settings, seed 42, 150 epochs, beam size 20, `n_eval=57439`, and evaluation protocol remain unchanged. Only the iteration-local wrapper mechanism label and iter29 SID/results paths differ.
- Stage2 SID-derived metrics are descriptive-only: no outer or inner hard gate; all candidates run to `MAX_GLOBAL_STEPS`, and `should_early_stop()` returns exactly `(False, "")`.
- No iter28/rejected-draft mechanism is inherited or stacked.

## POLICY_HYGIENE_DIFFERENCES — approved reporting migration

Direct iter29 source evidence establishes an active call chain: `curvature_RQ-VAE.py` invokes `evaluate_sid_quality_full(sids, EMB_NPY)` at the existing checkpoint SID-reporting boundary; `modules/sid_quality.py` writes a temporary SID file, launches the project-local `scripts/sid_metrics_any.py` with positional arguments, and parses HitRate@K=50. This project-local CLI path violates root `CLAUDE.md` §1. Root §2 removes HR@50 because it does not read SID and is nondiscriminating, requires all Stage2 metrics to be descriptive-only/no hard gates, and requires `should_early_stop` to return `(False, "")` unconditionally. The existing in-process `evaluate_sid_quality(sids)` already computes the supported SID-derived descriptions.

The bounded S06 migration is:

- In `curvature_RQ-VAE.py`, replace the import/call of `evaluate_sid_quality_full(sids, EMB_NPY)` with the existing `evaluate_sid_quality(sids)`. Preserve checkpoint reporting cadence, SID generation and raw-SID writes, printed SID-derived metrics, warning-only metric exception handling, training behavior, and final 4-token Stage3 export.
- In `modules/sid_quality.py`, remove the subprocess/temp-file HitRate path (`evaluate_sid_quality_full`, `_parse_sid_metrics_stdout`, and subprocess-only imports). Remove `hitrate_k50` from `SidMetrics`, its NaN placeholder, validation, formatting, and report serialization. Retain and report the existing SID-derived descriptive metrics: full and per-layer Gini, unique SID count, item count, L0/L1 unique pairs, and `H(L1|L0)`. Remove stale HitRate/gate claims and update nearby comments/docstrings to state that metrics are descriptive only. Make `should_early_stop()` exactly return `(False, "")` unconditionally, with no type/metric validation or gate branch. Keep unrelated APIs such as `write_quality_report` only if still appropriate; do not broaden cleanup.
- Delete `scripts/sid_metrics_any.py` from this iter29 subtree only after migrating its active caller. Do not create or move a replacement CLI or touch another iteration tree.

This migration removes a forbidden, non-SID-dependent reporting path while preserving the SID-derived descriptions and all training/SID behavior. It is strictly policy/hygiene; it does not change the experimental treatment, formula inputs, outcomes, protocol, or scientific interpretation. Keep the existing warning-only behavior for metric-collection exceptions and the existing final SID export failure behavior unchanged.

## Expected S06 implementation paths and hygiene

The following is the complete expected iter29 S06 implementation scope, not changes made in S05:

1. `scripts/compute_closed_form_curvature.py` — implement the exact registered mapping, explicit plural raw-key consumption, and fail-closed validation.
2. `scripts/computed_behavior_branching.json` — preserve branching, explicit raw medians, and their historical source record; remove ambiguous legacy aliases and obsolete old-map fields; record the new equation's intermediates and fixed vector.
3. `curvature_RQ-VAE.py` — load/validate the canonical fixed vector; remove the unused legacy layer-scale fallback helper and obsolete old mapping constants/docs/labels; perform only the approved reporting caller migration described above. Do not change model/training behavior, data loading, loss, optimizer, Sinkhorn, seed, steps, warm-start, SID generation/cadence, or final export.
4. `curvature_config.py` — change only the full iter29 mechanism label and iter29 result paths; retain upstream input paths and all seed/GPU/model/runtime settings.
5. `scripts/run_stage3_iter29.py` — iteration-local wrapper with iter29 SID/output paths and full mechanism variant; no Stage3 trainer or setting change.
6. `scripts/mvg_check.py` — use the accepted vector; check fixed/invariant curvature and one real `loss.backward()` path; add same-checkpoint/same-batch counterfactual against the iter26 vector `[0.6145357379232853,0.5333020920777128,0.3814078098431606]`, requiring measurable change in a curvature-consuming assignment/geometry/loss signal.
7. `modules/sid_quality.py` — implement only the bounded reporting migration above: remove subprocess/HitRate/parser/field/reporting, retain existing SID-derived descriptive metrics, correct stale docs/comments, and make `should_early_stop()` unconditionally return `(False, "")`.
8. `scripts/sid_metrics_any.py` — delete only after the iter29 caller/import is migrated.

Conditional deletions only if still present and unused: stale copied iter-specific Stage3 wrappers, unauthorized `calibrate_residual_scales.py` (not an authorized S06 input generator), and `grad_check.sh`. Retain compatible no-argument `grad_check.py` and `export_sids_for_stage3.py`. Deletions must be limited to this enumerated stale/unauthorized scope.

Stored deterministic intermediates, strict key/contract validation, iteration labels, isolated output paths, and lightweight verification are path/output/verification hygiene, not additional scientific mechanisms. The `effective_eps` value may change only as a mediated consequence of the fixed-curvature mapping; its formula and configured parameters stay unchanged.

```text
OPTIMIZER_DIFFERENCES=None
LOSS_DIFFERENCES=None
STAGE1_DIFFERENCES=None
STAGE3_DIFFERENCES=None for trainer/source/settings/protocol; only wrapper label and iteration-local SID/results paths differ
POLICY_HYGIENE_DIFFERENCES=Replace project-local HitRate CLI reporting with in-process SID-only descriptive metrics; remove HitRate/parser/report field and delete script after caller migration; unconditional no-gate should_early_stop
```

## One-factor and falsification risks

The experiment is scientifically confounded if any parent condition other than the mapping/input-to-curvature mapping changes, if another mechanism is introduced, if a different input/key/value/order or fallback is used, or if Stage1/Stage3/protocol settings drift. The explicit raw key's absence is a hard failure, not permission to reconstruct or substitute it. The S03 provenance/replay limitations must remain explicit. A stale producer can overwrite the required key; stale path labels/wrappers can route to another iteration's SID/results. Old mapping helpers/constants must not remain live as an alternative path.

The registered mapping/contract is falsified by incorrect recomputation, non-finite/out-of-range values, trainable or time-varying curvature, or failure of the matched counterfactual to show a measurable effect on a curvature-consuming signal. A direct-effect failure cannot be repaired by retuning this mapping in iter29; follow the feasibility/abort contract. A valid Stage3 result at or below iter26 falsifies the registered retrieval-improvement hypothesis for this mapping; a result below the user's inclusive `test_recall@10 >= 0.065` target fails promotion separately. A single stochastic run does not establish run-to-run noise, broad FCCR-1 failure, or checkpoint-level provenance/replay. The policy/hygiene migration is not evidence about this hypothesis and must not be counted as another scientific factor.
