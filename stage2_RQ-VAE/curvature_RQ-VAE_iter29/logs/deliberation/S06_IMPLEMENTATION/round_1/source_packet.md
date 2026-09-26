# S06 Implementation Source Packet — iter29

STAGE_ID=S06_IMPLEMENTATION
ROUND=1

## Canonical authority

Read the canonical S00–S05 artifacts and Judge decisions, especially `logs/hypothesis_iter29.md`, `logs/mechanism_manifest_iter29.md`, `logs/mechanism_contract_iter29.json`, and `logs/one_factor_diff_iter29.md`. The sole conceptual change is replacing iter26's per-layer fixed-curvature mapping with:

```text
x_l = B_l / (B_l + 2.0)
y_l = m_l_raw / (m_l_raw + 0.1)
u_l = (x_l + y_l) / 2
c_l = 0.05 + 1.45 * u_l
```

using the existing, ordered values:

```text
behavior_branching = [19.324911558712664, 1.4605688962651735, 1.0148104414712726]
raw_residual_medians = [1.0, 0.10941, 0.09331]
fixed_curvature = [1.3660953164241916, 0.7347829661951981, 0.6439958072706683]
```

FCCR-1 stays fixed/precomputed/non-trainable/time-invariant. No cyclic schedule, curvature regularization, new optimizer/loss, or second mechanism. Iter26 is sole direct control. Its inherited `effective_eps = sk_eps * (c / c_cyclic_max)` Sinkhorn rule stays unchanged; only realized epsilon may differ through the changed `c`. S03 approves historical method/value provenance only, not current checkpoint replay or historic byte identity. S06 must consume the JSON key `raw_residual_medians` explicitly and fail closed if absent; never use `residual_norm`, normalized scales, aliases, or fallback. Do not run the old behavior writer, which erases this key and requires absent iter8 raw SIDs.

## Source-state findings

The current iter29 scaffold is copied from iter26 and has not yet been patched:

- `scripts/compute_closed_form_curvature.py` is absent from iter29 (present in iter26). The current `scripts/computed_behavior_branching.json` stores branching, raw medians, a legacy equal-valued `residual_norm` alias, old `raw_capacity`/`branch_mid`, and iter26 `s_l`/`z_l`/`closed_form_c_l` metadata.
- `curvature_RQ-VAE.py::_load_closed_form_curvatures()` trusts only the stored `closed_form_c_l`, checks range, and has an obsolete iter16 formula docstring. `CLOSED_FORM_C_BASE` and `CLOSED_FORM_ALPHA` are obsolete. `_load_layer_norms()` is unreferenced; main currently passes `[1.0] * N_LAYERS` directly, so deleting this stale fallback helper does not change training behavior. Current-iteration curvature log labels/docs still say iter16/iter26.
- `curvature_config.py` currently labels iter26 and routes Stage2 outputs into `results/stage2_RQ-VAE/curvature_RQ-VAE_iter26/`; these must move to short iter29 result paths while `MECHANISM_NAME` remains a full `iter29_<description>` label.
- There is no current iter29 Stage3 wrapper. The iter26 wrapper is the template; Stage3 trainer/settings/source remain unchanged. Stage3 `LOG_PATH`/`SAVE_PATH` must use the short results directory `curvature_RQ-VAE_iter29`, and `RQVAE_VARIANT` must use the full descriptive mechanism label.
- `scripts/mvg_check.py` expects iter26's vector, checks invariance and autograd gradients, but does not explicitly exercise a standalone `loss.backward()` contract path or matched iter26-vs-iter29 counterfactual. Its model setup already loads the exact iter8 checkpoint and one 640-example batch with the Stage2 parent settings.
- The iter29 `scripts/` folder contains obsolete copied run wrappers for iter4/7/8/11/18/27, plus `sid_metrics_any.py`, `calibrate_residual_scales.py`, and `grad_check.sh`. S05 authorizes removing only these if unused; retain compatible no-argument `grad_check.py` and `export_sids_for_stage3.py`.
- `curvature_RQ-VAE.py` writes raw/final SID artifacts using `curvature_config` paths; its built-in export is authoritative. The retained export helper must remain no-argument and use the same config.

## Required minimal implementation

A/B independently produce a complete patch plan and behavior-focused verification plan. Judge C writes canonical `logs/implementation_plan_iter29.md`. No worker edits source. The Judge-approved plan should cover:

1. Add `scripts/compute_closed_form_curvature.py` as a no-CLI, CPU-capable mapping module/runner. Use exact constants and layer order; validate three-element finite vectors with `B >= 0` and `m_raw > 0`; explicitly require/read `raw_residual_medians`; fail if missing even when `residual_norm` exists; preserve the input record; store `x_l`, `y_l`, `u_l`, constants/mapping identity, and exact `closed_form_c_l`; remove obsolete alias/old-map fields, not the approved inputs.
2. Keep input provenance semantically separated in the JSON: retain the behavior producer metadata as behavior-branching provenance and explicitly state it used raw iter8 SIDs plus Stage0 train parquet (the checkpoint path in that metadata was not loaded by the producer); separately record the iter10 residual calibration script/log, source label `iter1`, step 100000, historical checkpoint and Stage1 embedding paths, and the S03 replay/hash limitations. Do not imply one source produced both inputs or claim replay.
3. Update `curvature_RQ-VAE.py` to compute/revalidate the registered equation from the explicit input fields at load time, compare the result with both stored `closed_form_c_l` and S04 `final_curvature_values`, enforce finite supported range/shape, and fail closed on missing/invalid data. Reuse a single formula implementation rather than introduce an alternate mapping. Remove only dead old mapping constants/helper/docs/labels; preserve model, loss, optimizer, Sinkhorn rule/parameters, warm-start, data paths, seed, steps, export logic, and checks.
4. Update `curvature_config.py` with full label `iter29_bounded_rational_additive_mapping`, `_CONFIG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter29`, `RQVAE_OUT_DIR` beneath that short path, and iter29 `SIDS_NPY`/`ITEM_SIDS_JSON` paths. All Stage1/Stage0 inputs and runtime/seed/GPU settings remain unchanged.
5. Add `scripts/run_stage3_iter29.py` from the parent wrapper pattern: use iter29 `item_sids.json`, full `RQVAE_VARIANT`, and short `results/stage3_T5Train/curvature_RQ-VAE_iter29/{logs,ckpt}` paths. Do not modify Stage3 trainer/settings.
6. Update `scripts/mvg_check.py` expected candidate vector and add FCCR-1 Layer A–E checks: exact finite fixed vector; non-grad buffer absent from optimizer parameters; time/train-eval invariance; finite nonzero model gradients including explicit `loss.backward()` on one warm-start checkpoint/one batch; fixed vector unchanged after optimizer updates; matched candidate-vs-iter26 counterfactual using the same checkpoint, same batch, and identical non-curvature weights, requiring measurable difference in assignment, geometry, or curvature-consuming loss/output. Preserve the direct control vector `[0.6145357379232853,0.5333020920777128,0.3814078098431606]`; do not train a full model.
7. Retarget only iteration-specific source labels/result paths and delete only S05-authorized stale copies. No `argparse`, `sys.argv` path arguments, environment override, new training behavior, or Stage1/Stage3 code changes.

## Verification scope

No project-wide test suite/build/formatter. Recommend only observable, bounded checks before Stage2: run the no-argument mapping script once with the approved CPU interpreter and verify exact output/intermediates and no legacy fallback key; exercise the missing-plural-key failure in a throwaway check without altering the canonical input; run the official no-argument `preflight_contract.py` and deliberation gate; run `mvg_check.py` on one checkpoint and one real batch, including `loss.backward()` and counterfactual evidence. Do not add permanent tests unless a genuinely uncertain edge justifies one. Root CLAUDE additionally requires `grep RQVAE_OUT_DIR` to verify the iter29 result subtree before launch, and exact locked input hashes must be rechecked.

No Stage2/Stage3 launch is authorized at S06. Full GPU execution occurs once only after S07/S08/S09 gates and Judge approvals.