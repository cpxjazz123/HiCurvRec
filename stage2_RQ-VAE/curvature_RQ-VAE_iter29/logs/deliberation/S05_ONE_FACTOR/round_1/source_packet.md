# S05 One-Factor Source Packet — iter29

STAGE_ID=S05_ONE_FACTOR
ROUND=1

## Canonical lock

Use only canonical S00–S04 artifacts: `logs/source_snapshot_iter29.md`, `logs/protocol_manifest_iter29.md`, `logs/hypothesis_iter29.md`, `logs/mechanism_manifest_iter29.md`, `logs/mechanism_contract_iter29.json`, and their Judge decisions. Do not use rejected candidate drafts as active instructions.

S01 declares the parent and sole protocol-valid direct control to be iter26, with `PARENT_COMMIT=612a5a41dfe524205b6afa46370ad0d0ce377882`, `EXPERIMENT_TYPE=single_factor`, and `CANONICAL_BASELINE_ITER=iter26`. Direct-control Stage3 score is `test_recall@10=0.057017009349048554` (`n_eval=57439`). The inclusive user target is `test_recall@10 >= 0.065`. S14 authorized one bounded alternative FCCR-1 mapping from the same behavior-branching and raw-residual-median inputs; all other conditions remain locked to iter26.

S02/S04 canonically register the sole new mapping for each layer:

```text
x_l = B_l / (B_l + 2.0)
y_l = m_l_raw / (m_l_raw + 0.1)
u_l = (x_l + y_l) / 2
c_l = 0.05 + 1.45 * u_l
```

Canonical input vectors in `[L0,L1,L2]` order:

```text
behavior_branching = [19.324911558712664, 1.4605688962651735, 1.0148104414712726]
raw_residual_medians = [1.0, 0.10941, 0.09331]
```

The accepted output is `[1.3660953164241916, 0.7347829661951981, 0.6439958072706683]`. S03 approved historical method/value provenance only, not checkpoint-level reproduction or historical byte identity. Implementation must explicitly consume the plural data key `raw_residual_medians` and fail closed if absent; no `residual_norm` alias, normalized scale, or other fallback is allowed. Do not rerun the old behavior writer, which overwrites this key and depends on missing iter8 raw SIDs.

## Parent evidence: iter26

Canonical iter26 files establish the preceding mapping and inherited mechanisms:

- `stage2_RQ-VAE/curvature_RQ-VAE_iter26/scripts/compute_closed_form_curvature.py` and `logs/mechanism_manifest_iter26.md`: `m_min=min(m_raw)`, `s_l=log1p(B_l)/log1p(m_l_raw/m_min)`, `z_l=(s_l-mean(s))/(std(s)+1e-12)`, `c_l=clip(0.5*exp(0.2*z_l),0.05,1.5)`, yielding `[0.6145357379232853,0.5333020920777128,0.3814078098431606]`.
- `modules/quantize.py` registers fixed `_fixed_c` buffers; `get_c()` returns them; cyclic and learnable branches are unreachable when fixed curvature is supplied; curvature regularization is zero.
- `modules/rqvae.py` and `curvature_RQ-VAE.py`: inherited behavior-loss weight `0.20`, temperature `0.07`, Sinkhorn configured `eps=0.05`, `iters=3`, including the unchanged parent rule `effective_eps = sk_eps * (c / c_cyclic_max)` (so the new curvature changes its realized epsilon without changing that rule), 3 quantizer layers × codebook size 256, seed 42, 100,000 Stage2 global steps, iter8 warm-start checkpoint (curvature state excluded), and AdamW `lr=1e-3`, `weight_decay=1e-4`.
- Stage1/Stage0 inputs and hashes, Stage3 source/settings, warm-start identity, and test protocol are locked in the iter29 S01 manifest. The Stage3 trainer source is unchanged; only the per-iteration wrapper's SID path, variant label, and results directory differ.
- Existing iter29 code scaffold is copied from iter26. `curvature_RQ-VAE.py` reads `closed_form_c_l` from `scripts/computed_behavior_branching.json`; the current copied data also contains a legacy `residual_norm` alias and old mapping fields. These must not be inputs or fallback for the new mapping.

## Expected one-factor delta to adjudicate

The only conceptual change is replacing iter26's per-layer closed-form equation with the S02/S04 bounded rational/additive equation above, using the same approved `behavior_branching` and explicit `raw_residual_medians` values, then passing the resulting fixed vector through the unchanged curvature-consuming quantization path. The parent’s existing curvature-dependent Sinkhorn `effective_eps = sk_eps * (c / c_cyclic_max)` will respond deterministically to the new `c`; this is a mediated consequence of the sole mapping change, not a new or edited mechanism. The following are not additional mechanisms: updating the stored deterministic mapping result/intermediates, strict contract/input validation, current-iteration labels, result-directory isolation, or lightweight verification.

Expected iteration-local implementation paths:

- `scripts/compute_closed_form_curvature.py`: implement the new equation with explicit plural raw-key consumption and fail-closed validation.
- `scripts/computed_behavior_branching.json`: preserve branching, explicit raw medians and their historical source record; remove ambiguous legacy aliases/obsolete old-map fields; record new intermediates and fixed vector.
- `curvature_RQ-VAE.py`: load/validate the canonical fixed vector; remove the unused legacy layer-scale fallback helper and old closed-form constants/docs/labels; do not change architecture, losses, optimizer, Sinkhorn, warm-start, seed, steps, or data loading.
- `curvature_config.py`: change only the full iter29 mechanism label and iter29 result paths; keep upstream data paths, seeds, GPU, model and runtime settings.
- `scripts/run_stage3_iter29.py`: add the iteration-local wrapper with iter29 SID/output paths and full mechanism variant; do not change Stage3 trainer code/settings.
- `scripts/mvg_check.py`: use the accepted iter29 vector; check fixed/invariant curvature and one real `loss.backward()` path; add a same-checkpoint/same-batch counterfactual against iter26's declared fixed vector `[0.6145357379232853,0.5333020920777128,0.3814078098431606]`, requiring measurable change in a curvature-consuming assignment/geometry/loss signal.
- Remove only stale copied iter-specific Stage3 wrappers, `sid_metrics_any.py` (project-local CLI utility / obsolete HitRate presentation), `calibrate_residual_scales.py` (not an authorized S06 input generator), and `grad_check.sh` if they remain unused. Keep the no-argument `grad_check.py` and `export_sids_for_stage3.py` where their current code is compatible.

The implementation and Judge must distinguish housekeeping/path-label changes from the single scientific delta. Do not alter any other formula, constant, input, loss, optimizer, existing Sinkhorn rule/parameters, Stage1, Stage3 trainer, or protocol setting. A changed realized Sinkhorn epsilon from the unchanged inherited `c`-dependent rule is part of the mapping’s downstream effect, not a second source change.

## Candidate task

Agents A and B independently identify parent, inherited mechanisms, exact one-factor delta, and source/path changes. Each writes only its own proposal at `logs/deliberation/S05_ONE_FACTOR/round_1/agent_a.md` or `agent_b.md`, beginning with the required role/independence/source/stage header and including evidence, risks, self-rejection, and an autonomous next action. Judge C adjudicates against this packet and canonical S00–S04, then writes `logs/one_factor_diff_iter29.md` and `logs/deliberation/S05_ONE_FACTOR/round_1/judge.md`. No source implementation, tests, build, formatter, GPU, or training at S05.