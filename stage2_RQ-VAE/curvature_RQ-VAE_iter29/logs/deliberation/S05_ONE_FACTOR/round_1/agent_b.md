ROLE=AGENT_B
INDEPENDENCE_DECLARATION=I did not read the other candidate before completing this artifact.
SOURCE_PACKET=stage2_RQ-VAE/curvature_RQ-VAE_iter29/logs/deliberation/S05_ONE_FACTOR/round_1/source_packet.md
STAGE_ID=S05_ONE_FACTOR

# Iter29 S05 One-Factor Proposal — Agent B

## Registration

```text
PARENT_ITER=iter26
PARENT_COMMIT=612a5a41dfe524205b6afa46370ad0d0ce377882
CANONICAL_BASELINE_ITER=iter26
EXPERIMENT_TYPE=single_factor
ACTIVE_MECHANISMS_BEFORE=iter26 FCCR-1 fixed closed-form per-layer curvature; fixed non-trainable curvature buffers; inherited behavior-loss and Sinkhorn quantization mechanisms
NEW_MECHANISM=replace only iter26's per-layer closed-form mapping with the canonical S02/S04 bounded rational/additive mapping, consuming the same behavior_branching and explicit raw_residual_medians inputs
ACTIVE_MECHANISMS_AFTER=the iter26 inherited mechanisms unchanged, with the canonical iter29 fixed-curvature vector produced by the new mapping
```

The direct-control result is iter26 `test_recall@10=0.057017009349048554`, `n_eval=57439`; the protocol lock marks it as the sole direct comparator. Iter18 is historical-only (`HISTORICAL_NONCOMPARABLE`), not this experiment's baseline. The iter29 user success threshold is inclusive `test_recall@10 >= 0.065` (S01 protocol manifest).

## Exact single-factor equation and substitution

Iter26's previous mapping was:

```text
m_min = min_l(m_l_raw)
s_l = log1p(B_l) / log1p(m_l_raw / m_min)
z_l = (s_l - mean(s)) / (std(s) + 1e-12)
c_l = clip(0.5 * exp(0.2 * z_l), 0.05, 1.5)
```

It used `m_min=0.09331`, `c_base=0.5`, `alpha=0.2`, and yielded `[0.6145357379232853, 0.5333020920777128, 0.3814078098431606]` (iter26 `scripts/compute_closed_form_curvature.py`, mechanism manifest, and one-factor record).

The sole conceptual delta is replacing that equation with the canonical S02/S04 mapping, independently for each layer `l`:

```text
x_l = B_l / (B_l + 2.0)
y_l = m_l_raw / (m_l_raw + 0.1)
u_l = (x_l + y_l) / 2
c_l = 0.05 + 1.45 * u_l
```

Constants are `B_ref=2.0`, `m_ref=0.1`, `c_min=0.05`, and `c_max=1.50` (`c_max-c_min=1.45`). Use precisely these S03-approved historical input values, in `[L0,L1,L2]` order:

```text
behavior_branching = [19.324911558712664, 1.4605688962651735, 1.0148104414712726]
raw_residual_medians = [1.0, 0.10941, 0.09331]
x = [0.9062129756321139, 0.42206034326942476, 0.3366083742817442]
y = [0.9090909090909091, 0.5224678859653312, 0.48269618747090165]
u = [0.9076519423615115, 0.4722641146173780, 0.40965228087632294]
fixed_curvature = [1.3660953164241916, 0.7347829661951981, 0.6439958072706683]
```

The residual input MUST be obtained from the explicit plural JSON key `raw_residual_medians`; fail closed if absent. Do not use `residual_norm` (even if numerically equal in a copied artifact), normalized layer scales, or another alias/fallback. Keep the layer order and values unchanged. S03 approves historical method/value provenance only; the substitution is not checkpoint-level reproduction and does not establish historical byte identity.

The new mapping is the only scientific/experimental change. The inherited curvature-dependent Sinkhorn rule will continue to compute `effective_eps = sk_eps * (c / c_cyclic_max)`: its rule and configured parameters are unchanged, while the realized per-layer epsilon changes deterministically as a downstream consequence of the new `c`. Updating deterministic stored intermediates/vector, validating required input keys/values, and checking the fixed-curvature contract are implementation/verification hygiene, not extra mechanisms.

## Inherited mechanisms and controlled invariants

Carry forward iter26 without modification except the mapping-derived fixed vector:

- FCCR-1 closed-form curvature is computed before Stage2, stored as fixed/non-trainable buffers, and invariant to training step, optimizer updates, and train/eval mode. No learnable curvature, cyclic schedule, or curvature regularization is added; the quantization path consumes fixed curvature as before.
- Existing behavior mechanism remains at weight `0.20` and temperature `0.07`.
- Existing Sinkhorn configuration remains `sk_eps=0.05`, `iters=3`, and the parent rule `effective_eps = sk_eps * (c / c_cyclic_max)` is unchanged. Its realized epsilon changes only because the sole mapping change supplies a different fixed `c`; do not treat this mediated downstream effect as a second code/mechanism delta.
- Three quantizer layers, codebook size 256 per layer; seed 42; 100,000 Stage2 global steps; iter8 warm-start checkpoint with curvature state excluded; existing data inputs and training/runtime settings remain locked to the iter26 protocol.
- No mechanism is stacked from iter28 or any rejected draft. No Stage1 representation, Stage3 trainer, evaluation protocol, loss, optimizer, Sinkhorn rule/parameters, manifold, architecture, or input definition changes.

## Expected changed implementation files and path hygiene

The source packet's S06 implementation-path list is the expected scope; this S05 proposal does not edit any implementation file:

1. `scripts/compute_closed_form_curvature.py` — implement only the registered equation; explicitly read plural `raw_residual_medians`, validate it and `behavior_branching`, and fail closed without fallback.
2. `scripts/computed_behavior_branching.json` — preserve the approved branching and historical provenance record plus explicit raw medians; remove ambiguous legacy aliases/obsolete old-map fields and store the new equation's intermediates/constants/fixed vector. This is deterministic input/result bookkeeping, not a second mechanism. Do not rerun the old behavior writer: it overwrites this JSON without the plural raw key and requires missing iter8 raw SIDs.
3. `curvature_RQ-VAE.py` — load and validate the canonical fixed vector; remove the unused legacy layer-scale fallback helper and obsolete old-mapping constants/docs/labels. Do not alter architecture, losses, optimizer, Sinkhorn, warm-start, seed, steps, or data loading.
4. `curvature_config.py` — update only the full iter29 mechanism label and iteration-specific result paths; preserve upstream data paths, seed, GPU, model and runtime configuration.
5. `scripts/run_stage3_iter29.py` — iteration-local Stage3 wrapper using the iter29 SID/output paths and full mechanism variant label; keep Stage3 trainer source and settings unchanged.
6. `scripts/mvg_check.py` — use the accepted iter29 vector; verify fixed/time-invariant curvature and one real `loss.backward()` path; compare same-checkpoint/same-batch against iter26's declared fixed vector `[0.6145357379232853,0.5333020920777128,0.3814078098431606]` and require a measurable change in a curvature-consuming assignment/geometry/loss signal. This is verification of the registered intervention, not another change to the training mechanism.

Iteration-local label/output-path changes and the wrapper are provenance/output isolation hygiene, not scientific changes. Remove only if still present and unused: stale copied iter-specific Stage3 wrappers, project-local `sid_metrics_any.py` (obsolete HitRate presentation/CLI utility), unauthorized `calibrate_residual_scales.py` input generator, and `grad_check.sh`. Retain compatible no-argument `grad_check.py` and `export_sids_for_stage3.py`. No other code/path cleanup is authorized by this proposal.

## Explicit comparison dimensions

```text
OPTIMIZER_DIFFERENCES=None. Preserve AdamW lr=1e-3 and weight_decay=1e-4; no optimizer-conditioned mechanism.
LOSS_DIFFERENCES=None. Preserve existing losses, behavior-loss weight/temperature, and zero curvature regularization; no auxiliary loss.
STAGE1_DIFFERENCES=None. Preserve locked Stage1 embedding/input identities and paths.
STAGE3_DIFFERENCES=No trainer/source/settings/evaluation-protocol difference. Only iter29 wrapper variant label and iteration-local SID/output paths differ.
```

**Explicit one-factor statement:** the only conceptual change from iter26 is replacing its per-layer closed-form mapping with the canonical S02/S04 mapping of the same `behavior_branching` and explicit `raw_residual_medians` values. No second mechanism or protocol change is proposed.

## Evidence basis

- S01 protocol manifest and Judge identify iter26 as parent/direct control, give the parent commit, direct baseline score/path and locked Stage1/Stage2/Stage3 settings; iter18 is explicitly historical noncomparable.
- S02 canonical hypothesis and Judge specify the exact equation, constants, input vectors, layerwise intermediates and output vector.
- S03 canonical manifest and Judge distinguish raw medians `[1.0,0.10941,0.09331]` from normalized scales `[0.001,0.932889,1.0]`, approve historical method/value provenance only, and require plural-key-only consumption with no fallback.
- S04 canonical contract/Judge fix FCCR-1 behavior: closed-form, fixed, non-trainable/time-invariant, with no cyclic schedule, curvature regularization, new curvature-conditioned optimizer or auxiliary loss.
- Parent iter26 primary script/log evidence supplies the preceding mapping and existing mechanisms/settings. The shared S05 packet enumerates expected S06 implementation paths and forbids source edits, validation, or training in this stage.

## Risks and constraints

- Re-running the copied iter26 writer could erase `raw_residual_medians`; absence of this key must be an immediate input failure, not a reason to regenerate from unavailable historical SIDs or substitute another residual field.
- Historical provenance is limited: neither this proposal nor the canonical values prove replay from historical checkpoint bytes. Preserve that limitation in implementation/evidence.
- Old constants/helpers/docs left live in `curvature_RQ-VAE.py` could accidentally keep or imply the prior map or an alternate fallback; remove only obsolete mapping plumbing while preserving shared invariant checks and training behavior.
- A wrapper/path/variant mistake could point Stage3 at the wrong iteration's SID or results. Isolate paths/labels without changing Stage3 trainer or protocol.
- The accepted vector differs materially from iter26's vector; the later matched-input counterfactual must test a real curvature-consuming signal. A missing direct effect cannot be repaired by retuning this mapping within iter29; follow the feasibility/abort contract.
- No conclusion about retrieval benefit or Stage3 outcome follows from this S05 proposal.

## Self-rejection conditions

Reject this proposal or block propagation if any primary canonical artifact contradicts the declared parent/commit, mapping/constants/output, input semantics, S03 provenance boundary, or locked protocol; if implementation would need `residual_norm`, a normalized scale, missing-key fallback, recalibration, checkpoint replay claims, another mechanism, or a non-mapping protocol change; if optimizer/loss/Stage1/Stage3 trainer changes are needed; or if any listed source/path scope cannot be justified as the expected implementation or iteration-local hygiene. A contract, provenance, one-factor, or direct-activation failure must be handled under the skill's fail-closed/abort rules, never by changing the registered mapping in this iteration.

## Autonomous next action

Submit this candidate only to Judge C for comparison with Agent A against the shared packet and canonical S00–S04 evidence. If adjudicated as the canonical S05 artifact, advance to S06 with the equation, inputs, vector, provenance limits, and file/path boundaries unchanged; do not implement, run tests, launch GPU work, or train during S05.
