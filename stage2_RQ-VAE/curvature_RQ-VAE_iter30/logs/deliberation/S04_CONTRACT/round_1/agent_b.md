ROLE=AGENT_B
INDEPENDENCE_DECLARATION=I did not read the other candidate before completing this artifact.
SOURCE_PACKET=stage2_RQ-VAE/curvature_RQ-VAE_iter30/logs/deliberation/S04_CONTRACT/round_1/source_packet.md
STAGE_ID=S04_CONTRACT
ROUND=1
ITERATION=30

# Independent FCCR-1 contract proposal

## Assumptions and evidence

I treat the S04 source packet, the canonical S02 hypothesis and its Judge decision, and the Judge-approved S03 mechanism manifest and Judge decision as controlling. The contract is for the registered iter29 candidate mapping only. S03 passes narrowly for historical method/value provenance; this is not evidence of checkpoint replay, current recomputation, historic byte identity, or historic runtime consumption.

The registered common input vectors are ordered `[L0,L1,L2]`:

- `B = [19.324911558712664, 1.4605688962651735, 1.0148104414712726]`.
- `raw_residual_medians = [1.0, 0.10941, 0.09331]`.

Primary evidence is the iter29 mapping implementation (`scripts/compute_closed_form_curvature.py`), which fixes `B_REF=2.0`, `M_REF=0.1`, `C_MIN=0.05`, and `C_MAX=1.50`, and its preserved `scripts/computed_behavior_branching.json`, which records the input vectors, intermediate values, and mapped outputs. These agree with S02's registered substitution. For each layer, `x_l = B_l/(B_l+2.0)`, `y_l = m_l_raw/(m_l_raw+0.1)`, `u_l=(x_l+y_l)/2`, and `c_l=0.05+1.45*u_l`. The recorded ordered outputs are `[1.3660953164241916, 0.7347829661951981, 0.6439958072706683]`.

S03 defines `B_l` as effective branching `exp(H_l)`, not a literal distinct-child count. `m_l_raw` means the per-layer median over Stage1 items of Euclidean residual-vector L2 magnitudes (norm over embedding dimensions, then median over items). The manifest explicitly distinguishes this raw magnitude from the normalized layer scales `[0.001, 0.932889, 1.0]` and from any learnable curvature-scale parameter. Preserve the S03 confidence and provenance limits: branching has HIGH confidence for recorded method/value, raw residual MEDIUM confidence for historical method/value; neither input has historic byte-identity/replay proof.

## Exact proposed JSON

This is exactly the required 10-field object, with no comments or extra fields:

```json
{
  "contract_version": "FCCR-1",
  "curvature_source": "closed_form",
  "curvature_trainable": false,
  "curvature_time_varying": false,
  "uses_cyclic_schedule": false,
  "uses_curvature_regularization": false,
  "new_curvature_conditioned_optimizer": false,
  "new_curvature_conditioned_aux_loss": false,
  "formula_inputs": ["behavior_branching", "raw_residual_median"],
  "final_curvature_values": [1.3660953164241916, 0.7347829661951981, 0.6439958072706683]
}
```

## Field contract and exact ordered output

- `contract_version` (string): identifies the governing FCCR-1 contract.
- `curvature_source` (string): curvature is produced by the registered deterministic closed-form map, not learned or selected from runtime state.
- `curvature_trainable` (boolean): `false`; curvature values are fixed, not parameters updated by gradients or an optimizer.
- `curvature_time_varying` (boolean): `false`; the three layer values are computed before Stage2 and remain invariant over time and execution mode.
- `uses_cyclic_schedule` (boolean): `false`; no cycle or schedule changes curvature.
- `uses_curvature_regularization` (boolean): `false`; there is no curvature-regularization term in this mapping contract.
- `new_curvature_conditioned_optimizer` (boolean): `false`; the candidate adds no curvature-conditioned optimizer mechanism.
- `new_curvature_conditioned_aux_loss` (boolean): `false`; the candidate adds no curvature-conditioned auxiliary loss.
- `formula_inputs` (ordered array of strings): `behavior_branching` names the per-layer `B_l` semantic input; `raw_residual_median` names each layer's scalar `m_l_raw` semantic quantity. The residual formula input is sourced from the explicit JSON vector key `raw_residual_medians` (plural), in `[L0,L1,L2]` order. The singular schema label is not an alternate data key and does not permit aliases or extra contract fields.
- `final_curvature_values` (ordered array of numbers): exactly three finite fixed values in `[L0,L1,L2]` order, the S02 iter29 candidate output shown above.

## Residual-key safety, scope, and risks

Any S06 consumer must require the explicit `raw_residual_medians` key and fail closed if it is missing. It must never alias or fall back to `residual_norm`, normalized layer scales, another checkpoint, or a newly inferred/recomputed vector. Numeric equality of an ambiguous historical alias is not semantic evidence. This explicit-key hygiene preserves S03's historical method/value-only provenance and its replay/identity limitations; it does not upgrade those claims.

S04 covers the iter29 candidate map. The iter26 paired-control vector `[0.6145357379232853, 0.5333020920777128, 0.3814078098431606]` is not a second mechanism contract and must not be inserted into this schema. S07/S08 separately verify the two arms and the registered iter26 control vector.

Risks are the limited historical provenance of the recorded inputs, especially the medium-confidence raw-residual method/value chain without original checkpoint replay or historic byte identity, and any future implementation's risk of confusing the singular semantic formula-input name with the plural required payload key. The mapping outputs are registered arithmetic, not new measurements or evidence of runtime behavior. I self-reject this proposal if its schema, types, values, or order differ from the required object; if the residual is treated as normalized scale or read through a legacy alias; if it asserts stronger S03 provenance; or if it conflates the control with this candidate contract.

## Autonomous next action

Submit this independent candidate for Judge C adjudication against the exact schema and canonical S02/S03 evidence. No implementation, S05–S09 authorization, MVG, Stage2/Stage3 run, or GPU work is authorized by this proposal; only after adjudication may the workflow advance through its separately adjudicated stages.
