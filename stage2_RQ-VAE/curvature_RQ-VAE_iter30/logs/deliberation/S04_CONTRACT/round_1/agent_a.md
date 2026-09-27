ROLE=AGENT_A
INDEPENDENCE_DECLARATION=I did not read the other candidate before completing this artifact.
SOURCE_PACKET=stage2_RQ-VAE/curvature_RQ-VAE_iter30/logs/deliberation/S04_CONTRACT/round_1/source_packet.md
STAGE_ID=S04_CONTRACT
ROUND=1
ITERATION=30

# Agent A — Independent FCCR-1 Contract Proposal

## Assumptions and scope

- The contract represents the S02-approved iter29 candidate mapping, with ordered layer vectors `[L0,L1,L2]`; it is not a contract for the paired-control arm.
- S03's `PASS_LIMITED_TO_HISTORICAL_METHOD_VALUE_PROVENANCE` approves the recorded historical method/value lineage only. It does not establish checkpoint replay, current recalculation, historical byte identity, or historical runtime use. The branching method/value confidence is HIGH and raw-residual method/value confidence is MEDIUM; those confidence qualifiers do not expand the provenance claim.
- The semantic formula-input name `raw_residual_median` denotes one per-layer raw-residual scalar. The source data vector/key is exactly the plural `raw_residual_medians`. S06 must consume that explicit key and fail closed if absent; it must never alias `residual_norm`, use normalized layer scales, or fall back to a normalized-scale or inferred/recomputed value.
- S04 covers the iter29 candidate map only. The iter26 paired-control vector `[0.6145357379232853,0.5333020920777128,0.3814078098431606]` remains outside this schema and is to be separately verified for both arms at S07/S08.

## Evidence and independent value/semantic check

The active FCCR-1 contract requires closed-form, fixed, non-trainable, time-invariant curvature using behavior branching plus raw residual magnitude, and disallows cyclic schedule, curvature regularization, a new curvature-conditioned optimizer, or a new curvature-conditioned auxiliary loss. The exact 10-field schema below follows the S04 packet and skill §8 without additional fields.

Primary iter29 `scripts/compute_closed_form_curvature.py` defines `B_REF=2.0`, `M_REF=0.1`, `C_MIN=0.05`, and `C_MAX=1.50`; its mapping function takes the ordered branching and `raw_residual_medians` vectors, computes `x=B/(B+2.0)`, `y=m/(m+0.1)`, `u=(x+y)/2`, and `c=0.05+(1.50-0.05)u`. Its primary `computed_behavior_branching.json` records the corresponding inputs, intermediates, and outputs in the same `[L0,L1,L2]` order. Cross-checking the source constants/equations against that record confirms:

```text
B = [19.324911558712664, 1.4605688962651735, 1.0148104414712726]
raw_residual_medians = [1.0, 0.10941, 0.09331]
x = [0.9062129756321139, 0.42206034326942476, 0.3366083742817442]
y = [0.9090909090909091, 0.5224678859653312, 0.48269618747090165]
u = [0.9076519423615115, 0.472264114617378, 0.40965228087632294]
fixed_curvature = [1.3660953164241916, 0.7347829661951981, 0.6439958072706683]
```

The input record explicitly distinguishes `raw_residual_medians` from the normalized scales `[0.001,0.932889,1.0]`. S03 further establishes that iter26's historical `residual_norm` alias is ambiguous and is not a permitted future input. Thus the JSON semantic name and concrete key are intentionally distinct without adding a key/schema field.

## Exact proposed JSON

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

## Field contract: types, meaning, and exact order

| Field | Type and proposed value | Contract meaning |
|---|---|---|
| `contract_version` | string: `"FCCR-1"` | Identifies the governing Fixed Closed-Form Curvature Research Contract. |
| `curvature_source` | string: `"closed_form"` | The values come from the registered deterministic algebraic map, not a learned or scheduled source. |
| `curvature_trainable` | boolean: `false` | Curvature is a fixed value, not an optimizable parameter. |
| `curvature_time_varying` | boolean: `false` | Values are computed before training and do not vary by step/time. |
| `uses_cyclic_schedule` | boolean: `false` | No cyclic curvature schedule is part of this mechanism. |
| `uses_curvature_regularization` | boolean: `false` | No curvature regularization is introduced. |
| `new_curvature_conditioned_optimizer` | boolean: `false` | No new optimizer behavior conditioned on curvature is introduced. |
| `new_curvature_conditioned_aux_loss` | boolean: `false` | No new curvature-conditioned auxiliary loss is introduced. |
| `formula_inputs` | array of strings, exactly `["behavior_branching", "raw_residual_median"]` | Ordered semantic formula-input names. `raw_residual_median` is the per-layer quantity; actual vector data is read only from explicit plural key `raw_residual_medians`. This semantic-to-storage distinction does not add a schema field. |
| `final_curvature_values` | array of numbers, exactly `[1.3660953164241916, 0.7347829661951981, 0.6439958072706683]` | Candidate output in `[L0,L1,L2]` order, from the S02 iter29 mapping and its recorded inputs. |

The object contains exactly ten top-level fields, with no comments or additional metadata. The field and array order above is intentional and matches the required schema and candidate layer order.

## Risks and self-rejection conditions

- The residual provenance is bounded to S03's historical method/value evidence at MEDIUM confidence; the exact decimals do not establish replay, historical byte identity, present-day recalculation, or runtime use.
- The unsafe `residual_norm` alias and normalized vector are a material future-consumer hazard. Any implementation that omits the explicit-key requirement, accepts an alias/fallback, or substitutes normalized scales fails the contract and must be rejected at the applicable later gate.
- I would reject this proposal if primary S02/S03 evidence did not support the exact inputs, layer order, mapping outputs, or restricted historical provenance; if the object differed in field count, type, value, or order from the required schema; or if it represented the iter26 control as another contract mechanism.

## Autonomous next action and authorization boundary

Submit this candidate and the canonical source evidence to Judge C for S04 adjudication. If accepted, Judge C may materialize only the canonical mechanism contract and Judge record. Continue only through the separately adjudicated pipeline stages; this proposal authorizes no implementation, S05–S09 progression, MVG, preflight, Stage2/Stage3 execution, or GPU work.
