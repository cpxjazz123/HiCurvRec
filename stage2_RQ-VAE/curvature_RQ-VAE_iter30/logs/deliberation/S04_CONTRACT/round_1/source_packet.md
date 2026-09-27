# S04_CONTRACT — canonical source packet

STAGE_ID=S04_CONTRACT
ROUND=1
ITERATION=30

## Objective
Independently encode the FCCR-1 machine-readable mechanism contract for the S02 candidate (iter29 mapping) using the exact required schema and S03-approved historical method/value inputs. Produce no code change or run authorization. The fresh iter26-map paired control is not a second conceptual mechanism contract; it must be checked separately later by S07/S08 against its already registered vector.

## Governing canonical sources
1. Root `CLAUDE.md`, active `skill://curvature-rqvae-iter` §§3, 7–9, 17–19.
2. S00 `stage2_RQ-VAE/curvature_RQ-VAE_iter30/logs/source_snapshot_iter30.md`.
3. S01 `stage2_RQ-VAE/curvature_RQ-VAE_iter30/logs/protocol_manifest_iter30.md` and S01 Judge.
4. S02 `stage2_RQ-VAE/curvature_RQ-VAE_iter30/logs/hypothesis_iter30.md` and S02 Judge.
5. Judge-approved S03 `stage2_RQ-VAE/curvature_RQ-VAE_iter30/logs/mechanism_manifest_iter30.md` and `logs/deliberation/S03_PROVENANCE/round_1/judge.md`.
6. Primary current/historical source references listed within those canonical artifacts, particularly iter29 mapping code/output and fixed-curvature implementation.

## Canonical candidate mapping and values
The primary S04 contract covers the iter29 mapping, not the iter26 paired control. Layer order `[L0,L1,L2]`:
```text
B = [19.324911558712664, 1.4605688962651735, 1.0148104414712726]
raw_residual_medians = [1.0, 0.10941, 0.09331]
```
S03 verdict is `PASS_LIMITED_TO_HISTORICAL_METHOD_VALUE_PROVENANCE`, with HIGH confidence in recorded branching method/value and MEDIUM confidence in raw-residual method/value. It is not checkpoint replay, current recalculation, historic byte identity, or proof of historic runtime use. The distinct `[0.001,0.932889,1.0]` normalized layer scales remain forbidden as a substitute. Every new iter30 consumer must require the explicit `raw_residual_medians` key and fail closed if absent; never fall back to `residual_norm` or any normalized vector.

The S02 candidate equation and outputs are fixed:
```text
x_l = B_l / (B_l + 2.0)
y_l = m_l_raw / (m_l_raw + 0.1)
u_l = (x_l + y_l) / 2
c_l = 0.05 + 1.45 * u_l
fixed_curvature = [1.3660953164241916, 0.7347829661951981, 0.6439958072706683]
```
This output is fixed closed-form, non-trainable, and time-invariant. The mapping adds no cyclic schedule, curvature regularization, new curvature-conditioned optimizer, or new curvature-conditioned auxiliary loss.

The control map remains the already approved iter26 equation/vector in S02, `[0.6145357379232853,0.5333020920777128,0.3814078098431606]`; do not place it as a second contract mechanism or alter the required FCCR-1 schema. S07/S08 must separately verify both arms and the control vector.

## Exact required JSON schema
The canonical contract must contain exactly these 10 top-level fields, with no extras or comments:
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
`raw_residual_median` is the FCCR-1 semantic name for the per-layer scalar quantity. The actual data vector key is plural `raw_residual_medians`; S06 must preserve explicit-key/fail-closed hygiene. This semantic/key distinction does not justify extra JSON schema fields.

## Required candidate work and boundary
Each independent candidate should propose the exact JSON object above, explain its field types/meaning and mapping/value source, state the explicit residual data-key safety requirement and S03 limitation, note how the iter26 control will be separately verified without adding a second contract, and list risks/self-rejection/next action. Candidates write only their own `agent_a.md` or `agent_b.md` in this directory. Judge C must adjudicate against the schema and canonical artifacts, then materialize `logs/mechanism_contract_iter30.json` and `logs/deliberation/S04_CONTRACT/round_1/judge.md`.

S04 does not authorize S05–S09 or implementation, MVG, Stage2, Stage3, or GPU work. If the accepted contract cannot be reconciled with S02/S03 or exact required schema, reject/replan without changing the hypothesis or values.