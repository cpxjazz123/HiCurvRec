ROLE=AGENT_B
INDEPENDENCE_DECLARATION=I did not read the other candidate before completing this artifact.
SOURCE_PACKET=stage2_RQ-VAE/curvature_RQ-VAE_iter29/logs/deliberation/S04_CONTRACT/round_1/source_packet.md
STAGE_ID=S04_CONTRACT

## Candidate mechanism contract

The following JSON has exactly the ten required FCCR-1 fields and no additional or contradictory fields:

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

## Flag and input interpretation

- `contract_version: FCCR-1` identifies the binding fixed closed-form curvature contract; it is not a new contract revision.
- `curvature_source: closed_form` means the S02 mapping computes curvature directly from the registered per-layer inputs and constants before Stage2.
- `curvature_trainable: false` means curvature is not a parameter to optimize and cannot receive optimizer updates.
- `curvature_time_varying: false` means values do not depend on training step or time; the precomputed vector remains fixed throughout the run.
- `uses_cyclic_schedule: false` rules out any cyclic or other time schedule for curvature.
- `uses_curvature_regularization: false` rules out a regularization term whose purpose is to alter or pull curvature.
- `new_curvature_conditioned_optimizer: false` means no new optimizer behavior conditioned on curvature (including curvature-driven learning-rate or beta changes) is part of this mechanism.
- `new_curvature_conditioned_aux_loss: false` means no new curvature-conditioned auxiliary loss is added.
- `formula_inputs` names the two authorized semantic inputs in S02 order: per-layer behavior branching and raw residual median. The S03 data artifact's field is plural, `raw_residual_medians`, because it stores the three layer-ordered observations as a vector; the contract's singular `raw_residual_median` names the per-layer scalar quantity `m_l^{raw}` used by the formula. These are compatible semantic names, not interchangeable JSON keys. The input values are exactly `[19.324911558712664, 1.4605688962651735, 1.0148104414712726]` and `[1.0, 0.10941, 0.09331]`, respectively, in `[L0,L1,L2]` order.
- `final_curvature_values` records the S02 accepted mapping outputs in `[L0,L1,L2]` order, unchanged: `[1.3660953164241916, 0.7347829661951981, 0.6439958072706683]`.

The mapping constants/equation are those already accepted in S02, not additional schema fields: `x=B/(B+2.0)`, `y=m_raw/(m_raw+0.1)`, `u=(x+y)/2`, and `c=0.05+1.45u`. No value or formula is proposed for revision.

## Evidence and limitations

- The S04 packet requires these exact ten fields and values, cites the S02-approved equation/constants, and gives the layer-ordered inputs and final curvature vector (source packet, lines 8–12 and 16–30).
- Canonical S02 `hypothesis_iter29.md` lines 11–23 registers the equation and constants; lines 36–49 record the intermediate terms and final values. S02 Judge C accepted that mapping in `logs/deliberation/S02_HYPOTHESIS/round_1/judge.md`.
- Canonical S03 `mechanism_manifest_iter29.md` lines 22–27 preserves the exact input vectors and order. Lines 61–67 explicitly distinguish `raw_residual_median`, `normalized_layer_scale`, and `learnable_c_layer_scale`; lines 91–99 record the S06 key requirement and the audit's limited conclusion. S03 Judge C's verdict is `MERGE_AB` and its canonical decision is limited to historical method/value provenance.
- S03 does not establish checkpoint-level reproduction, independent recomputation, or historical input-byte identity. The referenced calibration checkpoint and iter8 raw SID export are absent, and no historical hashes establish their bytes. These limits do not authorize changing the registered values or semantics.

## Required S06 hygiene and risks

S06 must read the explicit data field `raw_residual_medians` and validate its presence and exact registered vector/order before using it for the schema's `raw_residual_median` formula input. If that key is absent, reject/fail the input: no fallback to `residual_norm`, normalized `layer_norms`/scales, or any other field is permitted, even if an alias currently has numerically equal values. Do not rerun the iter26 behavior writer: it overwrites the same JSON without `raw_residual_medians` and depends on the unavailable iter8 raw SIDs. Any S06 writer touching that JSON must preserve and validate the explicit raw key.

Risks are provenance reproducibility limits, not license to substitute data: S03 approved historical method/value provenance only; it did not replay the original checkpoint or reconstruct the missing raw SID export. A consumer using the ambiguous legacy alias or normalized scales would violate the registered mechanism despite similar-looking names or possible numeric equality. The contract itself must remain fixed/non-trainable/time-invariant and must not be expanded with fields or stacked mechanisms.

## Self-rejection conditions

I would reject this candidate if any JSON key/value differs from the exact S04 schema or the accepted S02/S03 values; if the field list is expanded; if `raw_residual_median` is treated as permission to consume a data key other than the explicit `raw_residual_medians`; if normalized scales or `residual_norm` are substituted; or if the provenance limits are described as checkpoint-level reproduction. Any such defect must be resolved by Judge C against the canonical packet and records, not by silently changing this contract.

## Autonomous next action

Judge C should adjudicate both independent S04 candidates against the shared packet and canonical S02/S03 evidence, then materialize exactly this ten-field JSON at `stage2_RQ-VAE/curvature_RQ-VAE_iter29/logs/mechanism_contract_iter29.json` before S05. S06 must preserve the explicit `raw_residual_medians` key and fail closed if it is absent; do not rerun the iter26 writer or claim checkpoint-level reproduction.