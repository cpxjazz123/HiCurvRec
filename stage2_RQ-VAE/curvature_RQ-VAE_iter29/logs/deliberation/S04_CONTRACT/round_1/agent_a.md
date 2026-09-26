ROLE=AGENT_A
INDEPENDENCE_DECLARATION=I did not read the other candidate before completing this artifact.
SOURCE_PACKET=stage2_RQ-VAE/curvature_RQ-VAE_iter29/logs/deliberation/S04_CONTRACT/round_1/source_packet.md
STAGE_ID=S04_CONTRACT

# Agent A — iter29 FCCR-1 mechanism contract candidate

## Candidate JSON

This candidate contains exactly the ten required FCCR-1 fields. Values are copied from the canonical, adjudicated S02/S03 record and the S04 packet; no schema fields or values are added or changed.

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

## Contract semantics and evidence

The three values are ordered `[L0,L1,L2]` and are the exact S02 mapping outputs for the canonical inputs `behavior_branching=[19.324911558712664,1.4605688962651735,1.0148104414712726]` and `raw_residual_medians=[1.0,0.10941,0.09331]`. S02 registered `x_l=B_l/(B_l+2.0)`, `y_l=m_l_raw/(m_l_raw+0.1)`, `u_l=(x_l+y_l)/2`, and `c_l=0.05+1.45*u_l`; its listed fixed outputs are `[1.3660953164241916,0.7347829661951981,0.6439958072706683]` (canonical `logs/hypothesis_iter29.md`, §B and numeric substitution; S04 packet lines 8–12, 16–30). These outputs are precomputed before Stage2 and remain fixed.

Field interpretation, aligned with the active FCCR-1 schema:

- `contract_version: "FCCR-1"` identifies the controlling fixed closed-form curvature contract.
- `curvature_source: "closed_form"` states that curvature is obtained from the registered deterministic mapping, not learned.
- `curvature_trainable: false` forbids a trainable curvature parameter or optimizer-updated curvature.
- `curvature_time_varying: false` requires the precomputed values to remain invariant with training step, optimizer state, and train/eval mode.
- `uses_cyclic_schedule: false` rules out cyclic, scheduled, or curriculum-dependent curvature.
- `uses_curvature_regularization: false` rules out a curvature regularizer, including one intended to pull trainable curvature toward a prior.
- `new_curvature_conditioned_optimizer: false` rules out a new optimizer-side curvature-conditioned mechanism (for example, a new curvature-conditioned LR or beta2 rule).
- `new_curvature_conditioned_aux_loss: false` rules out a new curvature-conditioned auxiliary loss or any second mechanism stacked with the mapping.
- `formula_inputs` declares the two semantic quantities in the S02 equation, in the listed order; it does not assert that the singular schema label is a literal data-object key.
- `final_curvature_values` fixes the exact registered per-layer output vector; no other values, recalibration, or fallback mapping are authorized.

### Raw residual key and provenance limitation

`raw_residual_median` in the contract schema names the per-layer semantic input `m_l_raw`. The canonical recorded data field is plural, `raw_residual_medians`, because it stores the three layer-ordered values. These names are compatible: singular schema terminology denotes the quantity for one layer; plural storage key denotes the vector across layers. The consumer must explicitly read `raw_residual_medians` and map its entries to the corresponding layer. It must not infer this input from `residual_norm`, normalized layer scales, or a similarly named field. S06 must fail if `raw_residual_medians` is absent; there is no fallback. The S03 record also warns not to rerun the iter26 behavior writer: it overwrites the shared JSON without this explicit raw key and requires the unavailable iter8 raw SID export (`logs/mechanism_manifest_iter29.md`, §§29–45, 91–99; S03 Judge, `logs/deliberation/S03_PROVENANCE/round_1/judge.md`).

S03 passed only for historical method/value provenance. It did not establish checkpoint-level replay, historic input-byte identity, or a new measurement: the historical calibration checkpoint and iter8 raw SID export are absent at their referenced paths, and hashes were not recorded. The exact values and their historical method are approved to proceed with these reproducibility limits preserved; this contract does not upgrade the evidence to reproduction (canonical S03 manifest §§11–13, 69–76, 95–99; S03 Judge lines 19–25).

## Evidence / risks

Evidence for this candidate is the S04 packet's exact ten-field required JSON, canonical S02 equation and numeric values, and canonical S03's approved input semantics and limits. S00 and the active skill make FCCR-1 controlling and forbid trainable, scheduled, regularized, optimizer-conditioned, auxiliary-loss, or stacked curvature mechanisms. No formula, protocol, code, or source data is modified or inferred here.

Primary risk is downstream semantic-key drift: a consumer could mistakenly use legacy `residual_norm` or normalized values because the data key is plural and the schema field is singular. The explicit S06 key/no-fallback requirement above is mandatory hygiene, not a schema addition. Another risk is overclaiming reproducibility from S03's PASS; it only approves historical method/value provenance, with missing-checkpoint/raw-SID and historic-byte-identity limitations. Neither risk justifies altering the approved contract values or inventing contract fields.

## Self-rejection conditions

I would reject this candidate if any field were missing, any extra schema field appeared, either input name or any curvature value differed from the S04/S02 canonical specification, the output order were not `[L0,L1,L2]`, or any flag authorized trainability, time variation, cyclic scheduling, curvature regularization, a new curvature-conditioned optimizer/loss, or mechanism stacking. I would also reject any interpretation that treats `residual_norm` or normalized layer scales as a fallback for `raw_residual_medians`, or claims S03 proves checkpoint-level reproduction. None applies to the JSON proposed here.

## Autonomous next action

Judge C should adjudicate this candidate against the same packet and primary canonical S00–S03 artifacts. After adjudication, materialize only the judge-approved exact JSON at `stage2_RQ-VAE/curvature_RQ-VAE_iter29/logs/mechanism_contract_iter29.json` before S05. Preserve the plural `raw_residual_medians` consumer-key/no-fallback requirement and S03's reproduction limitation for S06; do not alter the hypothesis, schema, values, or source files.