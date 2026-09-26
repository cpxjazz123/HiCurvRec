# S04 Contract Source Packet — iter29

STAGE_ID=S04_CONTRACT
ROUND=1

## Canonical context

- S00 source snapshot, S01 protocol, S02 accepted hypothesis, and S03 PASS manifest/Judge are canonical. Read these; do not use rejected candidate drafts as active instructions.
- S02 accepted: `B_ref=2.0`, `m_ref=0.1`, `c_min=0.05`, `c_max=1.50`; `x=B/(B+2.0)`, `y=m_raw/(m_raw+0.1)`, `u=(x+y)/2`, `c=0.05+1.45u`.
- S03 approved the exact historical method/value provenance with reproducibility limitations only. Its input vector is `[1.0,0.10941,0.09331]`; this is not checkpoint-level replay. The data consumer must use the explicit `raw_residual_medians` field, never the ambiguous `residual_norm` alias. Do not rerun the iter26 behavior writer; it overwrites the JSON without the raw key and requires missing iter8 raw SIDs.
- Canonical values in `[L0,L1,L2]` order: `behavior_branching=[19.324911558712664,1.4605688962651735,1.0148104414712726]`, `raw_residual_medians=[1.0,0.10941,0.09331]`, `fixed_curvature=[1.3660953164241916,0.7347829661951981,0.6439958072706683]`.
- FCCR-1 requires fixed, precomputed, non-trainable, time-invariant curvature. No cyclic schedule, curvature regularization, new curvature-conditioned optimizer, new curvature-conditioned auxiliary loss, or any other mechanism. Iter26 is the sole direct control and all its settings remain fixed apart from the accepted mapping. User success criterion is `test_recall@10 >= 0.065`; Stage2 SID metrics are descriptive only.

## Contract construction task

Agent A and Agent B independently propose the exact required machine-readable JSON contract, using the FCCR-1 schema in the shared skill and the canonical S02/S03 values above. The candidate JSON must have exactly these required fields (no experimental extras or contradictory values):

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

Check formula input semantics against S03; the JSON key name `raw_residual_medians` and schema semantic field `raw_residual_median` are compatible and must not be replaced by `residual_norm` or normalized scales. Do not alter the formula, output values, protocol, code, or source data. Each worker writes only its own candidate at `logs/deliberation/S04_CONTRACT/round_1/agent_a.md` or `agent_b.md`, begins with required role/independence/source/stage lines, explains each flag, and gives evidence, risks, self-rejection, and autonomous next action. No source edits, tests, build, formatter, GPU, or training. Judge C must adjudicate and materialize `logs/mechanism_contract_iter29.json` before S05.