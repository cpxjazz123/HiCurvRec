# S07_PREFLIGHT round 2 source packet

ROLE=AGENT_CANDIDATE
ROUND=2
STAGE_ID=S07_PREFLIGHT

## Canonical authority

The round-1 S07 Judge artifact is `logs/deliberation/S07_PREFLIGHT/round_1/judge.md`, `VERDICT=REJECT_BOTH`, and `REPLAN_CONSTRAINTS` therein are binding. The confirmed defect is not current-data mismatch: runtime `_load_closed_form_curvatures()` recomputes the FCCR-1 map and compares derived arrays and final curvature, but does not pin physical JSON inputs to the exact S02/S03 approved ordered vectors. Equality of final curvature is insufficient because the mapping averages transformed inputs and distinct input pairs can preserve the same output.

Primary contract references:
- `logs/mechanism_manifest_iter29.md:22-27` (approved semantic vectors and order).
- `logs/mechanism_contract_iter29.json:1-12` (unchanged ten-field S04 contract).
- `logs/implementation_plan_iter29.md` and round-1 Judge (S06 constraints).
- `scripts/computed_behavior_branching.json` (physical JSON keys and historical provenance).
- `scripts/compute_closed_form_curvature.py` (shared formula helper).
- `curvature_RQ-VAE.py:258-350` (active runtime consumer).
- Root `CLAUDE.md` and `skill://curvature-rqvae-iter`.

## Round-2 proposal task

Each candidate independently proposes the smallest fail-closed implementation that enforces the registered exact values and layer order for both physical inputs before FCCR-1 values can reach training. The approved physical keys remain `branching` (semantic name `behavior_branching`) and `raw_residual_medians` (semantic formula name `raw_residual_median`). The approved ordered values are:
- branching: `[19.324911558712664, 1.4605688962651735, 1.0148104414712726]`
- raw_residual_medians: `[1.0, 0.10941, 0.09331]`

Describe the exact source location and comparison semantics, and a focused bounded acceptance scenario: canonical vectors pass; a valid changed input pair that preserves the final curvature must fail closed before Stage2 can use it. Keep formula/constants/derived arrays/output unchanged for canonical inputs. State whether the assertion belongs in the shared iter29 formula helper, runtime consumer, or both, with the fewest duplicated sources of truth.

## Hard limits

- Do not edit production source or other canonical files; write only your candidate artifact.
- Do not run official `preflight_contract.py`, `deliberation_gate.py`, MVG, any tests/build/formatter, GPU work, Stage2 or Stage3.
- Do not change the S04 ten-field schema unless you prove it is strictly necessary; round-1 Judge explicitly prefers keeping it unchanged.
- No recalibration/replay, input substitution, new mechanism, protocol/Stage1/Stage3/training/Sinkhorn/warm-start/export behavior change, S05 reporting expansion, or report-helper cleanup.
- Warm-start identity/use and fresh Stage3 export remain S09 launch blockers, not this S07 correction.
- Do not read the other round-2 candidate artifact. Cite exact evidence, separate observed facts from recommendation, and do not claim validation that was not run.

## Acceptance criteria for Judge C

A proposal is acceptable only if it (1) pins both exact ordered vectors with fail-closed numeric/type/length handling, (2) preserves the distinct physical/semantic names, (3) leaves canonical FCCR-1 outputs and contract unchanged, (4) rejects a compensated noncanonical input pair even when final curvature is unchanged, and (5) stays within the hard limits above.
