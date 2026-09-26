ROLE=CANONICAL_SOURCE_PACKET
STAGE_ID=S12_RESULT_CLASSIFICATION
ROUND=2

# Trigger
The required closure-mode `deliberation_gate.py` has not passed. After the S11 round-2 Agent A/B `SOURCE_PACKET` fields were changed from absolute `/home/...` to a project-relative packet path (only headers; content unchanged), the checker advanced to S12 and failed:

`DELIBERATION_GATE_FAIL: .../S12_RESULT_CLASSIFICATION/round_1/agent_a.md SOURCE_PACKET does not match stage packet`

S12 round-1 Agent A and Agent B headers currently declare the identical absolute `/home/wlia0047/ar57/wenyu/GeneRec/.../S12_RESULT_CLASSIFICATION/round_1/source_packet.md` path. The checker executes with repository path resolved under `/fs04/ar57/wenyu/GeneRec`; it accepts exact path or a relative declaration that is a suffix of the resolved packet path, but rejects the equivalent `/home` symlink alias.

# Primary evidence and decision boundary
- `logs/deliberation/S12_RESULT_CLASSIFICATION/round_1/agent_a.md:1-5` and `agent_b.md:1-5` contain the required role/independence/stage fields and identical `/home/...` `SOURCE_PACKET` aliases.
- `logs/deliberation/S12_RESULT_CLASSIFICATION/round_1/source_packet.md` is the actual shared packet, and the A/B analysis and S12 Judge `round_1/judge.md` are complete and valid. The only blocker shown by the checker is string-path mismatch; do not alter candidate analysis, judge decision, scientific classification, outcome numbers, S14 trigger, or any Stage2/Stage3 product.
- Existing S12 round-1 Judge decision is `MERGE_AB`, `MECHANISM_STATUS=ACTIVE_NEUTRAL`, `PROMOTION_STATUS=PROMOTION_FAIL`, `USER_INPUT_REQUIRED=NO`, and authorizes S14 after closure. It declares `logs/stage3_outcome_iter29.md`, `logs/failure_attribution_iter29.md`, and `logs/gate_decision_iter29.md` canonical.
- `require_worker` in the checker compares the declared header with resolved packet path; a project-relative `SOURCE_PACKET=stage2_RQ-VAE/curvature_RQ-VAE_iter29/logs/deliberation/S12_RESULT_CLASSIFICATION/round_1/source_packet.md` ends with the canonical path and is accepted.

# Required adjudication
Independently determine whether a metadata-only normalization of the two S12 round-1 candidate `SOURCE_PACKET` header values is safe and necessary for the required closure gate, without changing their analyses or retroactively changing the S12 result. If safe, provide the exact two-line-only modification and explicitly preserve candidate text/independence declarations and the existing S12 round-1 Judge decision. Reconfirm that classification/outcome/S14 direction remain supported by existing S12 primary evidence. Do not propose a gate bypass, skill/checker edit, Stage3 rerun/restart, third S12 round, user input, or changes to any model/data/result.

# Canonical outputs and limits
This is S12 round 2, the final allowed S12 adjudication round. A/B independently write only their own round-2 candidate files. Judge C may authorize and materialize the two metadata-only round-1 header corrections, write `round_2/judge.md`, and reaffirm the existing three canonical outputs: `logs/stage3_outcome_iter29.md`, `logs/failure_attribution_iter29.md`, `logs/gate_decision_iter29.md`. Do not change their contents unless primary evidence demonstrates a factual error. No Git staging/commit/push is in scope.

# Candidate contract
Each candidate states assumptions, primary evidence, risks, self-rejection conditions, and an autonomous action. Use the round-2 project-relative source packet path in your own required header. No tests, formatters, linters, broad validation.