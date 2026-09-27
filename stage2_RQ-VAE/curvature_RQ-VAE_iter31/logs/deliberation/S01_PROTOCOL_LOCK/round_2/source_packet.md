# Iter31 S01 Protocol Lock — Operational Repair Round 2 Source Packet

```text
STAGE_ID=S01_PROTOCOL_LOCK
ROUND=2
ROUND_TYPE=OPERATIONAL_REPAIR
ITERATION=31
LOCKED_SCIENCE_CHANGED=NO
SAME_ITERATION_REPAIR=AUTHORIZED
USER_INPUT_REQUIRED=NO
```

## Objective

Repair only the current gate's inability to validate the already completed S01 MERGE_AB record. The round-1 Judge and `logs/protocol_manifest_iter31.md` are canonical and remain unchanged. The round-1 Judge contains `VERDICT=MERGE_AB` and the required evidence, hard-gate, decision, and canonical-artifact fields, but does not contain the gate's current `MERGE_COMPONENTS_A=` and `MERGE_COMPONENTS_B=` markers. The existing S01 Agent A and Agent B reports and canonical manifest supply the underlying evidence; this is a gate-evidence serialization omission, not a new protocol decision.

## Repair boundary

Do not edit round-1 source packet, A/B reports, Judge, protocol manifest, or any Iter31 science/protocol facts. Do not create a new A/B pair or re-adjudicate the parent, comparator, protocol ID, or evidence. In this operational round's `repair_record.md`, record the exact original primary evidence from each candidate that the Judge merged and identify the already-canonical protocol manifest. This is an audit supplement only; it does not amend or rewrite the historical Judge decision.

Preserve Iter29 as the S01 parent and sole comparator, the locked protocol and all caveats, and the current no-training boundary. No code or checker changes are in scope.

## Required completion

Write the evidence-only repair record, documenting the missing literal merge-component fields as the gate defect, exact cited Agent A/Agent B contributions, unchanged round-1 Judge and manifest hashes or other stable identities if recorded, and confirmation that no scientific/protocol choice changed. A repair verifier then checks that record against the frozen round-1 evidence and confirms the canonical manifest still exists. Only after verification may this round conclude with `REPAIR_PASS`.

No A/B deliberation, source-history rewriting, protocol modification, baseline rerun, code/checker execution, Stage2, Stage3, GPU, or training is authorized.

## Primary evidence reviewed

- `.claude/skills/curvature-rqvae-iter/scripts/deliberation_gate.py`: current `require_judge` requires both `MERGE_COMPONENTS_A=` and `MERGE_COMPONENTS_B=` for MERGE_AB; operational rounds use the repair-record path.
- `stage2_RQ-VAE/curvature_RQ-VAE_iter31/logs/deliberation/S01_PROTOCOL_LOCK/round_1/judge.md`: completed `MERGE_AB`; no literal merge component markers; existing evidence/problem/why-not and canonical-decision content remain intact.
- `stage2_RQ-VAE/curvature_RQ-VAE_iter31/logs/deliberation/S01_PROTOCOL_LOCK/round_1/agent_a.md` and `agent_b.md`: completed independent evidence underlying the merge.
- `stage2_RQ-VAE/curvature_RQ-VAE_iter31/logs/protocol_manifest_iter31.md`: existing canonical protocol manifest selected by S01.
