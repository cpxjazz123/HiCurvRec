STAGE_ID=S00_SOURCE_TRUTH
ROUND=1
VERDICT=MERGE_AB

HARD_GATE_A=PASS
HARD_GATE_B=PASS

EVIDENCE_FOR_A=agent_a.md correctly identified the CAO-1 contract switch from FCCR-1, the iter18 R@10=0.0599 baseline, and the iter18→iter27 inheritance boundary.
EVIDENCE_FOR_B=agent_b.md independently confirmed the same contract switch, the same iter18 baseline, and the same protocol-inheritance pattern.
PROBLEMS_A=None
PROBLEMS_B=None

WHY_NOT_A=Both candidates agree; MERGE_AB is more honest than arbitrarily choosing one.
WHY_NOT_B=See above.

CANONICAL_DECISION=Both candidates agree on the canonical source snapshot.  The canonical artifact is the union of both readings, captured in logs/source_snapshot_iter27.md.
CANONICAL_ARTIFACT=source_snapshot_iter27.md
CONFIDENCE=HIGH

REPLAN_CONSTRAINTS=
MERGE_COMPONENTS_A=agent_a.md (independent reading of source packet)
MERGE_COMPONENTS_B=agent_b.md (independent reading of source packet)
