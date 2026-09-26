STAGE_ID=S03_PROVENANCE
ROUND=1
VERDICT=MERGE_AB

HARD_GATE_A=PASS
HARD_GATE_B=PASS

EVIDENCE_FOR_A=Traced every formula input (c_l(t), v_k^old, v_k*, p_k^old, p_k*, δ_k, D_l, τ_l, s_k) to its source: c_l(t) from Quantize.get_c() (inherited iter18), distances from modules/hyperbolic._poincare_distance_t, projection from new modules/quantize.apply_hyperbolic_trust_region.
EVIDENCE_FOR_B=Independently verified semantic distinction: raw_residual_median and behavior_branching (FCCR-1 inputs) are NOT inputs to iter27.  Mechanism is independent of assignment rule and curvature formula.  D_l computed from post-step codebook, not fixed reference.
PROBLEMS_A=None
PROBLEMS_B=None

WHY_NOT_A=Both drafts agree.
WHY_NOT_B=See above.

CANONICAL_DECISION=Both drafts agree on provenance manifest.  Canonical artifact is logs/mechanism_manifest_iter27.md.
CANONICAL_ARTIFACT=mechanism_manifest_iter27.md
CONFIDENCE=HIGH

REPLAN_CONSTRAINTS=
MERGE_COMPONENTS_A=agent_a.md (independent reading of source packet)
MERGE_COMPONENTS_B=agent_b.md (independent reading of source packet)
