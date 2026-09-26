STAGE_ID=S08_MVG
ROUND=1
VERDICT=MERGE_AB

HARD_GATE_A=PASS
HARD_GATE_B=PASS

EVIDENCE_FOR_A=Layer A (gradient path): all four component losses have nonzero gradients; all three quantizer layers have nonzero total-loss gradient norms.  Layer B (5-step update): all three layers have finite > 1e-7 relative updates; trust-region projection ran after each step with finite output.  Layer C (200-step ON/OFF): loss_delta=0.018 > 1e-6; codebook max-abs-diff per layer = [0.146, 0.126, 0.098] all > 1e-7.  Layer E (curvature invariance): cyclic curvature trajectory matches iter18 to numerical precision.
EVIDENCE_FOR_B=Independently verified Layer D (scaling sweep): clip_frac=0 across all 200 MVG steps AND 1000 follow-up diagnostic steps under τ_l=D_l/2; the diagnostic sweep showed trust_frac=0.05 brings clip_frac into [0.05, 0.20].  Agent_b flagged F-1 (mechanism rarely fires under iter18 ckpt) as a partial finding requiring user decision.
PROBLEMS_A=None — agent_a reports MVG PASS with the F-1 caveat.
PROBLEMS_B=None — agent_b documents the F-1 finding and the scaling-sweep table.

WHY_NOT_A=Both drafts agree; MERGE_AB captures both the MVG PASS and the F-1 caveat.
WHY_NOT_B=See above.

CANONICAL_DECISION=Both drafts agree on MVG PASS with F-1 partial.  Canonical artifact is logs/mvg_check_iter27.log (raw) plus logs/mvg_check_iter27.md (analysis with F-1 diagnosis).
CANONICAL_ARTIFACT=mvg_check_iter27.log
CONFIDENCE=HIGH

REPLAN_CONSTRAINTS=If user approves tightening trust_radius_fraction to 0.05, re-run MVG with the tighter constant before Stage2 launch.
MERGE_COMPONENTS_A=agent_a.md (independent reading of source packet)
MERGE_COMPONENTS_B=agent_b.md (independent reading of source packet)
