STAGE_ID=S06_IMPLEMENTATION
ROUND=2
VERDICT=ACCEPT_A

CANDIDATE_A=stage2_RQ-VAE/curvature_RQ-VAE_iter29/logs/deliberation/S06_IMPLEMENTATION/round_2/agent_a.md
CANDIDATE_B=stage2_RQ-VAE/curvature_RQ-VAE_iter29/logs/deliberation/S06_IMPLEMENTATION/round_2/agent_b.md
SOURCE_PACKET=stage2_RQ-VAE/curvature_RQ-VAE_iter29/logs/deliberation/S06_IMPLEMENTATION/round_2/source_packet.md

HARD_GATE_A=PASS
HARD_GATE_B=FAIL

EVIDENCE_FOR_A=Candidate A preserves the canonical mapping/constants/ordered outputs, explicitly identifies the actual JSON fields `branching` and `raw_residual_medians`, makes the plural residual key fail-closed with no fallback, keeps S03's historical method/value provenance limit, and proposes one shared formula helper plus strict JSON/S04 contract comparisons (agent_a.md:12-27, 31-54). It covers the iter29-only configuration and wrapper routing, the FCCR-1 fixed-buffer/invariance plus explicit backward and matched-control MVG requirements, and the full S05-authorized descriptive-SID migration with bounded conditional cleanup (agent_a.md:56-102). The physical input key and scope are confirmed by the current iter29 `computed_behavior_branching.json`, S03/S04 canonical records, the amended S05 diff/Judge, and the round-2 source packet.
EVIDENCE_FOR_B=Candidate B preserves the accepted equation and values, fixed-curvature MVG requirements, and the S05 policy-hygiene cutover and later-stage boundaries (agent_b.md:10-23, 27-31, 42-80, 86-104). However, it describes `behavior_branching` as an explicit JSON input property to preserve/read (agent_b.md:31, 40, 45), while the actual iter29 JSON property is `branching`; `behavior_branching` is the S03/S04 semantic name. The S04 Judge expressly distinguishes semantic input names from literal data-object keys. That distinction is confirmed by the current primary JSON and S05's requirement to preserve branching. A literal implementation of B's key instruction would fail on the canonical JSON or require an unapproved key migration.

PROBLEMS_A=None material to a hard gate. Candidate A's exact on-disk key instruction is consistent with the current JSON and canonical semantic contract; its plan retains the S05 migration and all required fixed-curvature MVG evidence.
PROBLEMS_B=Hard-gate failure on semantic/provenance-to-source specificity: B conflates the S04 semantic formula name `behavior_branching` with the actual JSON property `branching`, and directs the implementer to preserve/read `behavior_branching`. This is not the canonical source field. The plan is not safe to apply literally without introducing a data-key change outside the accepted S05/S04 contract. The remaining proposal is otherwise substantially aligned.

WHY_NOT_A=No substantive reason to reject A. It names the physical branching property accurately and its complete file-level plan covers the accepted equation, provenance boundary, contract checks, wiring, MVG, policy migration, stale-file bounds, and later-stage authorization limits.
WHY_NOT_B=Do not propagate B's `behavior_branching` property instruction. S03/S04 use that name for the semantic quantity, but the on-disk vector is `branching`; the canonical S05 diff explicitly preserves branching. A's explicit property/key distinction is required to avoid a source mismatch. The other generally compatible requirements do not cure this hard-gate defect.

CANONICAL_DECISION=ACCEPT_A. A passes the active FCCR-1 contract, provenance/semantic correctness, one-factor isolation, implementation specificity, and S05 amendment gates. B's literal input-key instruction fails semantic/source correctness. The canonical plan retains A's exact source-key distinction and strict fail-closed validation; it also explicitly keeps the S04 ten-field contract unchanged, separates behavior-producer and raw-residual provenance, preserves the existing literal `residual_layer_norms=[1.0] * N_LAYERS`, and bounds all cleanup to the S05-approved iter29 subtree. No source changes or verification runs are authorized by this S06 adjudication.

CANONICAL_ARTIFACT=stage2_RQ-VAE/curvature_RQ-VAE_iter29/logs/implementation_plan_iter29.md; stage2_RQ-VAE/curvature_RQ-VAE_iter29/logs/deliberation/S06_IMPLEMENTATION/round_2/judge.md
CANONICAL_PLAN_CONTENT_IDENTITY=SHA256:eef646457f01d0c531c74d4cd97536c1dffd1e2abb02a00a824d29c365306486
CONFIDENCE=HIGH
USER_INPUT_REQUIRED=NO
AUTONOMOUS_NEXT_ACTION=The orchestrator applies only the Judge-approved canonical plan once. Then open S07_PREFLIGHT with fresh independent A/B work against this plan and primary source, followed by its Judge; run S07's official preflight/deliberation gates only after their required canonical artifacts and S07 authorization exist. Only after S07 passes, open and adjudicate S08_MVG independently, then run the single bounded checkpoint/batch MVG. Do not launch Stage2/Stage3 before the later required adjudications and gates. S07/S08 gates remain unrun in this S06 task.

REPLAN_CONSTRAINTS=No replan is required because Candidate A is accepted. Preserve the exact registered mapping, inputs/physical keys, provenance limits, contract values, parent/control and all locked protocol settings. Do not read or use Candidate B's `behavior_branching` as a JSON key; do not retune, add a second mechanism, change Stage3, add SID-metric gates, or advance past the S07/S08 authorization boundaries.
