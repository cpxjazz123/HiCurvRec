# Iter31 S07 Preflight — Judge C Adjudication

```text
STAGE_ID=S07_PREFLIGHT
ROUND=1
VERDICT=REJECT_BOTH

HARD_GATE_A=FAIL
HARD_GATE_B=FAIL

EVIDENCE_FOR_A=Agent A provides a detailed static audit consistent with the frozen packet and S06 plan: HRA/FCCR remain separate, the Step6 review is static only, and no execution was claimed. The parent-side checker outcomes now supersede A's unavailable executions. Primary evidence: source_packet.md:48-61; S06 Judge judge.md:26-28; S06 plan implementation_plan_iter31.md:46-48,83-91,110-114.
EVIDENCE_FOR_B=Agent B provides a detailed static audit consistent with the frozen packet and S06 plan: the checker source was inspected but neither command was run in that candidate session, and no execution was claimed. The parent-side checker outcomes now supersede B's unavailable executions. Primary evidence: Agent B report lines 40-48,72-93; source_packet.md:48-61; S06 Judge judge.md:26-28; S06 plan implementation_plan_iter31.md:46-48,83-91,110-114.
PROBLEMS_A=The authorized checker outputs were reported unavailable, so A could not establish either mandatory checker pass. Parent-side runs subsequently produced failures for both checks. Static evidence cannot satisfy the two-checker gate.
PROBLEMS_B=The authorized checker outputs were reported unavailable, so B could not establish either mandatory checker pass. Parent-side runs subsequently produced failures for both checks. Static evidence cannot satisfy the two-checker gate.

WHY_NOT_A=The static implementation findings are useful, but A's S07 conclusion is incomplete for canonical adjudication: its checker results were NOT RUN, whereas actual authorized parent-side executions have now failed. It cannot pass or propagate.
WHY_NOT_B=The static implementation findings are useful, but B's S07 conclusion is incomplete for canonical adjudication: its checker results were NOT RUN, whereas actual authorized parent-side executions have now failed. It cannot pass or propagate.

CANONICAL_DECISION=REJECT_BOTH for S07 round 1; both mandatory preflights failed in the parent-side executions. Authorize only the two narrowly scoped, non-mechanism repairs below, then require a fresh independent S07 round 2 and rerun both checkers there. The shared FCCR checker is immutable and must not be changed, bypassed, or substituted. No S08/MVG, Stage2, Stage3, GPU, or training is authorized.
CANONICAL_ARTIFACT=stage2_RQ-VAE/curvature_RQ-VAE_iter31/logs/deliberation/S07_PREFLIGHT/round_1/judge.md; stage2_RQ-VAE/curvature_RQ-VAE_iter31/logs/preflight_iter31.md
CONFIDENCE=HIGH
USER_INPUT_REQUIRED=NO
ITERATION_PURPOSE=PERFORMANCE_SEEKING_MECHANISM
SWEEP_OR_REPLICATION_ITERATION=NO
ROOT_CAUSE_ITERATION=NO
AUTONOMOUS_NEXT_ACTION=Apply only the two authorized minimal repairs, then complete S07 round 2 with independent A/B review and successful separate executions of both authorized checkers; keep S08 and all later stages blocked until that adjudication.

REPLAN_CONSTRAINTS=Repair 1: add only an additive semantic/provenance clarification subsection to stage2_RQ-VAE/curvature_RQ-VAE_iter31/logs/mechanism_manifest_iter31.md. It must include the literal semantic terms “behavior branching”, “raw residual”, and “normalized layer scale”; faithfully restate the existing B_l provenance/input, exact historical JSON key raw_residual_medians for m_l_raw, and that normalized layer scale is a distinct quantity and not a substitute. Preserve all existing S03 facts, values, provenance limitations, schema, formula, and mechanism; do not revise or delete existing text. This is authorized because the existing canonical manifest already states the distinctions and underlying definitions at lines 54-55, 79-87; this is terminology clarification, not a new scientific claim or contract transition. Repair 2: edit only stage2_RQ-VAE/curvature_RQ-VAE_iter31/scripts/preflight_hra_step6_iter31.py::_find_method so the comprehension captures `child` (the matching FunctionDef/AsyncFunctionDef) rather than `node` (the containing RqVae ClassDef), retaining its exact-one-match guard and return. This is a checker correctness fix, not a model/mechanism change. S07 round 2 must independently inspect both repairs and independently run both checks, documenting exact commands, cwd, full outputs, and exit statuses. No checker rerun is part of round 1 adjudication. Do not edit the unchanged shared FCCR checker or FCCR contract/loader, do not weaken/bypass either checker, and do not alter the mechanism or any other canonical S00-S06 record. No S08/MVG, MVG scripts, Stage2/Stage3, GPU, or training before S07 round 2 is independently adjudicated PASS.
```

## Primary checker outcomes and root causes

### Shared FCCR-1 preflight — FAIL

The source packet required the unchanged shared checker and explicitly prohibited a bypass (source_packet.md:48-55). The canonical S06 plan likewise requires the exact FCCR schema and input-key discipline to remain intact and prohibits editing or bypassing the shared checker (implementation_plan_iter31.md:46-48,112). The checker source `/.claude/skills/curvature-rqvae-iter/scripts/preflight_contract.py:76-80` lowercases the manifest and requires the literal phrases `raw residual`, `behavior branching`, and `normalized layer scale`; it fails on the first missing phrase. The existing S03 manifest distinguishes the quantities and records provenance, but its wording uses forms such as `B_l` / `behavior_branching`, `m_l^{raw}` / `raw_residual_medians`, and `normalized_layer_scale` rather than all required space-separated phrases (mechanism_manifest_iter31.md:54-55,79-87). The actual failure was therefore a missing semantic phrase in the manifest, not evidence that the registered inputs, values, or mechanism are incorrect.

### Iter31 HRA preflight — FAIL due checker defect

In `scripts/preflight_hra_step6_iter31.py:53-64`, `_find_method` filters matching child functions but the list comprehension emits `node` at line 55, which is the containing `RqVae` ClassDef; the matching `child` appears at lines 59-61. Consequently `_find_method` returns a ClassDef rather than the requested method, and downstream AST checks operate on the whole class. The parent-side AST inspection of the actual Step6 method found no banned loss/optimizer names; this is a false failure caused by the checker implementation, not by an unauthorized model loss/optimizer/auxiliary mechanism. The narrowly authorized correction is to capture `child` while retaining the count guard and return behavior.

## Candidate reconciliation

Agents A and B both performed substantive static source audits and explicitly reported that their candidate sessions could not execute commands. Their reports correctly avoid claiming PASS, but their checker verdicts were unavailable and are superseded by the actual parent-side executions summarized above. Neither candidate's static audit overrides either failed checker. The S06 Judge authorizes the HRA checker only as listed tooling (judge.md:39-43; implementation_plan_iter31.md:83-85); it does not authorize changing the model or the shared FCCR checker.

S07 round 1 is not complete and cannot propagate to S08. Round 2 is a repair-verification round only. It must independently review the exact additive manifest wording and `_find_method` correction; run the unchanged FCCR command using its valid installed skill path and run the HRA checker from the Iter31 directory; record both complete outputs and exit codes; confirm both pass; and ensure no unrelated files/contracts/mechanism changed. Even a round 2 S07 PASS would authorize only progression to separately adjudicated S08, not MVG, Stage2, Stage3, GPU execution, or training.
