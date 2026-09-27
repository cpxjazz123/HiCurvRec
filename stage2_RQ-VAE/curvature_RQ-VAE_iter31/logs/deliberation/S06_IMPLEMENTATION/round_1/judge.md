# Iter31 S06 Implementation Design — Judge C

```text
STAGE_ID=S06_IMPLEMENTATION
ROUND=1
VERDICT=MERGE_AB

HARD_GATE_A=PASS
HARD_GATE_B=PASS

EVIDENCE_FOR_A=Agent A's Step6 equation, tangent-versus-ball types, exact right-nested L0/L1/L2 order, explicit (L,D,B) to (B,D) conversion, contract separation, source inventory, HRA checker scope, replacement MVG, and S07-S14 gates agree with canonical S02/S04/S05 and direct parent evidence. Its direct Stage3 constants-only/no-wrapper choice is correct; however, its treatment of outer nohup redirection as an unmodified §5 example conflicts with §11 and the task's explicit path reconciliation, resolved canonically below.
EVIDENCE_FOR_B=Agent B independently reaches the same Step6/FCCR/HRA/checker/MVG decisions and recommends direct trainer constants without a wrapper. It correctly demands exact Stage3 path verification at S11 and a hard blocker rather than wrapper fallback. Its outer-log claim is incomplete/underspecified, not evidence against direct entry; this Judge resolves the path explicitly.
PROBLEMS_A=No hard-gate failure. Candidate A's route paragraph says retain whatever outer redirection §5 specifies and implies it must not be altered; root §5's illustrative source-cwd `logs/_stage3_run.log` conflicts with root §11's results-only Stage3 artifacts. Use the task's explicit narrow path-only reconciliation, directing outer stdout/stderr into Iter31's results `logs/` subtree. A static plan cannot verify runtime output consumers or activation.
PROBLEMS_B=No hard-gate failure. Candidate B says the §5 outer redirection remains whatever §5 specifies and does not name the explicit results-root outer log. This otherwise coherent route needs the same narrow reconciliation. It also states future S11 must verify all actual output consumers; no present runtime-route proof is implied.

WHY_NOT_A=Not rejected: A provides an exhaustive classified file inventory and well-specified checker and replacement MVG. Its outer-log conflict is resolved by path-only reconciliation, with no change to cwd, executable, direct script, no-argument invocation, prescribed launcher, or protocol.
WHY_NOT_B=Not rejected: B's contract-aware algorithm, direct-entry choice, and S11 blocker boundary are supported. The outer-log omission is resolved in the canonical route; no model/evaluator change or wrapper is warranted by current evidence.

MERGE_COMPONENTS_A=Use Agent A's selective copy/omit inventory; exact HRA static checker authorizations and requirements; explicit root §6 loss/backward/component-gradient checks; same-model/state/checkpoint/batch Step6-versus-Euclidean replacement MVG; separate S07-S14 gate list; Iter31 descriptive mechanism identity; and the requirement that execution is not implied.
MERGE_COMPONENTS_B=Use Agent B's clear L0/L1/L2 source/target curvature and helper call roles; direct Stage3 trainer with constants-only route and no wrapper; actual-consumer check at S11 and hard-blocker rather than fallback; exact short result-root interpretation. Normalize outer nohup output to the results-root path required by §11 and the task.

CANONICAL_DECISION=MERGE_AB. Approve `logs/implementation_plan_iter31.md` as the sole canonical S06 plan. The one model-behavior delta is HRA-STEP6-1 in Iter31 `modules/rqvae.py::RqVae._step6_sum_embeddings`; all other listed work is classified as path/identity wiring or audit tooling. The FCCR-1 ten-field `mechanism_contract_iter31.json` and fixed vector remain unchanged and separate from `hra_step6_contract_iter31.json`; the copied loader must continue requiring `branching` and `raw_residual_medians`, recomputing the fixed mapping, and enforcing the exact schema. Primary evidence: Iter29 `modules/rqvae.py:14–20,290–326,354–365` (helper imports, `(L,D,B)` tensor creation, current sum and decoder consumer); `modules/hyperbolic.py:9–28,66–73` (projection/clamp/denominator semantics); `curvature_RQ-VAE.py:305–382` (loader inputs and exact ten-field validation); Iter31 canonical contracts as cited in the plan. For Stage3, primary `stage3_T5Train/train_HG-Rec.py:121–159,215–258,827–849,1143–1154,1175–1212` supports constants-only SID/output/launcher wiring as a feasible design: absolute `CODE_PATH` is accepted, training/test outputs derive from `LOG_PATH`/`SAVE_PATH`, and child output derives from `_LAUNCHER['log']`; this is not a full proof of actual run destinations. Retain root §5's cwd, absolute Python 3.10 executable, direct `train_HG-Rec.py`, no CLI, nohup, and prescribed launcher; direct outer stdout/stderr to `/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter31/logs/_stage3_run.log`, because root §11 requires run outputs only under that results root. Internal `_LAUNCHER['log']` must be `/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter31/_stage3_launcher.log`; `CODE_PATH` must be exactly Iter31's `results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/item_sids.json`; `RQVAE_VARIANT` is `iter31_hra_step6_common_reference`; `LOG_PATH` and `SAVE_PATH` use the short Iter31 results root. This is a narrow outer-log path reconciliation, not a wrapper or protocol/model/evaluator edit. S11 alone verifies the applied full route/output behavior and authorizes one Stage3 run.

The exact Step6 implementation must receive `(3,D,B)`, transpose each layer to `(B,D)` before helpers (which operate on final dimension), use `q_l=exp0^(c_l)(e_l)`, `q_l^0=exp0^(c0)(log0^(c_l)(q_l))`, then `q_0^0 ⊕_{c0}(q_1^0 ⊕_{c0} q_2^0)`, and return `log0^(c0)(h)` with `(B,D)` shape and finiteness validation. No telescope, paper-equivalence, helper changes, losses, d-HSTE, reorder/reassociation or other scientific delta.

Plan inventory excludes Iter29 wrapper, old logs/contracts, generated artifacts/checkpoints/SIDs; retains unchanged copied module/data/init sources and frozen input JSON; identifies all edited/copy targets, one new checker, and direct Stage3 trainer. Approved HRA checker is authorized for creation exactly once in implementation; checker execution remains S07-only, separate from the unchanged FCCR preflight. S08 replaces Iter29's alternate-curvature counterfactual with a same fixed-curvature/state/checkpoint/batch Step6-vs-Euclidean comparison, finite/domain/clipping/curvature invariance checks, and intended model gradient checks. Root §6 actual `loss.backward()` and relevant component gradient requirements remain mandatory before Stage2. Stage2 `RQVAE_OUT_DIR` exact short-root and iter number must be directly verified under S09 before launch.

No implementation or runtime evidence is established by these plans. S07 static/preflight, S08 MVG, S09 Stage2 launch, S10 Stage2 analysis, S11 Stage3 wiring/evaluation and one-run gate, S12 result classification, S13 Git closure, and triggered S14 Global Review remain separate required stages. Any S11 route blocker must be recorded; no wrapper fallback. S06 authorizes neither checker/MVG execution nor Stage2/Stage3.

CANONICAL_ARTIFACT=stage2_RQ-VAE/curvature_RQ-VAE_iter31/logs/implementation_plan_iter31.md; stage2_RQ-VAE/curvature_RQ-VAE_iter31/logs/deliberation/S06_IMPLEMENTATION/round_1/judge.md
CONFIDENCE=HIGH for registered design/contract boundary and current source semantics; MEDIUM for eventual applied route, runtime activation and domain pending S07/S08/S11
USER_INPUT_REQUIRED=NO
ITERATION_PURPOSE=PERFORMANCE_SEEKING_MECHANISM
SWEEP_OR_REPLICATION_ITERATION=NO
ROOT_CAUSE_ITERATION=NO
AUTONOMOUS_NEXT_ACTION=Materialize both canonical S06 artifacts; then proceed to one-time application of this approved plan only. Do not run checker, MVG, Stage2 or Stage3 until their separate canonical gates and authorizations.
REPLAN_CONSTRAINTS=No HRA/FCCR schema merge or FCCR loader/preflight bypass; no change to registered equation/constants, helper operations, model loss/optimizer/Sinkhorn/decoder, data, Stage1, Stage3 model/evaluator/protocol, or additional mechanism. No Iter29 wrapper. Do not leave the outer Stage3 run log in source-tree `logs/`; use the mandated Iter31 Stage3 results root. Do not assert Stage3 route verification before S11. No execution authorization is conferred here.

S06_PLAN_APPROVAL=APPROVED
HRA_CHECKER_CREATION_AUTHORIZATION=APPROVED_ONCE; EXECUTION=NOT_AUTHORIZED
MVG_AUTHORIZATION=NOT_AUTHORIZED (S08 required)
STAGE2_AUTHORIZATION=NOT_AUTHORIZED (S09 required)
STAGE3_AUTHORIZATION=NOT_AUTHORIZED (S11 required)
```
