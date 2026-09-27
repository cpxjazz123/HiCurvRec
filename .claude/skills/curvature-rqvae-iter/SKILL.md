---
name: curvature-rqvae-iter
description: Controlled research workflow for HiCurvRec curvature-aware RQ-VAE experiments. Scientific decisions use one canonical Research Agent; deterministic checks and operational repairs use a fast repair path without duplicate independent-agent review. Independent deterministic tasks execute in parallel unless a real dependency, shared-state write conflict, or resource constraint requires serialization. GPU-heavy Stage2/Stage3 execution occurs once after adjudication. The current research contract is FCCR-1 fixed closed-form curvature.
---

# curvature-rqvae-iter

## 0. Purpose

This skill exists to produce **causally interpretable curvature experiments**, not an endless sequence of score tweaks.

Every iteration must answer one clean question.

**Non-negotiable feasibility rule:** if direct evidence shows that the registered iteration is clearly infeasible, operationally inactive, mathematically invalid, numerically unsustainable, or can only be rescued by changing the registered mechanism/protocol, the orchestrator must terminate the iteration autonomously with `ABORT_ITERATION`. It must not wait for the user to decide whether to continue, must not spend a full Stage2/Stage3 run on a known-infeasible specification, and must not retune a locked mechanism inside the same iteration. Any revised mechanism starts as a new iteration.

This rule has priority over pipeline-completeness pressure: **a clean early abort is preferable to an expensive invalid run.**

**Non-negotiable autonomous-execution rule:** within every research workflow governed by this skill, **no state may require the user to decide the next step**. No Agent, Judge, or orchestrator may pause, stop, wait, or terminate merely to ask the user which option to choose, whether to continue, which parameter/mechanism to try, whether to change direction, or whether to launch the next valid stage.

All such decisions must be made autonomously from repository evidence, the active contract, the single-agent canonical decision protocol, hard gates, one-factor rules, and the project objective.

There is no valid workflow state named or equivalent to:

```
ASK_USER
WAIT_FOR_USER
NEED_USER_DECISION
PAUSE_FOR_DIRECTION
CONFIRM_NEXT_STEP
CONFIRM_NEXT_ITERATION
```

When multiple choices exist, the Research Agent evaluates them against primary evidence and hard gates and records one canonical action. When no option is valid, use the replan/abort rules. When the current iteration is infeasible, abort it cleanly and autonomously open the next justified iteration. When an external hard blocker makes execution impossible, record `EXTERNAL_BLOCKER` and terminate that path cleanly; do **not** turn the blocker into an open-ended request for user direction.

**Never use user consultation as a substitute for evidence-based decision making.** Uncertainty, low confidence, conflicting evidence, multiple plausible mechanisms, failed MVG, parameter choice, negative evidence, or research-direction choice are internal decisions for the Research Agent under the locked contract and hard gates.

## 0.1 Mandatory parallel execution

**Parallelism is the default, not an optimization.** Before executing any stage, the orchestrator must identify independent work items and launch all work that has no data dependency, shared-state write conflict, or exclusive-resource conflict concurrently.

Required parallel behavior:

- Do not spawn duplicate independent research agents for the same scientific decision. Each stage has one canonical Research Agent decision path.
- Independent repository reads, searches, file fetches, provenance lookups, and static inspections must be batched/concurrent.
- Independent read-only checker commands and preflight checks must run concurrently when they do not mutate shared files, consume the same exclusive GPU resource, or depend on each other's output.
- Independent post-run analyses of already materialized artifacts must run concurrently.
- Independent artifact writes to different paths may run concurrently when no shared mutable state is touched.
- A scientific decision may be finalized only after all required deterministic evidence for that decision has completed.
- Stage2 and Stage3 remain single canonical executions and are not duplicated for parallelism; downstream stages that depend on their outputs wait for those outputs.
- Mutations to the same source file, Git ref, checkpoint, shared log, or other shared mutable resource must be serialized.
- GPU tasks that compete for the same reserved GPU/memory budget may be serialized; this is a resource dependency, not permission to serialize unrelated CPU/read-only work.

If two tasks are executed serially despite being apparently independent, the orchestrator must record:

```text
SERIALIZATION_REASON=<actual dependency/write conflict/resource constraint>
```

"Easier to implement", "for caution", "to keep order", or "because this is how previous iterations ran" are not valid serialization reasons.

For each stage packet, record a compact execution DAG or equivalent grouping:

```text
PARALLEL_GROUP_1=<independent tasks launched together>
PARALLEL_GROUP_2=<tasks unblocked by group 1>
SERIAL_DEPENDENCIES=<only true dependencies>
```

The goal is to minimize wall-clock latency without weakening scientific gates.
**Non-negotiable forward-progress rule:** every new iteration must test **one forward-looking structural mechanism intended to improve downstream performance**. An iteration may not exist primarily to measure uncertainty, reproduce a previous result, search parameters, compare seeds, perform ablations, identify why a prior run behaved as it did, or isolate the root cause of a small delta. The iteration budget is reserved for mechanisms with a plausible path to materially improve the target metric.

The following are forbidden as the primary purpose or mechanism of any iteration:

- parameter sweep, grid search, random search, Bayesian search, or trying several values/ranges of the same mechanism;
- different-seed / multi-seed / matched-seed replication;
- rerunning the same mechanism only to estimate variance or noise;
- hyperparameter sensitivity studies;
- ablation-only iterations whose purpose is to identify which component caused an earlier result;
- reverse-mapping / counterfactual / control iterations whose main purpose is causal attribution rather than improvement;
- root-cause investigations of why a previous mechanism succeeded, failed, or produced a small delta;
- diagnostic iterations whose output is explanation rather than a new performance-seeking mechanism.

A small, neutral, or ambiguous delta is recorded as such and the workflow **moves forward to a distinct structural mechanism**. Do not spend a new iteration proving whether the small delta was noise.

Diagnostics are allowed only as lightweight preflight/MVG checks needed to verify that the newly registered mechanism is active, valid, and executable. They must not become a performance sweep, seed study, parameter search, or substitute for the registered end-to-end experiment.



Every non-aborted iteration must answer one clean question:

[
	ext{hypothesis}
ightarrow
	ext{one controlled mechanism}
ightarrow
	ext{contract verification}
ightarrow
	ext{Stage2}
ightarrow
	ext{Stage3 evidence}
]

The project-level target remains downstream `test_R@10 > 0.065`, but promotion failure and mechanism failure are not the same thing.

---

# 1. Source of truth

Read current repository rules before every iteration.

Priority:

1. repository root `CLAUDE.md`;
2. this skill;
3. current iteration's machine-readable mechanism contract;
4. current protocol manifest;
5. iteration-local hypothesis/audit files;
6. historical reference files.

If a lower-priority source conflicts with a higher-priority source, the higher-priority source wins.

Historical files never override an actual protocol-compatible `test_final.json`.

---

# 2. Single-Agent Canonical Decision Protocol

Every stage that makes a **scientific choice, interpretation, mechanism decision, contract decision, causal classification, or go/no-go judgment** uses one canonical Research Agent. Do not create parallel independent candidate agents for the same decision.

```
canonical source packet
        ↓
single Research Agent
        ↓
ACCEPT | REPAIR_AND_RERUN | ABORT_ITERATION
        ↓
canonical artifact
        ↓
deterministic gate/checker where applicable
        ↓
next pipeline stage
```

This is the top-level protocol for scientific decisions. Deterministic repository reads, checker commands, provenance collection, and post-run analyses may still execute concurrently when they are independent. Parallelism applies to **work items**, not to duplicate scientific decision makers.

Historical completed rounds that already contain `agent_a.md`, `agent_b.md`, and `judge.md` remain valid audit evidence. They are legacy format only. New or reopened stages must use the single-agent format below.

## 2.1 Research Agent rules

The Research Agent must:

- receive the frozen source packet and exact stage objective;
- use primary repository evidence rather than trusting historical summaries when exact evidence exists;
- state assumptions, evidence, proposed action, risks, and self-rejection conditions;
- apply the active research contract, protocol lock, one-factor rule, and hard gates before making a decision;
- write one stage record only; do not generate competing candidate drafts;
- never silently modify a locked mechanism or Stage3 protocol to rescue a result;
- never directly create duplicate Stage2 or Stage3 executions.

Each normal stage agent artifact must begin with:

```
ROLE=RESEARCH_AGENT
SOURCE_PACKET=<path>
STAGE_ID=<stage id>
```

## 2.2 Canonical decision rules

The stage decision must be one of:

```
ACCEPT
REPAIR_AND_RERUN
ABORT_ITERATION
```

`ACCEPT` means the current canonical stage output passes the applicable hard gates.

`REPAIR_AND_RERUN` is mandatory when the registered scientific mechanism remains viable but the stage failed because of a repairable implementation, checker, logging, provenance-capture, serialization, routing, cleanup, or other operational defect. The repair must preserve the registered hypothesis, mechanism equation, key constants, parent, protocol, one-factor delta, seed policy, data identity, and evaluation definition.

`ABORT_ITERATION` is mandatory only when direct evidence shows that the **registered iteration itself is no longer scientifically or operationally viable** and continuing would require changing a locked scientific factor.

When the current proposal is unsupported but the iteration remains potentially viable, the same Research Agent may open a new numbered scientific-decision round with explicit `REPLAN_CONSTRAINTS`. Maximum automatic research-proposal rounds per stage: **2**. Operational repairs do not consume a research-proposal round.

## 2.3 Hard gates and rubric

Hard gates are evaluated before qualitative preference:

- active contract compliance;
- protocol compatibility;
- semantic/provenance correctness;
- one-factor causal isolation;
- falsifiability;
- reproducibility / sufficient implementation specificity;
- no unsupported factual claims;
- iteration purpose is a forward performance-seeking structural mechanism, not sweep/replication/root-cause analysis.

A stage that fails a hard gate cannot be accepted merely because its narrative is persuasive.

For a hard-gate-valid stage, the Research Agent considers scientific rationale, direct prior evidence, causal interpretability, implementation clarity, hidden-confound risk, expected information gain, and cost proportionality.

## 2.4 Canonical-only propagation

After the stage decision:

- exactly one canonical artifact is written to the normal pipeline path;
- downstream stages read only the canonical artifact and `decision.md`;
- scratch notes and superseded drafts must not become active instructions;
- a stage is not complete until the canonical artifact actually exists.

## 2.5A Fast operational repair — no duplicate research-agent review

A repair that does **not** change the registered scientific experiment stays inside the same stage and does not trigger duplicate research-agent review.

This path applies to localized operational defects such as checker/parser/AST bugs, missing logging fields, wrong paths, malformed serialization, missing hashes/device/batch-ID instrumentation, cleanup/process-exit defects, or source code that fails to implement the already accepted equation exactly.

Fast-repair procedure:

1. Record the exact repair scope as `REPAIR_AND_RERUN`.
2. Scan the entire failing deterministic surface once and collect all independently detectable operational defects before editing.
3. Independent repair edits to different files may be prepared concurrently; conflicting shared-state edits are serialized.
4. Apply the minimal repair batch.
5. Run all independent deterministic checks for that stage in parallel and capture stdout/stderr/exit status plus required provenance.
6. Write `repair_record.md` containing `ROUND_TYPE=OPERATIONAL_REPAIR`, `LOCKED_SCIENCE_CHANGED=NO`, exact changes, commands/checks, outputs/provenance, and `CHECKS_PASS=YES|NO`.
7. The Research Agent reviews the repair record and writes the canonical `decision.md`.
8. If the repair passes, continue the same iteration. If it reveals true mechanism infeasibility requiring a locked scientific change, use `ABORT_ITERATION`.

A fast-repair rerun replaces an invalid/incomplete verification attempt. It is not a seed replication, performance replication, sweep, ablation, or new iteration.

### Consolidated deterministic checks

For S07/S08, deterministic evidence must be gathered before scientific interpretation whenever possible:

```
freeze source / contract
        ↓
identify all deterministic checks
        ↓
run independent checks concurrently
        ↓
collect all failures
        ↓
one consolidated operational repair batch if needed
        ↓
rerun failed/affected checks concurrently
        ↓
single Research Agent decision
```

Do not create separate research agents to discover deterministic facts that executable checkers can establish directly.

### Pre-run provenance completeness for MVG

Before the canonical MVG invocation, the checker must already be instrumented to emit contemporaneously:

- exact input paths and pre-run hashes where required;
- checkpoint path and hash;
- seed and deterministic selection rule;
- actual ordered batch/sample/item IDs;
- batch shape/count;
- actual runtime device string and, when available, physical device identity;
- complete mechanism diagnostics;
- component-wise gradient evidence required by the active contract;
- stdout/stderr and exit status.

A missing field discovered after execution is an evidence-capture repair: invalidate that verification attempt for gate purposes, fix instrumentation, and rerun S08 in the same iteration.

## 2.5 Side-effect / GPU-heavy stages

For stages that mutate shared code, launch jobs, write checkpoints, evaluate Stage3, commit, or push:

- the Research Agent produces one complete method / patch plan / wiring audit;
- the orchestrator applies the canonical method **once**;
- Stage2 and Stage3 full GPU runs are never duplicated;
- multi-seed / matched-seed replication is not an allowed iteration type under this skill.

No worktree may be used. This section does not override repository `CLAUDE.md` Git or no-Plan-Agent rules.

## 2.6 Stage-record artifact layout

New or reopened stages use:

```
logs/stage_records/<STAGE_ID>/round_<R>/
    source_packet.md
    agent.md
    decision.md
    repair_record.md   # only when applicable
```

`decision.md` must contain:

```
STAGE_ID=
ROUND=
VERDICT=ACCEPT | REPAIR_AND_RERUN | ABORT_ITERATION

HARD_GATE=PASS | FAIL
CANONICAL_DECISION=
CANONICAL_ARTIFACT=
CONFIDENCE=HIGH | MEDIUM | LOW
USER_INPUT_REQUIRED=NO
ITERATION_PURPOSE=PERFORMANCE_SEEKING_MECHANISM
SWEEP_OR_REPLICATION_ITERATION=NO
ROOT_CAUSE_ITERATION=NO
AUTONOMOUS_NEXT_ACTION=

REPLAN_CONSTRAINTS=
```

For `REPAIR_AND_RERUN`, `SAME_ITERATION_REPAIR=AUTHORIZED` is mandatory.  
For `ABORT_ITERATION`, `ABORT_REASON`, `ABORT_EVIDENCE`, and `NEXT_ITERATION_CONSTRAINTS` are mandatory; `CANONICAL_ARTIFACT` must point to `logs/iteration_abort_iter<N>.md`.  
For every verdict, `USER_INPUT_REQUIRED=NO`, `ITERATION_PURPOSE=PERFORMANCE_SEEKING_MECHANISM`, `SWEEP_OR_REPLICATION_ITERATION=NO`, `ROOT_CAUSE_ITERATION=NO`, and a concrete `AUTONOMOUS_NEXT_ACTION` are mandatory.

The checker may accept legacy `logs/deliberation/.../agent_a.md + agent_b.md + judge.md` only for already materialized historical rounds. The legacy layout must not be generated for new work.

## 2.7 Stage map

| Stage ID | Pipeline stage | Single Research Agent responsibility | Canonical output |
|---|---|---|---|
| `S00_SOURCE_TRUTH` | Read rules / source of truth | extract constraints, current repo facts, unresolved conflicts | `logs/source_snapshot_iter<N>.md` |
| `S01_PROTOCOL_LOCK` | Baseline + protocol | reconstruct comparable protocol and baseline from primary files | `logs/protocol_manifest_iter<N>.md` |
| `S02_HYPOTHESIS` | Research hypothesis | register exact falsifiable hypothesis/equation under active contract | `logs/hypothesis_iter<N>.md` |
| `S03_PROVENANCE` | Semantic/provenance | trace every formula input and semantic definition | `logs/mechanism_manifest_iter<N>.md` |
| `S04_CONTRACT` | Mechanism contract | encode expected implementation invariants | `logs/mechanism_contract_iter<N>.json` |
| `S05_ONE_FACTOR` | One-factor diff | identify parent, inherited mechanisms, and exact delta | `logs/one_factor_diff_iter<N>.md` |
| `S06_IMPLEMENTATION` | Implementation design | produce complete patch plan/diff and tests | `logs/implementation_plan_iter<N>.md`; orchestrator applies once |
| `S07_PREFLIGHT` | Static/contract preflight | audit source against hypothesis + contract after deterministic checks | `logs/preflight_contract_iter<N>.log` + decision |
| `S08_MVG` | MVG | run/interpret lightweight mechanism verification | `logs/mvg_check_iter<N>.log` + decision |
| `S09_STAGE2_EXECUTION` | Stage2 run | audit launch command, inputs, outputs, invariants | `logs/stage2_execution_plan_iter<N>.md`; Stage2 runs once |
| `S10_STAGE2_ANALYSIS` | SID/geometry analysis | analyze the Stage2 outputs | `logs/sid_geometry_iter<N>.md` |
| `S11_STAGE3_EVALUATION` | Stage3 wiring/eval | audit SID wiring, checkpoint, frozen eval protocol and expected outputs | `logs/stage3_evaluation_plan_iter<N>.md`; Stage3 runs once |
| `S12_RESULT_CLASSIFICATION` | Causal/result interpretation | classify mechanism effect, confounds, promotion status | `logs/failure_attribution_iter<N>.md` + `logs/gate_decision_iter<N>.md` |
| `S13_GIT_CLOSURE` | Commit/push closure | audit required artifacts, paths, git state, remote hash | `logs/git_closure_iter<N>.md` |
| `S14_GLOBAL_REVIEW` | Direction selection when triggered | synthesize evidence and select next research direction | `logs/global_review_after_iter<N>.md` |

If a stage has multiple canonical files, `decision.md` must list all of them in `CANONICAL_ARTIFACT`.

## 2.8 Stage-specific decision emphasis

- **S00–S01 factual stages:** primary-source correctness and protocol comparability dominate.
- **S02 hypothesis:** contract compliance, falsifiability, information gain, and causal isolation dominate.
- **S03–S05 audit stages:** semantic exactness, provenance, and one-factor integrity dominate.
- **S06 implementation:** fidelity to the canonical hypothesis/contract and minimal diff dominate.
- **S07–S08 verification:** evidence beats intention. Run deterministic/read-only checks first and concurrently; consolidate operational defects into one repair batch.
- **S09/S11 execution:** reproducibility, exact wiring, frozen protocol, and no unintended protocol changes dominate.
- **S10/S12 interpretation:** separate observed facts from causal inference; do not overgeneralize a mapping failure into a family-level failure without evidence.
- **S13 closure:** repository truth and remote verification dominate.
- **S14 review:** compare only protocol-valid evidence and explicitly distinguish validated findings from unresolved hypotheses.

## 2.9 Forward-Only Iteration Policy — no sweep, replication, or root-cause iterations

Every iteration must answer:

> What **new structural mechanism** are we introducing that could materially improve downstream performance?

It must not answer:

> Was the last +Δ real?  
> Which seed is better?  
> Which constant is best?  
> Why did the last mechanism fail?  
> Which component caused the gain/loss?

### Forbidden iteration forms

```
SWEEP
PARAMETER_SEARCH
SEED_REPLICATION
MATCHED_SEED_COMPARISON
NOISE_ESTIMATION
HYPERPARAMETER_SENSITIVITY
ABLATION_ONLY
ROOT_CAUSE_ANALYSIS
MICRO_DELTA_ATTRIBUTION
REVERSE_CONTROL_ONLY
```

If a Global Review or previous artifact proposes one of these as the next iteration, that proposal is automatically superseded by this section. The single-agent canonical decision protocol must instead select a distinct performance-seeking structural mechanism.

### Parameter policy

A new mechanism may contain fixed constants, but they must be chosen **before registration** from theory, geometry, scale analysis, prior evidence, or a deterministic rule. Do not create an iteration whose scientific contribution is trying several values.

After S02/S04 lock, no in-iteration retuning is allowed. If the registered constant makes the mechanism infeasible, abort the iteration. The next iteration must not simply be another value of the same constant unless the changed value is inseparable from a genuinely new structural mechanism.

### Seed policy

Use the protocol-locked seed for comparability. Do not create extra runs solely with different seeds. A small effect that cannot be distinguished from historical noise is classified `ACTIVE_NEUTRAL`; the next iteration moves to a new structural mechanism instead of estimating variance.

### Root-cause policy

Failure attribution is descriptive bookkeeping only. It may state what is directly supported by evidence, but it must not launch a new iteration whose goal is to discover the cause of the previous result. Unknown causes may remain unresolved.

---

## 2.10 Feasibility Abort — self-terminate clearly infeasible iterations

The orchestrator must **end the current iteration without asking the user to rescue it** when direct evidence shows the registered mechanism cannot be meaningfully executed as specified.

This is a scientific hygiene rule, not performance early stopping.

### Mandatory abort conditions

Use `ABORT_ITERATION` when one or more of the following is directly demonstrated:

1. **Mechanism inactivity under the registered specification**  
   The preregistered mechanism produces effectively zero intervention / no measurable direct effect at MVG or during execution, and making it active would require changing a registered mechanism constant, equation, mapping, or threshold.

2. **Activation requires a qualitatively different regime**  
   The mechanism can only be made active by moving into saturation, near-always-on clipping/gating, unstable dynamics, or another regime that changes the scientific interpretation of the registered experiment.

3. **Unavoidable contract or one-factor violation**  
   Continuing requires adding a second mechanism, changing Stage1/Stage3, changing the parent/protocol, or otherwise violating the registered single-factor experiment.

4. **Mathematical / semantic impossibility discovered after registration**  
   A required quantity is undefined, degenerate, unavailable with the declared provenance, or the registered mapping cannot produce the claimed behavior on the actual data.

5. **Persistent mechanism-caused numerical invalidity**  
   NaN/Inf, divergence, invalid manifold state, corrupted assignment, or equivalent failure persists after one implementation-level repair that does not alter the registered scientific mechanism.

6. **Execution infeasibility that changes the experiment if worked around**  
   The registered computation cannot fit available resources / runtime constraints, and reducing it would materially change the mechanism or locked protocol.

7. **Direct evidence of destructive saturation before useful training**  
   A mechanism-specific direct signal is already saturated/degenerate to the point that the intended comparison is no longer meaningful.

### Not valid reasons to abort

Do **not** abort merely because:

- Gini, collision rate, entropy, utilization, or another Stage2 proxy looks worse;
- an active, numerically valid mechanism appears unlikely to beat the target;
- an intermediate loss is higher but finite and training is otherwise valid;
- a completed candidate has `R@10 < 0.065`;
- the result may be negative;
- a small implementation bug can be repaired while preserving the exact registered mechanism;
- a preflight/MVG run omitted batch IDs, runtime device, hashes, checkpoint identity, or other required audit fields;
- the first verification attempt must be invalidated and rerun after instrumentation is fixed;
- a prior incomplete verification observation cannot be historically reconstructed, when the same registered verification can simply be rerun faithfully.

### Repairable verification / evidence failures

The following are **not feasibility aborts** when the registered mechanism itself is still valid:

- missing or incomplete logging of batch IDs, ordered sample IDs, runtime device, checkpoint hash, input hashes, or environment metadata;
- a checker forgot to print or serialize an already-computable required field;
- a verification artifact is truncated, malformed, routed to the wrong path, or missing a non-scientific audit field;
- a small implementation bug prevents the code from faithfully executing the already registered equation;
- a preflight/MVG script needs instrumentation so that required evidence is captured contemporaneously.

Classify these as one of:

```text
IMPLEMENTATION_REPAIR_REQUIRED
CHECKER_REPAIR_REQUIRED
EVIDENCE_CAPTURE_REPAIR_REQUIRED
```

and the stage decision must record:

```text
VERDICT=REPAIR_AND_RERUN
SAME_ITERATION_REPAIR=AUTHORIZED
```

### Mandatory same-iteration repair procedure

1. Mark the defective verification attempt as **non-canonical for gate purposes**; do not reinterpret missing fields as observed facts.
2. Scan the full deterministic verification surface and collect all independently detectable operational defects before making the repair.
3. Patch only implementation/checker/logging/capture paths necessary to execute or document the **already registered** experiment.
4. Preserve the locked mechanism equation, constants, parent, one-factor delta, seed policy, data identity, checkpoint identity policy, Stage1/Stage3 protocol, and evaluation definition.
5. Before rerun, make the checker capture required provenance **contemporaneously**, including pre-run input hashes when required, actual ordered batch/sample IDs, selected checkpoint/hash, batch shape, and actual runtime device.
6. Rerun all independent affected checks **in parallel** in the same iteration; do not create duplicate research-agent review merely to confirm deterministic outputs.
7. Record the repair in a repair-only round with `repair_record.md`; the Research Agent may accept the repaired stage directly if `LOCKED_SCIENCE_CHANGED=NO`.
8. Only the first complete, contract-valid repaired verification run becomes canonical evidence for that stage.
9. Proceed normally if the repaired run passes. If it reveals true mechanism infeasibility that would require changing a locked scientific factor, then and only then return to the normal scientific-decision path and consider `ABORT_ITERATION`.

A repaired MVG rerun is **not** a forbidden replication or seed study because it does not estimate performance variance and does not create a second scientific condition. It replaces an invalid/incomplete verification attempt before Stage2.

### No artificial single-invocation rule

Do not declare an otherwise lightweight and repeatable preflight/MVG check to be permanently "single-invocation only" merely for audit purity. If its evidence capture is defective, invalidate that attempt, repair the capture path, and rerun the same verification stage. Historical reconstruction is unnecessary when the registered verification can be rerun faithfully under the locked protocol.

Missing runtime evidence is therefore an `EVIDENCE_CAPTURE_REPAIR_REQUIRED` condition, not `ITERATION_ABORTED_INFEASIBLE`, unless the missing fact is intrinsically unrecoverable **and** a faithful rerun would itself change the registered scientific experiment.

### Registered-spec immutability after hypothesis lock

After S02/S04 are canonically accepted, the following are part of the registered experiment:

- mechanism equation;
- key mechanism constants / thresholds;
- mapping direction;
- parent and one-factor delta;
- activation definition.

If feasibility testing shows one of those choices is bad, **do not retune it inside the same iteration**.

Example:

```
registered: trust_radius_fraction = 0.5
MVG: mechanism intervention ≈ 0
```

If changing `0.5 → 0.05` is needed to obtain meaningful activation, the current iteration must end as:

`ITERATION_ABORTED_INFEASIBLE`

and `0.05` must be registered as a **new iteration** with its own S00–S08 stage records.

Implementation repairs that merely make the code match the already registered equation do not require a new iteration.

### Abort procedure

At the stage where infeasibility becomes clear:

1. reopen that stage as the next scientific-decision round if needed;
2. the Research Agent assesses the abort evidence against the locked contract;
3. the Research Agent records `ABORT_ITERATION` when the infeasibility gate is met;
4. stop pending/full Stage2 or Stage3 jobs for this iteration;
5. do not launch later pipeline stages;
6. create `logs/iteration_abort_iter<N>.md` containing:
   - `STATUS=ITERATION_ABORTED_INFEASIBLE`;
   - exact stage and step where abort occurred;
   - direct evidence;
   - why a same-iteration repair would alter the registered experiment;
   - what remains scientifically unresolved;
   - constraints for the next iteration;
7. commit and push the mechanism code and all audit/abort evidence;
8. start any revised mechanism as a new iteration from the last appropriate valid parent.

An aborted iteration:

- is **not** a clean scientific negative;
- does **not** count toward the 3-clean-iteration Global Review trigger;
- does **not** require Stage2/Stage3 completion;
- must not be used as evidence that the broader mechanism family failed.

---

## 2.11 Autonomous Continuation — user-decision states are forbidden

This rule applies to every new stage and iteration governed by this skill.

At every decision point:

1. The Research Agent assesses the available actions from primary evidence.
2. The Research Agent applies the hard gates and evidence rubric.
3. The Research Agent must output exactly one concrete `AUTONOMOUS_NEXT_ACTION`.
4. The orchestrator executes that action without asking the user for a preference.
5. If the current iteration is infeasible, use `ABORT_ITERATION`, close it cleanly, and autonomously register the next justified iteration.
6. If accumulated evidence warrants changing the research contract, perform the transition **between iterations**, document it through the single-agent canonical decision protocol, and continue autonomously.
7. If confidence is low, record `CONFIDENCE=LOW`; low confidence is not a reason to pause.
8. If an external hard blocker prevents execution, record `EXTERNAL_BLOCKER`, preserve all evidence, and terminate that blocked path. Do not ask the user to choose an alternative path; the Research Agent selects the best available unblocked action, or closes the workflow if none exists.

Forbidden output/actions include:

- "Which option do you want?"
- "Should I continue?"
- "Do you want me to try X?"
- "Which parameter should I use?"
- "What should the next iteration be?"
- any equivalent pause awaiting user preference.

A workflow is invalid if it reaches a discretionary state requiring user choice.

---

# 3. CURRENT RESEARCH CONTRACT — FCCR-1

Until a higher-priority instruction or a canonically adjudicated **between-iteration autonomous contract transition** changes the research hypothesis, the active contract is:

`FCCR-1 = Fixed Closed-Form Curvature Research Contract`

The scientific claim under test is:

[
(B_l,;m_l^{raw})
ightarrow
f(cdot)
ightarrow
c_l
]

where:

- (B_l) = behavior branching / effective branching statistic for RQ layer (l);
- (m_l^{raw}) = **raw residual magnitude**, not normalized layer scale;
- (c_l) = final per-layer curvature.

## 3.1 Required curvature behavior

For FCCR-1:

```
CURVATURE_SOURCE=closed_form
CURVATURE_TRAINABLE=false
CURVATURE_TIME_VARYING=false
USES_CYCLIC_SCHEDULE=false
USES_CURVATURE_REGULARIZATION=false
NEW_CURVATURE_CONDITIONED_OPTIMIZER=false
NEW_CURVATURE_CONDITIONED_AUX_LOSS=false
```

The three final curvature values must be computed **before Stage2 training** and remain unchanged for the entire run.

Allowed implementation pattern:

```python
self.register_buffer("fixed_c", torch.tensor(c_l, dtype=torch.float32))
```

`get_c()` may return `fixed_c`, but must not depend on training step, optimizer state, learnable curvature parameters, or a curriculum.

## 3.2 Forbidden under FCCR-1

Unless the contract is changed by a higher-priority instruction or a canonically adjudicated **between-iteration autonomous contract transition**, do not introduce:

- `nn.Parameter` for curvature, `c_layer_scale`, `log_c`, or equivalent;
- learned curvature priors;
- cyclic / scheduled `c(t)`;
- curvature regularization whose purpose is to pull a trainable curvature toward a prior;
- optimizer-side curvature learning such as curvature-conditioned LR or beta2 as the **new mechanism**;
- a second new Sinkhorn / behavior / auxiliary-loss mechanism in the same iteration;
- any mechanism whose purpose is to modify how curvature is learned, because FCCR-1 curvature is not learned.

Existing downstream computations may **consume the fixed curvature** (e.g. Poincaré distance or a baseline quantization path) if those computations are held identical between parent and candidate. They must not change the curvature itself.

## 3.3 What counts as a clean FCCR-1 iteration

The only conceptual change should be the closed-form mapping:

[
(B_l,m_l^{raw})ightarrow[c_0,c_1,c_2]
]

or one clearly specified component of that mapping.

Do not simultaneously add a new optimizer, behavior loss, Sinkhorn rule, manifold, or Stage3 change.

---

# 4. Protocol Lock

Before mechanism implementation, create:

`logs/protocol_manifest_iter<N>.md`

It must record:

```
PROTOCOL_ID
dataset/version
Stage1 embedding path + immutable hash/id
parent iteration + commit
canonical baseline iteration
canonical baseline test_final.json path
canonical baseline test_R@10
Stage2 seed
Stage2 max steps
RQ layers / codebook size
Stage3 code commit
Stage3 seed
Stage3 epochs
beam size
n_eval
```

Two runs may be directly ranked only if their protocol manifests are compatible.

If protocol compatibility is uncertain, mark the historical result:

`HISTORICAL_NONCOMPARABLE`

and do not call it the current best.

If repository records disagree about a baseline number, read the exact `test_final.json` for the declared baseline and document the discrepancy.

---

# 5. Canonical parent and one-factor rule

The previous iteration is **not automatically the parent**.

Every new iteration must declare:

```
PARENT_ITER=
PARENT_COMMIT=
CANONICAL_BASELINE_ITER=
EXPERIMENT_TYPE=single_factor
ACTIVE_MECHANISMS_BEFORE=
NEW_MECHANISM=
ACTIVE_MECHANISMS_AFTER=
```

For the current FCCR-1 core-hypothesis phase, interaction/stacking experiments are disabled.

A failed exploratory mechanism must never silently become the parent of the next experiment.

Create:

`logs/one_factor_diff_iter<N>.md`

It must list:

- changed source files;
- changed equations/constants;
- inherited mechanisms;
- optimizer differences;
- loss differences;
- Stage1/Stage3 differences;
- explicit statement that the only conceptual change is the registered mechanism.

If an unexplained second mechanism is present, stop before training.

---

# 6. Hypothesis registration

Create:

`logs/hypothesis_iter<N>.md`

It must contain:

## A. Research question

One sentence.

## B. Exact equation

Write the full mapping from (B_l,m_l^{raw}) to final (c_l).

## C. Actual numeric substitution

Before training, substitute real values and print:

```
B = [...]
raw_residual = [...]
intermediate_terms = [...]
fixed_curvature = [c0, c1, c2]
```

## D. Direct effects

For FCCR-1, direct effects include:

- final fixed curvature values equal the preregistered numbers;
- curvature is non-trainable;
- curvature is invariant to step;
- curvature is invariant to optimizer updates;
- curvature is identical in train/eval mode.

## E. Downstream rationale

Explain:

[
	ext{branching/residual structure}
ightarrow
	ext{fixed geometry}
ightarrow
	ext{quantization behavior}
ightarrow
	ext{why Stage3 could benefit}
]

## F. Falsification

State observations that would invalidate the mechanism implementation or the scientific hypothesis.

---

# 7. Semantic / Provenance Gate

Create:

`logs/mechanism_manifest_iter<N>.md`

For every formula input record:

| Field | Required |
|---|---|
| symbol | e.g. (B_l,m_l) |
| semantic meaning | exact definition |
| source file/script | provenance |
| source iteration | producing run |
| raw value | before transformation |
| transformation | log/normalize/clip/etc. |
| transformed value | actual formula input |
| expected range | sanity bound |

For FCCR-1, the manifest must explicitly distinguish:

```
raw_residual_median
!=
normalized_layer_scale
!=
learnable_c_layer_scale
```

Hard FAIL if:

- raw residual is replaced by normalized layer scale;
- provenance is unknown;
- a historical fallback is used without explicit justification;
- the formula name implies one quantity while code supplies another;
- actual substituted numbers differ from preregistration.

A provenance FAIL blocks MVG and Stage2.

---

# 8. Machine-readable Mechanism Contract Gate

This gate exists to prevent an experiment from implementing a different parameterization than the scientific hypothesis.

Create:

`logs/mechanism_contract_iter<N>.json`

Required FCCR-1 schema:

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
  "final_curvature_values": [0.0, 0.0, 0.0]
}
```

Replace the curvature values with the actual preregistered values.

Before MVG, run from the iteration directory:

```bash
python /home/wlia0047/ar57/wenyu/GeneRec/.claude/skills/curvature-rqvae-iter/scripts/preflight_contract.py
```

No CLI arguments are allowed.

Expected output:

`MECHANISM_CONTRACT_PASS`

The preflight must block Stage2 if it detects any of the following:

- missing mandatory preflight files;
- trainable curvature parameter;
- `get_c()` depends on curriculum step / time schedule;
- missing fixed curvature buffer;
- nonzero curvature regularization in the active loss;
- contract JSON contradicts the source implementation.

A textual statement such as “fixed curvature” is not enough. The code must satisfy the contract.

---

# 9. MVG — contract-aware mechanism verification

MVG verifies implementation. It does not decide whether the scientific idea is good.

The old rule “every new mechanism parameter must have a gradient and update” is **not valid for fixed curvature**.

## 9.1 FCCR-1 MVG

For fixed closed-form curvature, MVG must verify:

### Layer A — Formula

- code-computed (c_l) matches preregistered values;
- values are finite and valid for the manifold.

### Layer B — Immutability

- curvature tensors have `requires_grad=False`;
- curvature tensors are not in optimizer parameter groups;
- no trainable curvature parameter exists;
- after multiple optimizer steps, every (c_l) is unchanged within numerical tolerance.

### Layer C — Time invariance

Evaluate `get_c()` at multiple training steps, e.g. 0 / 25k / 50k / 100k.

Require:

[
c_l(0)=c_l(25k)=c_l(50k)=c_l(100k)
]

within tolerance.

### Layer D — Model gradient health

The **rest of the model** must still train:

- total loss requires grad;
- intended RQ-VAE/model parameters receive finite nonzero gradients;
- no accidental detach was introduced by fixed-curvature implementation.

### Layer E — Counterfactual mechanism activation

Using the same checkpoint/batch, compare:

- candidate fixed closed-form curvature;
- declared baseline curvature configuration.

Require the registered direct output (distance/assignment/loss or another preregistered signal) to differ measurably.

This proves the fixed curvature affects the system without making curvature trainable.

## 9.2 MVG interpretation

- `MVG PASS` = implementation matches the mechanism contract, required evidence was captured, and the mechanism affects computation;
- `MVG IMPLEMENTATION FAIL` = the checker/model implementation does not yet faithfully execute the registered mechanism; if repairable without changing locked science, issue `REPAIR_AND_RERUN`;
- `MVG EVIDENCE CAPTURE FAIL` = the mechanism may be numerically valid, but required runtime/provenance evidence was not captured; issue `REPAIR_AND_RERUN`, fix instrumentation, and rerun S08 in the same iteration;
- `MVG MECHANISM FAIL` = direct evidence shows the registered mechanism itself is inactive/invalid and making it viable would require changing a locked scientific factor; only this class may support `ABORT_ITERATION`;
- `MVG PASS` does not imply better Stage3 performance.

If MVG was written for a learnable-curvature mechanism and checks curvature gradient/update, it is incompatible with FCCR-1 and must be rewritten before use.

---

# 10. Stage2 policy

Run Stage2 according to current `CLAUDE.md`.

Stage2 descriptive metrics are **not performance gates**:

- Gini;
- collision rate;
- unique SID count;
- per-layer utilization;
- (H(L1|L0));
- (H(L2|L0));
- coarse/fine ratios.

They describe what happened; they do not replace Stage3.

A healthy run should not be stopped because a proxy looks worse.

Early termination is reserved for actual invalid execution **or a 2.9 Feasibility Abort**:

- crash;
- NaN/Inf;
- corrupted checkpoint/export;
- contract violation discovered during training;
- fixed curvature unexpectedly changes;
- registered mechanism is proven inactive and activation would require changing the registered spec;
- continuing would require a new mechanism/protocol regime.

When a Feasibility Abort condition is met, stop the current iteration rather than retuning it in place.

For FCCR-1, log fixed curvature at multiple checkpoints to prove invariance.

---

# 11. Stage2 geometry analysis

Create:

`logs/sid_geometry_iter<N>.md`

Separate:

1. **Contract compliance** — fixed curvature stayed fixed;
2. **mechanism direct effect** — geometry/assignment changed as preregistered;
3. **SID observations** — descriptive metrics;
4. **interpretation limit** — Stage2 proxies are not assumed to predict Stage3.

Do not call a run good because Stage2 metrics look cleaner.

---

# 12. Stage3 evaluation

Every numerically valid, contract-valid, **non-aborted** candidate proceeds to the unchanged Stage3 protocol. A candidate with a confirmed `ITERATION_ABORTED_INFEASIBLE` status does not proceed merely to satisfy pipeline completeness.

Record exact:

- run directory;
- checkpoint;
- n_eval;
- R@5;
- R@10;
- NDCG@5;
- NDCG@10;
- delta vs canonical baseline.

Only protocol-compatible results may be directly compared.

The adoption target remains:

[
test_R@10 > 0.065
]

---

# 13. Result classification

Separate scientific effect from promotion.

## Mechanism status

Choose one:

- `PROTOCOL_INVALID`
- `PROVENANCE_INVALID`
- `CONTRACT_INVALID`
- `IMPLEMENTATION_INVALID`
- `MECHANISM_INACTIVE`
- `ACTIVE_POSITIVE`
- `ACTIVE_NEUTRAL`
- `ACTIVE_NEGATIVE`
- `PIPELINE_INVALID`
- `ITERATION_ABORTED_INFEASIBLE`

A run where curvature was intended to be fixed but became learnable or time-varying is `CONTRACT_INVALID`, not a scientific failure of the closed-form hypothesis.

## Promotion status

Separately record:

- `PROMOTION_PASS`
- `PROMOTION_FAIL`

A mechanism can be `ACTIVE_POSITIVE + PROMOTION_FAIL`.

Do not use `R@10 < 0.065` alone to label a mechanism failed.

---

# 14. Small-delta / noise rule — move forward, do not replicate

Tiny one-run deltas are not discoveries.

If candidate-baseline difference is comparable to historical run noise:

- classify it as near-parity / `ACTIVE_NEUTRAL`;
- report the uncertainty honestly;
- **do not** create a multi-seed, matched-seed, repeated-run, or replication iteration;
- **do not** create a root-cause or attribution iteration;
- autonomously select a distinct structural mechanism with a plausible path to a materially larger downstream gain.

The purpose of the workflow is performance progress, not precise variance estimation.

---

# 15. Global Review

Run a Global Review after **3 clean protocol-valid iterations**, not merely after 3 iteration numbers.

Also trigger it when:

- a mechanism family repeatedly gives neutral/negative results;
- protocol/baseline inconsistency is discovered;
- a core hypothesis has still not received a clean test.

Output:

`logs/global_review_after_iter<N>.md`

Report:

- protocol-compatible results only;
- which hypotheses were cleanly tested;
- which were confounded/invalid;
- strongest evidence-supported **new structural performance mechanism**;
- paused directions;
- unresolved core hypothesis.

Global Review must never select replication, multi-seed comparison, parameter sweep, sensitivity testing, ablation-only work, or root-cause investigation as the next iteration. If the evidence is ambiguous, ambiguity is recorded and the Judge still selects the strongest distinct forward mechanism.

Invalid-contract runs do not count as evidence against the scientific hypothesis.

---

# 16. Current compatibility matrix

Under FCCR-1:

| Mechanism | Status |
|---|---|
| branching + raw residual → fixed (c_l) | **ACTIVE / REQUIRED** |
| alternative bounded closed-form mapping | **ALLOWED** as one-factor experiment |
| learnable (c_l) / `c_layer_scale` | **FORBIDDEN** |
| cyclic/scheduled (c(t)) | **FORBIDDEN** |
| curvature regularization | **FORBIDDEN** |
| curvature-conditioned LR / beta2 as new mechanism | **DEFERRED** |
| new curvature-dependent Sinkhorn rule | **DEFERRED** |
| new behavior-loss mechanism | **DEFERRED** |
| manifold replacement | **DEFERRED** |
| Stage1/Stage3 modification | **OUT OF SCOPE** unless scope is changed by a higher-priority instruction or a canonically adjudicated between-iteration autonomous contract transition |

“Deferred” means it may be studied later, after the fixed closed-form hypothesis has received a clean test or an evidence-backed between-iteration autonomous contract transition canonically activates it.

---

# 17. Required artifacts and stage-decision evidence

Before Stage2, all canonical artifacts must exist:

```
logs/source_snapshot_iter<N>.md
logs/protocol_manifest_iter<N>.md
logs/hypothesis_iter<N>.md
logs/mechanism_manifest_iter<N>.md
logs/mechanism_contract_iter<N>.json
logs/one_factor_diff_iter<N>.md
logs/implementation_plan_iter<N>.md
logs/preflight_contract_iter<N>.log
logs/mvg_check_iter<N>.log
logs/stage2_execution_plan_iter<N>.md
```

In addition, stages `S00_SOURCE_TRUTH` through `S09_STAGE2_EXECUTION` must each have a completed single-agent stage record with a final `ACCEPT` decision before the Stage2 full run is launched. Already materialized legacy A/B/Judge rounds may be accepted only for historical compatibility. If an earlier stage canonically returns `ABORT_ITERATION`, no later pre-Stage2 stage is required and Stage2 must not launch.

After Stage2/Stage3, add:

```
logs/sid_geometry_iter<N>.md
logs/stage3_evaluation_plan_iter<N>.md
logs/stage3_outcome_iter<N>.md
logs/failure_attribution_iter<N>.md
logs/gate_decision_iter<N>.md
logs/git_closure_iter<N>.md
```

The corresponding single-agent stage records for `S10_STAGE2_ANALYSIS` through `S13_GIT_CLOSURE` are also mandatory before the iteration is considered closed.

`S14_GLOBAL_REVIEW` stage record is mandatory only when the Global Review trigger fires.

For an aborted iteration, `logs/iteration_abort_iter<N>.md` plus the single-agent decision evidence for the aborting stage replace the requirement to manufacture downstream Stage2/Stage3 artifacts that were never validly run.

A rule written in this skill but not checked before launch/closure is not considered enforced.

Before Stage2, run the skill-level deliberation checker from the iteration directory:

```bash
/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 \
  /home/wlia0047/ar57/wenyu/GeneRec/.claude/skills/curvature-rqvae-iter/scripts/deliberation_gate.py
```

Expected output for a runnable iteration:

`DELIBERATION_GATE_PASS`

For a canonically aborted iteration, the valid terminal output is:

`DELIBERATION_ABORT_CONFIRMED`

and no Stage2 launch is allowed.

---

# 18. Iteration loop

Scientific-decision arrows below mean: **one Research Agent evaluates the frozen evidence → records one canonical decision → canonical artifact proceeds**. Deterministic verification/repair arrows use §2.5A fast repair. At any stage, a scientifically justified `ABORT_ITERATION` exits into the abort-closure path.

```
S00  Source-of-truth extraction
   ↓
S01  Resolve canonical baseline + Protocol Lock
   ↓
S02  Register FCCR-1 hypothesis
   ↓
S03  Semantic / Provenance Gate
   ↓
S04  Mechanism Contract JSON
   ↓
S05  One-Factor Diff
   ↓
S06  Single implementation plan → apply canonical patch once
   ↓
S07  Concurrent deterministic preflight/checkers → consolidated repair if needed → single Research Agent decision
      + preflight_contract.py → MECHANISM_CONTRACT_PASS
   ↓
S08  Pre-instrument provenance → canonical MVG → concurrent deterministic evidence collection → single Research Agent decision; abort only for true mechanism infeasibility
   ↓
S09  Single Stage2 launch/wiring audit
      + deliberation_gate.py → DELIBERATION_GATE_PASS
      + Stage2 full run ONCE
   ↓
S10  Single Stage2/SID analysis
   ↓
S11  Single Stage3 wiring/eval audit against the frozen protocol
      + Stage3 full evaluation ONCE
   ↓
S12  Single causal/result classification
   ↓
S13  Single Git/artifact closure audit
      + commit + push + remote hash verification
   ↓
S14  If triggered: single Global Review → select a NEW structural performance mechanism (never sweep/replication/root-cause)
```

A stage is not complete merely because the Research Agent states a decision. The decision must be supported by primary evidence, and the canonical artifact must actually be materialized at the path defined in the stage map.

---

# 19. Git / artifact closure

Follow current `CLAUDE.md` for exact artifact paths and Git rules.

A **normal** iteration is not closed until:

- Stage2 completed;
- Stage3 completed;
- mandatory audit files exist;
- mechanism code + Stage2 artifacts + Stage3 artifacts are committed;
- push to `origin/main` succeeds;
- local and remote main hashes match.

An **aborted** iteration is closed when:

- the Research Agent has recorded `ABORT_ITERATION` from direct evidence;
- `logs/iteration_abort_iter<N>.md` exists;
- no later invalid pipeline stage was launched;
- mechanism code + all evidence up to the abort point are committed;
- push to `origin/main` succeeds;
- local and remote main hashes match.

Do not fabricate missing Stage2/Stage3 outputs for an aborted iteration.

Before declaring closure, rerun the same deliberation checker from the iteration directory:

```bash
/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 \\
  /home/wlia0047/ar57/wenyu/GeneRec/.claude/skills/curvature-rqvae-iter/scripts/deliberation_gate.py
```

For a normal completed iteration, require `DELIBERATION_GATE_PASS` with `phase=CLOSURE` (or `phase=GLOBAL_REVIEW` when S14 is triggered).

For an aborted iteration, require `DELIBERATION_ABORT_CONFIRMED` with `phase=ABORTED`. The checker stops at the aborting stage; correctly unlaunched downstream stages are not required.

Never claim an iteration is complete before the applicable normal or aborted closure condition is met.

---

# 20. Deprecated guidance

The following older skill ideas are explicitly retired under FCCR-1:

- generic “learnable/cyclic curvature remains competitive, so keep trying it” guidance;
- generic candidate rotation across learnable/cyclic/optimizer/Sinkhorn/behavior mechanisms;
- MVG rules requiring the curvature parameter itself to receive gradient/update;
- using Stage2 proxy thresholds as a reason to skip Stage3;
- treating every run below 0.065 as `TRUE_MECHANISM_FAIL`;
- historical baseline numbers from incompatible protocols;
- chaining the immediately previous experimental mechanism by default.

Historical experiments may still be analyzed, but they do not define the active contract.

---

# 21. Anti-patterns

Never:

- say “fixed curvature” while implementing a trainable prior;
- use raw-residual terminology for a normalized layer scale;
- keep cyclic scheduling active in a fixed-curvature experiment;
- retain curvature regularization for a non-trainable curvature;
- require fixed curvature to have nonzero gradient;
- let an optimizer update curvature under FCCR-1;
- silently inherit behavior/Sinkhorn/optimizer mechanisms from a failed parent;
- compare incompatible protocol results;
- declare a core hypothesis failed from a contract-invalid run;
- launch Stage2 with missing mandatory preflight files;
- spawn duplicate independent research agents for the same scientific decision;
- accept a hard-gate-failing stage because its narrative is more persuasive;
- propagate scratch or superseded drafts as active instructions to the next stage;
- run duplicate full Stage2/Stage3 jobs;
- skip the closure-mode deliberation gate before marking an iteration complete;
- keep an iteration alive by retuning a registered mechanism constant after MVG proves the registered value infeasible;
- abort an otherwise viable iteration because a verification script omitted recoverable runtime/provenance fields;
- impose a "sole MVG invocation" rule that prevents repairing and rerunning a defective verification capture;
- treat a same-stage verification repair as a forbidden replication, seed study, or new scientific iteration;
- serialize independent repository reads, source audits, deterministic checkers, or post-run analyses without a real dependency/resource conflict;
- create duplicate research-agent review solely to confirm a checker/logging/path/provenance repair;
- discover deterministic checker defects one-at-a-time when they could have been collected in one static/concurrent preflight pass;
- spend a full Stage2/Stage3 run on a mechanism already proven inactive under its registered specification;
- ask the user to choose the next research action, parameter, mechanism, direction, or whether to continue;
- pause an authorized workflow waiting for user preference;
- treat low confidence or multiple plausible options as requiring user input;
- use an iteration for parameter sweep/grid/random search or trying several values of one mechanism;
- use an iteration for different-seed, matched-seed, or variance-estimation replication;
- use an iteration to identify the root cause of a previous gain/loss;
- use an iteration only for ablation, reverse-control, or micro-delta attribution;
- let Global Review choose replication/noise estimation instead of a new structural performance mechanism.

---

# 22. Success definition

A scientifically successful iteration leaves a defensible statement:

> Under protocol P and FCCR-1, closed-form (f(B_l,m_l^{raw})) produced fixed curvature ([c_0,c_1,c_2]). The curvature was verified non-trainable and time-invariant, the mechanism changed the intended quantization behavior, and downstream R@10 changed by Δ relative to the canonical baseline.

Only after such a clean run may the project conclude whether the fixed branching+raw-residual curvature hypothesis is supported, neutral, or negative.
