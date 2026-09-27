---
name: curvature-rqvae-iter
description: Controlled research workflow for HiCurvRec curvature-aware RQ-VAE experiments. Only irreducible scientific judgments use a Research Agent. Anything decidable by explicit protocol, equality, file/hash checks, or executable checkers is a deterministic workflow phase and must not become an Agent stage. Independent deterministic tasks execute in parallel unless a real dependency, shared-state write conflict, or resource constraint requires serialization. GPU-heavy Stage2/Stage3 execution occurs once after deterministic gates. The current research contract is FCCR-1 fixed closed-form curvature.
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

### New iteration directory copy

When creating a new iteration by copying a parent or template directory, immediately delete the copied destination's entire `logs/` directory before doing any iteration work. Do not delete or modify the parent/source directory's `logs/`. Recreate a fresh destination `logs/` directory and populate it only with records generated for the new iteration under the active phase model; do not carry forward prior-round deliberations, decisions, metrics, or closure claims. Complete this cleanup before starting the P01 Research Agent or writing any new iteration artifacts.

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

# 2. Agent-Minimal Workflow — programmatic facts are not Agent stages

**Core rule:** if a decision can be made from an explicit contract, exact equality, repository state, hashes, paths, process state, numeric validity, or a deterministic checker, it must be implemented as a programmatic gate/action. Do **not** create a Research Agent stage merely to restate or approve deterministic facts.

A Research Agent is used only when the output requires irreducible scientific judgment, such as selecting/formulating a mechanism, interpreting causal evidence, or choosing a new research direction from ambiguous evidence.

The workflow uses seven numbered phases; P07 is conditional:

| Phase | Type | Purpose | Research Agent? |
|---|---|---|---|
| `P01_RESEARCH_DESIGN` | scientific | hypothesis + mechanism semantics + one-factor design + falsification | **Yes, first** |
| `P02_BUILD_VERIFY` | deterministic/action | source/baseline/protocol/hash lock, then implement accepted design + preflight + provenance + MVG | **No** |
| `P03_STAGE2` | deterministic/action | Stage2 run once + automatic SID/geometry metrics | **No** |
| `P04_STAGE3` | deterministic/action | exact Stage3 protocol diff + Stage3 run once | **No** |
| `P05_RESULT_DECISION` | scientific | causal/result interpretation + promotion status + next action | **Yes** |
| `P06_CLOSURE` | deterministic/action | artifact completeness + commit/push + remote hash | **No** |
| `P07_GLOBAL_REVIEW` | scientific, conditional | select a new structural direction when review trigger fires | **Yes** |

A normal iteration therefore has **two mandatory Agent decisions** (`P01`, `P05`) plus an optional third (`P07`). Start P01 directly with its Research Agent; do not require source, baseline, protocol, or hash locks first. Perform those deterministic locks in P02 before implementation and any Stage2/Stage3 launch.

## 2.1 Research Agent rule

For `P01_RESEARCH_DESIGN`, `P05_RESULT_DECISION`, and conditional `P07_GLOBAL_REVIEW`, use one canonical Research Agent. The Agent must use primary evidence, apply the active contract and hard gates, state assumptions/risks/self-rejection conditions, and never override a deterministic gate.

If a deterministic gate says `FAIL`, the Agent cannot reinterpret it as `PASS`. The only valid responses are operational repair (when locked science is unchanged) or iteration abort (when a locked scientific factor must change).

## 2.2 Agent record layout

Only the three scientific phases use agent records:

```
logs/stage_records/<P01_RESEARCH_DESIGN|P05_RESULT_DECISION|P07_GLOBAL_REVIEW>/round_<R>/
    source_packet.md
    agent.md
    decision.md
```

`agent.md` begins with:

```
ROLE=RESEARCH_AGENT
SOURCE_PACKET=<path>
STAGE_ID=<phase id>
```

`decision.md` contains:

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

For `REPAIR_AND_RERUN`, `SAME_ITERATION_REPAIR=AUTHORIZED` is mandatory. For `ABORT_ITERATION`, record `ABORT_REASON`, `ABORT_EVIDENCE`, and `NEXT_ITERATION_CONSTRAINTS`.

Historical `logs/deliberation/Sxx/.../agent_a.md + agent_b.md + judge.md` directories remain audit evidence only. New or reopened work must not create them.

## 2.3 Deterministic phase rule

The following are explicitly **not Agent stages**:

- source-of-truth extraction and protocol-field collection;
- baseline lookup when source precedence is explicit;
- hashes, paths, exact config comparisons, and provenance field presence;
- one-factor source diff verification after the scientific design is locked;
- implementation conformance to the accepted mechanism contract;
- preflight and MVG pass/fail conditions that are explicitly defined;
- Stage2 launch wiring and output checks;
- Stage2 SID/geometry metric calculation;
- Stage3 config/wiring comparison against the locked protocol;
- Stage3 execution and test-artifact completeness;
- Git status, required-artifact existence, commit, push, and local/remote hash equality.

These may run concurrently when independent. They do not gate or delay P01; the Research Agent begins directly from the registered research question and available primary evidence.

## 2.4 Fast operational repair

A repair that does **not** change the registered scientific experiment stays inside the same phase and does not create another Agent decision. Checker/parser bugs, missing logging/provenance fields, wrong paths, malformed serialization, stale workers, cleanup failure, or code that does not faithfully implement the already accepted equation are operational repairs.

Scan the full deterministic surface once, apply the minimal repair batch, rerun affected checks, and continue when they pass. If repair would change locked science, stop the repair path and use the appropriate scientific Agent phase or `ABORT_ITERATION`.

## 2.5 Deterministic Stage3 protocol gate

Stage3 is an evaluator, not a research-design stage. Immediately before Stage3, run:

```bash
python /home/wlia0047/ar57/wenyu/GeneRec/.claude/skills/curvature-rqvae-iter/scripts/stage3_protocol_gate.py
```

Expected output: `STAGE3_PROTOCOL_PASS`.

The checker compares the authoritative Stage3 lock against both the shared Stage3 trainer and the current iteration launcher. It fails on training-protocol drift, including epochs, early-stop behavior, evaluation mode, test skipping, seed, beam size, disabled screen, trainer hash, SID wiring, or output paths.

The iteration launcher may change only iteration-specific wiring/metadata (`CODE_PATH`, `RQVAE_VARIANT`, `LOG_PATH`, `SAVE_PATH`, launcher metadata). It must not override Stage3 training hyperparameters. A mismatch is `PROTOCOL_GATE_FAIL`; no Agent may waive it.

## 2.6 Canonical phase outputs

| Phase | Required canonical evidence |
|---|---|
| `P01_RESEARCH_DESIGN` | Research Agent decision: `hypothesis_iter<N>.md`, `mechanism_manifest_iter<N>.md`, `mechanism_contract_iter<N>.json`, `one_factor_diff_iter<N>.md` |
| `P02_BUILD_VERIFY` | deterministic `source_snapshot_iter<N>.md` + `protocol_manifest_iter<N>.md` lock, source patch, `preflight_contract_iter<N>.log`, and `mvg_check_iter<N>.log` |
| `P03_STAGE2` | canonical Stage2 artifacts + `sid_geometry_iter<N>.md` |
| `P04_STAGE3` | `stage3_protocol_gate_iter<N>.log` + canonical Stage3 `test_final.json` + `stage3_outcome_iter<N>.md` |
| `P05_RESULT_DECISION` | `failure_attribution_iter<N>.md` + `gate_decision_iter<N>.md` |
| `P06_CLOSURE` | `git_closure_iter<N>.md` + remote hash verification |
| `P07_GLOBAL_REVIEW` | `global_review_after_iter<N>.md` when triggered |

New iterations do not require `implementation_plan_iter<N>.md`, `stage2_execution_plan_iter<N>.md`, or `stage3_evaluation_plan_iter<N>.md`.

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

After P01_RESEARCH_DESIGN is accepted, no in-iteration retuning is allowed. If the registered constant makes the mechanism infeasible, abort the iteration. The next iteration must not simply be another value of the same constant unless the changed value is inseparable from a genuinely new structural mechanism.

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

After P01_RESEARCH_DESIGN is canonically accepted, the following are part of the registered experiment:

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

and `0.05` must be registered as a **new iteration** and pass the new iteration's P01–P02 workflow.

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

# 4. P02 Execution Lock — deterministic, no Agent stage

P01 starts directly with the Research Agent; no source, baseline, protocol, or hash lock is a precondition. After P01 accepts the design, P02 must create `logs/source_snapshot_iter<N>.md` and `logs/protocol_manifest_iter<N>.md` before implementing the mechanism or launching any Stage2/Stage3 work. These records are P02 evidence, not a separate workflow phase.

Apply the source-priority rules mechanically. If same-priority authoritative sources conflict, emit `PROTOCOL_CONFLICT` and block execution rather than asking an Agent to rationalize the conflict.

The manifest must contain:

```text
PROTOCOL_ID=
DATASET_VERSION=
STAGE1_EMBEDDING_PATH=
STAGE1_EMBEDDING_SHA256=
PARENT_ITER=
PARENT_COMMIT=
CANONICAL_BASELINE_ITER=
CANONICAL_BASELINE_TEST_FINAL=
CANONICAL_BASELINE_TEST_R10=
STAGE2_SEED=
STAGE2_MAX_STEPS=
RQ_LAYERS=
CODEBOOK_SIZE=

STAGE3_TRAINER_PATH=stage3_T5Train/train_HG-Rec.py
STAGE3_TRAINER_SHA256=
STAGE3_SEED=
STAGE3_EPOCHS=
STAGE3_EARLY_STOP=DISABLED
STAGE3_NO_EVAL=
STAGE3_SKIP_TEST=
STAGE3_BEAM_SIZE=
STAGE3_SCREEN_BASELINE_LOG=
STAGE3_CODE_PATH=
STAGE3_LOG_PATH=
STAGE3_SAVE_PATH=
```

The authoritative Stage3 values come from the repository-level frozen Stage3 protocol, not from whatever values happen to be present in a modified trainer. The trainer SHA is captured at lock time and checked again immediately before Stage3.

Two runs may be directly ranked only if their protocol manifests are compatible. If compatibility is uncertain, mark the historical result `HISTORICAL_NONCOMPARABLE`. If repository records disagree about a baseline number, read the exact declared baseline `test_final.json`.

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
- `MVG EVIDENCE CAPTURE FAIL` = the mechanism may be numerically valid, but required runtime/provenance evidence was not captured; issue `REPAIR_AND_RERUN`, fix instrumentation, and rerun P02_BUILD_VERIFY in the same iteration;
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

# 11. Stage2 geometry analysis — deterministic

Generate this from canonical Stage2 artifacts; it is not a Research Agent stage.

Create:

`logs/sid_geometry_iter<N>.md`

Separate:

1. **Contract compliance** — fixed curvature stayed fixed;
2. **mechanism direct effect** — geometry/assignment changed as preregistered;
3. **SID observations** — descriptive metrics;
4. **interpretation limit** — Stage2 proxies are not assumed to predict Stage3.

Do not call a run good because Stage2 metrics look cleaner.

---

# 12. Stage3 evaluation — deterministic frozen-protocol execution

This is not a Research Agent stage. Immediately before launch, run `stage3_protocol_gate.py` and save stdout/stderr to `logs/stage3_protocol_gate_iter<N>.log`. Only `STAGE3_PROTOCOL_PASS` authorizes launch.

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

# 13. Result classification — P05 Research Agent

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

# 15. Global Review — P07 Research Agent (conditional)

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

# 17. Required evidence — Agent records only for scientific judgment

Before Stage2, require:

```
logs/source_snapshot_iter<N>.md
logs/protocol_manifest_iter<N>.md
logs/hypothesis_iter<N>.md
logs/mechanism_manifest_iter<N>.md
logs/mechanism_contract_iter<N>.json
logs/one_factor_diff_iter<N>.md
logs/preflight_contract_iter<N>.log
logs/mvg_check_iter<N>.log
```

`P01_RESEARCH_DESIGN` requires an Agent record and runs first. Source/protocol/hash locking and `P02_BUILD_VERIFY` are deterministic and happen after P01 acceptance, before Stage2.

Run the no-argument workflow gate before Stage2:

```bash
/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 \
  /home/wlia0047/ar57/wenyu/GeneRec/.claude/skills/curvature-rqvae-iter/scripts/deliberation_gate.py
```

Expected runnable output remains `DELIBERATION_GATE_PASS` for compatibility.

After Stage2/Stage3 require:

```
logs/sid_geometry_iter<N>.md
logs/stage3_protocol_gate_iter<N>.log
logs/stage3_outcome_iter<N>.md
logs/failure_attribution_iter<N>.md
logs/gate_decision_iter<N>.md
logs/git_closure_iter<N>.md
```

Only `P05_RESULT_DECISION` requires an Agent record. `P03_STAGE2`, `P04_STAGE3`, and `P06_CLOSURE` are deterministic/action phases.

`P07_GLOBAL_REVIEW` requires an Agent record only when triggered. For an aborted iteration, `logs/iteration_abort_iter<N>.md` replaces downstream artifacts that were correctly never produced.

Historical completed iterations using old S00–S14 deliberation directories remain readable as legacy evidence; new work must use the reduced phase model.

---

# 18. Iteration loop — reduced phase model

```
P01 RESEARCH_DESIGN          [RESEARCH AGENT — FIRST]
    hypothesis + mechanism semantics + one-factor design + falsification
        ↓
P02 BUILD_VERIFY             [DETERMINISTIC + ACTION]
    source/baseline/protocol/hash lock
    implement accepted design once
    preflight_contract.py → MECHANISM_CONTRACT_PASS
    MVG → MVG PASS
        ↓
P03 STAGE2                   [DETERMINISTIC + GPU ACTION]
    wiring checks → Stage2 RUN ONCE → SID export/metrics
        ↓
P04 STAGE3                   [DETERMINISTIC + GPU ACTION]
    stage3_protocol_gate.py → STAGE3_PROTOCOL_PASS
    Stage3 RUN ONCE → final test artifact
        ↓
P05 RESULT_DECISION          [RESEARCH AGENT]
    scientific effect + promotion + autonomous next action
        ↓
P06 CLOSURE                  [DETERMINISTIC + ACTION]
    artifact check → commit → push → local/remote hash equality
        ↓
P07 GLOBAL_REVIEW            [RESEARCH AGENT, CONDITIONAL]
    select a new structural performance mechanism
```

No Agent is used to approve file existence, hashes, paths, exact config equality, process cleanup, preflight/MVG conditions, Stage2 metrics, Stage3 wiring, or Git closure. A deterministic `FAIL` cannot be overruled by an Agent.

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

For a normal completed iteration, require `DELIBERATION_GATE_PASS` with `phase=CLOSURE` (or `phase=GLOBAL_REVIEW` when P07 is triggered).

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
