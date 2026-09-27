---
name: curvature-rqvae-iter
description: Main-only research workflow for HiCurvRec curvature-aware RQ-VAE experiments. Main performs research design, implementation, verification, Stage2, Stage3, result interpretation, next-direction selection, and Git closure end-to-end. No delegated reasoning workflow is permitted. Independent tool operations may run concurrently when safe. The active curvature contract is FCCR-1 fixed closed-form curvature.
---

# curvature-rqvae-iter

## 0. Purpose

This workflow produces causally interpretable, forward-moving experiments.

Every iteration must answer one clean question:

```
one hypothesis
→ one controlled structural mechanism
→ contract / implementation verification
→ Stage2
→ Stage3
→ result classification
```

Project target: downstream `test_R@10 > 0.065`.

Promotion failure and mechanism failure are different concepts.

### Main-only rule

There is exactly one reasoning owner from start to finish: **main**.

Main itself:

- chooses and registers the research mechanism;
- reads repository evidence;
- locks parent, baseline, protocol, paths, and hashes;
- edits code;
- runs deterministic checks;
- launches Stage2 and Stage3;
- interprets results;
- chooses the next direction;
- writes canonical artifacts;
- commits, pushes, and verifies the remote hash.

Do not create or invoke any secondary reasoning workflow, role-based reviewer, delegated scientific-decision process, or separate sign-off layer.

Independent **tool operations** may run concurrently when they do not conflict on mutable state or exclusive GPU resources. Concurrency never creates another scientific decision-maker.

### Autonomous execution

No workflow state may require the user to choose the next step.

Invalid states include:

```
ASK_USER
WAIT_FOR_USER
NEED_USER_DECISION
PAUSE_FOR_DIRECTION
CONFIRM_NEXT_STEP
CONFIRM_NEXT_ITERATION
```

Main resolves uncertainty from repository evidence, the active contract, hard gates, one-factor rules, and the project objective.

If an external blocker makes execution impossible, record `EXTERNAL_BLOCKER` and close that path cleanly.

### New iteration directory

When creating a new iteration from a parent/template:

1. copy the source tree;
2. immediately delete the destination's copied `logs/`;
3. recreate an empty `logs/`;
4. never copy prior decisions, metrics, closure claims, or historical review files into the new iteration.

Do not modify the parent's logs.

---

# 1. Source of truth

Priority:

1. repository root `CLAUDE.md`;
2. this skill;
3. current iteration machine-readable mechanism contract;
4. current protocol manifest;
5. current iteration hypothesis / provenance records;
6. historical files.

Higher-priority sources override lower-priority sources.

A historical summary never overrides a protocol-compatible primary `test_final.json`.

---

# 2. Main-only phase model

| Phase | Main responsibility |
|---|---|
| `P01_RESEARCH_DESIGN` | hypothesis, exact mechanism, one-factor delta, falsification |
| `P02_BUILD_VERIFY` | source/protocol/hash lock, implementation, preflight, provenance, MVG |
| `P03_STAGE2` | Stage2 once, SID export, descriptive geometry metrics |
| `P04_STAGE3` | Stage3 protocol gate, Stage3 once, final test |
| `P05_RESULT_DECISION` | scientific interpretation, promotion, next action |
| `P06_CLOSURE` | artifact completeness, commit/push, remote hash |
| `P07_GLOBAL_REVIEW` | conditional cross-iteration review and new direction |

No phase is delegated.

## 2.1 Canonical records

Do not create role-specific workflow directories, role-specific drafts, vote files, or separate decision wrappers.

Canonical records are:

```
logs/source_snapshot_iter<N>.md
logs/protocol_manifest_iter<N>.md
logs/hypothesis_iter<N>.md
logs/mechanism_manifest_iter<N>.md
logs/mechanism_contract_iter<N>.json
logs/one_factor_diff_iter<N>.md
logs/preflight_contract_iter<N>.log
logs/mvg_check_iter<N>.log
logs/sid_geometry_iter<N>.md
logs/stage3_protocol_gate_iter<N>.log
logs/stage3_outcome_iter<N>.md
logs/failure_attribution_iter<N>.md
logs/gate_decision_iter<N>.md
logs/git_closure_iter<N>.md
logs/global_review_after_iter<N>.md   # only when P07 triggers
```

Each record is written directly by main or by a deterministic checker invoked by main.

Historical review directories may remain in old iterations as immutable audit history. New work does not create them and the active workflow does not use them as authorization evidence.

## 2.2 Parallel tool execution

Main batches independent reads, searches, hashes, static checks, and post-run metric extraction.

Serialize only for:

- true data dependency;
- same-file/shared-artifact write conflict;
- Git ref conflict;
- exclusive GPU/memory conflict.

Stage2 and Stage3 each execute once per valid scientific condition.

---

# 3. Forward-only policy

Every new iteration tests one forward-looking structural mechanism with a plausible path to better downstream recommendation performance.

Forbidden as the primary purpose of a new iteration:

- parameter sweep / grid search / random search / Bayesian search;
- multi-seed or matched-seed replication;
- rerunning the same mechanism only to estimate variance;
- hyperparameter sensitivity study;
- ablation-only experiment;
- reverse/control experiment whose main purpose is attribution;
- root-cause investigation of a prior small delta;
- diagnostic-only iteration.

A small or ambiguous result is recorded as such. The next iteration moves to a distinct structural mechanism.

Diagnostics are lightweight P02 checks only.

## 3.1 Registered design immutability

Once P01 registers the mechanism, freeze:

- equation;
- mechanism constants;
- parent;
- one-factor delta;
- data identity;
- seed;
- Stage3 protocol.

Do not retune the same iteration after seeing MVG or downstream results.

If a changed constant/equation is required, close the current iteration and register a new one.

---

# 4. Active curvature contract — FCCR-1

`FCCR-1 = Fixed Closed-Form Curvature Research Contract`

Required:

```
CURVATURE_SOURCE=closed_form
CURVATURE_TRAINABLE=false
CURVATURE_TIME_VARYING=false
USES_CYCLIC_SCHEDULE=false
USES_CURVATURE_REGULARIZATION=false
NEW_CURVATURE_CONDITIONED_OPTIMIZER=false
NEW_CURVATURE_CONDITIONED_AUX_LOSS=false
```

Per-layer curvature values are computed before Stage2 and remain unchanged for the full run.

Allowed pattern:

```python
self.register_buffer("fixed_c", torch.tensor(c_l, dtype=torch.float32))
```

Without an explicit between-iteration contract transition, do not introduce:

- learnable curvature parameters;
- learned curvature prior;
- scheduled / cyclic curvature;
- curvature regularization intended to update curvature;
- optimizer-side curvature learning;
- a second unrelated scientific mechanism in the same iteration.

A structural mechanism may consume fixed curvature while leaving curvature itself unchanged.

---

# 5. P01 — Research design

Main creates:

```
logs/hypothesis_iter<N>.md
logs/mechanism_manifest_iter<N>.md
logs/mechanism_contract_iter<N>.json
logs/one_factor_diff_iter<N>.md
```

P01 records:

1. research question;
2. exact equation / algorithm;
3. coordinate and tensor semantics;
4. parent iteration;
5. active mechanisms before;
6. active mechanisms after;
7. exactly one conceptual delta;
8. unchanged factors;
9. downstream rationale;
10. direct-effect expectation;
11. falsification conditions.

Do not implement until the scientific mechanism is unambiguous.

---

# 6. P02 — Lock, build, verify

After P01, main creates:

```
logs/source_snapshot_iter<N>.md
logs/protocol_manifest_iter<N>.md
```

Then main implements the registered design.

## 6.1 Protocol manifest

Record at least:

```
PROTOCOL_ID=
DATASET_VERSION=
PARENT_ITER=
PARENT_COMMIT=
CANONICAL_BASELINE_ITER=
CANONICAL_BASELINE_TEST_FINAL=
CANONICAL_BASELINE_TEST_R10=
STAGE1_EMBEDDING_PATH=
STAGE1_EMBEDDING_SHA256=
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
STAGE3_CODE_PATH=
STAGE3_LOG_PATH=
STAGE3_SAVE_PATH=
```

Two runs may be directly ranked only when relevant protocol fields are compatible.

Otherwise mark the historical result `HISTORICAL_NONCOMPARABLE`.

## 6.2 One-factor gate

Record:

```
PARENT_ITER=
ACTIVE_MECHANISMS_BEFORE=
ITERATION_DELTA=
ACTIVE_MECHANISMS_AFTER=
UNCHANGED_FACTORS=
```

Do not add a second scientific mechanism to rescue the current one.

## 6.3 Preflight

Run:

```bash
python /home/wlia0047/ar57/wenyu/GeneRec/.claude/skills/curvature-rqvae-iter/scripts/preflight_contract.py
```

Required:

`MECHANISM_CONTRACT_PASS`

## 6.4 MVG

MVG verifies implementation, not scientific merit.

It must establish, as applicable:

- registered equation is actually executed;
- source/target coordinate semantics are correct;
- fixed curvature is non-trainable and time-invariant;
- tensors are finite;
- intended gradients are finite/non-zero;
- the registered intervention has a measurable direct computational effect;
- required hashes, IDs, device, checkpoint identity, and runtime provenance are captured.

Interpretation:

- `MVG PASS`: proceed.
- `MVG IMPLEMENTATION FAIL`: repair implementation while preserving registered science.
- `MVG EVIDENCE CAPTURE FAIL`: repair instrumentation and rerun.
- `MVG MECHANISM FAIL`: the mechanism itself is inactive/invalid and making it viable requires changing locked science; abort.

`MVG PASS` does not imply better Stage3 performance.

## 6.5 Operational repair

Operational defects stay in the same iteration when locked science is unchanged.

Examples:

- checker/parser bug;
- missing logging field;
- wrong path;
- missing hash/device/batch-ID capture;
- stale worker / cleanup issue;
- code that can be corrected to the already registered equation.

Main applies the smallest repair and reruns affected checks.

## 6.6 Abort

Use `ABORT_ITERATION` only when direct evidence shows the registered scientific condition cannot proceed without changing locked science.

Not valid abort reasons:

- unattractive Stage2 proxy;
- worse Gini/collision/entropy;
- repairable provenance omission;
- checker/logging bug.

Create:

`logs/iteration_abort_iter<N>.md`

with:

```
STATUS=ITERATION_ABORTED_INFEASIBLE
ABORT_STAGE=
ABORT_EVIDENCE=
WHY_SAME_ITERATION_REPAIR_INVALID=
NEXT_ITERATION_CONSTRAINTS=
```

Do not launch later invalid stages.

---

# 7. P03 — Stage2

Main runs Stage2 once under the locked protocol.

Stage2 descriptive metrics are not performance gates:

- Gini;
- collision statistics;
- code utilization;
- unique tuples;
- L0/L1 pairs;
- conditional entropy;
- geometry summaries.

Create:

`logs/sid_geometry_iter<N>.md`

Stage2 proxies never replace Stage3.

---

# 8. P04 — Stage3

Stage3 is a frozen evaluator.

Immediately before launch run:

```bash
python /home/wlia0047/ar57/wenyu/GeneRec/.claude/skills/curvature-rqvae-iter/scripts/stage3_protocol_gate.py
```

Save stdout/stderr to:

`logs/stage3_protocol_gate_iter<N>.log`

Required:

`STAGE3_PROTOCOL_PASS`

A mismatch is `PROTOCOL_GATE_FAIL`; main must not bypass it.

Repository-level Stage3 lock includes:

```
NUM_EPOCHS=150
EARLY_STOP=DISABLED
NO_EVAL=true
SKIP_TEST=false
SEED=42
BEAM_SIZE=20
```

Run Stage3 once to the locked completion condition and require the canonical final test.

Create:

`logs/stage3_outcome_iter<N>.md`

---

# 9. P05 — Result decision

Main creates:

```
logs/failure_attribution_iter<N>.md
logs/gate_decision_iter<N>.md
```

Mechanism status:

- `ACTIVE_POSITIVE`
- `ACTIVE_NEUTRAL`
- `ACTIVE_NEGATIVE`
- invalid implementation / contract / pipeline classifications where applicable

Promotion:

- `PROMOTION_PASS`
- `PROMOTION_FAIL`

A mechanism may be `ACTIVE_POSITIVE + PROMOTION_FAIL`.

Do not overstate causality from a single small point delta.

A small / ambiguous delta becomes `ACTIVE_NEUTRAL`; do not create a replication iteration to estimate noise.

Main records one concrete autonomous next action.

---

# 10. P06 — Closure

A normal iteration is closed only when:

- Stage2 completed;
- Stage3 completed;
- required canonical artifacts exist;
- mechanism code and required results are committed;
- push to `origin/main` succeeds;
- local and remote `main` hashes match.

Create:

`logs/git_closure_iter<N>.md`

Before closure run:

```bash
/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 \
  /home/wlia0047/ar57/wenyu/GeneRec/.claude/skills/curvature-rqvae-iter/scripts/deliberation_gate.py
```

Required:

`WORKFLOW_GATE_PASS`

For an aborted iteration, close with the abort record and all valid evidence up to the stopping point. Do not fabricate downstream outputs.

---

# 11. P07 — Conditional global review

Trigger after 3 clean protocol-valid iterations, or when a higher-priority repository rule explicitly requires it.

Create:

`logs/global_review_after_iter<N>.md`

Main synthesizes protocol-valid evidence and chooses a new forward structural mechanism.

Do not choose replication, seed comparison, parameter sweep, sensitivity study, ablation-only work, or root-cause-only work.

---

# 12. Workflow gate

Run before Stage2 and again before closure.

The gate reads canonical artifacts only. Historical review directories are not authorization inputs.

Possible outputs:

```
WORKFLOW_GATE_PASS
WORKFLOW_ABORT_CONFIRMED
WORKFLOW_GATE_FAIL
```

---

# 13. Anti-patterns

Do not:

- delegate any phase or scientific decision;
- create role-based review/sign-off files;
- add a second scientific mechanism in the same iteration;
- retune a locked mechanism after seeing results;
- use Stage2 proxies to skip a valid Stage3 run;
- change Stage3 for an iteration-level mechanism;
- reintroduce metric-triggered Stage3 early stopping;
- run duplicate Stage2/Stage3 jobs for the same condition;
- spend a new iteration on replication/noise estimation/root-cause-only work;
- ask the user to choose the next research step.

---

# 14. Success definition

A successful iteration has:

1. one registered mechanism;
2. one-factor integrity;
3. implementation matching the contract;
4. MVG evidence of the intended intervention;
5. Stage2 completion under the locked protocol;
6. Stage3 completion under the frozen evaluator;
7. result classification without overstated causality;
8. committed and pushed evidence;
9. an autonomously selected next action by main.
