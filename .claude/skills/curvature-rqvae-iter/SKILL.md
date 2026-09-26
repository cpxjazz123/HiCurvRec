---
name: curvature-rqvae-iter
description: Controlled research workflow for HiCurvRec curvature-aware RQ-VAE experiments. Every pipeline stage uses independent 2+1 deliberation: Agent A and Agent B solve the same stage independently, Judge C adjudicates from primary evidence, and only the judge-approved canonical artifact may propagate. GPU-heavy Stage2/Stage3 execution occurs once after adjudication. The current research contract is FCCR-1 fixed closed-form curvature.
---

# curvature-rqvae-iter

## 0. Purpose

This skill exists to produce **causally interpretable curvature experiments**, not an endless sequence of score tweaks.

Every iteration must answer one clean question:

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

# 2. Independent 2+1 Deliberation Protocol — mandatory at every pipeline stage

Every pipeline stage must use:

```
same canonical source packet
        ├──> Agent A (independent)
        └──> Agent B (independent)
                    ↓
              Judge Agent C
                    ↓
    ACCEPT_A | ACCEPT_B | MERGE_AB | REJECT_BOTH
                    ↓
             canonical artifact
                    ↓
              next pipeline stage
```

This is the top-level execution protocol for the entire skill.

**Prospective scope:** this 2+1 requirement applies to new pipeline stages/iterations started after this protocol is introduced. Historical completed iterations are not invalidated solely because they predate 2+1 deliberation. If a historical mechanism is reopened or rerun, the new work must use 2+1.

## 2.1 Independence rules

Agent A and Agent B must:

- receive the same source packet and the same stage objective;
- work independently and in parallel when the runtime supports parallel agents;
- not read, summarize, quote, or react to the other worker's draft before Judge C decides;
- use primary repository evidence rather than trusting historical summaries when exact evidence exists;
- state assumptions, evidence, proposed output, risks, and self-rejection conditions;
- write only candidate artifacts under the stage's `logs/deliberation/` directory;
- never directly overwrite the canonical artifact or shared production code.

The two workers should differ only by role identity (`A` vs `B`), not by hidden hints that steer one toward a preferred answer.

Each worker artifact must begin with:

```
ROLE=AGENT_A | AGENT_B
INDEPENDENCE_DECLARATION=I did not read the other candidate before completing this artifact.
SOURCE_PACKET=<path>
STAGE_ID=<stage id>
```

## 2.2 Judge C rules

Judge C receives:

1. the same canonical source packet;
2. Agent A's completed artifact;
3. Agent B's completed artifact;
4. direct access to primary repository evidence needed to verify disputed facts.

Judge C must not choose based on verbosity, writing style, or novelty alone.

Allowed verdicts:

```
ACCEPT_A
ACCEPT_B
MERGE_AB
REJECT_BOTH
```

`MERGE_AB` is allowed only when the merged result is internally consistent and does not violate the one-factor rule or the active research contract.

`REJECT_BOTH` is mandatory when both candidates are unsupported, contract-invalid, confounded, irreproducible, or materially incomplete.

Judge C must not silently invent a third research mechanism after `REJECT_BOTH`. Instead it writes `REPLAN_CONSTRAINTS`, then fresh independent A2/B2 workers retry the same stage.

Maximum automatic adjudication rounds per stage: **2**. If round 2 is also `REJECT_BOTH`, the stage is blocked and the orchestrator must report the unresolved decision rather than silently continuing.

## 2.3 Judge hard gates and rubric

Hard gates are evaluated before comparative quality:

- active contract compliance;
- protocol compatibility;
- semantic/provenance correctness;
- one-factor causal isolation;
- falsifiability;
- reproducibility / sufficient implementation specificity;
- no unsupported factual claims.

A candidate failing a hard gate cannot win merely because its narrative is stronger.

For candidates that pass hard gates, Judge C compares:

- scientific rationale;
- direct use of prior experimental evidence;
- causal interpretability;
- implementation clarity;
- risk of hidden confounding;
- expected information gain from the experiment;
- cost proportionality.

The judge may use qualitative ratings, but no simple additive score overrides hard gates.

## 2.4 Canonical-only propagation

After Judge C decides:

- one canonical artifact is written to the normal pipeline path;
- downstream workers may read the canonical artifact and `judge.md`;
- downstream workers must not use rejected A/B drafts as active instructions;
- rejected drafts remain in the repository only as audit evidence.

This prevents later stages from blending mutually incompatible proposals.

## 2.5 Side-effect / GPU-heavy stages

For stages that mutate shared code, launch jobs, write checkpoints, evaluate Stage3, commit, or push:

- Agent A and Agent B independently produce a complete **method / patch plan / wiring audit**, not two competing shared-state executions;
- Judge C selects or merges the method;
- the orchestrator applies the canonical method **once**;
- Stage2 and Stage3 full GPU runs are never duplicated merely to satisfy the 2+1 protocol;
- true scientific replication is a separate registered experiment and may intentionally run multiple seeds.

No worktree may be used. This section does not override repository `CLAUDE.md` Git or no-Plan-Agent rules.

## 2.6 Deliberation artifact layout

Every stage uses:

```
logs/deliberation/<STAGE_ID>/round_<R>/
    source_packet.md
    agent_a.md
    agent_b.md
    judge.md
```

`judge.md` must contain:

```
STAGE_ID=
ROUND=
VERDICT=ACCEPT_A | ACCEPT_B | MERGE_AB | REJECT_BOTH

HARD_GATE_A=PASS | FAIL
HARD_GATE_B=PASS | FAIL

EVIDENCE_FOR_A=
EVIDENCE_FOR_B=
PROBLEMS_A=
PROBLEMS_B=

WHY_NOT_A=
WHY_NOT_B=

CANONICAL_DECISION=
CANONICAL_ARTIFACT=
CONFIDENCE=HIGH | MEDIUM | LOW

REPLAN_CONSTRAINTS=
```

For `ACCEPT_A`, `WHY_NOT_B` is required.  
For `ACCEPT_B`, `WHY_NOT_A` is required.  
For `MERGE_AB`, the judge must state exactly which parts came from A and B.  
For `REJECT_BOTH`, `REPLAN_CONSTRAINTS` is mandatory.

## 2.7 Stage map

| Stage ID | Pipeline stage | A/B independently complete | Judge-approved canonical output |
|---|---|---|---|
| `S00_SOURCE_TRUTH` | Read rules / source of truth | extract constraints, current repo facts, unresolved conflicts | `logs/source_snapshot_iter<N>.md` |
| `S01_PROTOCOL_LOCK` | Baseline + protocol | reconstruct comparable protocol and baseline from primary files | `logs/protocol_manifest_iter<N>.md` |
| `S02_HYPOTHESIS` | Research hypothesis | propose exact falsifiable hypothesis/equation under active contract | `logs/hypothesis_iter<N>.md` |
| `S03_PROVENANCE` | Semantic/provenance | independently trace every formula input and semantic definition | `logs/mechanism_manifest_iter<N>.md` |
| `S04_CONTRACT` | Mechanism contract | independently encode expected implementation invariants | `logs/mechanism_contract_iter<N>.json` |
| `S05_ONE_FACTOR` | One-factor diff | independently identify parent, inherited mechanisms, and exact delta | `logs/one_factor_diff_iter<N>.md` |
| `S06_IMPLEMENTATION` | Implementation design | independently produce complete patch plan/diff and tests | `logs/implementation_plan_iter<N>.md`; orchestrator applies once |
| `S07_PREFLIGHT` | Static/contract preflight | independently audit source against hypothesis + contract | `logs/preflight_contract_iter<N>.log` + judge decision |
| `S08_MVG` | MVG | independently design/run lightweight verification and interpret evidence | `logs/mvg_check_iter<N>.log` + judge decision |
| `S09_STAGE2_EXECUTION` | Stage2 run | independently audit launch command, inputs, outputs, invariants | `logs/stage2_execution_plan_iter<N>.md`; Stage2 runs once |
| `S10_STAGE2_ANALYSIS` | SID/geometry analysis | independently analyze the same Stage2 outputs | `logs/sid_geometry_iter<N>.md` |
| `S11_STAGE3_EVALUATION` | Stage3 wiring/eval | independently audit SID wiring, checkpoint, eval protocol and expected outputs | `logs/stage3_evaluation_plan_iter<N>.md`; Stage3 runs once |
| `S12_RESULT_CLASSIFICATION` | Causal/result interpretation | independently classify mechanism effect, confounds, promotion status | `logs/failure_attribution_iter<N>.md` + `logs/gate_decision_iter<N>.md` |
| `S13_GIT_CLOSURE` | Commit/push closure | independently audit required artifacts, paths, git state, remote hash | `logs/git_closure_iter<N>.md` |
| `S14_GLOBAL_REVIEW` | Direction selection when triggered | independently synthesize evidence and propose next research direction | `logs/global_review_after_iter<N>.md` |

If a stage has multiple canonical files, Judge C must list all of them in `CANONICAL_ARTIFACT`.

## 2.8 Stage-specific judge emphasis

- **S00–S01 factual stages:** primary-source correctness and protocol comparability dominate.
- **S02 hypothesis:** contract compliance, falsifiability, information gain, and causal isolation dominate.
- **S03–S05 audit stages:** semantic exactness, provenance, and one-factor integrity dominate.
- **S06 implementation:** fidelity to the canonical hypothesis/contract and minimal diff dominate.
- **S07–S08 verification:** evidence beats intention; a claimed PASS without direct evidence is a FAIL.
- **S09/S11 execution:** reproducibility, exact wiring, and no unintended protocol changes dominate.
- **S10/S12 interpretation:** separate observed facts from causal inference; do not overgeneralize a mapping failure into a family-level failure without evidence.
- **S13 closure:** repository truth and remote verification dominate.
- **S14 review:** compare only protocol-valid evidence and explicitly distinguish validated findings from unresolved hypotheses.

---

# 3. CURRENT RESEARCH CONTRACT — FCCR-1

Until the user explicitly changes the research hypothesis, the active contract is:

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

Unless the user explicitly changes the contract, do not introduce:

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

- `MVG PASS` = implementation matches the mechanism contract and affects computation;
- `MVG FAIL` = implementation/activation invalid;
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

Early termination is reserved for actual invalid execution:

- crash;
- NaN/Inf;
- corrupted checkpoint/export;
- contract violation discovered during training;
- fixed curvature unexpectedly changes.

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

Every numerically valid, contract-valid candidate proceeds to the unchanged Stage3 protocol.

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

A run where curvature was intended to be fixed but became learnable or time-varying is `CONTRACT_INVALID`, not a scientific failure of the closed-form hypothesis.

## Promotion status

Separately record:

- `PROMOTION_PASS`
- `PROMOTION_FAIL`

A mechanism can be `ACTIVE_POSITIVE + PROMOTION_FAIL`.

Do not use `R@10 < 0.065` alone to label a mechanism failed.

---

# 14. Replication / noise rule

Tiny one-run deltas are not discoveries.

If candidate-baseline difference is comparable to historical run noise:

- call it near-parity / neutral;
- replicate before claiming improvement;
- use matched protocol and seed policy;
- preferably report multiple seeds and spread.

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
- strongest evidence-supported direction;
- paused directions;
- unresolved core hypothesis.

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
| Stage1/Stage3 modification | **OUT OF SCOPE** for this skill unless user explicitly changes scope |

“Deferred” means it may be studied later, after the fixed closed-form hypothesis has received a clean test or the user explicitly changes the contract.

---

# 17. Required artifacts and deliberation evidence

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

In addition, stages `S00_SOURCE_TRUTH` through `S09_STAGE2_EXECUTION` must each have a completed A/B/Judge deliberation directory with a non-`REJECT_BOTH` final verdict before the Stage2 full run is launched.

After Stage2/Stage3, add:

```
logs/sid_geometry_iter<N>.md
logs/stage3_evaluation_plan_iter<N>.md
logs/stage3_outcome_iter<N>.md
logs/failure_attribution_iter<N>.md
logs/gate_decision_iter<N>.md
logs/git_closure_iter<N>.md
```

The corresponding deliberation directories for `S10_STAGE2_ANALYSIS` through `S13_GIT_CLOSURE` are also mandatory before the iteration is considered closed.

`S14_GLOBAL_REVIEW` deliberation is mandatory only when the Global Review trigger fires.

A rule written in this skill but not checked before launch/closure is not considered enforced.

Before Stage2, run the skill-level deliberation checker from the iteration directory:

```bash
/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 \
  /home/wlia0047/ar57/wenyu/GeneRec/.claude/skills/curvature-rqvae-iter/scripts/deliberation_gate.py
```

Expected output:

`DELIBERATION_GATE_PASS`

---

# 18. Iteration loop

Every arrow below means: **A and B independently complete the stage → Judge C adjudicates → canonical artifact only proceeds**.

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
S06  Independent implementation plans → Judge → apply canonical patch once
   ↓
S07  Independent static audits → Judge
      + preflight_contract.py → MECHANISM_CONTRACT_PASS
   ↓
S08  Independent MVG verification → Judge → MVG PASS
   ↓
S09  Independent Stage2 launch/wiring audits → Judge
      + deliberation_gate.py → DELIBERATION_GATE_PASS
      + Stage2 full run ONCE
   ↓
S10  Two independent Stage2/SID analyses → Judge
   ↓
S11  Two independent Stage3 wiring/eval audits → Judge
      + Stage3 full evaluation ONCE
   ↓
S12  Two independent causal/result classifications → Judge
   ↓
S13  Two independent Git/artifact closure audits → Judge
      + commit + push + remote hash verification
   ↓
S14  If triggered: two independent Global Reviews → Judge
```

A stage is not complete merely because A and B agree. Judge C must still verify the agreement against primary evidence.

A stage is not complete merely because Judge C chooses a candidate. The judge-approved canonical artifact must actually be materialized at the path defined in the stage map.

---

# 19. Git / artifact closure

Follow current `CLAUDE.md` for exact artifact paths and Git rules.

An iteration is not closed until:

- Stage2 completed;
- Stage3 completed;
- mandatory audit files exist;
- mechanism code + Stage2 artifacts + Stage3 artifacts are committed;
- push to `origin/main` succeeds;
- local and remote main hashes match.

Before declaring closure, rerun the same deliberation checker from the iteration directory. Once result-classification artifacts exist it automatically enters `CLOSURE` mode and verifies S00–S13:

```bash
/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 \\
  /home/wlia0047/ar57/wenyu/GeneRec/.claude/skills/curvature-rqvae-iter/scripts/deliberation_gate.py
```

Require `DELIBERATION_GATE_PASS` with `phase=CLOSURE` (or `phase=GLOBAL_REVIEW` when S14 is triggered).

Never claim an iteration is complete before that point.

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
- let Agent A or Agent B read the other's draft before both candidate artifacts are complete;
- allow Judge C to select a hard-gate-failing candidate because it is more persuasive;
- let Judge C invent an unreviewed third mechanism after `REJECT_BOTH`;
- propagate rejected A/B drafts as active instructions to the next stage;
- run duplicate full Stage2/Stage3 jobs merely to satisfy the 2+1 protocol;
- skip the closure-mode deliberation gate before marking an iteration complete.

---

# 22. Success definition

A scientifically successful iteration leaves a defensible statement:

> Under protocol P and FCCR-1, closed-form (f(B_l,m_l^{raw})) produced fixed curvature ([c_0,c_1,c_2]). The curvature was verified non-trainable and time-invariant, the mechanism changed the intended quantization behavior, and downstream R@10 changed by Δ relative to the canonical baseline.

Only after such a clean run may the project conclude whether the fixed branching+raw-residual curvature hypothesis is supported, neutral, or negative.
