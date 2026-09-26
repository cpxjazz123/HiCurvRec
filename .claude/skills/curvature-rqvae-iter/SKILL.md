---
name: curvature-rqvae-iter
description: Scientifically controlled iteration workflow for curvature-aware RQ-VAE experiments in HiCurvRec. The goal is not merely to run successive variants, but to produce causally interpretable evidence about which curvature mechanisms help downstream generative recommendation. The workflow enforces protocol locking, canonical-baseline comparison, one-factor diffs, semantic/provenance validation before MVG, full Stage2→Stage3 evaluation, effect classification, periodic global review, and GitHub auditability.
---

# curvature-rqvae-iter

## 0. Purpose

This skill exists to answer **clean research questions** about curvature-aware RQ-VAE.

The objective is **not** “keep creating new iterations until one score is high.”  
The objective is:

[
	ext{one hypothesis} ightarrow 	ext{one controlled change} ightarrow
	ext{verified activation} ightarrow 	ext{downstream evidence}
]

A useful iteration must tell us **why** the result changed, not only whether the final score changed.

The hard downstream target remains:

- Amazon-2023 Instruments
- Stage3 final `test_R@10 > 0.065`

But target failure and mechanism failure are **different concepts**. A mechanism may be active and mildly positive while still failing the promotion threshold.

---

# 1. Source-of-truth hierarchy

Before every iteration, read the current repository rules.

Priority:

1. repository root `CLAUDE.md`
2. this skill
3. current protocol manifest / experiment registry
4. iteration-local hypothesis / audit files
5. historical reference files

If this skill conflicts with current `CLAUDE.md`, **CLAUDE.md wins**.

Historical files such as `references/baseline_metrics.md`, old manuscript numbers, old Stage3 runs, or results from another protocol are **not automatically comparable** to the current run.

Never use historical “best” values as the active baseline until protocol compatibility is proven.

---

# 2. Non-negotiable project constraints

Follow current `CLAUDE.md` exactly for:

- GitHub-only remote policy
- `main` branch only
- no worktrees
- no unauthorized force push
- Stage2 / Stage3 artifact paths
- no editing Stage1 embeddings
- Stage3 trainer read-only during mechanism iterations
- no unsupported CLI/config override patterns
- mandatory Stage2 gradient-path validation
- mandatory commit + push before an iteration is considered closed

Do not duplicate volatile environment or launcher details here when `CLAUDE.md` already defines them.

An iteration is not “complete” merely because training finished locally. It is complete only after its code, audit files, Stage2 artifacts, Stage3 artifacts, and final decision have been committed and pushed to `origin/main`.

---

# 3. Core experimental rule: CONTROL BEFORE NOVELTY

## 3.1 Canonical baseline

Every experiment must name a **canonical baseline under the same protocol**.

The baseline is not “whatever the previous iteration was.”

Default behavior:

- branch the new experiment from the canonical baseline mechanism;
- add exactly one new mechanism;
- keep all unrelated settings identical.

A failed or exploratory mechanism must **not silently become the parent** of the next experiment.

Before editing code, record:

```
PARENT_ITER=
PARENT_COMMIT=
CANONICAL_BASELINE_ITER=
CANONICAL_BASELINE_RUN=
ACTIVE_MECHANISMS_BEFORE=
NEW_MECHANISM=
ACTIVE_MECHANISMS_AFTER=
```

If `ACTIVE_MECHANISMS_AFTER - ACTIVE_MECHANISMS_BEFORE` contains more than one conceptual change, this is not a single-factor iteration.

## 3.2 Interaction experiments

Mechanism stacking is allowed only when explicitly declared as an **interaction experiment**.

Example:

```
EXPERIMENT_TYPE=interaction
PARENT=iter18
MECHANISM_A=curvature_conditioned_beta2
MECHANISM_B=<new mechanism>
QUESTION=Does B add value on top of A?
```

Interaction experiments must never be confused with clean single-mechanism ablations.

## 3.3 One-factor diff audit

Before MVG, compare the new iteration against its declared parent and write:

`logs/one_factor_diff_iter<N>.md`

It must contain:

- changed source files;
- changed constants;
- changed losses;
- changed optimizer groups;
- changed curvature equations;
- inherited mechanisms;
- explicit statement that no unrelated mechanism was inherited accidentally.

If the diff contains an unexplained second mechanism, stop and fix it before training.

---

# 4. Protocol Lock — mandatory before mechanism design

Before comparing scores, create:

`logs/protocol_manifest_iter<N>.md`

It must record at minimum:

```
PROTOCOL_ID
dataset/version
Stage1 embedding path + hash or immutable identifier
Stage2 parent commit
Stage2 seed(s)
Stage2 max steps
codebook size / number of RQ layers
Stage3 code commit/hash
Stage3 seed(s)
Stage3 epochs
beam size
n_eval
baseline run directory
baseline test_final.json path
baseline test_R@10
baseline protocol ID
```

Two runs may be directly ranked only if their protocol manifests are compatible.

If a historical result uses a different Stage1 embedding, Stage3 trainer/configuration, evaluation population, beam setting, seed policy, dataset version, or other material protocol component, label it:

`HISTORICAL_NONCOMPARABLE`

and do not call it the current best.

## 4.1 Baseline disagreement rule

If two repository files disagree about the same baseline score:

1. resolve the exact run directory;
2. read the actual `test_final.json`;
3. verify the protocol manifest;
4. use that run as the source of truth;
5. document the discrepancy.

Never select whichever number is more convenient.

---

# 5. Global Review — do not optimize only against the previous iteration

A **Global Review** is mandatory:

- every 3 completed iterations;
- before switching mechanism families;
- whenever 3 consecutive iterations remain within a narrow downstream band;
- whenever historical baseline/protocol inconsistencies are discovered.

Output:

`logs/global_review_after_iter<N>.md`

The review must group experiments by mechanism family, for example:

- curvature parameterization;
- optimizer-side curvature adaptation;
- quantization/Sinkhorn;
- behavior contrastive;
- manifold replacement;
- product-manifold / mixed curvature.

For every family, report:

- exact comparable `test_R@10`;
- delta vs canonical baseline;
- activation status;
- whether results are positive / neutral / negative;
- whether another experiment in the same family is justified.

The review must identify:

1. **current strongest evidence-supported family**;
2. **families to pause**;
3. **unanswered core hypothesis**;
4. **next experiment type**: ablation, replication, interaction, or new mechanism.

Do not continue a local chain merely because the immediately previous iteration suggests another tweak.

---

# 6. Mechanism proposal

## 6.1 Literature search

Literature search remains useful, but literature novelty does not override experimental cleanliness.

Agent A / literature search should answer:

- what mechanism has prior evidence;
- what exact failure mode it addresses;
- what variable it changes;
- whether it is compatible with the current curvature formulation.

At least 3 candidates may be collected, but candidate count is less important than relevance.

## 6.2 Direction selection

Direction selection must prioritize:

1. clean testability;
2. relation to the current dominant unresolved question;
3. ability to isolate one causal change;
4. curvature relevance;
5. literature support;
6. implementation risk.

A mechanism must not be selected merely because it is novel.

If the previous family already produced repeated neutral/negative results, a new candidate from that family requires a specific structural reason why it is different.

---

# 7. Hypothesis registration

Before implementation, write:

`logs/hypothesis_iter<N>.md`

It must contain five sections.

## A. Research question

One sentence only.

Example:

> Does curvature-conditioned AdamW beta2 improve the learned SID representation compared with the same baseline using uniform beta2?

## B. Mechanism equation

Write the actual formula, not only prose.

## C. Direct-effect predictions

These are implementation/activation expectations.

Examples:

- `beta2_0 != beta2_1 != beta2_2`;
- `c0(t) < c1(t) < c2(t)`;
- reciprocal Sinkhorn epsilon decreases as curvature increases.

## D. Downstream rationale

Explain the full chain:

[
	ext{mechanism}
ightarrow
	ext{optimization / geometry change}
ightarrow
	ext{SID change}
ightarrow
	ext{why Stage3 could benefit}
]

Do not use “may improve” as the entire rationale.

## E. Falsification

State what observation would contradict the hypothesis.

Do not preregister Stage2 proxy thresholds as hard downstream success criteria.

---

# 8. Semantic / Provenance Gate — mandatory before MVG

This gate exists because code can be mathematically executable while using the **wrong semantic variable**.

Create:

`logs/mechanism_manifest_iter<N>.md`

For every variable entering the new mechanism, record:

| Field | Required content |
|---|---|
| symbol | e.g. `m_l`, `B_l`, `u_l` |
| semantic meaning | e.g. raw residual median |
| source file/script | exact provenance |
| source iteration | which run generated it |
| raw value | before transformation |
| transformation | normalization/log/clip/etc. |
| transformed value | actual value passed to formula |
| units/scale | if meaningful |
| expected range | sanity bound |

Then substitute the actual values into the mechanism equation and print the resulting numbers.

Example:

```
raw_residual_median = [1.000, 0.10941, 0.09331]
normalized_layer_scale = [0.001, 0.932889, 1.0]

formula expects: raw_residual_median
actual input: raw_residual_median
PASS
```

Hard failure conditions:

- variable name and semantic meaning disagree;
- raw and normalized quantities are confused;
- source iteration is unknown;
- formula expects current-run statistics but receives a historical fallback without explicit justification;
- actual numeric result is inconsistent with the intended mechanism.

If provenance fails, **do not run MVG or Stage2**.

---

# 9. MVG — implementation verification only

MVG means mechanism verification.

MVG answers:

> Did we implement the proposed mechanism correctly, and does it actually affect computation?

MVG does **not** answer:

> Is the mechanism scientifically good?

Required checks:

## Layer 1 — Graph

- total loss requires grad;
- required mechanism losses have valid grad functions;
- no accidental detach breaks the intended path.

## Layer 2 — Gradient

- required mechanism terms produce finite, nonzero gradients on intended parameters;
- gradients reach every intended layer.

## Layer 3 — Update

- after a few optimizer steps, intended parameters actually move;
- optimizer groups cover exactly the expected parameters;
- no accidental duplicate parameter groups.

## Layer 4 — Counterfactual activation

Using same seed / same batch / same starting checkpoint:

- mechanism ON;
- mechanism OFF;

must produce a measurable difference in the registered direct effect.

MVG output:

`MVG PASS` or `MVG FAIL`

Interpretation:

- `PASS` = implementation and activation are valid enough for training;
- `FAIL` = implementation/activation problem;
- `PASS` never means downstream improvement.

---

# 10. Stage2 policy

Run Stage2 according to current `CLAUDE.md`.

## 10.1 No proxy performance gate

The following are descriptive:

- Gini;
- collision rate;
- unique SID count;
- per-layer utilization;
- `H(L1|L0)`;
- `H(L2|L0)`;
- coarse/fine ratio;
- historical hitrate fields.

They answer:

> What structure did this mechanism create?

They do **not** answer:

> Is this mechanism good enough to skip Stage3?

Do not stop a numerically healthy run only because a descriptive SID metric looks worse.

Early termination is permitted only for genuine execution invalidity such as:

- crash;
- NaN / Inf;
- impossible curvature values caused by numerical failure;
- corrupted checkpoint/export;
- mechanism proven inactive contrary to MVG assumptions.

Do not use loss plateau, Gini, collision, or unique-code heuristics as a substitute for downstream evaluation.

## 10.2 Checkpoint trajectory

For cyclic or time-varying curvature mechanisms, final-step SID may not represent the best learned geometry.

At minimum preserve/analyze checkpoints around:

- 25%;
- 50%;
- 75%;
- 100%

of the Stage2 schedule when available.

Write:

`logs/checkpoint_trajectory_iter<N>.md`

Record descriptive SID statistics at those points.

Important:

- trajectory analysis is diagnostic;
- do not choose a Stage2 checkpoint using **test** performance;
- checkpoint selection must use a preregistered validation criterion if downstream checkpoint selection is performed.

---

# 11. Stage2 geometry analysis

After Stage2, write:

`logs/sid_geometry_iter<N>.md`

The analyst must separate:

### Direct mechanism effects

Did the registered mechanism produce the expected immediate behavior?

### Geometry observations

What changed in the SID structure?

### Interpretation limits

Explicitly state:

> Stage2 geometry is descriptive and is not assumed to correlate with Stage3 unless historical correlation analysis supports that claim.

Do not label a run “good” merely because SID metrics appear cleaner.

---

# 12. Proxy usefulness audit

At each Global Review, compute or summarize the relationship between available Stage2 proxies and comparable Stage3 outcomes.

Examples:

[
corr(	ext{Gini}, R@10)
]

[
corr(H(L1|L0), R@10)
]

[
corr(	ext{collision}, R@10)
]

Use only protocol-compatible iterations.

If a proxy has weak or inconsistent relation to downstream results, Agent G must not use that proxy as the dominant justification for the next experiment.

This prevents a self-reinforcing loop where the workflow optimizes a convenient Stage2 statistic that Stage3 does not care about.

---

# 13. Stage3 evaluation

All numerically valid and correctly activated candidates proceed to Stage3 under the same locked protocol.

Stage3 is the primary downstream evidence.

Record exact:

- run directory;
- checkpoint;
- `n_eval`;
- R@5;
- R@10;
- NDCG@5;
- NDCG@10;
- baseline delta.

Never compare a current test result against a value copied from a different protocol.

---

# 14. Result classification — separate mechanism effect from promotion

Do not use one label for both scientific interpretation and target attainment.

## 14.1 Mechanism status

Choose one:

- `IMPLEMENTATION_INVALID`
- `PROVENANCE_INVALID`
- `MECHANISM_INACTIVE`
- `ACTIVE_POSITIVE`
- `ACTIVE_NEUTRAL`
- `ACTIVE_NEGATIVE`
- `INTERACTION_UNRESOLVED`
- `PIPELINE_INVALID`

Suggested interpretation against the canonical baseline:

- positive: consistent improvement larger than expected run noise / replicated gain;
- neutral: within the uncertainty/noise band;
- negative: reproducible regression;
- unresolved: cannot isolate contribution because mechanisms are stacked or protocol changed.

Do not call a mechanism negative solely because `R@10 < 0.065`.

## 14.2 Promotion status

Separately record:

- `PROMOTION_PASS` if strict target is achieved under the locked protocol;
- `PROMOTION_FAIL` otherwise.

Example:

```
MECHANISM_STATUS=ACTIVE_POSITIVE
PROMOTION_STATUS=PROMOTION_FAIL
```

is valid.

---

# 15. Replication and noise policy

Tiny score differences should not be treated as discoveries.

If a mechanism appears to beat the canonical baseline by a small amount comparable to historical run-to-run variation:

1. do not promote the claim immediately;
2. repeat baseline and candidate under the same protocol/seed policy;
3. preferably use multiple seeds;
4. report mean and spread.

A difference such as `+0.0001 R@10` is evidence of “near parity” unless replicated.

---

# 16. Failure ledger

The failure ledger should store **scientifically interpretable evidence**, not every run below the hard target.

Add a mechanism to the failed-mechanism ledger only when:

- protocol is valid;
- provenance is valid;
- MVG passes;
- the mechanism is isolated or the interaction is explicitly identified;
- Stage3 comparison is protocol-compatible;
- the negative result is meaningful enough to constrain future work.

Do not write implementation bugs, ambiguous stacked experiments, or non-comparable runs into the mechanism-failure ledger as if they were scientific failures.

For neutral results, use a separate “tested / inconclusive” record if needed.

---

# 17. Family-level stopping rules

Pause a mechanism family when:

- 3 clean experiments in the same family are neutral/negative;
- further variants only tune strength without a new structural hypothesis;
- the family repeatedly changes Stage2 proxies without downstream gain.

A paused family can be reopened only with a structural exception documented in the next Global Review.

Examples of a structural exception:

- corrected semantic/provenance error;
- new causal pathway;
- different manifold;
- interaction with a previously validated positive mechanism.

---

# 18. Core-hypothesis audit

The workflow must continuously track the project's central unanswered claims.

Example unresolved claim:

[
	ext{behavior branching} + 	ext{raw residual geometry}
ightarrow 	ext{layer curvature}
]

If previous experiments used a transformed quantity in place of raw residual, they do **not** count as a clean test of this claim.

Global Review must distinguish:

- hypothesis tested cleanly;
- hypothesis tested with confound;
- hypothesis not yet tested.

Do not abandon a central hypothesis based on an invalid or semantically mismatched experiment.

---

# 19. Recommended iteration loop

```
0. Read CLAUDE.md
   ↓
1. Protocol Lock
   ↓
2. Global Review check
   ↓
3. Select canonical parent
   ↓
4. Literature / direction review
   ↓
5. Register one research hypothesis
   ↓
6. Semantic / Provenance Gate
   ↓
7. One-Factor Diff Audit
   ↓
8. MVG
   ↓
9. Stage2 full run
   ↓
10. Checkpoint trajectory + SID geometry
   ↓
11. Stage3 locked-protocol evaluation
   ↓
12. Mechanism-status classification
   ↓
13. Promotion decision
   ↓
14. Commit + push + remote hash verification
   ↓
15. Every 3 iters: Global Review
```

---

# 20. Required files per iteration

Each iteration should contain or reference:

```
logs/protocol_manifest_iter<N>.md
logs/global_review_after_iter<N>.md          # when triggered
logs/lit_search_iter<N>.md                   # if literature search used
logs/direction_decision_iter<N>.md
logs/hypothesis_iter<N>.md
logs/mechanism_manifest_iter<N>.md
logs/one_factor_diff_iter<N>.md
logs/mvg_check_iter<N>.log
logs/train_migrated.log
logs/checkpoint_trajectory_iter<N>.md
logs/sid_geometry_iter<N>.md
logs/stage3_outcome_iter<N>.md
logs/failure_attribution_iter<N>.md
logs/gate_decision_iter<N>.md
```

The exact filename may vary for legacy iterations, but new iterations should follow this structure.

---

# 21. Gate decision template

`logs/gate_decision_iter<N>.md` should contain:

```
# Iter<N> Decision

PROTOCOL_ID:
PARENT_ITER:
NEW_MECHANISM:
EXPERIMENT_TYPE: single-factor | interaction | replication

PROVENANCE: PASS | FAIL
ONE_FACTOR_DIFF: PASS | FAIL
MVG: PASS | FAIL
STAGE2_COMPLETED: yes | no
STAGE3_COMPLETED: yes | no

CANONICAL_BASELINE_R@10:
ITER_R@10:
DELTA_R@10:

MECHANISM_STATUS:
PROMOTION_STATUS:

INTERPRETATION:
- what this iteration actually established
- what it did not establish

NEXT_ACTION:
- continue family / replicate / interaction test / pause family / global review

AUDIT_COMMIT:
```

---

# 22. Curvature-specific design principles

Curvature mechanisms remain the research focus, but the skill must not force every new idea into the same control path.

Potential families include:

- layer-wise learned curvature;
- cyclic / scheduled curvature;
- residual-calibrated curvature;
- behavior-calibrated curvature;
- optimizer-side curvature adaptation;
- Riemannian optimization;
- curvature-conditioned quantization;
- manifold replacement;
- mixed/product manifolds.

Key rule:

> Curvature must play a mathematically explicit role, and the mechanism must state exactly where curvature enters the computation.

For any formula of the form:

[
c_l=f(B_l,m_l,ldots)
]

the Semantic / Provenance Gate must verify the exact meaning and numeric source of every input before training.

---

# 23. Current lessons that must influence future iterations

The workflow should treat these as **empirical observations under the current protocol**, not universal truths:

1. Fixed or aggressively prescribed layer curvature has not produced a clear downstream gain.
2. Learnable/cyclic curvature remains competitive with fixed alternatives.
3. Strong curvature intervention inside Sinkhorn or behavior contrastive objectives can alter SID structure without improving Stage3.
4. Optimizer-side curvature adaptation has been comparatively promising and should be studied with clean controls.
5. Stage2 “cleaner” geometry does not reliably imply better Stage3 recall.
6. Chaining failed mechanisms creates attribution ambiguity.
7. Small `R@10` differences near the current ceiling require replication before being treated as meaningful.
8. The branching + **raw** residual → curvature hypothesis has not been cleanly validated if prior runs used normalized layer scales in place of raw residual magnitudes.

These lessons guide experiment selection; they do not replace protocol-compatible evidence.

---

# 24. Anti-patterns

Never:

- silently inherit the previous failed mechanism;
- call a stacked experiment a single-mechanism test;
- treat MVG PASS as evidence of effectiveness;
- treat Stage2 proxy improvement as proof of downstream improvement;
- compare results across incompatible protocols;
- use a transformed variable while naming it as the raw physical/statistical quantity;
- declare a tiny one-run gain as a breakthrough;
- continue the same mechanism family indefinitely through parameter tweaks;
- let historical baseline files override an actual locked `test_final.json`;
- mark every `R@10 < 0.065` result as `TRUE_MECHANISM_FAIL`.

---

# 25. Success definition

A successful research iteration is not only one that crosses 0.065.

It is one that leaves the repository with a defensible statement such as:

> Under protocol P, adding mechanism X to baseline B produced direct effect D, changed SID geometry in way G, and changed downstream R@10 by Δ. The mechanism was isolated, its inputs had verified provenance, and the comparison is reproducible.

The project-level promotion target remains `test_R@10 > 0.065`, but the iteration system should optimize for **reliable scientific evidence first** and score second.
