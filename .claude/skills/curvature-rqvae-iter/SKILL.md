---
name: curvature-rqvae-iter
description: Main-only, zero-bookkeeping-artifact workflow for HiCurvRec curvature-aware RQ-VAE experiments. Main performs research design, implementation, verification, Stage2, Stage3, result interpretation, next-direction selection, and Git closure end-to-end. Every condition edits stage2_RQ-VAE/curvature_RQ-VAE in place under a Git-gated accept/rollback policy. The workflow itself creates no Markdown, JSON, JSONL, LOG, TXT, report, manifest, decision, audit, review, or closure files. Only source-code changes and the experiment programs' native Stage2/Stage3 outputs are retained.
---

# curvature-rqvae-iter

## 0. Core rule

There is one reasoning owner from start to finish: **main**.

Main performs:

```
research design
→ source/protocol inspection
→ implementation
→ MVG
→ Stage2
→ Stage3
→ result interpretation
→ next-direction selection
→ Git closure
```

No phase is delegated.

---

## 0.2 Single working tree

The mechanism has exactly one working tree:

```
stage2_RQ-VAE/curvature_RQ-VAE/
results/stage2_RQ-VAE/curvature_RQ-VAE/
results/stage3_T5Train/curvature_RQ-VAE/
```

Every condition edits these paths directly. Never create a parallel
directory, and never put a counter into any path, launcher filename,
`MECHANISM_NAME`, or `RQVAE_VARIANT`.

Git is the boundary and the undo mechanism. Main notes the current commit hash
before editing so a losing condition can be reverted exactly.

## 0.3 Accept / rollback policy

Compare against the parent by Stage3 `test_recall@10` read from each run's own
native `test_final.json`, on a protocol-compatible population
(`n_eval = 57439`).

- **Better** → keep it: commit the source with its native Stage2/Stage3
  outputs, push, verify the remote hash.
- **Worse** → `git revert` the implementation commit (never
  `git reset --hard`; pushed history stays linear and auditable), push the
  revert, verify the remote hash. The tree returns to the parent, which is the
  baseline for the next condition.
- **Equal or within noise** → treat as not better, revert, and pick a
  genuinely different structural mechanism.

A rejected condition still runs Stage2 and Stage3 exactly once; the decision
comes from their native outputs, never from Stage2 proxies. The revert commit
is its only durable trace — no failure archive, ledger, or report is created.

Reverting restores the source, so the next condition always starts from the
best known state rather than from a pile of half-working variants.

---

## 0.1 Zero workflow-artifact rule

The workflow itself must create **no bookkeeping artifact files of any kind**.

Do not create any workflow-generated:

- Markdown;
- JSON;
- JSONL;
- LOG;
- TXT;
- manifest;
- hypothesis file;
- mechanism description file;
- protocol snapshot;
- source snapshot;
- audit file;
- decision file;
- result-summary file;
- closure file;
- review file;
- abort file;
- repair record;
- direction record.

Do not create a workflow `logs/` directory merely to store process records.

Scientific design, assumptions, one-factor constraints, repair decisions, abort decisions, result interpretation, promotion decisions, and next-direction choices remain in the active main context only.

Static repository documentation may be **read**.

### Important distinction

This zero-artifact rule applies to **workflow bookkeeping**.

Stage2 / Stage3 programs may still produce their **native experiment outputs required for training/evaluation**, such as checkpoints, SID files, training metrics, and the evaluator's native final-test output. Do not create additional wrapper summaries around those outputs.

---

# 1. Source of truth

Priority:

1. repository root `CLAUDE.md`;
2. this skill;
3. current source code;
4. current experiment configuration;
5. native Stage2/Stage3 outputs.

Higher-priority sources override lower-priority sources.

Always read the actual native final-test output rather than any summary of it.

---

# 2. Forward-only research policy

Each condition tests one forward-looking structural mechanism with a plausible
path to improving downstream `test_R@10`.

Project target:

`test_R@10 > 0.065`

Forbidden as the primary purpose:

- parameter sweep;
- grid/random/Bayesian search;
- multi-seed or matched-seed replication;
- rerunning the same condition only to estimate variance;
- hyperparameter sensitivity study;
- ablation-only work;
- reverse/control work whose main purpose is attribution;
- root-cause-only work;
- diagnostic-only work.

A small or ambiguous effect is recorded mentally by main and the workflow moves
to a distinct structural mechanism.

---

# 3. One-factor rule

Before editing code, main must establish in its active reasoning context:

- parent condition and its parent commit hash;
- active mechanisms before;
- exact new mechanism;
- active mechanisms after;
- unchanged factors;
- exact equation / algorithm;
- coordinate / tensor semantics;
- direct-effect expectation;
- falsification condition.

No file is written for this.

Once implementation begins, the registered scientific condition is frozen.

Do not retune the same condition after MVG or downstream results.

If the mechanism turns out to need a changed equation, scientific constant,
data identity, seed, or Stage3 protocol, abandon this condition and start the
next one in the same directory. Abandoning a condition is a normal outcome of
the accept/rollback policy, not an exceptional case.

---

## 4. Curvature contracts

### FCCR-1 — Fixed Closed-Form Curvature Research Contract (default)

Required:

```
CURVATURE_SOURCE=closed_form
CURVATURE_TRAINABLE=false
CURVATURE_TIME_VARYING=false
USES_CYCLIC_SCHEDULE=false
USES_CURVATURE_REGULARIZATION=false
USES_CURVATURE_CONDITIONED_OPTIMIZER=false
NEW_CURVATURE_CONDITIONED_AUX_LOSS=false
```

Per-layer curvature is computed before Stage2 and stays fixed for the full run.
FCCR-1 is always in force unless a contract transition is explicitly
authorized.

### DCCR-1 — Dynamic Cyclic Curvature Research Contract

DCCR-1 is available only after explicit user authorization, and applies only
to the specifically authorized work. It does not change the default contract.

```
CURVATURE_SOURCE=cyclic_schedule_with_trainable_layer_scale
CURVATURE_TRAINABLE=true
CURVATURE_TIME_VARYING=true
USES_CYCLIC_SCHEDULE=true
USES_CURVATURE_REGULARIZATION=true
USES_CURVATURE_CONDITIONED_OPTIMIZER=true
NEW_CURVATURE_CONDITIONED_AUX_LOSS=false
```

Such a condition must name its DCCR-1 parent and preserve the inherited cyclic
schedule, layer scales, curvature regularization, and curvature-conditioned
optimizer exactly unless one of those is the explicitly registered single
mechanism. A transition to DCCR-1 does not authorize any other curvature or
Stage3 protocol changes.

Without an explicit contract transition, do not introduce:

- learnable curvature;
- learned curvature prior;
- cyclic/scheduled curvature;
- curvature regularization intended to move curvature;
- optimizer-side curvature learning;
- a second unrelated scientific mechanism in the same condition.

A structural mechanism may consume the curvature defined by the selected
contract while leaving that contract unchanged.

---

# 5. Phase A — Design in main

Main chooses one mechanism and checks that it is:

- structurally distinct from the mechanism that failed or did nothing;
- one-factor relative to the current condition;
- compatible with the explicitly selected curvature contract;
- plausibly relevant to downstream recommendation quality;
- falsifiable;
- implementable without changing Stage3.

Do not create a hypothesis, mechanism, protocol, or decision file.

---

# 6. Phase B — Inspect, implement, verify in main

Main directly inspects:

- current source;
- current repository rules;
- actual Stage1 path / identity;
- Stage2 seed / steps / architecture;
- current Stage3 trainer;
- current Stage3 frozen protocol;
- the current native Stage3 result.

No source/protocol snapshot file is created.

Then implement the mechanism.

## 6.1 MVG

MVG verifies implementation, not scientific merit.

Main checks, as applicable:

- intended equation executes;
- coordinate semantics are correct;
- the selected curvature contract remains satisfied throughout the run;
- tensors are finite;
- intended gradients are finite/non-zero;
- intervention has measurable direct computational effect;
- runtime identity/path/device information is correct.

Do not save an MVG report file.

Interpretation exists only in main:

- `MVG PASS`: proceed;
- implementation failure: repair code while preserving the registered mechanism, then rerun;
- evidence/instrumentation failure: repair instrumentation, then rerun;
- mechanism failure requiring changed science: abandon the condition and revert.

## 6.2 Operational repair

Operational repairs stay within the same condition when the registered science is unchanged.

Examples:

- checker/parser bug;
- wrong path;
- missing runtime print;
- stale process;
- source implementation mismatch that can be corrected to the already chosen equation.

Do not create a repair record.

---

# 7. Phase C — Stage2

Run Stage2 once under the locked condition.

Stage2 native outputs are allowed because they are experiment outputs, not workflow bookkeeping.

Descriptive metrics such as Gini, collision, entropy, code utilization, unique tuples, and SID geometry are **not performance gates**.

Main may inspect them directly from native outputs or commands.

Do not generate a separate SID/geometry report.

Every valid, non-aborted condition proceeds to Stage3.

---

# 8. Phase D — Stage3

Stage3 is a frozen evaluator.

Immediately before launch, run:

```bash
python /home/wlia0047/ar57/wenyu/GeneRec/.claude/skills/curvature-rqvae-iter/scripts/stage3_protocol_gate.py
```

Read its stdout/stderr directly. Do not redirect it into a workflow log file.

Required:

`STAGE3_PROTOCOL_PASS`

Current frozen protocol includes:

```
NUM_EPOCHS=150
EARLY_STOP=DISABLED
NO_EVAL=true
SKIP_TEST=false
SEED=42
BEAM_SIZE=20
```

Main must not bypass `PROTOCOL_GATE_FAIL`.

Run Stage3 once to completion.

Use the evaluator's native final-test output as the downstream evidence. Do not generate an additional Stage3 outcome summary.

---

# 9. Phase E — Result decision in main

Main reads the native Stage2/Stage3 outputs and classifies the condition internally as one of:

- `ACTIVE_POSITIVE`
- `ACTIVE_NEUTRAL`
- `ACTIVE_NEGATIVE`
- invalid implementation / contract / pipeline condition

Promotion is internally:

- `PROMOTION_PASS`
- `PROMOTION_FAIL`

Do not write a result, failure-attribution, gate-decision, promotion, direction, or summary file.

Do not overstate causality from one small point delta.

Main immediately selects the next valid action from current evidence.

---

# 10. Phase F — Closure

No workflow closure artifact is created.

An **accepted** condition is closed when:

- Stage2 completed;
- Stage3 completed;
- native required Stage2/Stage3 outputs exist;
- source changes and required native experiment outputs are committed;
- push to `origin/main` succeeds;
- local `main` hash equals remote `main` hash.

A **rejected** condition is closed by rollback, not by deletion:

- do not fabricate or keep downstream outputs as if they were the new best;
- `git revert` the implementation commit, restoring the parent source;
- commit the revert, push, and verify the remote hash;
- the parent's native outputs, already in history, remain the baseline;
- continue with the next justified condition in the same working tree.

Never delete history, never `git reset --hard` a pushed branch, and never leave
a rejected mechanism in the tree. Do not create a Git-closure record.

---

# 11. Conditional global review

Main may periodically step back and reassess the open question, entirely in
active context.

Do not create a global-review file.

The next direction must still be a distinct forward structural mechanism, not
replication, seed comparison, parameter sweep, sensitivity study,
ablation-only work, or root-cause-only work.

---

# 12. Git policy

Follow root `CLAUDE.md`.

Commit:

1. mechanism source/config/script changes;
2. native Stage2 outputs required by the project;
3. native Stage3 outputs required by the project.

Commit the implementation together with its native outputs, so a single
`git revert` removes the mechanism and its artifacts as one unit.

The commit message states the mechanism in one line and the measured
`test_recall@10` against the parent, so the accept/rollback decision is
readable from the commit itself.

Rollback uses `git revert` only. Do not rewrite pushed history and do not
delete a losing run's commit; the tree returns to the parent.

Do not commit newly generated workflow bookkeeping files because none should exist.

---

# 13. Anti-patterns

Do not:

- delegate any phase;
- create a parallel mechanism directory;
- put a counter into any path, launcher filename, `MECHANISM_NAME`, or `RQVAE_VARIANT`;
- keep a rejected mechanism in the tree instead of reverting it;
- rewrite pushed history or `git reset --hard` to undo a condition;
- generate workflow Markdown/JSON/JSONL/LOG/TXT records;
- create hypothesis/protocol/manifest/audit/decision/closure/review files;
- create a workflow logs directory;
- stack a second scientific mechanism;
- retune a locked mechanism after results;
- use Stage2 proxies to block a valid Stage3 run;
- modify Stage3 for a mechanism under test;
- reintroduce Stage3 early stopping;
- run duplicate Stage2/Stage3 jobs for one condition;
- spend a condition on replication/noise estimation/root-cause-only work;
- ask the user to choose the next research step.
