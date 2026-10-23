# The runbook: starting, restarting and monitoring the research controller

This is the operational half of issue #107. The controller is a **resumable worker**: its
state is a file, every transition is checkpointed, and a restart continues where the work
stopped. Nothing here promises uninterrupted execution -- a controller that is not running
makes no progress, and the state file says exactly what it was doing when it stopped.

## Where everything lives

| what | where |
|---|---|
| the state | `results/harness3_runs/controller/state.json` |
| the audit log (append-only) | `results/harness3_runs/controller/audit.jsonl` |
| each round's portfolio | `results/harness3_runs/controller/rounds/round_NNN/portfolio.json` |
| the evidence store (one immutable manifest per run) | `results/harness3_runs/store/runs/<run id>/manifest.json` |
| the hypothesis ledger | `results/harness3_runs/hypotheses/ledger.json` |
| the transfer analysis and the design requirements | `results/harness3_runs/transfer_analysis/` |
| the per-arm table | `results/harness3_runs/final/final_per_arm_table.md` |
| the resource configuration | `results/harness3_runs/controller/resource_config.json` |

## Starting the worker

```bash
cd /fs04/ar57/wenyu/GeneRec
export PYTHONPATH=/fs04/ar57/wenyu/GeneRec

# one transition at a time (useful while you watch it)
python3 -m tools.genrec_agents.cli controller tick

# the worker loop
setsid nohup /tmp/controller_worker.sh > /tmp/controller_worker.log 2>&1 < /dev/null &
```

The worker script ticks the controller, waits, and ticks again. It stops only when the
controller reaches `done` (a confirmed Stage-3 success) or an operator pause it cannot
resume from; the state file is the authority on what happened.

## Monitoring

```bash
# the queue, the phase, the next action, the pause reason
python3 -m tools.genrec_agents.cli controller status

# the last twenty audit events
tail -20 results/harness3_runs/controller/audit.jsonl

# what is running
pgrep -af "train_rqvae.py|run_stage3_param"

# the arms measured so far, and the downstream deltas
python3 -m tools.genrec_agents.cli final-table --out results/harness3_runs/final
```

## Restarting after a crash

Nothing to repair. Start the worker again: the controller reads the state file, finds the
arm it was working on, and continues. A job whose process is gone is **classified, not
retried** -- the classification is in the audit log. A dispatch that was already made is
never made twice: the state records the dispatch *before* the launch, so a restart in that
window does not duplicate a cold run.

## Pausing and resuming

```bash
# an operator pause: the state is persisted and the issue stays OPEN
python3 -m tools.genrec_agents.cli controller pause --reason "maintenance window"

# resume
python3 -m tools.genrec_agents.cli controller resume
```

A pause is operational, not scientific. The controller never turns a negative result into a
pause, and never turns a pause into a success.

## The phases

```
read -> hypothesise -> screen -> train -> evaluate -> stage3 -> learn -> (next portfolio)
```

* **read** -- take in the store and the ledger; name the gap the round is about; decide
  whether the scientific stop condition is met.
* **hypothesise** -- the agent proposes a portfolio of mathematically distinct mechanisms
  from the store, the ledger and the measured design requirements; the controller freezes
  each proposal and computes its specification digest.
* **screen** -- the unchanged independent checks, the active-ramp probe and the export
  sensitivity, with up to `max_concurrent_screening` at a time. A rejection here costs no
  cold-run GPU time.
* **train** -- one cold 3000-epoch Stage-2 per admitted arm, at the frozen seed, with the
  ports and the output directories isolated. A cold run waits for the GPUs to be free.
* **evaluate** -- the frozen Hard-L1 ER, read once per arm, and the arm is recorded in the
  store with the measurement it just took.
* **stage3** -- Stage-3 once per ER-qualified arm, waiting for the GPUs; the result is
  recorded in the store.
* **learn** -- the round's portfolio is archived, the ledger is updated with what was
  falsified, and the next portfolio starts automatically. A negative result does not stop
  the loop.

## The prohibitions, and where they are enforced

`tools/genrec_agents/guardrails.py`, consulted before every dispatch and covered by
`tools/genrec_agents/tests/test_guardrails.py`:

| prohibition | refusal |
|---|---|
| a second seed | `single_seed` |
| re-training the control | `no_control_retrain` |
| running an archived policy | `no_archived_policy_execution` |
| re-running a failed arm | `no_failed_arm_rerun` |
| renaming an unchanged mechanism | `no_semantic_duplicate` |
| reading the frozen ER twice for one arm | `no_repeated_frozen_evaluation` |
| running Stage-3 twice for one arm | `no_repeated_stage3` |

## The bar for closing the issue

The issue is closed **only** when all three hold:

1. a candidate clears the frozen Hard-L1 ER gate 16.38953942232631;
2. its Stage-3 recall@10 **and** NDCG@10 are both above the archived frozen control
   (0.06406796775709883 and 0.03530651701940677);
3. a human explicitly approves the closure.

A negative result, an operational pause, a quota, a passing check suite or a single ER win
does not close it. The controller reports the first condition on its own; the second and
the third are the author's.
