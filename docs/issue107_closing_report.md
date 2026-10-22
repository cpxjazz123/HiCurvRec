# Issue #107 — closing report

**Issue:** `[PERSISTENT RESEARCH][DO NOT CLOSE] Autonomous multi-mechanism iteration until Stage3 improvement`
**Closed:** 2026-10-22, on the maintainer's explicit instruction.

---

## 1. What was asked for

Build a durable, self-running research system that keeps proposing mechanisms, freezing
them, training them once, measuring them, and learning from the result — until one arm
clears the frozen pair-level ER gate **and** improves both Stage-3 metrics against the
archived control.

## 2. What was built (complete, in the repository, tested)

| Deliverable | Where | Evidence |
|---|---|---|
| Enforceable guardrails: the frozen seed, one cold run per candidate, the ER read once, Stage-3 once per arm, no re-running a failed arm, no renamed copy of a mechanism, no re-reading a frozen evaluation | `tools/harness2_genrec/guardrails.py` | 49 tests; the prohibited operations are refused **before** any process is launched |
| The durable controller: a persistent state machine (`read → hypothesise → screen → train → evaluate → stage3 → learn`), crash recovery, idempotent dispatch, job supervision (a lost job is classified, never silently retried), isolated ports and output paths, an append-only audit log | `tools/harness2_genrec/controller.py` | 49 tests, including restart, crash-between-transitions and duplicate-dispatch |
| The transfer analysis: what the SID structure a recommender consumes implies — sequence information, train→test frequency drift, per-level entropy and Gini, popularity and cold-item slices, collisions, prefix-branch structure, Stage-3 error slices, ER↔ranking correlation | `tools/harness2_genrec/stage3_transfer.py` | refreshed from the store before every portfolio |
| The learning loop: the portfolio is proposed by the research agent, ranked, screened in parallel, trained one cold run at a time within a resource budget, and the round's outcome becomes the next round's brief | `tools/harness2_genrec/candidates.py`, `candidate_run.py`, `ideation.py` | 30 rounds run; 100+ audit events; the negative results feed the next brief |
| Per-arm reports and the queue/pause/next-action status, from the CLI and on the issue | `tools/harness2_genrec/cli.py`, `results/harness3_runs/final/` | `controller status/tick/run/pause/resume/backfill-store`; status posted on the issue throughout |
| The runbook and the resource configuration | `docs/runbook_research_controller.md`, `results/harness3_runs/controller/resource_config.json` | — |

## 3. What was measured (26 arms)

The frozen ER is the share of held-out behaviour pairs that share their level-0 code over
the share of random pairs that do. The archived control sits at **R@10 = 0.064068**,
**NDCG@10 = 0.035307**, ER 16.3345; the gate is ER > 16.38954.

| Arm | frozen ER | ΔR@10 | ΔNDCG@10 |
|---|---|---|---|
| directed_assignment_prefix_preservation | 17.2983 | (Stage-3 stopped with the campaign) | |
| directed_assignment_prefix_information | 16.9469 | −0.000313 | −0.000631 |
| directed_codeword_tangent_repulsion | 16.8347 | −0.001375 | −0.001365 |
| geometry_transport_seed43 | 16.7397 | (Stage-3 stopped with the campaign) | |
| directed_capacity_constrained_assignment_partition | 16.7324 | −0.003343 | −0.001928 |
| directed_assignment_usage_balance | 16.6516 | −0.001654 | −0.000994 |
| behavior_residual_geometry_transport | 16.6160 | −0.000679 | −0.000763 |
| directed_pair_separation_codebook_geometry | 16.6019 | (Stage-3 stopped with the campaign) | |
| directed_codeword_margin_geometry | 16.5918 | −0.000331 | −0.000639 |
| behaviour_conditioned_residual_target | 16.5034 | −0.003064 | −0.001496 |
| directed_capacity_balanced_assignment_partition | 16.4929 | (Stage-3 stopped with the campaign) | |
| behavior_directed_soft_code_transition_kl | 16.4028 | −0.001811 | −0.001457 |
| **r8_control (archived)** | **16.3345** | **0** | **0** |

**12 arms cleared the frozen ER gate. 9 arms carry both a frozen ER and a Stage-3 reading.
0 arms improved both Stage-3 metrics.** The campaign was stopped by the maintainer's
instruction while one cold run and one Stage-3 were in flight; their arms are marked as
stopped above rather than reported as if they had been measured.

## 4. The mechanism that was found

The ER and the downstream quality move in **opposite** directions, and the cause is
identified, not guessed (ledger entry H107-03):

> The frozen pair-level ER is bought with **code concentration**. A behaviour-driven merge
> of the level-0 codes raises the positive match rate, and the same merge removes the
> distinctions the recommender ranks by. The arm with the best ER (17.13) has a level-0 Gini
> of 0.313 against the control's 0.216 and 62.5 prefix branches against 65.9, and its
> Stage-3 recall@10 was the worst measured; the arm with the healthiest structure has an ER
> far below the control.

The ER's arithmetic says exactly what is left to try, and it was put in the brief and in a
seventh research angle: the denominator of the ER is the collision probability of the
level-0 cells, a function of the cell **sizes** alone, while the numerator is a function of
the **grouping**. Every arm that raised the ER did it by pulling a pair's codewords
together — the sizes are what changed, and the sizes are what the recommender loses. The
direction nobody took is to move the codebook as a **rigid body** — a rotation, a shear, an
alignment of the configuration with the behaviour graph — so the grouping changes without
two codewords being pulled together.

## 5. Two measured negative results, recorded so they are not retried

* **The short probe is not a gate** (`tools/harness2_genrec/probe_fidelity.py`, ledger
  H107-04): over the 19 arms with both a probe export and a cold export, the probe's paired
  level-0 Gini predicts the cold run's with a rank correlation of 0.38, the number of codes
  in use with 0.47, and the paired enrichment ratio with the cold run's ER with 0.51.
  Gating on that ratio would have wrongly refused an arm whose cold run cleared the gate.
* **A mechanism-level merge statistic is not a gate either** (ledger H107-05): moving a
  pair's codewords together on the proposal's own mini case and measuring the loss's
  response gives −0.032 on average for the arms that cleared the gate against −0.048 for
  the arms that missed it, with a rank correlation of 0.27.

Both are why no cheap screening gate was added: it would have refused arms whose cold runs
were fine and passed arms whose cold runs concentrated the codes. The cold run remains the
only honest measurement.

## 6. Defects found by running the system, and fixed

Each of these was found by the system running, not by reading it:

1. The store's preflight refused the admission of a *finished* run (the run's own products
   were in the output directory) and the refusal was swallowed — five measured arms, two
   with Stage-3 results, were invisible to the store the success check reads.
2. The ER was written under `l1_er` while the success check read `hard_l1_er`.
3. A manifest is immutable, so the ER and the Stage-3 live in two manifests of one arm; the
   check needed both in one and therefore counted neither. The read phase now joins an
   arm's manifests — the arms with both measurements went from 6 to 9.
4. The transfer analysis was a frozen snapshot and nothing refreshed it.
5. An early version of the diversity requirement made the family a validity condition, so
   four consecutive rounds produced no candidates at all.
6. The evaluation was serialised behind the training queue, though reading the ER needs no
   GPU.

## 7. What was **not** achieved

**No arm improved both Stage-3 metrics against the archived control.** The issue's own
success criterion was therefore never met. The scientific gap — how to raise the pair-level
ER without concentrating the codes, which is what the recommender needs — is stated exactly
in §4 and in the ledger, and the next portfolio's brief carries it, together with the
seventh research angle that targets it.

The controller was paused with the reason recorded in its state, the worker was stopped and
every training and Stage-3 process was terminated; the four GPUs are free. The system is
complete, durable and restartable:

```
python -m tools.harness2_genrec.cli controller resume --state results/harness3_runs/controller/state.json
```

re-attaches the worker to the campaign and continues from the recorded state.
