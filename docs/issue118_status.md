# Issue #118 — status: the P0/P1 chain is in place, the replay generator runs again, and one repeat the policy forbids is now blocked

## What was already delivered (verified in the current tree)

| item | where | state |
|---|---|---|
| versioned contract compiler | `tools/research/science/contract.py` | thresholds read from the authoritative gate, metric registry, canonical suite ids, staged sections with digests |
| deterministic preflight | `tools/research/science/preflight.py` | four layers, verdicts `MACHINE_INVALID` / `MATH_COUNTEREXAMPLE` / `SCIENTIFIC_REVIEW_REQUIRED`, `COLLAPSE_UNGUARDED` at the math layer |
| four-suite contract | `tools/research/verification/` | independent agents, reviewer `APPROVE_TESTS`, `verification_check` re-runs all four |
| failure registry | `tools/research/core/failures.py`, `results/research/issue117/failures/registry.json` | 14 entries, fingerprint-deduplicated, each with fix commit and regression |
| defect replay | `tools/research/integrations/defect_report.py` → `results/research/issue117/reports/defect_report.json`, published copy `docs/issue118_defect_report.json` | runs again in this round (see below) |

## What this round changed

### 1. The defect replay generator was broken and is fixed

`python3 -m tools.research.integrations.defect_report` crashed with

```text
KeyError: 'valid_l1_hard_er_strict_gt'
```

The generator still read the promotion thresholds that issue #120 removed, so the evidence the issue publishes could not be regenerated at all. It now reads the frozen **reference** values, and the class `DUPLICATED_THRESHOLD` is **retired with a reason** instead of being silently dropped: with no promotion floor there is no threshold for a candidate to duplicate. The report carries `retired_defect_classes` next to the active ones.

```text
all_defects_detected  true
active defect classes 10     retired 1
legal spellings wrongly rejected  0
```

### 2. A repeated scientific rejection was not blocked

The registry showed one candidate (`geometry_content_rq_poincare_sq_v1`) sent to review **ten** times, every time with the same `FATAL`. CLAUDE.md §12 and §70 say a scientific verdict is final and the same reviewer must not be asked again; the registry recorded the verdict but nothing consulted it before scheduling the next review.

- `tools/research/core/failures.py::prior_scientific_rejection` returns the recorded verdict for a candidate.
- `tools/research/workflows/pipeline.py::settled_scientific_verdict` consults it before the review agent runs and blocks the task with `blocked: true`.
- Regression test: `tools/research/tests/test_issue118_scientific_guard.py` (a rejected candidate is settled, another candidate is not, a machine failure does not settle a candidate, the repeat is one entry with two occurrences).

The DoD counters now separate the two kinds of avoidable work:

```text
avoidable_review_calls        25   (machine-invalid candidates that were reviewed)
repeated_scientific_reviews    9   (re-reviews of a candidate already judged FATAL)
```

### 3. The deployed-interpreter check, run locally

The GitHub Actions research CI was **removed at the repository owner's request** (commit `5bfe34c2`, "Remove GitHub Actions research CI workflow"), so the CI job named in the earlier issue comment no longer exists in the tree. The same check was therefore executed locally against the interpreter the research package actually runs under:

| interpreter | result |
|---|---|
| `genrec_env/bin/python3.10` | 387 tests, OK |
| `genrec_env_v2/bin/python3.9` (the deployed one) | 387 tests, OK (17 skipped) |

No workflow file was re-added, because that decision belongs to the repository owner.

## Recomputed DoD counters (from the report, not asserted)

```text
runs                              34
review verdicts                   4 CPU_ELIGIBLE, 16 FATAL, 8 MATERIAL, 6 none
science rejections                24
avoidable review calls            25
repeated scientific reviews       9
known failure recurrence          1   (the FATAL above)
registry entries                  14
stage2 runs with exported SIDs    41
time to a verified candidate      23365 s
defect classes replayed           10 active + 1 retired, all caught at the expected layer
```

## Remaining unknowns

- The registry's `known_failure_recurrence` counts the repeated scientific FATAL; it is now also visible as `repeated_scientific_reviews`, but the registry still has no rule that stops a *machine* failure from recurring on the same fingerprint.
- The CI integration evidence is local only, because the owner's decision removed the workflows.
