# Native behavior-first research tools — Issue #117

The only active research orchestration lives in `tools/research/`; formal training and job status remain in `tools/genrec-mcp/server.py`.

## What is live, and what is the Issue #117 surface

Two research surfaces live in this package and they are not the same thing.

**Live research path (Issue #121 rounds):** `rounds/` (the round state machine,
its phases and its jobs), `stage2/` (the frozen engineering contract: paths, the
launcher, the conformance checks and their regression tests) and `agents/` (the
real OpenCode subprocess and its receipts). A candidate carries its mechanism;
everything else about running it belongs to `stage2/`.

**Shared infrastructure:** `core/` (the canonical ledger and the atomic evidence
helpers) and `science/` (train-only measurements, the SID layer record and the
forensics study). Both are used by the live path.

**Issue #117 surface, still mounted by the MCP:** `mcp/` (the async handlers and
the detached workers behind `research_start` and `research_verify_*`),
`workflows/` (DATA, LITERATURE, GEOMETRY, REVIEW), `verification/` (independent
test authoring, review and admission) and `integrations/` (read-only preflight).
They stay because `tools/genrec-mcp/server.py` imports them; they are not part of
a round.

`tests/` holds the offline regressions of all of it. A test that needs torch runs
only in the Stage 2 interpreter and skips elsewhere; a skipped check is not a pass.

## Source structure

```text
tools/
├── genrec-mcp/
│   └── server.py              Official Stage2/Stage3 execution and status
└── research/
    ├── settings.py            Source-defined paths and safe defaults
    ├── __main__.py            No-argument inspect/step
    ├── core/
    │   ├── dag.py             Canonical task ledger, leases and events
    │   └── evidence.py        Hashes and atomic JSON evidence
    ├── agents/
    │   ├── runner.py          Real OpenCode CLI subprocess receipts
    │   └── prompts.py         Candidate-specific role prompts
    ├── science/
    │   ├── behavior.py        Train-only user-sequence measurements
    │   └── contracts.py       Behavioral and candidate identity rules
    ├── mcp/
    │   ├── handlers.py       Async start/status/results/cancel/resume
    │   ├── worker.py         Detached DAG worker; stdin-only run identity
    │   ├── verification_handlers.py  Async verifier authoring and review
    │   └── verify_worker.py  Real detached verification sessions
    ├── workflows/
    │   └── pipeline.py        DATA + LITERATURE → GEOMETRY → REVIEW
    ├── verification/
    │   ├── workflow.py        Independent test authoring and review
    │   ├── admission.py       Deterministic evidence checks and test execution
    │   ├── cases/             Reviewed per-candidate test suites
    │   └── tests/             Fail-closed verification regressions
    ├── integrations/
    │   └── preflight.py       Read-only training and verifier path checks
    └── tests/                 Offline DAG, prompt and behavior tests
```

## How it works

`python3 -m tools.research` safely initializes the research ledger and reports state without model or GPU access; `python3 -m unittest discover -s tools/research/tests -v` and `python3 -m unittest discover -s tools/research/verification/tests -v` exercise offline regressions.

DATA must produce measurable train-only behavior evidence, LITERATURE must cite applicable methods, and only then can GEOMETRY propose a candidate and an independent REVIEW assess it.

Independent MATH_VERIFY, GRADIENT_VERIFY and RUNTIME_VERIFY Agents prepare mathematical, gradient, real DDP and actual entrypoint tests; a separate VERIFY_REVIEW Agent checks their source, expected results, evidence and exact hashes.

After independent review, the mechanism-specific `test_math.py`, `test_gradient.py`, `test_ddp.py` and `test_entrypoint.py` are stored in `verification/cases/<candidate_id>/` and remain fail-closed until actual execution.

The MCP `verification_check(source_dir, verification_id)` verifies all source, proposal, test and agent evidence and re-runs the tests; `stage2_start(source_dir, verification_id)` requires the same verification and executes a real trainer smoke before full Stage2.

The research package never starts GPU training directly, never uses validation/test future behavior to build SID, and never accepts textual Agent approval as proof of test execution.

## Boundaries

Output state is isolated to `results/research/issue117/`, live OpenCode Agents are disabled by default, and the model/execution environment requires real filesystem isolation before live work.

The standard Stage2/Stage3 source trees are unchanged. Stage2-to-Stage3 admission is a **technical validity** decision (Issue #120): `stage2_start` still requires its independent executable tests and smoke, and `stage3_start` refuses only an unreadable, malformed or ambiguous SID. Hard-L1 ER, level-2 prefix counts, L3 collision, code usage and redundancy are still measured on the frozen independent validation pairs and reported as diagnostics, but **no quality floor may refuse a Stage 3 launch**; the Resample values are a historical reference, not a boundary. The final Stage3 success rule remains in `CLAUDE.md`.

`tools/stage2_stability_probe.py` has been retired because it imported modules from the removed legacy verification package and the active MCP no longer exposes the probe.

No actual HPC Agents, multi-GPU tests or Stage2 training have been executed merely by creating these tools; verify those separately before interpreting a candidate as admitted.

## Native MCP research API

The existing GenRec MCP Server now exposes `research_start()` → `run_id`, `research_status(run_id)`, `research_results(run_id)`, `research_cancel(run_id)` and `research_resume(run_id)` through the single `tools/research/mcp/handlers.py` adapter.

`research_start` creates a run under `results/research/issue117/runs/<run_id>/` and schedules DATA and LITERATURE independently from the sole `results/research/issue117/dag/state.json` authority, while GEOMETRY and REVIEW cannot start until their real evidence dependencies are completed.

Detached workers are real Python processes; MCP returns immediately and the user queries status instead of awaiting long OpenCode sessions inside the MCP request.

The distinct `research_verify_start(run_id, source_dir)` and `research_verify_review(run_id, source_dir)` MCP operations launch actual verification Agent work for the run's **currently frozen** candidate, and `research_verify_status(run_id)` reports their real worker progress.

Test drafts are only written to the run evidence area and require independent source review and tracked-code promotion before `research_verify_review` may approve their exact committed test files; completion does not launch GPU training.

The existing `verification_check(source_dir, verification_id)` and `stage2_start(source_dir, verification_id)` remain separate, deterministic, privileged MCP tools with their source-bound four-suite tests and live Trainer smoke checks.

**Deployment safety**: `LIVE_AGENTS_ALLOWED = False` disables all OpenCode child processes by default, while a research run can still execute DATA using the configured train-data interpreter; running Agent code requires operating-system-level filesystem/tool isolation before enabling live mode.

**Cancellation**: `research_cancel` marks the global DAG and a run-local cancellation token, and active OpenCode subprocesses terminate cooperatively; actual worker death and incomplete cancellation are reported rather than silently claimed successful.

MCP exposes no options for changing training hyperparameters, bypassing source checks, launching GPU from research tools or accepting a Reviewer-only verdict as independent executable evidence.

## Mandatory per-SID Stage2 metric record

For **every new Stage2 SID** (including models that will not reach Stage3), immediately after training/export and **before consulting its Stage3 result**, use `tools.research.science.sid_layer_analysis.record_single_sid(sid_path)` with the same real train sequences and frozen independent-validation pairs. It writes an atomic, SID-hash-versioned report to `results/stage2_RQ-VAE/<ARM>/stage2_metrics/<sid_sha256>.json`. A saved report is required for the experiment's audit/review and must be linked in the Issue/run outcome; this does not create an additional Stage3 performance gate.

Every report includes numeric `metrics.valid_l1_hard_er`, `valid_l1_positive_hit_rate`, `valid_l1_negative_hit_rate`, `level2_distinct_prefixes`, `level3_distinct_triples`, `l3_collision`, `l3_refinement_gain_bits`, and `conditional_l3_gain`, together with frozen Resample thresholds, gate PASS/FAIL, methodology, validation sample count and SHA-256 provenance for SID/train/validation/baseline/source. Both L1 HitRates and all L3 measures are diagnostics only; **only ER > 16.38953942232631 and L2 prefixes > 16145 are hard Stage2 quality thresholds**. A negative G3 value is valid and must not be converted to zero or used as a cutoff. If any metric cannot be computed, record an explicit failure with the actual cause and do not claim a complete report. This measurement step requires the Stage3 Python environment with NumPy/Pandas for offline computation, not the stdlib-only MCP process.


## Versioned contract, deterministic preflight and the failure registry (Issue #118)

`science/contract.py` is the single source of truth. Admission thresholds are
**read** from `tools/genrec-mcp/stage2_quality_gate.py` rather than copied, the
frozen capacity/protocol and the metric registry live here, and a proposal
compiles into a staged contract (`behavior -> geometry -> math -> controls ->
falsifier -> stage3`) whose sections each carry their own digest. The contract
exposes `contract_sha256` and `prompt_input_digest`, so an agent, a reviewer and a
verification manifest can prove they saw the same version. Suite identifiers are
canonicalised in one place (`math` / `test_math.py` resolve to the same id).

`science/preflight.py` runs four layers in dependency order — output, contract,
math, falsifier — and stops at the first that cannot be trusted. Verdicts are
distinct on purpose: `MACHINE_INVALID` and `MATH_COUNTEREXAMPLE` are repairable
engineering defects whose diagnostics name the location, the expected value, the
observed value and a repair; only `SCIENTIFIC_REVIEW_REQUIRED` reaches the
independent reviewer. Entries with `verdict=None` are warnings (for example an
expected direction that contradicts measured evidence) and never mask a failure.
The math layer evaluates the audited reference, not the proposal's prose: the
flat limit is checked on the unclamped formula, the clamp gets its own
domain check, and a stated `4||x-y||^2`-style limit is compared textually.

`core/failures.py` records every reproducible failure under
`results/research/issue117/failures/registry.json`, deduplicated by a fingerprint
over `(stage, error_class, rule, expected, actual)` — never by prose similarity —
with occurrences, run ids, fix commit, regression test and status. The pipeline
records machine failures from the task error classifier and scientific
`FATAL`/`PARKED` verdicts, so a run can report `known_failure_recurrence`
instead of re-learning the same defect.

CI runs a deployed-Python 3.9 job that imports the research package, because the
MCP server uses the host's system `python3` and a 3.10-only annotation once took
the whole tool surface down at call time.

### Machine report and the failure ledger

`integrations/defect_report.py` replays every known engineering defect as a live
injection against the preflight and writes
`results/research/issue117/reports/defect_report.json`: one row per class with
the layer that caught it, the observed diagnostic code, the verdict and whether
a legal spelling was wrongly rejected. It also publishes the DoD counters —
`runs`, `review_verdicts`, `known_failure_recurrence`, `registry_entries`,
`machine_repair_attempts`, `avoidable_review_calls`, `science_rejections`,
`time_to_verified_candidate_seconds` — computed from the run directories and the
registry, so the baseline can be recomputed rather than asserted. CI runs it and
fails if any class escapes.

`integrations/backfill_failures.py` records the defects #117 actually hit, each
with the commit that fixed it and the regression that now proves it, so the
registry starts as a ledger instead of an empty counter.

## Multi-agent rounds (Issue #120)

A round replaces the single-proposal/single-review bottleneck with an
experiment-first pipeline:

```text
3 ideation agents (1-3 ideas each, 3-9 total)
  -> 3 independent scorers, each judging EVERY idea (full matrix, medians, disagreement)
  -> the rule keeps ONE mechanism: the single best qualified idea (rejections logged;
     an out-of-scope idea is never selected, not even as the fallback)
  -> one implementation agent builds that mechanism in its own copy of the baseline
  -> the verifier reviews it
     -> any correctness blocker: a dedicated repair agent, then the review again
     -> no progress after the revision budget: BLOCKED with evidence
  -> the verified mechanism enters the Stage 2 queue in the frozen four-card shape
  -> its Stage 2 hands over to Stage 3 automatically

A round is not a shortlist. It produces exactly one mechanism and carries it all
the way to a measured Stage 3 result; the next round proposes the next mechanism
(`max_selected = 1`, `keep_single_best`). Derived controls (the behaviour-off and
geometry-off ablations of that one mechanism) are optional and are recorded
separately from the mechanism.
```

`rounds/` holds it: `ideas.py` (idea contract, cross-agent duplicates with
provenance), `scoring.py` (matrix, aggregate, disagreement, the one-mechanism rule),
`pipeline.py` (durable round state machine with every event, rejection, revision
and failure signature), `worker.py` (the agent stages), `jobs.py` (Stage 2 /
Stage 3 queue). The research package never trains: it prepares and verifies the
candidates, and the MCP owns the GPU boundary.

Round state lives in `results/research/issue120/rounds/<round_id>/` (`state.json`,
`jobs.json`, `agents/<role>/`), implementations live in
`results/research/issue120/trees/<round_id>/<idea_id>/`, so two candidates can
never overwrite each other.

MCP tools: one per phase, so a round can be driven step by step or in one
call.

| tool | phase | GPU |
|---|---|---|
| `research_round_start(objective, issue, run)` | create a round (`run=false` only creates it) | no |
| `research_round_ideate(round_id)` | three ideation agents, the scope gate | no |
| `research_round_score(round_id)` | three scorers over every idea, the multi-select | no |
| `research_round_implement(round_id)` | one implementation agent per selected idea, each on a fresh copy of the one permitted baseline | no |
| `research_round_verify(round_id)` | the verifier of each implementation and the repair loop | no |
| `research_round_run(round_id)` | every remaining agent phase in one call | no |
| `research_round_launch(round_id, parallel)` | the Stage 2 queue: one candidate on all four cards, the frozen recipe | **yes** |
| `research_round_collect(round_id)` | collect the Stage 2 runs, hand every valid SID to Stage 3 | **yes** |
| `research_round_status` / `research_round_results` | where the round stands / the full audit record | no |

A phase is idempotent: a call for a phase the round has left reports the
current step and does nothing. One lock per round (`rounds/lock.py`) keeps two
runners from paying for the same agents twice. A Stage 2 launch for a round
candidate uses the verification id `round:<round_id>:<idea_id>`, whose evidence is
three verifier verdicts, no unresolved blocker, the source digest they reviewed
and a real smoke result.

## Joint behaviour x hyperbolic rounds (Issue #121)

The research target is a mechanism in which **training-only user behaviour and
an explicit curved operation are computationally coupled**, so that removing
either changes the core computation. The contract is hard coded in
`rounds/agents.py` (`RESEARCH_OBJECTIVE`, `SCOPE_CONTRACT`) and reaches every
role: the three ideation prompts, the three scoring prompts, the implementation
specification, the three verification reports and the repair prompt.

`rounds/scope.py` checks the contract against the **content** an agent wrote,
not against a flag it set: the behaviour signal and how it is computed, the
curved operator and where it acts, one connected pathway with the equation that
connects them, the three ablations and the minimal specification. A behaviour
suffix on an unchanged Euclidean quantizer, a hyperbolic-only distance swap, two
unrelated losses and inherited hyperbolic code that nothing calls are all out of
scope, whatever they score. The scope decision is audited against the
proposal's own content, so one reviewer's label never decides an idea; a
disagreement is recorded instead.

No trainer may contain an absolute path: the repository root is derived by
walking up to the `stage2_RQ-VAE` marker, and every artifact path derives from
it (`REPO_ROOT / "results" / "stage2_RQ-VAE" / ARM_NAME`). The preflight runs
at the **promoted** location, because a path built from `Path(__file__)`'s depth
is right in the round workspace and wrong after the copy, and it refuses an
absolute literal, a `parents[N]` anchor, an artifact inside the source tree, or
a result directory outside the arm.

`rounds/baseline.py` holds the one permitted source tree,
`stage2_RQ-VAE/TIGER_RQ-VAE`. Every selected idea gets a **fresh copy** of it,
with the resolved path and the source digest of both the baseline and the copy
recorded. The baseline is read-only: a missing or unreadable source fails
closed, and a candidate whose output paths would write into the baseline's
result directory or into another candidate is refused before it can train.

The stops are kept apart: `AGENT_INFRA_ERROR` (an agent crashed, timed out or
wrote an unusable answer) is retried and never blocks a mechanism,
`IMPLEMENTATION_NEEDS_FIX` is a real defect and goes to the repair agent,
`RESOURCE_BLOCKED` is a measured infeasibility (the frozen run projects beyond
the budget) and keeps the idea for an optimisation route, and `STAGE2_3_FAILED`
is a training or evaluation failure. A candidate stops asking for agents after
five repair rounds and is recorded with its findings.

## Pre-score evidence probe (Issue #123)

Before the reviewers see a proposal, the framework measures it:

```text
3 ideation agents -> scope screen
   -> evidence probe (per proposal, bounded):
        behaviour probe   train-only statistics of the stated hypothesis
        math checks       small CPU tensors: round trip, curvature sign, domain guard,
                          matched-Euclidean sensitivity (PASS/FAIL/INCONCLUSIVE/NOT_RUN)
        literature pass   OpenAlex (Semantic Scholar fallback), 2-3 mechanism queries,
                          <=15 verified papers, cached
   -> one bounded evidence-driven revision (new proposal hash, affected probes re-run)
   -> frozen evidence bundle (same bytes to all three reviewers)
   -> 3 independent reviews -> pairwise debate -> immutable decision.json
```

The probe is read-only, CPU-only and capped: **120 s** of data probe per
proposal, at most **3** literature queries and **15** papers per proposal, at
most **1** evidence revision per proposal, and **no** Stage 2 or Stage 3 run
inside the probe. Its artifacts are written per proposal under
`<round>/probes/<idea_id>/`: `probe.json`, `literature.json`,
`math_checks.json`, `evidence_bundle.json`.

What the probe cannot establish is stated, never filled in:
`DATA_UNAVAILABLE` (the train split lacks the field), `UNKNOWN` (not
established), `NOT_CHECKED` (the literature pass did not run), `NOT_RUN` (the
check could not execute) and `NOT_IMPLEMENTED` (no diagnostic exists for that
hypothesis family). A round trip `log_0(exp_0(v)) = v` shows the maps are
consistent, **not** that the curvature helps; a small proxy assignment is not
the candidate's own pathway; and a non-Euclidean effect is not evidence that
Stage 3 Recall will improve.

The probe never rewrites a score. It can lower one: evidence that contradicts a
hypothesis is evidence, not a promotion.

## Weighted scientific scoring (Issue #122)

Selection is not "0–5 scores, unweighted average, highest wins". A round

1. assembles one **frozen evidence bundle** per proposal (behavioural evidence,
   geometric necessity, prior work, the experiment ledger with its comparability
   tags, implementation and cost). A section that cannot be grounded says
   `UNKNOWN`; a literature search that was not run says `NOT_CHECKED`;
2. has the three independent reviewers score **every** proposal on the seven
   dimensions of the versioned rubric, with evidence ids, one objection and a
   counterexample when they claim a defect;
3. gives a proposal whose objection names a mathematical identity, a
   cancellation, a leakage or a dimension mismatch **one bounded revision**, then
   reviews the revised proposal afresh under a new proposal hash;
4. compares the finalists head to head (order-balanced A-vs-B), aggregates with
   Copeland and records ties, cycles and the measured position bias;
5. writes an **immutable decision record** (`decision.json`) with the rubric
   version and digest, the proposal and evidence digests, the three raw reviews,
   the weighted matrix, the pairwise result, the winner and its ordered backups.

The rubric lives in `rounds/rubric_v1.json`: seven dimensions whose weights sum
to 100% (behaviour evidence 20, geometry necessity 20, novelty 20, downstream
mechanism 15, falsifiability 10, implementability 10, exploration value 5), each
with anchored 0–5 meanings. **Change the weights only before a new round
starts** — a decision is scored with the rubric it pinned. The prompt, the
parser and the matrix read the same names, and an unknown, renamed or missing
dimension is a schema error, never a partial average.

There is no automatic fallback: when nobody qualifies the round reports
`NO_QUALIFIED_IDEA` and goes back to ideation. A later recalculation writes
`rescored_audit_<timestamp>.json` beside the decision and never rewrites it.

A round dispatches its Stage 2 runs in the frozen recipe's own shape:
`STAGE2_LAUNCH_STRATEGY = "ddp4"` gives one candidate all four cards, so the
global batch is 4096 and an epoch costs 0.16 s instead of the 1.5 s a single
card needs. Packing several one-card jobs (`"spread"`) is faster in wall clock
for many candidates, but their global batch is 1024, which is a different
optimisation regime and makes the comparison to the frozen baseline invalid --
that is why it is not the default.

Rounds are filed under the issue that opened them: `results/research/issue121/
rounds/<round_id>/` and `results/research/issue121/trees/<round_id>/<idea_id>/`.

**Admission is technical validity only.** ER, level-two prefix counts, L3
collision, code usage and redundancy are measured and reported as diagnostics; a
Stage 2 run with poor numbers still proceeds to Stage 3, and every Stage 3 result
- including the low ones - stays in the registry. There is no quality floor and
no best-only selection.
