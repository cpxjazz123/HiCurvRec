# Harness² × GeneRec

Google Research's **Harness²** (<https://github.com/google-research/harness2>) integrated into
this repository. The authoritative specification is **issue #101**; this file records what is
built, how to run it, and what is still missing.

Harness² iterates an agent **harness**, not a model. Here the harness under iteration is this
repository's iteration protocol -- `CLAUDE.md` + `.agent/skills/**/SKILL.md` + `scripts/*.py` --
the task is one Stage-2 mechanism arm, and the judge is `test_recall@10`.

## 1. Install (outside the repository)

| component | location | version |
|---|---|---|
| source | `/home/wlia0047/ar57_scratch/wenyu/harness2` (= `/fs04/scratch2/ar57/wenyu/harness2`) | HEAD `1cfe254e8d25b15f37048c3730134686abe60335` (2026-10-13) |
| venv | `<repo>/.venv` | CPython 3.12.15 (the framework needs `>=3.12,<3.14`) |
| package | editable `harness2==0.1.0` + `[render]` | `uv pip install -e '.[render]'` |
| uv | `/home/wlia0047/.local/bin/uv` | 0.13.3 |

Verified: `harness2-demo --output runs/demo --k 2` -> exit 0, `2 tasks completed; 0 failed`;
`pytest tests/ -q` -> `5 passed, 2 subtests passed`.

**The repository's own environments cannot host the framework**: `genrec_env` is 3.10 and
`genrec_env_v2` is 3.9. Everything under `tools/genrec_agents/` is therefore stdlib-only and
talks to the framework through its documented seams; the trainer never imports any of it.

## 2. P0 -- the offline pipeline (DONE)

No model, no network, no credentials, no GPU.

```
/home/wlia0047/ar57_scratch/wenyu/harness2/.venv/bin/python -m pytest tools/genrec_agents/tests -q
# 9 passed

PYTHONPATH=. /home/wlia0047/ar57_scratch/wenyu/harness2/.venv/bin/python \
    -m tools.genrec_agents.cli sweep --k 1 --modes parallel --offline --runs-dir /tmp/h2_p0
```

What runs:

| file | role | framework seam |
|---|---|---|
| `surface.py` | the editable surface: `CLAUDE.md`, `.agent/skills/<n>/SKILL.md`, `scripts/<n>.py` | `SurfaceSpec` (`core/tree.py:41`) |
| `adapter.py` | the benchmark seam: tasks, domains, rollouts, deliverables, the judgment | `BenchmarkAdapter` (`core/spec.py:201`) |
| `substrate.py` | runs a candidate's verifier against the frozen fixtures, in isolation | `Substrate` (`substrate/base.py:48`) |
| `proposer.py` | `GenrecProposer` (host provider) and `ScriptedProposer` (offline) | `Proposer` (`core/proposer.py:91`) |
| `oracle.py` | the independent reference; a candidate can never import it | -- |
| `tasks.py` + `fixtures/` | the task pool: a served spec and a private held-out set | -- |
| `prompts_ext.py` | registers this surface's prompt text (see §5) | `core/prompts.py:617` |
| `cli.py` | `tasks` and `sweep` | `core/schedule.py:780` |

### The tasks are a miniature of the real gates

Each task ships a frozen fixture (input + declared value) that the candidate may read, and a
**private** held-out set it never sees: same shape, different input, unchanged declared value.

The candidate must ship `scripts/verify_masked_sum.py`, which computes the masked weighted sum
from the input and reports `STATUS:` / `RESULT:`. Two gates decide whether that is acceptable:

* **static** (`adapter.validate_tree`): a verifier that restates a declared value is rejected
  before it is ever rolled -- it contains the literal, which is a forbidden substitution;
* **runtime** (`substrate.run_verifier`): the independent reference recomputes every fixture,
  so a verifier that claims success on a held-out fixture is caught as a **false success**, and
  a run with any false success is **not a deliverable**.

The offline sweep demonstrates both: the `dom1` candidate ships a restating verifier and is
rejected with

```
scripts/verify_masked_sum.py contains the literal 1.0, a declared fixture value:
the verifier must COMPUTE the value from the input, not restate it
```

while `inv1` and the composed `agg1` are adopted, reflected into an `EXPERIENCE.md`
(cue / lesson / evidence) and carried into the next task's library.

### Acceptance evidence (issue #101 §D P0)

1. the pipeline runs deterministically and writes `summary.json`, `selection.json`, the evidence
   tree, `EXPERIENCE.md` and the library `INDEX.md` -- `tests/test_p0_pipeline.py`;
2. a candidate is rejected for a false success, statically and at runtime -- two tests;
3. the library travels: the second task's evidence carries the first task's lesson -- one test.

## 3. The OpenCode backend (issue #102)

`opencode.py` is the execution backend. Everything in it was measured against the
installed CLI, not assumed:

* the flags come from `opencode run --help`, and the preflight refuses to run if the
  installed CLI does not have them;
* the event stream is JSONL (`step_start` / `tool_use` / `step_finish` / `text` / `error`),
  and the tool calls carry their inputs, their outputs, the cost and the tokens;
* the MCP server is configured in the workspace's own `opencode.json` under
  `mcp.servers.<name>`, and the genrec MCP is wired there;
* the provider is explicit: OpenCode's own stored credentials (the default), an
  operator-written config, or the operator's Anthropic-compatible endpoint. A credential
  travels through the child environment only, is never written to a file, and is redacted
  out of every log line that is kept.

`--standalone` is used for every run: the background service keeps the environment it was
started with, so a key exported afterwards is invisible to it, and a private server per run
is also the isolation the issue asks for.

Measured on this machine: `openai/gpt-6-luna` works on the stored ChatGPT OAuth;
`gpt-5.5` is listed by the CLI and refused by that account; the MiniMax key is a 401. A
model can be listed and still be unusable, which is why the preflight probes with a real
call.

### The commands

```bash
cd /home/wlia0047/ar57/wenyu/GeneRec
V=/home/wlia0047/ar57_scratch/wenyu/harness2/.venv/bin/python

$V -m tools.genrec_agents.cli preflight            # can the backend be used here?
$V -m tools.genrec_agents.cli mcp-check            # prove the genrec MCP is callable
$V -m tools.genrec_agents.cli repair-demo          # reject a defect, repair it, recheck
$V -m tools.genrec_agents.cli lifecycle --proposal ... --skip-stage2
$V -m tools.genrec_agents.cli sweep --k 1          # the harness-optimisation loop
```

## 4. The autonomous research campaign (issue #103)

Harness2 is the sole orchestrator: it decides which mechanisms are proposed, which are
qualified, which are run, and what the next round is. OpenCode is the only agent that
writes a mechanism or repairs one. Nothing in the campaign is scripted.

```bash
cd /home/wlia0047/ar57/wenyu/GeneRec
V=/home/wlia0047/ar57_scratch/wenyu/harness2/.venv/bin/python
$V -m tools.genrec_agents.cli campaign --config tools/genrec_agents/campaign_config.json
```

| stage | what happens | the evidence |
|---|---|---|
| preflight | the agent backend, the GPUs, the budget and the historical digests are checked **before** anything runs | `campaign_state.json` |
| ideation | OpenCode proposes one mechanism per angle, from the R8 record and its lessons | `ideation_round1.json` |
| rubric | a deterministic critique freezes the proposals that declare what they must, and selects a **diverse** portfolio | the critique in the ideation file |
| implementation | OpenCode implements each arm in its own tree, with the proposal frozen first | `arms/<arm>/implement.json` |
| verification | the declared invariants, the gradient targets and the forbidden substitutions, checked by a checker that does not know the mechanism's internals | `arms/<arm>/verification.json` |
| repair | bounded, with the findings handed back to the agent | `arms/<arm>/repair_*.json` |
| probe | the **active** probe: the mechanism is forced on, a matched control is run beside it, and the code's occupancy is compared | `arms/<arm>/probe/probe_report.json` |
| regression | the known collapse arm must be rejected by that probe before any cold run | `regression_control.json` |
| reflection | the diagnosis separates an incorrect implementation, a correct-but-harmful mechanism, an infrastructure defect and an uncertain result | `arms/<arm>/arm.json` |
| second round | the researcher is given the first round's diagnoses and proposes again | `ideation_round2.json` |
| harness evolution | a harness change is compared with the prior harness on the **held-back** tasks and adopted only if it wins | `harness_evolution.json` |
| cold Stage2 | the qualified arms, and only those, run the frozen cold 3000-epoch protocol | the MCP's own state and logs |
| evaluation | the frozen Hard-L1 ER, once per admitted arm, against a fresh cold control | `arms/<arm>/er_eval.json` |
| report | every arm, including the failures | `leaderboard.json`, `report.md` |

The budget is declared in `campaign_config.json` and enforced by the controller: the
parallelism, the number of cold runs and the wall clock. A run that was interrupted is
resumed from the evidence on disk, not from the beginning.

Two boundaries the campaign keeps: the checks need torch and the campaign needs the
framework's interpreter, so they cross as a subprocess and a JSON report; and the
mechanism's module and entry-point names are a campaign convention, because a checker
that has to guess a name is a checker that will sometimes look in the wrong place.

## 5. What came out of #102, and what #103 adds

`P0 reuse -> P1 real OpenCode -> P2 unified lifecycle -> P3 parallel research` (issue #102,
all done) and then the research campaign itself (issue #103, the section above).

| phase | content | state |
|---|---|---|
| P0 reuse | the offline adapter and its tests, and the independent checks as **callable judges** (`tools/mvg/judges.py`); CGV G1-G6 and M0-M5 are no longer a mandatory admission gate | done |
| P1 real OpenCode | `OpenCodeProposer` and `OpenCodeSubstrate`: OpenCode proposes/edits the harness and executes the mechanism, the genrec MCP connected through OpenCode's own configuration, full logs, a real failure -> diagnosis -> repair -> recheck | done |
| P2 unified lifecycle | frozen proposal -> adaptive independent checks -> repair -> real-trainer probe -> gated cold Stage2 -> frozen evaluation, with budgets, resume and failure states | done |
| P3 parallel research | several mechanisms in isolation with bounded budgets, a blind held-back task for the harness-optimization outer loop, lesson transfer | done |
| #103 campaign | the autonomous two-round research campaign on top of all of the above | running |

**The execution backend is OpenCode** (issue #102): it is the default agent for both the
Harness2 proposer and the task-execution substrate, and the production path must not depend
on Claude Code or Vertex credentials. The P0 scripted mode stays as the deterministic CI
path.

## 6. Known gaps in what is built

* **The static check is literal-based.** A candidate that hardcodes a value through an
  expression rather than a literal slips past `validate_tree`. The independent-reference
  comparison of the forward oracle (production against a reference implementation, forward
  value *and* intermediates) is what closes this; until then the runtime gate catches only
  the cases where the lie changes the reported verdict.
* **The runtime gate needs a mutated fixture to exist.** A task with no held-out fixture has
  no mutation detection at all; `LocalSubstrate.preflight` refuses to run such a task.
* **The offline proposer is scripted.** It demonstrates the plumbing and the gates; the real
  editor is OpenCode (§3), and the offline path stays as the deterministic CI path.
* **The independent checks read code, not prose.** A candidate that explains a defect in a
  comment is not substituting the operation; a static rule that fired on the explanation
  would reject a correct repair for saying why it was wrong.

## 7. The one extension point used

The framework resolves its per-surface prompt text from a module-level dict built at import time
(`core/prompts.py:617`) and exposes no public registration hook, so a surface it does not ship
text for cannot be prompted. `prompts_ext.py` adds this repository's surface text to that dict
at import time. No framework source is modified, no fork is kept, and the extension is
fail-closed: the framework raises on an unknown surface name, so an upgrade that renames the
dict breaks loudly instead of degrading silently. When the framework grows a public hook, this
file becomes one call and is deleted.

## 8. Rules this integration does not break

* no fork and no patch of the framework core; only the seams above are used;
* the trainer's environment never imports the framework (subprocess / MCP / JSON only);
* the frozen math, thresholds, references, judges and held-out data are not candidate-editable;
  `adapter.validate_tree` enforces it;
* `main` only, no worktree: candidate harnesses live in scratch copies, and only a reviewed
  adoption is written back into `main`;
* runs and libraries stay outside the repository (the zero-bookkeeping rule of CLAUDE.md §12);
  the repository receives code and reviewed harness edits only.
