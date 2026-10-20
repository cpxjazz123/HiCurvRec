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
`genrec_env_v2` is 3.9. Everything under `tools/harness2_genrec/` is therefore stdlib-only and
talks to the framework through its documented seams; the trainer never imports any of it.

## 2. P0 -- the offline pipeline (DONE)

No model, no network, no credentials, no GPU.

```
/home/wlia0047/ar57_scratch/wenyu/harness2/.venv/bin/python -m pytest tools/harness2_genrec/tests -q
# 9 passed

PYTHONPATH=. /home/wlia0047/ar57_scratch/wenyu/harness2/.venv/bin/python \
    -m tools.harness2_genrec.cli sweep --k 1 --modes parallel --offline --runs-dir /tmp/h2_p0
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

## 3. What comes next (issue #101 order)

`P0 -> independent MVG + mutation suite -> CGV/launch enforcement -> P1 -> P2 -> P3`

| phase | content | state |
|---|---|---|
| MVG | the `MechanismSpec` schema, the M0-M5 gates, the named mutation suite, `mvg_report.json` | not started |
| CGV enforcement | live G1-G6 wired to real seven-group specs; `stage2_start` refuses without a fresh certificate bound to the source/spec/dataset digests | not started |
| P1 | the real executor: Claude Code CLI + genrec MCP, isolated candidate workspaces, `k=1` | not started |
| P2 | real probes: MVG + CGV before bounded train-only stability probes; the R8 `#91-#100` split for development, a held-back split for acceptance | not started |
| P3 | adoption into `main` only after a held-back regression suite and human review | not started |

## 4. Known gaps in what is built

* **The static check is literal-based.** A candidate that hardcodes a value through an
  expression rather than a literal slips past `validate_tree`. The independent-reference
  comparison of MVG's M1 (production against a reference implementation, forward value *and*
  intermediates) is what closes this; until then the runtime gate catches only the cases where
  the lie changes the reported verdict.
* **The runtime gate needs a mutated fixture to exist.** A task with no held-out fixture has
  no mutation detection at all; `LocalSubstrate.preflight` refuses to run such a task.
* **The offline proposer is scripted.** It demonstrates the plumbing and the gates; it does not
  show a model improving a harness. That is P1.

## 5. The one extension point used

The framework resolves its per-surface prompt text from a module-level dict built at import time
(`core/prompts.py:617`) and exposes no public registration hook, so a surface it does not ship
text for cannot be prompted. `prompts_ext.py` adds this repository's surface text to that dict
at import time. No framework source is modified, no fork is kept, and the extension is
fail-closed: the framework raises on an unknown surface name, so an upgrade that renames the
dict breaks loudly instead of degrading silently. When the framework grows a public hook, this
file becomes one call and is deleted.

## 6. Rules this integration does not break

* no fork and no patch of the framework core; only the seams above are used;
* the trainer's environment never imports the framework (subprocess / MCP / JSON only);
* the frozen math, thresholds, references, judges and held-out data are not candidate-editable;
  `adapter.validate_tree` enforces it;
* `main` only, no worktree: candidate harnesses live in scratch copies, and only a reviewed
  adoption is written back into `main`;
* runs and libraries stay outside the repository (the zero-bookkeeping rule of CLAUDE.md §12);
  the repository receives code and reviewed harness edits only.
