# genrec_agents — the first-party agent framework

This package runs the repository's mechanism research with **real OpenCode CLI subprocess agents**
and one durable DAG. It imports nothing outside this repository: every type it used to borrow from a
third-party agent-harness framework (a surface specification, a benchmark adapter contract, a
proposer protocol, a substrate interface, a scheduler) is declared here, with the same fields and
the same semantics.

## The chain

```
campaign / cli
  -> dag.py        one durable truth: tasks, leases, attempts, hashes, reservations
  -> worker.py     the supervised dispatcher: per-role concurrency, heartbeats, backoff
  -> roles.py      L / A / M / I: fixed prompts built from frozen inputs, output contracts
  -> executor.py   task -> isolated workspace -> opencode.run_agent -> verifier -> a record
  -> opencode.py   subprocess.Popen([opencode, run, --standalone, --format, json, ...])
  -> a real OpenCode session, optionally with the genrec MCP server
  -> the session JSONL + the parsed summary + the verdict, written as evidence
  -> dag.complete / dag.fail   (atomic; a gate with no measurement cannot be completed)
```

Nothing in the chain spawns an agent implicitly. A task is DONE only when its execution record
exists; an idle tick, a `coordinator` status change or a hand-written ledger entry is not an
execution.

## Running the tests

```
/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 -m pytest tools/genrec_agents/tests -q
```

Zero GPU, zero network, zero credentials. The executor's tests use a mock OpenCode binary that emits
the same JSONL shape as the real CLI, so they prove the orchestration without a model.

## What lives where

| file | role |
|---|---|
| `spec.py` | `ChannelSpec`, `BenchPrompts`, `BenchSpec`, `RunView` |
| `surface.py` | `SurfaceSpec`, `GENREC_SURFACE`, and the editable-tree readers and writers |
| `prompts.py` | the surface text and the INV/DOM prompt parts |
| `proposer.py` | `Proposer`, `Reply`, and the three proposers (host, OpenCode, scripted) |
| `adapter.py` | the task pool's gates: a restating verifier and an edited threshold are refused |
| `sweep.py` | the propose → gate → select loop, and the library that carries the winner onward |
| `library.py` | the per-mode, per-domain improvement library |
| `substrate.py` | the isolated verifier runner |
| `opencode.py` | the ONLY place a CLI process is started |
| `executor.py` | one task → one isolated session → one evidence record |
| `roles.py` | the L/A/M/I bindings |
| `worker.py` | the supervised production worker |
| `dag.py` | the scheduler, the leases and the experiment reservations |
| `controller.py`, `campaign.py`, `lifecycle.py` | the research state machines |
| `partition_degeneracy.py` | the gate that refuses a collapsed codebook partition |
| `tools/cgv/**` | the six-gate independent verifier (a separate package: it imports neither this one nor the arm it checks) |
