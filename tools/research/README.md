# Native behavior-first research tools — Issue #116

The only active research orchestration lives in `tools/research/`; formal training and job status remain in `tools/genrec-mcp/server.py`.

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

Output state is isolated to `results/research/issue116/`, live OpenCode Agents are disabled by default, and the model/execution environment requires real filesystem isolation before live work.

The standard Stage2/Stage3 source trees are unchanged, Stage2 metrics remain diagnostic only, and the current formal Stage3 success rule is defined in `CLAUDE.md`.

`tools/stage2_stability_probe.py` has been retired because it imported modules from the removed legacy verification package and the active MCP no longer exposes the probe.

No actual HPC Agents, multi-GPU tests or Stage2 training have been executed merely by creating these tools; verify those separately before interpreting a candidate as admitted.
