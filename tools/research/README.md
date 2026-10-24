# GeneRec research tools — new clean architecture for Issue #116

This package is a *new* small tool system against the cleaned repository. It does
not import or restore any historical research controller, experiment campaign, or
old agent-Harness framework. The Stage2/Stage3 trainers, CGV engine, and MCP
server stay exactly where they are, unmodified.

## Directory tree (one clear owner per concern)

    tools/
    ├── cgv/                         EXISTING verification G1..G6; unchanged
    ├── genrec-mcp/server.py         EXISTING authenticated training service; unchanged
    ├── stage2_stability_probe.py    EXISTING diagnostic; unchanged
    └── research/                    NEW sole Issue116 research orchestration
        ├── README.md                architecture and operational contract
        ├── settings.py              source-locked paths, model, operation mode
        ├── __main__.py              no-argument inspect/one-step entry
        ├── core/
        │   ├── dag.py               ONLY task ledger, locks, leases, recoverable jobs
        │   └── evidence.py          sha256 and atomic artifact writes
        ├── agents/
        │   ├── prompts.py           behavior-first role prompts from frozen inputs
        │   └── runner.py            direct OpenCode CLI + process evidence (no library)
        ├── science/
        │   ├── behavior.py          train-only transition measurement (lazy pandas)
        │   └── contracts.py         candidate/SID and reviewer identity validation
        ├── workflows/
        │   └── pipeline.py          DATA+LITERATURE -> GEOMETRY -> REVIEW
        ├── integrations/
        │   └── preflight.py         read-only CGV/MCP/trainer registry; no bypass
        └── tests/                    stdlib unittest; no network/model/GPU

Output/state ONLY in results/research/issue116, NEVER under source trees.

## Execution / boundaries

    python3 -m tools.research
    python3 -m unittest discover -s tools/research/tests -v

The default ACTION = "inspect" is *read-only about science*: it initializes a
fresh issue116 task ledger and prints statuses; it never trains or invokes a model.
To generate real train-only measurements, edit settings.py to ACTION = "behavior",
then use the genrec_env Python 3.10 interpreter. The behavior analyzer reads
ONLY results/stage0_build_parquet/train.parquet, not valid/test or raw full-user
sequences. It collapses repeated prefix-expanded rows to each user's longest
training prefix before counting transitions (so repeated history is not counted
multiple times).

The worker mode is opt-in *by source edit*, not by CLI flags or environment
hyperparameter overrides, per repository CLAUDE.md. LIVE_AGENTS_ALLOWED defaults
False. Before enabling, deploy real filesystem confinement for the agent
processes: separate cwd is not a security sandbox. Agent JSON output proves only
the process ran, NOT that the scientific conclusion is correct. A true scientific
verifier must check a candidate's schema, hashes and frozen evidence.

The workflow stages are DATA and LITERATURE (independent, can be assigned by
separate processes); GEOMETRY waits for BOTH measured/train-only evidence and
literature; REVIEW is bound to the candidate ID, SHA and CURRENT formula. These
are scientific preparation tasks, not automatic Stage2 permission.

The research package has NO direct trainer subprocess or GPU launcher. Use the
surviving GenRec MCP cgv_check and stage2_start / stage3_start tools for official
jobs only after explicit approval and all relevant current controls pass.
CGV G1..G6 is the code contract gate. Stage2 descriptive metrics are NOT a hard
early-stop gate. Current CLAUDE.md defines the official Stage3 outcome and
frozen training settings; do not blindly import thresholds from closed Issues.

## Acceptance scope

This initial implementation establishes separated modules, train-only data
collection, atomic task evidence, a live-disabled agent runtime, prompt
identity guards and offline test fixtures. It does NOT claim:
- agent model sessions have run;
- Stage2 or Stage3 has been launched;
- an Idea has passed scientific review, CGV or final recommendation evaluation;
- robust sandboxing / service uptime has been deployed;
- that a JSON-formatted LLM answer is scientifically validated.

A subsequent production integration must add domain-specific scientific
verifiers and the real HPC worker supervisor before enabling live research.
