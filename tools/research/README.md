# GeneRec research tools — new clean architecture for Issue #116

This package is a *new* small tool system against the cleaned repository. It does
not import or restore any historical research controller, experiment campaign, or
old agent-Harness framework. The Stage2/Stage3 trainers, CGV engine, and MCP
server stay exactly where they are, unmodified.

## Directory tree (one clear owner per concern)

    tools/
    ├── cgv/                         RETIRED for new training admission; not restored
    ├── genrec-mcp/server.py         CURRENT training service; verification_id only
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
        ├── verification/
        │   ├── admission.py        source/evidence checks and real unittest re-runs
        │   ├── cases/             separately reviewed mechanism test modules
        │   └── tests/             offline evidence fault-injection tests
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

The research package NEVER launches formal training. Official Stage2 now accepts
`stage2_start(source_dir, verification_id)`, with `verification_check` as the
test-only endpoint. The MCP verifies four independent real OpenCode session
receipts and a distinct review, binds the source tree and four content-addressed
test files to a frozen proposal, then actually runs math, gradient, DDP and
entrypoint suites under the Stage2 Python environment. Missing or stale evidence
and zero executed tests always BLOCK. The MCP's actual one-epoch trainer
gradient smoke is still required AFTER these tests and BEFORE full 3000-epoch
training. There is no active CGV or historical admission alternate launch path.

Legacy `tools/cgv/` remains physically present temporarily for audit/reference,
but its declarations are NOT accepted by current Stage2 start; remove only after
the real HPC test chain and fault-injection regression are verified. The
existing Stage2/Stage3 trainers were not modified, and descriptive Stage2
metrics are not used as additional early-stop gates.

Validation Agents author mechanism-specific executable suites and an independent
Reviewer approves exact source and test hashes; passing tests alone are not
proof that the test assertions cover the claimed mathematics. Agent-generated
files require source review before promotion into `verification/cases/`.
Read `tools/research/verification/cases/README.md` for evidence requirements.

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
