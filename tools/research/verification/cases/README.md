# Agent-authored verification cases

Every **new** mechanism gets its own reviewed directory:

    tools/research/verification/cases/<candidate_id>/
      test_math.py
      test_gradient.py
      test_ddp.py
      test_entrypoint.py

MATH, GRADIENT and RUNTIME OpenCode Agents independently author the relevant
tests in source control, and an independent REVIEW Agent examines source, math
oracle independence, runtime and DDP coverage, and frozen proposal evidence.

No generic placeholder cases ship with this repository: until a real candidate
has four passing and reviewed test suites, official research Stage2 is blocked.

For every new candidate, write:
  results/research/issue116/verification/<candidate_id>/manifest.json

The manifest binds:
- schema=1, candidate_id, source_tree, source_sha256 (full active code tree);
- proposal_sha256 (current frozen agents/geometry/candidate.json);
- tests: math, gradient, ddp, entrypoint = exact sha256 of reviewed test files.

The evidence root must contain four independent real OpenCode sessions:
  agents/<candidate_id>_math/
  agents/<candidate_id>_gradient/
  agents/<candidate_id>_runtime/
  agents/<candidate_id>_review/

Each session needs execution.json, session.jsonl, and the named JSON output;
execution.json itself is a process receipt, NOT a valid scientific judgement.
Outputs must name candidate_id; the first three also name source_sha256; the
review output must contain verdict=APPROVE_TESTS, source_sha256,
proposal_sha256 and the exact tests sha mapping.

At each MCP stage2_start(source_dir, verification_id) call the server rechecks
these hashes and RE-RUNS the four suites with the project's Stage2 interpreter.
It then runs its existing real trainer gradient smoke, preserving runtime
integration evidence and output isolation before the 3000-epoch job.

Do not generate passing trivially-true tests: agents must derive independent
reference values from the frozen mathematics, test actual gradient paths,
exercise multi-rank distributed updates, and prove the real Stage2 entrypoint
calls the new mechanism; these substantive properties require code review and
fault-injection examples in the individual suite.

The verifier does not inspect CGV declarations or require G1–G6 contracts; it
deliberately fails closed on missing cases, unverifiable sessions, unknown
candidates, source changes and test changes.

No GPU is automatically launched from tools/research.
