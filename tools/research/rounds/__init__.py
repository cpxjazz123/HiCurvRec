"""The multi-agent research round of Issue #120.

A round runs five independent ideation agents, then three independent scorers
that each judge every idea, then selects several ideas, implements each of them
in its own isolated tree, has **five independent verifiers** read each
implementation, and finally trains the accepted candidates - in parallel where
the resources allow - and hands every technically valid Stage 2 result to
Stage 3 automatically.

The verifiers report; the **main agent decides**. `research_round_verify`
returns every verdict and every finding, together with how many reviewers
raised each finding, and dispatches nothing: the main agent accepts a candidate
with `research_round_accept`, or asks for a repair plan with
`research_round_plan_repair`. Only an accepted candidate is trained.

The repair role does not touch code. When the verifiers reject an
implementation, the repair agent reads the findings and the tree and writes a
**plan** - the file, the line, the exact change and the check that proves it.
The operator (the main agent) applies that plan to the tree and registers the
execution through `research_round_apply_repair`, which is what reopens the
review. The rule is enforced, not merely requested: the tree's sources are
copied aside before the planner runs, and a planner that edits anyway has its
edit rolled back and its plan rejected.

The modules here are deliberately split so that the decision logic is testable
without an agent: `ideas` and `scoring` are pure functions over JSON, `agents`
holds the agent contracts, `pipeline` is the round state machine and `jobs` is the
resource-aware Stage 2 / Stage 3 queue.
"""
