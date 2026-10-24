"""The multi-agent research round of Issue #120.

A round runs three independent ideation agents, then three independent scorers
that each judge every idea, then selects several ideas, implements each of them
in its own isolated tree, verifies each implementation with three agents, repairs
what they reject, and finally trains the verified candidates - in parallel where
the resources allow - and hands every technically valid Stage 2 result to
Stage 3 automatically.

The modules here are deliberately split so that the decision logic is testable
without an agent: `ideas` and `scoring` are pure functions over JSON, `agents`
holds the agent contracts, `pipeline` is the round state machine and `jobs` is the
resource-aware Stage 2 / Stage 3 queue.
"""
