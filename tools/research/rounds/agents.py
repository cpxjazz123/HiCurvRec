"""Agent contracts for the Issue #120 round: ideate, score, implement, verify, repair.

The prompts state what each role must return and what it must not do. The
verification role is a correctness reviewer of an implementation, not a scientific
paper referee: the issue is explicit that scientific paperwork completeness is
not a pre-training gate, while actual correctness, numerical stability and a
runnable trainer are.
"""
from __future__ import annotations

IDEA_FIELDS = (
    "idea_id, title, motivation, novelty, key_change, specification, cost_estimate, "
    "falsifiable_hypothesis, risks"
)

IDEA_PROMPT = (
    "Propose {minimum} to {maximum} DISTINCT, implementable ideas for improving the "
    "item semantic IDs that a generative recommender consumes. You are one of three "
    "independent ideation agents: aim for a different part of the design space than the "
    "obvious one, and do not copy an existing mechanism in the repository.\n"
    "Each idea MUST be a JSON object with: " + IDEA_FIELDS + ".\n"
    "  idea_id: short safe identifier, unique inside your answer.\n"
    "  title: one line.\n"
    "  motivation: the user-behaviour or recommendation problem it addresses.\n"
    "  novelty: what is different from the mechanisms that already exist here.\n"
    "  key_change: the computational change, in one or two sentences.\n"
    "  specification: the minimal mathematics or pseudocode that an implementer can "
    "write without guessing.\n"
    "  cost_estimate: the compute or wall-clock cost, relative to the baseline, and the "
    "per-step cost at the FROZEN scale (24556 training items, 24 steps per epoch, 3000 "
    "epochs). An idea whose per-step cost at that scale makes 3000 epochs infeasible on one "
    "GPU is rejected, not deferred.\n"
    "  falsifiable_hypothesis: what must be true for the idea to be worth keeping, and "
    "how it would be observed.\n"
    "  risks: what could make it fail.\n"
    "Unconventional ideas and unfinished derivations are welcome: only reject an idea "
    "that cannot be made minimally specified. Do NOT propose a proof, a paper section or "
    "a quality floor; the ideas are selected for innovation and then implemented and "
    "measured.\n"
    "Write JSON: {{\"ideas\": [ ... ]}} with between {minimum} and {maximum} ideas."
)

SCORE_PROMPT = (
    "You are one of three independent evaluators. Score EVERY idea below on five "
    "dimensions from 0 to 5: novelty, scientific_motivation, feasibility, "
    "downstream_impact, experimental_distinctness. Give the numeric scores, a short "
    "rationale and the known uncertainties for each idea. Judge the idea as written: an "
    "unfinished derivation is not a defect, but an idea that cannot be implemented is.\n"
    "Do not rank the ideas against each other and do not veto: your report is one of "
    "three, and the aggregate decides. Do not score on Stage 2 quality diagnostics such "
    "as prefix counts or collision rates; selection is innovation-first.\n"
    "Write JSON: {{\"scorer\": \"<your id>\", \"scores\": [{{\"idea_id\": ..., \"scores\": "
    "{{novelty, scientific_motivation, feasibility, downstream_impact, "
    "experimental_distinctness}}, \"rationale\": \"...\", \"uncertainties\": \"...\"}}]}} "
    "with one entry per idea."
)

IMPLEMENT_PROMPT = (
    "Implement ONE idea as a standalone Stage 2 candidate tree. You own this tree and "
    "nobody else writes into it.\n"
    "The idea:\n{idea}\n"
    "Requirements:\n"
    "  1. Copy the frozen baseline tree and change only what the idea needs. Keep the "
    "frozen protocol: latent 32, three levels of 256 codes, 3000 epochs, the existing "
    "optimizer, schedule, batch size, seeds, the data split and the exporter.\n"
    "  2. The training must run: a forward and a backward, finite losses, gradients that "
    "reach the parameters the idea changes, a valid SID export, and a minimal smoke test "
    "that proves it.\n"
    "  3. Record every deviation from the idea and every assumption you had to make, "
    "with the reason.\n"
    "  4. Do not change the shared repository outside your own tree. Do not modify the "
    "baseline trees, the MCP server or the verification code.\n"
    "Write JSON: {{\"tree\": \"<absolute path of the tree>\", \"changed_files\": [...], "
    "\"deviations\": [{{\"item\": ..., \"reason\": ...}}], \"assumptions\": [...], "
    "\"smoke\": {{\"command\": ..., \"result\": ...}}, \"notes\": \"...\"}}."
)

VERIFY_PROMPT = (
    "You are one of three independent verifiers of an implementation. Inspect the "
    "code, do not trust a passing test.\n"
    "The idea it implements:\n{idea}\n"
    "The tree:\n{tree}\n"
    "Check three things and report each with file and line references:\n"
    "  a) mechanism fidelity: does the code implement the idea, and only the idea?\n"
    "  b) trainability: does a real forward and backward produce finite losses and "
    "gradients that reach the intended parameters, with no dead term and no "
    "numerically unstable step?\n"
    "  c) integration: does the real trainer entry point run, is the data path the "
    "frozen one, is the SID export valid and correctly indexed, and does the "
    "distributed launcher still work?\n"
    "Report PASS or NEEDS_FIX per check. A NEEDS_FIX must name the defect, give a "
    "concrete reproduction or a minimal counterexample, say how severe it is and what "
    "the correction is. Do not demand a proof, a literature review or a quality floor: "
    "this is a correctness review of code, not a scientific paper review.\n"
    "Write JSON: {{\"verifier\": \"<your id>\", \"verdict\": \"PASS\"|\"NEEDS_FIX\", "
    "\"checks\": [{{\"name\": ..., \"verdict\": ..., \"findings\": [{{\"severity\": ..., "
    "\"location\": \"file:line\", \"detail\": ..., \"reproduction\": ..., "
    "\"correction\": ...}}]}}]}}"
)

REPAIR_PROMPT = (
    "You are the dedicated repair engineer for one implementation. The verifiers "
    "rejected it with these findings:\n{findings}\n"
    "Fix exactly the reported defects in this tree:\n{tree}\n"
    "Keep the frozen protocol and everything the idea requires that is not reported as "
    "defective. For every finding, state what you changed and how you know it is fixed. "
    "Run the smoke test and report its real output. Do not modify anything outside "
    "this tree.\n"
    "Write JSON: {{\"tree\": \"<path>\", \"repairs\": [{{\"finding\": ..., \"change\": ..., "
    "\"verification\": ...}}], \"remaining_issues\": [...], \"smoke\": {{\"command\": ..., "
    "\"result\": ...}}}}."
)
