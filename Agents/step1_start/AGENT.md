# The `start` step

Creates the round. This step runs NO agent: it writes the round's state (the objective, the issue, the step it is in, the empty ideas, the empty job queues) and takes the round's lock. Everything the later steps need to know is fixed here: the objective the agents are given, the scope contract they must meet, the rubric the reviewers use and the one-mechanism rule that decides the selection.

## Where it sits

- **step**: 1 - research_round_start
- **agents in this step**: 0 - this step runs no model

## What it fixes for the later steps

- **ideation agents**: 5
- **rubric**: `rank-2026-10-v1` (sha256 `f50a1334d619338f`)
- **selection rule**: {"max_selected": 1, "min_reviewers": 2, "tie_break_order": ["scope", "total_rank", "first_places", "idea_id"], "declared_hours_budget": 72.0}
- **objective**: Discover new mechanisms in which TRAINING-ONLY user behaviour and HYPERBOLIC geometry are computationally coupled to improve the RQ-VAE Semantic IDs used by a generative next-item recommender, with the real target of Stage3 test Recall@10 above 0.
- **scope contract**: 2359 characters, 6 clauses
- **experiment policy**: 1139 characters

## Files

- `config.json` - the machine-readable configuration
- `prompt.md` - the code that creates the round, verbatim
