# The `probe` step

Measures the behaviour behind each proposal before anyone scores it. The probe is DETERMINISTIC - no language model runs in this step - it is train-only, runs on CPU and has a hard budget; it reports UNKNOWN or NOT_CHECKED for what it could not establish, and it never rewrites a score.

## Where it sits

- **step**: 3 - research_round_probe
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
