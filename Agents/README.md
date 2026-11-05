# The agents of one round, one directory each

Every directory here is DERIVED from the code that runs the agent:
`tools/research/agents/export_configs.py` writes them, and
`tools/research/tests/test_agent_configs.py` fails when a file and
the code disagree. Edit the code, then re-export; edit these files
and the test tells you the framework no longer does what they say.

| step | role | agents | what it decides |
|---|---|---|---|
| 1 - research_round_start | `step1_start` | 0 | Creates the round. |
| 2 - research_round_ideate | `step2_ideate` | 5 | Proposes mechanisms. |
| 3 - research_round_probe | `step3_probe` | 0 | Measures the behaviour behind each proposal before anyone scores it. |
| 4 - research_round_score | `step4_score` | 3 | Ranks the proposals. |
| 5 - research_round_implement (formulation) | `step5_formulate` | 1 | Writes the mechanism as mathematics. |
| 5 - research_round_implement (formulation) | `step5_review_math` | 2 | Judges the specification against the idea it came from. |
| 5 - research_round_implement (coding) | `step5_code` | 3 | Implements the frozen specification. |
| 5 - research_round_implement (derivation) | `step5_derive` | 3 | Derives the mechanism independently. |
| 5 - research_round_implement (implementation) | `step5_implementation` | 1 | The candidate's own implementation path, used by the earlier framework generation. |
| the earlier verification path | `verify` | 1 | Runs an independent check suite against a candidate tree. |

The flow those steps belong to:

1. `research_round_start` - the round is created
2. `research_round_ideate` - five proposals
3. `research_round_probe` - the train-only evidence behind them
4. `research_round_score` - three reviewers rank, one mechanism is kept
5. `research_round_implement` - formulation (draft, two reviews, the eval gate, the freeze), coding (three implementations and the differential), derivation (three independent derivations), and the candidate tree copied from the baseline
6. `research_round_register_implementation` - the main agent's mechanism against the frozen specification
7. `research_round_accept` - the main agent's decision
8. `research_round_launch` - Stage 2 on four cards
9. `research_round_collect` - Stage 3, and the layered metrics

The gates that can stop the flow are the ones that refuse a
specification, a candidate or a SID; they are listed in CLAUDE.md.
