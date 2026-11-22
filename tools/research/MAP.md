# The map: which step, which agent, which function, which file

The research framework runs one round in nine steps. Every step is a call, every
call runs agents of named roles, and every role's work lives in a module named
after the step and the agent it runs.

## The flow

| step | the call | agent role | agents | the functions | the module |
|---|---|---|---|---|---|
| 1 | `research_round_start` | *(none)* | 0 | `new_round` | `rounds/pipeline.py` |
| 2 | `research_round_ideate` | `ideate` | 5 | `ideate_one`, `run_ideation`, `phase_ideation` | `rounds/step2_ideation.py` |
| 3 | `research_round_probe` | *(none: deterministic)* | 0 | `run_probes`, `phase_probe`, `phase_probe_step` | `rounds/step3_probe.py` |
| 4 | `research_round_score` | `score` | 3 | `rank_one`, `run_scoring`, `run_rebuttal`, `phase_scoring` | `rounds/step4_scoring.py` |
| 5a | `research_round_implement` | `formulate` | 1 | `_formula_prompt`, `_parse_math_spec`, `phase_formulation` | `rounds/step5_formulation.py` |
| 5b | `research_round_implement` | `review_math` | 2 | `_review_math_prompt`, `_parse_math_review`, `_retain_spec_tests` | `rounds/step5_formulation.py` |
| 5c | `research_round_implement` | `code` | 3 | `_coding_prompt`, `_parse_coding`, `phase_coding` | `rounds/step5_coding.py` |
| 5d | `research_round_implement` | `derive` | 3 | `_derivation_prompt`, `run_derivations`, `phase_derivation` | `rounds/step5_derivation.py` |
| 6 | `research_round_register_implementation` | *(the MAIN agent writes the code)* | 0 | `run_implementation`, `phase_implementation`, `advance` | `rounds/step6_implementation.py` |
| 7 | `research_round_accept` | *(the main agent's decision)* | 0 | `accept_implementation` | `rounds/pipeline.py` |
| 8 | `research_round_launch` | *(no agent: the GPU job)* | 0 | `launch` | `stage2/launcher.py` |
| 9 | `research_round_collect` | *(no agent: the GPU job)* | 0 | `complete_stage2`, `enqueue_stage3` | `rounds/jobs.py` |

`rounds/worker.py` is the **facade**: it re-exports every function above, so
`worker.phase_formulation` and `from .worker import ideate_one` keep working.
`rounds/agent_calls.py` holds the shared call into an agent: the workspace, the
boundary fingerprint, the runner and the concurrent driver.

## The packages

| package | what it is |
|---|---|
| `rounds/` | the round framework: the steps above, the scope gate, the eval gate, the compiler, the differential and the job queue |
| `stage2/` | the Stage 2 engineering contract: the frozen constants, the launcher, the nine machine checks, the reference rebuild and the DDP and gradient probes |
| `agents/` | the agent runtime: the boundary policy, the progress channel, the instructions and the schemas, the runner, and the exporter that writes `Agents/` |
| `agent_flow/` | the one-shot runner (`python3 -m tools.research.agent_flow.run --round <id> --phases ...`) |
| `geometry/` | the canonical geometry module every candidate implements on top of |
| `science/` | the five measurements the framework itself calls (`sid_layer_analysis`, `preflight`, `contract`, `contracts`, `behavior`); the other measurements live in `analysis_sid_recall/` |
| `core/` | the evidence helpers: the atomic write and the digest |
| `integrations/`, `mcp/`, `workflows/`, `verification/`, `pipeline_v2/` | the earlier framework generation, still mounted by the MCP's `research_start`, `research_verify_*` and `research_v2_*` tools |
| `tests/`, `stage2/tests/` | the framework's own tests |

## The naming rule

A module of the round flow is named `step<N>_<what it does>`; the prompt
builders and the answer parsers carry the agent's role in their names
(`_formula_prompt` is the formula agent's, `_parse_math_review` the reviewer's).
The exported copies of these configurations live in `Agents/`, one directory per
role, named the same way (`Agents/step5_formulate/`, `Agents/step2_ideate/`),
and `tools/research/tests/test_agent_configs.py` fails when they drift away from
the code.
