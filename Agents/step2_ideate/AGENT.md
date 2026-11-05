# The `ideate` agent

Proposes mechanisms. Five agents answer independently, each with the round's objective, the scope contract and the evidence already gathered; the framework merges the answers, drops the duplicates and scores every surviving proposal. The agent may read nothing outside its own workspace: everything it needs is in the prompt.

## Where it sits

- **step**: 2 - research_round_ideate
- **agents in this step**: 5
- **model**: `deepseek/deepseek-v4.1-flash`
- **deadline**: 900 s (wrap-up from 60 s before it)
- **structured output**: on (`IdeaPayload`)
- **answer file**: `ideas.txt (ideas_extra.txt for the second pass)`

## What it may do

- tools: `read_file`, `list_directory`, `append_progress`
- read roots: the agent's own workspace; the geometry contract (tools/research/contracts/HYPERBOLIC_GEOMETRY_CONTRACT.md)
- write root: the agent's own workspace (the answer file)
- shell: no

## Its contract

```
Answer with ONE JSON object and nothing else: no prose, no markdown fence. Its keys are: idea_id, title, motivation, novelty, key_change, specification, cost_estimate, falsifiable_hypothesis, risks, behavior_signal, behavior_computation, leakage_boundary, hyperbolic_operator, hyperbolic_location, coupling_path, coupling_equation, observables, ablation_behavior_off, ablation_geometry_off, ablation_both_on, spec_equations, spec_inputs_outputs, safeguards. Every key is a string except 'risks', which is a list of strings.
```

## Files

- `config.json` - the machine-readable configuration
- `output.schema.json` - the shape of the answer
- `prompt.md` - the prompt template, verbatim from the code
