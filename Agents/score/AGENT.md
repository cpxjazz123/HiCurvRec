# The `score` agent

Ranks the proposals. Three reviewers order every proposal by the versioned rubric, each with a reason and an evidence id; the aggregation adds the ranks, breaks the ties by the declared order and keeps exactly one mechanism (the one-mechanism cap).

## Where it sits

- **step**: 4 - research_round_score
- **agents in this step**: 3
- **model**: `deepseek/deepseek-v4.1-flash`
- **deadline**: 900 s (wrap-up from 60 s before it)
- **structured output**: on (`RankPayload`)
- **answer file**: `ranks.txt (rebuttal.txt for the bounded rebuttal)`

## What it may do

- tools: `read_file`, `list_directory`, `append_progress`
- read roots: the agent's own workspace; the geometry contract (tools/research/contracts/HYPERBOLIC_GEOMETRY_CONTRACT.md)
- write root: the agent's own workspace (the answer file)
- shell: no

## Its contract

```
Answer with ONE JSON object and nothing else: no prose, no markdown fence. Its keys are 'ranking' (a list of objects with 'idea_id', 'position' and 'reason') and 'objection' (a string).
```

## Files

- `config.json` - the machine-readable configuration
- `output.schema.json` - the shape of the answer
- `prompt.md` - the prompt template, verbatim from the code
