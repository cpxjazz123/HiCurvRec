# The `review_math` agent

Judges the specification against the idea it came from. Two reviewers check that the mathematics is faithful, complete and unambiguous, and every finding must arrive with a test that refuses the defect.

## Where it sits

- **step**: 5 - research_round_implement (formulation)
- **agents in this step**: 2
- **model**: `deepseek/deepseek-v4.1-flash`
- **deadline**: 900 s (wrap-up from 60 s before it)
- **structured output**: on (`MathReviewPayload`)
- **answer file**: `math_review.json`

## What it may do

- tools: `read_file`, `list_directory`, `append_progress`
- read roots: the agent's own workspace; the geometry contract (tools/research/contracts/HYPERBOLIC_GEOMETRY_CONTRACT.md)
- write root: the agent's own workspace (the answer file)
- shell: no

## Its contract

```
You REVIEW a mathematical specification against the idea it came from. You are not checking whether it can run: you are checking whether it FAITHFULLY expresses the idea, and whether the mathematics is correct, complete and unambiguous.
Answer with ONE JSON object and nothing else:
{"verdict": "PASS" or "REVISE", "findings": ["..."], "reason": "..."}
Rules: REVISE if the specification changes the idea, if a computation is missing, if a formula is ambiguous or dimensionally inconsistent, if a sign, a scale or a limit is wrong, if the gradient path is not stated, or if the switches do not do what they claim. Say in every finding WHICH part is wrong and what it must be instead. PASS only if the specification is a faithful, complete and correct formulation of the idea.
```

## Files

- `config.json` - the machine-readable configuration
- `output.schema.json` - the shape of the answer
- `prompt.md` - the prompt template, verbatim from the code
