# The `code` agent

Implements the frozen specification. Three coding agents work from the same frozen specification, in isolation; their agreement is the differential evidence, and a disagreement stops the candidate.

## Where it sits

- **step**: 5 - research_round_implement (coding)
- **agents in this step**: 3
- **model**: `deepseek/deepseek-v4.1-flash`
- **deadline**: 900 s (wrap-up from 60 s before it)
- **structured output**: on (`CodingPayload`)
- **answer file**: `module.py`

## What it may do

- tools: `read_file`, `list_directory`, `append_progress`
- read roots: the agent's own workspace; the geometry contract (tools/research/contracts/HYPERBOLIC_GEOMETRY_CONTRACT.md)
- write root: the agent's own workspace (the answer file)
- shell: no

## Its contract

```
You IMPLEMENT a frozen mathematical specification. The specification is the only authority: you may not reinterpret it, simplify it, or change a sign, a scale or a limit. If you believe the specification is wrong, say so in `notes` and implement it as written.
Answer with ONE JSON object and nothing else:
{"module": "import torch\n\ndef compute(...): ...",
 "notes": "anything the verifier must know"}
Rules: `module` is a complete Python module that defines the callable the specification's continuous_surface names, with exactly its arguments in its order, and it computes the specification's statements. Use torch only. Do not train, do not read a dataset, do not write files.
```

## Files

- `config.json` - the machine-readable configuration
- `output.schema.json` - the shape of the answer
- `prompt.md` - the prompt template, verbatim from the code
