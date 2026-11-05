# The `derive` agent

Derives the mechanism independently. Three agents write the same mathematics without seeing each other; the framework compares the derivations symbolically, numerically and by gradient sign.

## Where it sits

- **step**: 5 - research_round_implement (derivation)
- **agents in this step**: 3
- **model**: `deepseek/deepseek-v4.1-flash`
- **deadline**: 900 s (wrap-up from 60 s before it)
- **structured output**: on (`DerivationPayload`)
- **answer file**: `derivation.json`

## What it may do

- tools: `read_file`, `list_directory`, `append_progress`
- read roots: the agent's own workspace; the geometry contract (tools/research/contracts/HYPERBOLIC_GEOMETRY_CONTRACT.md)
- write root: the agent's own workspace (the answer file)
- shell: no

## Its contract

```
You DERIVE one formula, independently. Your reply is ONE JSON object and nothing else: no prose before it, no prose after it, no markdown fence. You are given a specification and nothing else: you do not see any other derivation, its codes or its test results, and you must not ask for them.
Answer with ONE JSON object and nothing else:
{"symbols": ["x0", "x1", "p0", "p1", "n0", "n1", "m"],
 "args": [{"name": "x", "width": 2, "radius": 0.4},
          {"name": "m", "scalar": 0.2}],
 "statements": ["dp = sqrt((x0-p0)**2 + (x1-p1)**2)",
                "L = log(1 + exp(-(dp - dn + m)))"],
 "output": "L",
 "code": "import torch\n\ndef compute(x, m):\n    ...",
 "excluded": ["the argmin assignment: not differentiable"]}
Rules: the statements are SymPy expressions, one `name = expression` per line, evaluated in order, and `output` names the last one. `symbols` lists every free symbol in the order the arguments consume them (a tensor argument with width w consumes w symbols, one per column). `code` is a complete Python module defining `compute(**kwargs)` with exactly the argument names in `args`, returning a one-dimensional tensor, and it must compute the SAME formula as the statements - the verifier compares them. Use torch only. Do not train, do not touch a dataset, do not write files.
```

## Files

- `config.json` - the machine-readable configuration
- `output.schema.json` - the shape of the answer
- `prompt.md` - the prompt template, verbatim from the code
