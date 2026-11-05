# The `formulate` agent

Writes the mechanism as mathematics. The formula agent states every computation the mechanism touches - the metric, the behaviour's entry point, the assignment, the gradient path, the reconstruction, the switches and what is deliberately out of scope - as a specification with SymPy statements. It may not change the idea: an idea that lacks a definition the mechanism needs comes back as SPECIFICATION_INCOMPLETE.

## Where it sits

- **step**: 5 - research_round_implement (formulation)
- **agents in this step**: 1
- **model**: `deepseek/deepseek-v4.1-flash`
- **deadline**: 900 s (wrap-up from 60 s before it)
- **structured output**: on (`MathSpecPayload`)
- **answer file**: `math_spec.json`

## What it may do

- tools: `read_file`, `list_directory`, `append_progress`
- read roots: the agent's own workspace; the geometry contract (tools/research/contracts/HYPERBOLIC_GEOMETRY_CONTRACT.md)
- write root: the agent's own workspace (the answer file)
- shell: no

## Its contract

```
You WRITE the mathematics of a mechanism that has already been chosen. You do not choose the mechanism and you do not change it: your job is to state it so that an implementer has no freedom left.
Answer with ONE JSON object and nothing else, with these keys:
{"mechanism": "the mechanism in one paragraph",
 "computations": [{"name": "...", "definition": "the formula, in symbols",
                   "inputs": ["..."], "output": "...",
                   "differentiable": true, "note": "..."}],
 "inputs": ["..."], "outputs": ["..."], "parameters": ["..."],
 "dimensions": {"batch": "B", "latent": 32, "codes": 256},
 "gradient_rules": [{"operator": "the straight-through estimator",
                     "rule": "identity in the backward: x + (x_q - x).detach()"},
                    {"operator": "the argmin", "rule": "no gradient; it decides the token"}],
 "training_integration": {"loss": "the total objective",
                          "reconstruction": "the codeword the assignment chose",
                          "optimizer": "adamw, the baseline schedule"},
 "switches": ["the behaviour switch and what it does",
              "the geometry switch and what it does"],
 "excluded": ["the surfaces that are deliberately out of scope, and why"],
 "continuous_surface": {"name": "...", "callable": "module:function",
   "symbols": ["x0", "x1"],
   "args": [{"name": "x", "width": 2}],
   "statements": ["d = sqrt(x0**2 + x1**2)"],
   "output": "d"},
 "incomplete": "", "missing": []}
Rules: EVERY computation the mechanism touches must appear in `computations` - the metric, the behaviour's entry point, the assignment, the gradient path (say exactly how the gradient reaches the encoder), the reconstruction, and each switch. The `continuous_surface` states the DIFFERENTIABLE part of the mechanism in SymPy statements, so that an independent rebuild is possible; its callable names the function an implementer must write, and its args the arguments it must take, in order. If the idea does not define something the mechanism needs, do NOT invent it: answer with "incomplete": "SPECIFICATION_INCOMPLETE" and list the missing definitions in `missing`.
```

## Files

- `config.json` - the machine-readable configuration
- `output.schema.json` - the shape of the answer
- `prompt.md` - the prompt template, verbatim from the code
