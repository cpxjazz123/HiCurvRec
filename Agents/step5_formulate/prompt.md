# The prompt template of the `formulate` agent

Source: `tools/research/rounds/step5_formulation.py` -> `_formula_prompt()`

```python
def _formula_prompt(idea: dict, findings: list = None) -> str:
    """The prompt of the Formula Agent: the idea, and what the reviewers said."""
    prompt = (
        "Write the mathematics of this mechanism. The mechanism has already been "
        "chosen; you are not choosing it and you may not change it.\n\n"
        "TITLE: " + str(idea.get("title")) + "\n"
        "MECHANISM: " + str(idea.get("mechanism")) + "\n"
        "SPECIFICATION: " + str(idea.get("specification")) + "\n"
        "SPEC EQUATIONS: " + str(idea.get("spec_equations")) + "\n"
        "INPUTS AND OUTPUTS: " + str(idea.get("spec_inputs_outputs")) + "\n"
        "BEHAVIOUR SIGNAL: " + str(idea.get("behavior_signal")) + "\n"
        "BEHAVIOUR COMPUTATION: " + str(idea.get("behavior_computation")) + "\n"
        "COUPLING: " + str(idea.get("coupling_path")) + "\n"
        "ABLATION BEHAVIOUR OFF: " + str(idea.get("ablation_behavior_off")) + "\n"
        "ABLATION GEOMETRY OFF: " + str(idea.get("ablation_geometry_off")) + "\n")
    prompt += ("\n\nTHE GEOMETRY CONTRACT IS THE AUTHORITY on every convention. "
               "Read it and quote its operators exactly; a formula written in an "
               "unstated parameterisation is wrong:\n\n"
               + _contract_text() +
               "\n\nANSWER THESE THREE QUESTIONS EXPLICITLY, each as its own JSON "
               "key (the review refuses a specification that leaves them out):\n"
               "  dimensions: which tensor has which width - the batch, the latent, "
               "the codebook, every intermediate;\n"
               "  gradient_rules: for EVERY operator of the mechanism, what the "
               "backward pass does - the straight-through estimator, the detach "
               "points, the operators with no gradient at all;\n"
               "  training_integration: what the total loss is, what the decoder "
               "receives (the chosen codeword or the transported one), and which "
               "optimizer and schedule train it.\n")
    prompt += ("\n\nThe specification MUST state, as JSON keys: mechanism, "
               "computations, inputs, outputs, parameters, dimensions, "
               "gradient_rules, training_integration, switches, excluded, and the "
               "continuous_surface. Every computation the mechanism touches "
               "belongs in computations; dimensions states every tensor width; "
               "gradient_rules states how the gradient flows for every operator "
               "(including the straight-through estimator and the argmin); "
               "training_integration states the loss, what the decoder receives "
               "and the optimizer.")
    if findings:
        prompt += ("\n\nTHE REVIEWERS FOUND THESE PROBLEMS. Fix them, and nothing "
                   "else:\n- " + "\n- ".join(str(finding) for finding in findings))
    return prompt
```
