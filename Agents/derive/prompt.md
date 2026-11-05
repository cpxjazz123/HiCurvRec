# The prompt template of the `derive` agent

Source: `tools/research/rounds/worker.py` -> `_derivation_prompt()`

```python
def _derivation_prompt(idea: dict) -> str:
    """One specification, no other derivation, and no way to ask for one."""
    instance = DERIVATION_INSTANCE
    args, symbols = derivation_signature()
    argument_lines = "\n".join(
        "      %s: %s" % (spec["name"],
                         ("a scalar" if spec.get("scalar") is not None
                          else "a vector of %d components" % int(spec["width"])))
        for spec in args)
    symbol_lines = ", ".join(symbols)
    return (
        "FIXED INSTANCE - the round fixes it and you must use exactly these "
        "arguments and these symbols. A derivation with its own signature is a "
        "different function and cannot be compared with the others.\n"
        "  * the arguments, in this exact order:\n" + argument_lines + "\n"
        "  * the free symbols, in this exact order (a vector argument of width w "
        "consumes w symbols, one per component): " + symbol_lines + "\n"
        "  * the output is a single per-item vector named '" + instance["output"] +
        "' - the quantity the mechanism changes in the training objective.\n"
        "  * the code is a complete Python module defining compute(**kwargs) with "
        "exactly these argument names, and it must compute the SAME formula as "
        "the statements.\n"
        "  * `symbols` and `args` are NOT part of your answer: the round owns "
        "them. You answer with the statements, the output name, the code and the "
        "excluded surfaces.\n\n"
        "Derive the formula of this mechanism from the specification alone.\n\n"
        "TITLE: " + str(idea.get("title")) + "\n"
        "MECHANISM: " + str(idea.get("mechanism")) + "\n"
        "SPECIFICATION: " + str(idea.get("specification")) + "\n"
        "SPEC EQUATIONS: " + str(idea.get("spec_equations")) + "\n"
        "INPUTS AND OUTPUTS: " + str(idea.get("spec_inputs_outputs")) + "\n\n"
        "FIXED INSTANTIATION - every derivation must use exactly this, or the "
        "two cannot be compared:\n"
        "  * latent width D = 2;\n"
        "  * the smallest complete instance: L = 2 levels, K = 2 codes per level;\n"
        "  * arguments, in this exact order and with these exact names:\n"
        "      z (width 2) - the encoder vector,\n"
        "      theta_1_1, theta_1_2 (width 2), theta_2_1, theta_2_2 (width 2) - "
        "the codebook vectors in (level, code) order,\n"
        "      s (scalar) - the frozen scale.\n"
        "  * output: a single per-item vector named `loss` (one value per row of z), "
        "the quantity the mechanism adds or changes in the training objective.\n"
        "  * a switch or a table that the specification needs may be carried as "
        "an extra argument (a scalar 0/1 or a width-2 vector) placed AFTER s, and "
        "must be named in `args` and `symbols` in the same order.\n"
        "  Symbols follow the arguments: a width-w argument consumes w symbols in "
        "column order (z -> z0, z1; theta_1_1 -> t110, t111 ...).\n\n"
        "Turn the equations into SymPy statements and into a complete torch "
        "module defining compute(**kwargs), as the answer format requires.")
```
