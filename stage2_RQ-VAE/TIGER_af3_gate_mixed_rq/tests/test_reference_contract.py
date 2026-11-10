"""The frozen statements of the reference contract, evaluated against the code.

The contract is documentation now (§87 removed the formula check from admission),
but documentation that quietly disagrees with the code is worse than none. This
file rebuilds each declared call from its own statements with SymPy and compares
the value with the implemented callable at sampled points, in double precision.
If SymPy is missing the file reports NOT_RUN instead of pretending to pass.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

TREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TREE))

CONTRACT = json.loads((TREE / "reference_contract.json").read_text(encoding="utf-8"))


def _callable(name: str):
    module, attribute = name.split(":")
    return getattr(__import__(module, fromlist=[attribute]), attribute)


def _reference(call: dict):
    sympy = pytest.importorskip("sympy")
    namespace = {"sqrt": sympy.sqrt, "atanh": sympy.atanh, "Min": sympy.Min,
                 "exp": sympy.exp, "erf": sympy.erf}
    symbols = {name: sympy.Symbol(name) for name in call["symbols"]}
    environment = dict(namespace, **symbols)
    substitutions = {}
    for statement in call["statements"]:
        target, expression = statement.split("=", 1)
        target = target.strip()
        value = sympy.sympify(expression, locals=dict(environment, **substitutions))
        substitutions[target] = value
    output = substitutions[call["output"]]
    return symbols, output


def _sample(call: dict, generator: torch.Generator):
    tensors = []
    scalars = []
    for spec in call["args"]:
        if "width" in spec:
            radius = float(spec.get("radius", 0.2))
            tensors.append((spec["name"], int(spec["width"]), radius))
        else:
            scalars.append((spec["name"], float(spec["scalar"])))
    points = {name: (torch.rand(4, width, generator=generator, dtype=torch.float64) * 2 - 1)
              * radius for name, width, radius in tensors}
    for name, value in scalars:
        points[name] = torch.tensor(value, dtype=torch.float64)
    return points


@pytest.mark.parametrize("call", CONTRACT["calls"], ids=[call["name"] for call in CONTRACT["calls"]])
def test_the_statements_reproduce_the_implemented_call(call):
    import sympy

    symbols, output = _reference(call)
    generator = torch.Generator().manual_seed(149)
    points = _sample(call, generator)
    arguments = [points[spec["name"]] for spec in call["args"]]
    measured = _callable(call["callable"])(*arguments)
    ordered = []
    for spec in call["args"]:
        value = points[spec["name"]]
        if "width" in spec:
            ordered.extend(value[:, column] for column in range(value.shape[1]))
        else:
            ordered.append(value)
    import numpy as np
    import scipy.special
    modules = [{"sqrt": np.sqrt, "atanh": np.arctanh, "Min": np.minimum,
                "exp": np.exp, "erf": scipy.special.erf}, "numpy"]
    expression = sympy.lambdify([symbols[name] for name in call["symbols"]], output, modules)
    expected = expression(*[tensor.numpy() for tensor in ordered])
    torch.testing.assert_close(torch.as_tensor(measured, dtype=torch.float64),
                               torch.as_tensor(expected, dtype=torch.float64),
                               rtol=1e-9, atol=1e-11)


def test_the_gate_call_is_the_module_gelu_of_the_implementation():
    torch_gelu = torch.nn.functional.gelu
    generator = torch.Generator().manual_seed(7)
    pre = torch.randn(32, generator=generator, dtype=torch.float64) * 3
    structure = torch.stack([pre, torch.zeros_like(pre)], dim=1)
    weight = torch.tensor([1.0, 0.0], dtype=torch.float64)
    bias = torch.tensor(0.0, dtype=torch.float64)
    scale = torch.tensor(1.0, dtype=torch.float64)
    shift = torch.tensor(0.0, dtype=torch.float64)
    from model.structure_gate import gate_alpha
    torch.testing.assert_close(gate_alpha(structure, weight, bias, scale, shift),
                               torch.sigmoid(torch_gelu(pre)), rtol=1e-12, atol=1e-14)
