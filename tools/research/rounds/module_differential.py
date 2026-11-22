"""The three implementations of a frozen specification, compared as programs.

The symbolic comparison in :mod:`tools.research.rounds.differential` decides
whether two *formulas* are the same function. A coding agent does not answer
with a formula: it answers with a module. The question the round actually needs
to answer is therefore "do these three modules behave the same?", and the only
honest way to ask it is to run them.

This module does that: it imports each implementation, calls its entry point on
the same inputs, and compares the outputs elementwise in all three switch arms
(BOTH_ON, GEOMETRY_OFF, BEHAVIOUR_OFF). Two modules that compute the frozen
specification agree; a module that mis-reads a sign, a scale or a switch does
not, and the disagreement names the field that differs.

A comparison that cannot run - an import error, a missing entry point, a raise
inside the module - is reported as an error, never as agreement.
"""
from __future__ import annotations

import importlib.util
import inspect
import tempfile
from pathlib import Path

#: The names an implementation may give its entry point, most specific first.
ENTRY_POINTS = ("level_score", "compute", "score", "level_1_score", "distance")

#: The tolerances of the behavioural comparison. Float32 arithmetic in three
#: independent spellings of the same formula lands well inside these.
ATOL = 1e-6
RTOL = 1e-5


def _load(implementation: dict, preferred: tuple = ()):
    """Import one implementation and return its entry point."""
    code = implementation.get("code") or ""
    if not code.strip():
        raise ValueError("the implementation carries no code")
    directory = Path(tempfile.mkdtemp(prefix="impl_"))
    path = directory / (str(implementation.get("name") or "impl") + ".py")
    path.write_text(code, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("impl_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in tuple(preferred) + ENTRY_POINTS:
        candidate = getattr(module, name, None)
        if callable(candidate):
            return name, candidate
    public = [name for name, value in vars(module).items()
              if callable(value) and not name.startswith("_")
              and getattr(value, "__module__", "") == module.__name__]
    if len(public) == 1:
        return public[0], getattr(module, public[0])
    raise ValueError("no entry point among %s (public callables: %s)"
                     % (list(ENTRY_POINTS), public))


def _call(function, args: list):
    """Call the entry point with the canonical arguments, tolerating naming."""
    parameters = list(inspect.signature(function).parameters)
    values = []
    for index, name in enumerate(parameters):
        values.append(args[min(index, len(args) - 1)])
    return function(*values)


def _tensors(result):
    """Every tensor a result exposes, in a stable order."""
    import torch

    if isinstance(result, torch.Tensor):
        # A bare tensor is the score matrix or the assignment: the entry point is
        # a level score, so the shape decides which of the two it is.
        if result.dim() == 2:
            return {"score_k": result}
        if result.dim() == 1:
            return {"k1": result}
        return {"out": result}
    if isinstance(result, dict):
        return {str(key): value for key, value in result.items()
                if isinstance(value, torch.Tensor)}
    if isinstance(result, (list, tuple)):
        return {"out_%d" % index: value for index, value in enumerate(result)
                if isinstance(value, torch.Tensor)}
    raise ValueError("the entry point returned %s, not a tensor or a mapping of "
                     "tensors" % type(result).__name__)


def _compare_inprocess(implementations: list, spec: dict, *, seed: int = 17) -> dict:
    """The comparison itself; it needs torch, so it runs in its own process."""
    import math
    import torch

    surface = spec.get("continuous_surface") or {}
    declared = str(surface.get("callable") or "")
    preferred = ()
    if "(" in declared:
        preferred = (declared.split("(", 1)[0].strip(),)
    loaded, invalid = [], []
    for implementation in implementations:
        try:
            name, function = _load(implementation, preferred)
            loaded.append({"name": implementation.get("name"),
                           "entry": name, "function": function})
        except Exception as exc:  # noqa: BLE001 - the report carries the reason
            invalid.append({"name": implementation.get("name"),
                            "status": "INVALID",
                            "reason": "the module does not load: %s" % str(exc)[:300]})
    if len(loaded) < 2:
        return {"status": "NOT_APPLICABLE",
                "reason": "fewer than two implementations load: " +
                          "; ".join(entry["reason"] for entry in invalid)[:300],
                "invalid": invalid}

    parameters = spec.get("parameters") or []
    scale = 1.0
    curvature = 0.25
    for entry in parameters:
        text = str(entry)
        if text.startswith("rho = 0.5") and "s = 1" not in text:
            continue
    inputs = _arms(spec, seed=seed)
    residual = inputs["__residual__"]
    codebook = inputs["__codebook__"]
    demand = inputs["__demand__"]
    weight = 0.35
    curvature = 0.25
    # The surface decides the arms: a score takes the residual, the codebook,
    # the curvature, the demand and the weight; a distance takes two points and
    # the curvature. The geometry switch is the same in both.
    takes_score = len(inspect.signature(loaded[0]["function"]).parameters) >= 5
    if takes_score:
        arms = {
            "BOTH_ON": (residual, codebook, curvature, demand, weight, False),
            "GEOMETRY_OFF": (residual, codebook, 0.0, demand, weight, False),
            "BEHAVIOUR_OFF": (residual, codebook, curvature, demand, weight, True),
        }
    else:
        # A distance takes two point sets of the SAME shape: the surface it is
        # compared on is elementwise, not a matrix of every pair, and the points
        # are inside the ball, where the surface is defined.
        arms = {
            "BOTH_ON": (inputs["BOTH_ON"][0], inputs["BOTH_ON"][1], curvature),
            "GEOMETRY_OFF": (inputs["BOTH_ON"][0], inputs["BOTH_ON"][1], 0.0),
        }
    pairs, failures = [], []
    for arm, values in arms.items():
        results = {}
        for entry in loaded:
            try:
                results[entry["name"]] = _tensors(_call(entry["function"], list(values)))
            except TypeError:
                # an implementation that takes fewer arguments than the arm
                results[entry["name"]] = _tensors(
                    _call(entry["function"], list(values[:3])))
            except Exception as exc:  # noqa: BLE001
                failures.append({"arm": arm, "name": entry["name"],
                                 "reason": "%s raised: %s"
                                           % (entry["name"], str(exc)[:200])})
                results[entry["name"]] = None
        names = [entry["name"] for entry in loaded]
        for index, left in enumerate(names):
            for right in names[index + 1:]:
                entry = {"arm": arm, "left": left, "right": right}
                left_values, right_values = results.get(left), results.get(right)
                if not left_values or not right_values:
                    entry["status"] = "ERROR"
                    entry["reason"] = "an implementation did not produce a result"
                    failures.append(entry)
                    pairs.append(entry)
                    continue
                shared = sorted(set(left_values) & set(right_values))
                if not shared:
                    entry["status"] = "ERROR"
                    entry["reason"] = "the two results share no tensor"
                    failures.append(entry)
                    pairs.append(entry)
                    continue
                worst = 0.0
                differing = []
                for key in shared:
                    left_tensor = left_values[key]
                    right_tensor = right_values[key]
                    if left_tensor.shape != right_tensor.shape:
                        differing.append("%s: shape %s against %s"
                                         % (key, tuple(left_tensor.shape),
                                            tuple(right_tensor.shape)))
                        continue
                    if left_tensor.dtype != right_tensor.dtype:
                        left_tensor = left_tensor.float()
                        right_tensor = right_tensor.float()
                    gap = (left_tensor - right_tensor).abs().max().item()
                    worst = max(worst, gap)
                    if not torch.allclose(left_tensor, right_tensor,
                                          atol=ATOL, rtol=RTOL):
                        differing.append("%s: max |difference| %g" % (key, gap))
                entry["fields"] = shared
                entry["worst_gap"] = worst
                entry["status"] = "AGREE" if not differing else "DISAGREE"
                if differing:
                    entry["reason"] = "; ".join(differing)[:300]
                    failures.append(entry)
                pairs.append(entry)
    return {"status": "AGREE" if not failures else "DISAGREE",
            "pairs": pairs, "invalid": invalid,
            "reason": "" if not failures else str(failures[0].get("reason"))[:300],
            "entries": [{"name": entry["name"], "entry": entry["entry"]}
                        for entry in loaded]}


#: The interpreter with torch: the round process runs on the standard library,
#: and a comparison of torch modules has to run where torch is.
TORCH_PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9"

DRIVER = """
import json, sys
from pathlib import Path
sys.path.insert(0, %(root)r)
from tools.research.rounds.module_differential import _compare_inprocess
payload = json.loads(Path(%(payload)r).read_text())
print("@@VERDICT@@" + json.dumps(_compare_inprocess(payload["implementations"],
                                                   payload["spec"],
                                                   seed=payload["seed"]),
                                 default=str))
"""


def _arms(spec: dict, *, seed: int = 17) -> dict:
    """The inputs every arm is called with.

    The points of a distance surface are sampled INSIDE the ball: outside it the
    artanh argument saturates, and correct implementations differ there on the
    spelling of the numerical guard, not on the mathematics.
    """
    import math
    import torch

    generator = torch.Generator().manual_seed(seed)
    B, D, K1 = 8, 6, 5
    curvature = 0.25
    residual = torch.randn(B, D, generator=generator)
    codebook = torch.randn(K1, D, generator=generator) * 0.5
    demand = torch.rand(K1, generator=generator) + 0.5
    left = torch.randn(B, D, generator=generator)
    right = torch.randn(B, D, generator=generator)
    bound = 0.5 / math.sqrt(curvature)
    left = left / left.norm(dim=-1, keepdim=True).clamp_min(1e-12) * bound * 0.9
    right = right / right.norm(dim=-1, keepdim=True).clamp_min(1e-12) * bound * 0.9
    return {"__residual__": residual, "__codebook__": codebook,
            "__demand__": demand,
            "BOTH_ON": (left, right, curvature),
            "GEOMETRY_OFF": (left, right, 0.0)}

def deviant_implementations(implementations: list, spec: dict, *, seed: int = 17) -> list:
    """The implementations that disagree with the majority.

    The comparison is run pairwise: an implementation that disagrees with EVERY
    other one is the deviant, and one that disagrees with only one of them is
    not - two implementations that agree with each other outvote the third.
    """
    if len(implementations) < 3:
        return []
    counts = {entry["name"]: 0 for entry in implementations}
    for index, left in enumerate(implementations):
        for right in implementations[index + 1:]:
            pair = compare_modules([left, right], spec, seed=seed)
            if pair.get("status") == "AGREE":
                continue
            counts[left["name"]] += 1
            counts[right["name"]] += 1
    worst = max(counts.values()) if counts else 0
    if worst < 2:
        return []
    return [entry for entry in implementations if counts[entry["name"]] == worst]

def compare_modules(implementations: list, spec: dict, *, seed: int = 17) -> dict:
    """Whether the modules behave the same in every switch arm.

    The comparison executes torch code, so it runs in the interpreter that has
    torch and returns its verdict as JSON. A comparison that cannot run is an
    error, never an agreement: the differential is what lets a candidate touch
    the baseline tree.
    """
    import json as _json
    import subprocess
    import sys

    root = str(Path(__file__).resolve().parents[3])
    directory = Path(tempfile.mkdtemp(prefix="diff_"))
    payload_path = directory / "payload.json"
    payload_path.write_text(_json.dumps({"implementations": implementations,
                                         "spec": spec, "seed": seed}),
                            encoding="utf-8")
    script = directory / "compare.py"
    script.write_text(DRIVER % {"root": root, "payload": str(payload_path)},
                      encoding="utf-8")
    try:
        done = subprocess.run([TORCH_PYTHON, str(script)], capture_output=True,
                              text=True, timeout=1200)
    except (OSError, subprocess.SubprocessError) as error:
        return {"status": "ERROR", "pairs": [], "invalid": [],
                "reason": "the comparison could not run: %s" % error}
    marker = "@@VERDICT@@"
    line = next((item for item in reversed(done.stdout.splitlines())
                 if item.startswith(marker)), "")
    if not line:
        return {"status": "ERROR", "pairs": [], "invalid": [],
                "reason": "the comparison did not finish: " +
                          (done.stderr or done.stdout or "no output").strip()[-500:]}
    return _json.loads(line[len(marker):])
