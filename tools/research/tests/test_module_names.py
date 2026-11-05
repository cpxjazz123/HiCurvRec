"""No module of the round framework may reference a name that does not exist.

The split of the 2,122-line worker into one module per step was mechanical, and
a mechanical split can drop a helper: the round then fails at runtime with a
NameError inside a phase, in the middle of a GPU-less run, after the agents have
already spent their budget. This test is the static version of that check - it
reads every module of the round framework and asks whether every global name it
uses is defined.
"""
from __future__ import annotations

import builtins
import symtable
from pathlib import Path

import pytest

ROUNDS = Path(__file__).resolve().parents[1] / "rounds"

#: The modules of the round framework that the split produced.
MODULES = (
    "agent_calls.py",
    "step2_ideation.py",
    "step3_probe.py",
    "step4_scoring.py",
    "step5_formulation.py",
    "step5_coding.py",
    "step5_derivation.py",
    "step6_implementation.py",
    "worker.py",
)


def _missing_globals(path: Path) -> set:
    """The global names a module uses but never defines."""
    source = path.read_text(encoding="utf-8")
    table = symtable.symtable(source, str(path), "exec")
    defined = set()
    for symbol in table.get_symbols():
        if symbol.is_assigned() or symbol.is_imported() or symbol.is_namespace():
            defined.add(symbol.get_name())
        elif symbol.is_parameter():
            defined.add(symbol.get_name())
    used = set()
    for symbol in table.get_symbols():
        if symbol.is_referenced() and not symbol.is_assigned() \
                and not symbol.is_imported() and not symbol.is_parameter():
            used.add(symbol.get_name())
    builtin = set(dir(builtins)) | {"__file__", "__name__", "__doc__", "__package__"}
    return {name for name in used if name not in defined and name not in builtin}


@pytest.mark.parametrize("module", MODULES)
def test_every_global_name_of_a_step_module_exists(module):
    path = ROUNDS / module
    assert path.is_file(), path
    missing = _missing_globals(path)
    assert not missing, "%s references names that do not exist: %s" % (
        module, ", ".join(sorted(missing)))


def test_the_facade_exports_every_phase():
    """The facade is what the rest of the framework imports."""
    from tools.research.rounds import worker

    for name in ("phase_ideation", "phase_probe", "phase_scoring",
                 "phase_formulation", "phase_coding", "phase_derivation",
                 "phase_implementation", "ideate_one", "rank_one",
                 "run_ideation", "run_scoring", "run_derivations",
                 "run_implementation", "advance"):
        assert hasattr(worker, name), name


def test_the_split_modules_carry_their_step():
    """The module names say which step they are."""
    from tools.research.rounds import (agent_calls, step2_ideation,
                                       step3_probe, step4_scoring,
                                       step5_coding, step5_derivation,
                                       step5_formulation, step6_implementation)

    assert hasattr(step2_ideation, "phase_ideation")
    assert hasattr(step3_probe, "phase_probe")
    assert hasattr(step4_scoring, "phase_scoring")
    assert hasattr(step5_formulation, "phase_formulation")
    assert hasattr(step5_coding, "phase_coding")
    assert hasattr(step5_derivation, "phase_derivation")
    assert hasattr(step6_implementation, "phase_implementation")
    assert hasattr(agent_calls, "_run_agent")


def test_the_runner_takes_the_round_lock():
    """Two runs of one round must not write the state at the same time."""
    import inspect

    from tools.research.agent_flow import flow

    source = inspect.getsource(flow.run_flow)
    assert "RoundLock" in source or "round_lock" in source, source[:400]
    assert "acquire" in source
    assert "release" in source
