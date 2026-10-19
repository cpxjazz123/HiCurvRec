"""Counterexample suite for CGV -- proof that a contract's gates can actually fail.

A certificate that says "six gates green" is worthless if the gates would also be green for a
broken mechanism. This suite injects a standard set of faults, one per run, and asserts that the
gate that is supposed to catch that fault does catch it.

Every fault is injected at RUNTIME into a temporary copy of the tree, never into the tree itself:

    detached_behaviour_gradient   the declared behaviour term stops carrying gradient   -> G2
    detached_reference_gradient   the declared reference term stops carrying gradient   -> G2
    inert_mechanism               the mechanism leaves the gradient unprojected         -> G3
    wrong_coefficient             the mechanism uses a coefficient it did not declare   -> G3
    wrong_direction               the mechanism projects against a negated reference    -> G3
    mechanism_unreachable         the declared activation never opens in real training  -> G6
    inert_optimizer               the real training loop stops updating parameters      -> G6

Faults 3-5 replace the mechanism's declared executor with a wrong implementation built from the
INDEPENDENT ORACLE, so the fault is defined by the mathematics rather than by the mechanism's
internals. Every entry point here takes the declaration as an argument, for the same reason the
engine does: the judgement is made against a claim the caller supplied, not one a file found.

Usage
-----
    negative_suite.py <tree>                 # run every fault, print a JSON report
    negative_suite.py <tree> <fault> --json  # run one fault (this is how the driver isolates it)
"""
from __future__ import annotations

import contextlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import adapters  # noqa: E402
import contract as C  # noqa: E402
import engine as E  # noqa: E402
import oracles  # noqa: E402

canonical_json = C.canonical_json

CGV_PY = E.CGV_PY

FAULTS = (
    ("detached_behaviour_gradient", "G2"),
    ("detached_reference_gradient", "G2"),
    ("inert_mechanism", "G3"),
    ("wrong_coefficient", "G3"),
    ("wrong_direction", "G3"),
    ("mechanism_unreachable", "G6"),
    ("inert_optimizer", "G6"),
)


# --------------------------------------------------------------------------------------
# fault injection
# --------------------------------------------------------------------------------------
def _faulty_executor(trainer, contract, mutation: dict):
    """A wrong implementation of the DECLARED rule, built from the independent oracle.

    It reads the same losses the real executor receives, computes their gradients with autograd,
    applies the declared rule with one deliberate mutation, and writes the result where the real
    executor writes it. Nothing here depends on the mechanism's internals, which is what makes the
    fault well defined.
    """
    gradient = contract["gradient_contract"]
    rule = gradient["rule"]
    reference = gradient.get("reference_term")
    alpha = float(adapters.declared_value(trainer, gradient, "alpha", 1.0))
    eps = float(adapters.declared_value(trainer, gradient, "eps", 1e-12))

    def faulty(base_loss, l1_loss, other_loss, encoder_parameters, behavior_weight,
               valid_levels: int = 3, **kwargs):
        params = list(encoder_parameters)

        def grads(term):
            if term is None:
                return [torch.zeros_like(p) for p in params]
            found = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
            return [torch.zeros_like(p) if g is None else g for p, g in zip(params, found)]

        # Every reference term the contract may name has to be available, or a rule that projects
        # against the reconstruction or the quantisation gradient would crash here instead of
        # producing a wrong gradient for the gate to catch.
        recon_loss = kwargs.get("recon_loss")
        quant_loss = kwargs.get("quant_loss")
        if quant_loss is None and recon_loss is not None:
            quant_loss = base_loss - recon_loss

        payload = {
            "rule": rule,
            "reference_term": reference,
            "alpha": alpha,
            "eps": eps,
            "scale": float(behavior_weight) / max(int(valid_levels), 1),
            "g_base": grads(base_loss),
            "g_l1": grads(l1_loss),
            "g_other": grads(other_loss),
            "g_recon": grads(recon_loss),
            "g_quant": grads(quant_loss),
            "gate_open": True,
        }
        payload.update(mutation(payload))
        wrong = oracles.assemble_gradient(payload)
        for parameter, value in zip(params, wrong):
            parameter.grad = value.detach().clone()

    return faulty


def _install_fault(trainer, contract, fault: str) -> None:
    """Apply one fault to the freshly imported trainer module.

    The "detached" faults keep the term a tensor that requires grad but has no connection to the
    model. Fully detaching it would make the mechanism's own autograd call raise before any gate
    could report the missing gradient, which would test the fault's ability to crash rather than
    the gate's ability to detect.
    """
    if fault == "detached_behaviour_gradient":
        original = trainer.behavior_contrastive_components

        def detached(residuals, *args, **kwargs):
            l1_loss, other_loss, valid = original(residuals, *args, **kwargs)
            return (_detached_term(l1_loss), _detached_term(other_loss), valid)

        trainer.behavior_contrastive_components = detached
    elif fault == "detached_reference_gradient":
        original = trainer.RQVAE.compute_loss

        def detached_loss(self, embeddings, reconstructed, quant_loss):
            base_loss, recon_loss = original(self, embeddings, reconstructed, quant_loss)
            return _detached_term(base_loss), recon_loss

        trainer.RQVAE.compute_loss = detached_loss
    elif fault == "inert_mechanism":
        setattr(trainer, contract["mechanism"]["executor"],
                _faulty_executor(trainer, contract, lambda payload: {"rule": "unprojected"}))
    elif fault == "wrong_coefficient":
        def mutation(payload):
            # A wrong coefficient on the declared behaviour term. Scaling the term itself changes
            # every rule this family declares -- the projection coefficient for the projected
            # rules, and the assembled term for the rest -- whereas perturbing a named alpha would
            # silently do nothing for the rules that take no alpha, and the fault would then be
            # reported as "caught" without having been injected at all.
            return {"g_l1": [value * 1.5 for value in payload["g_l1"]]}

        setattr(trainer, contract["mechanism"]["executor"],
                _faulty_executor(trainer, contract, mutation))
    elif fault == "wrong_direction":
        def mutation(payload):
            return {"g_base": [-value for value in payload["g_base"]]}

        setattr(trainer, contract["mechanism"]["executor"],
                _faulty_executor(trainer, contract, mutation))
    elif fault == "mechanism_unreachable":
        # switch the declared activation OFF, which is exactly "formal training never reaches the
        # mechanism". Lowering the behaviour weight would not do it: G6 opens the declared ramp on
        # purpose, so the fault has to disable the switch itself.
        activation = (contract.get("mechanism") or {}).get("activation") or {}
        flags = list(activation.get("requires") or [])
        if activation.get("flag"):
            flags.append(activation["flag"])
        if not flags:
            raise ValueError("this contract declares no activation flag to switch off")
        for flag in flags:
            setattr(trainer, flag, False)
    elif fault == "inert_optimizer":
        def no_step(self, *args, **kwargs):
            return None

        getattr(torch.optim, contract["update_contract"]["optimizer"]).step = no_step
    else:
        raise ValueError(f"unknown fault {fault!r}")


def _detached_term(term):
    """A tensor with the same value that requires grad but is connected to nothing."""
    if term is None:
        return None
    return term.detach().requires_grad_(True)


# --------------------------------------------------------------------------------------
# running one fault in its own process
# --------------------------------------------------------------------------------------
def run_fault(tree: Path, spec: Any, fault: str) -> dict:
    expected_gate = dict(FAULTS)[fault]
    tree = Path(tree)
    contract = C.normalise_spec(spec)
    trainer, saved_path, saved_mods = E.load_trainer(tree, module_name="cgv_fault_trainer")
    started = time.time()
    try:
        _install_fault(trainer, contract, fault)
        ctx = E.load_context(tree, contract, trainer=trainer)
        with contextlib.redirect_stdout(sys.stderr):
            if expected_gate == "G1":
                gate = E.gate_g1(ctx)
            elif expected_gate == "G2":
                gate = E.gate_g2(ctx)
            elif expected_gate == "G3":
                gate = E.gate_g3(ctx)
            elif expected_gate == "G4":
                gate = E.gate_g4(ctx)
            elif expected_gate == "G5":
                gate = E.gate_g5(ctx)
            else:
                with tempfile.TemporaryDirectory() as raw:
                    gate = E.gate_g6(ctx, Path(raw))
    finally:
        E.unload_trainer(saved_path, saved_mods, "cgv_fault_trainer")
    return {
        "fault": fault,
        "expected_gate": expected_gate,
        "expected_state": C.FAILED,
        "observed_state": gate.get("state"),
        "observed_detail": str(gate.get("detail"))[:400],
        "ok": gate.get("state") == C.FAILED,
        "elapsed_s": round(time.time() - started, 1),
    }


def run_all(tree: Path, spec: Any, faults=FAULTS) -> dict:
    """Run every fault in its own subprocess, so one injected fault cannot leak into the next."""
    tree = Path(tree)
    results = []
    for fault, _gate in faults:
        proc = subprocess.run(
            [CGV_PY, str(Path(__file__)), str(tree), fault, "--json", "--spec", "-"],
            input=canonical_json(spec), capture_output=True, text=True,
            timeout=E.CHILD_TIMEOUT_S * 2)
        line = ""
        for candidate in reversed(proc.stdout.strip().splitlines()):
            if candidate.strip().startswith("{"):
                line = candidate
                break
        if not line:
            results.append({"fault": fault, "ok": False,
                            "error": "the fault run produced no JSON",
                            "stderr_tail": proc.stderr.strip()[-400:]})
            continue
        results.append(json.loads(line))
    return {
        "suite": "negative",
        "tree": str(tree),
        "faults": results,
        "all_ok": all(item.get("ok") for item in results),
        "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _spec_from_argv(argv: list[str]) -> Any:
    """The declaration arrives the same way it arrives on the MCP call: as data."""
    if "--spec" not in argv:
        raise SystemExit("the declaration is required: pass --spec - (JSON on stdin) or "
                         "--spec <file.json>")
    value = argv[argv.index("--spec") + 1]
    if value == "-":
        return C.normalise_spec(sys.stdin.read(), source="the declaration on stdin")
    if value.lstrip().startswith("{"):
        return C.normalise_spec(value, source="the declaration on the command line")
    return C.normalise_spec(Path(value).read_text(encoding="utf-8"),
                            source=f"the declaration in {value}")


def main() -> int:
    if len(sys.argv) < 2:
        print('usage: negative_suite.py <tree> [fault] --spec <declaration.json|-> [--json]',
              file=sys.stderr)
        return 2
    tree = Path(sys.argv[1])
    as_json = "--json" in sys.argv
    spec = _spec_from_argv(sys.argv[2:])
    if len(sys.argv) > 2 and not sys.argv[2].startswith("--"):
        report = run_fault(tree, spec, sys.argv[2])
        print(json.dumps(report, default=str), flush=True)
        return 0 if report["ok"] else 1
    report = run_all(tree, spec)
    print(json.dumps(report, indent=2, default=str), flush=True)
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
