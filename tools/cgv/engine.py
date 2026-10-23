"""CGV -- Computational Graph & Gradient Verification.

CGV verifies a mechanism against the mathematical behaviour that mechanism DECLARES. It never
compares two versions of a mechanism, never needs a baseline checkpoint, and never requires the
mechanism to have an on/off switch. A declaration that mentions a switch is honoured; one that
mentions none is verified exactly the same way.

The declaration is an ARGUMENT. It arrives as data on the call and CGV reads nothing from the
mechanism's tree: a mechanism hands over no files to be judged by.

The three roles are kept apart on purpose:

    contract (contract.py)   what the mechanism claims, as data
    adapter  (adapters.py)   how to execute the real mechanism, and nothing else
    oracle   (oracles.py)    an independent mathematical reference, importing no mechanism

Gates
-----
    G1 forward contract        the observed forward satisfies the declared formula and invariants
    G2 gradient propagation    every declared loss path reaches the declared parameters, with the
                               declared zero/non-zero/finite character
    G3 gradient mathematics    the executed gradient equals the declared rule, and the mechanism
                               demonstrably changes it
    G4 parameter update        the declared optimizer and clipping produce the declared update
    G5 distributed consistency multi-rank execution matches the declared aggregation rule
    G6 real training entry     formal training executes the declared mechanism and moves a
                               parameter

Gate outcomes
-------------
A gate is ``passed``, ``failed``, ``not_applicable`` (the contract explicitly declares that the
mechanism has no effect in that dimension, and says why) or ``insufficient_evidence`` (the
contract leaves the gate nothing testable). ``not_applicable`` is a declaration and is reported
as such. ``insufficient_evidence`` NEVER counts as success: a mechanism whose behaviour cannot be
stated cannot be verified, and the certificate says so instead of passing it.

Output isolation
----------------
G6 executes the trainer's real entry point. :func:`isolated_outputs` redirects every path the
trainer WRITES into a temporary directory first, discovered from the filenames a trainer writes
rather than listed by hand. Without it, merely verifying a mechanism would overwrite the
artifacts of the run under audit.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import types
import warnings
from pathlib import Path
from typing import Any

import torch

import adapters
import contract as C
import oracles

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CGV_DIR = REPO_ROOT / "results/stage2_RQ-VAE/_cgv"
CGV_PY = "/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9"
if not Path(CGV_PY).exists():
    CGV_PY = sys.executable

CHILD_TIMEOUT_S = 900
G6_EPOCHS = 2
ATOL = 1e-5
RTOL = 1e-5

PROBE_SEEDS = (1, 4, 5, 12, 16, 30, 32, 36)

GATES = C.GATES
GATE_TITLES = C.GATE_TITLES
GATE_PREVENTS = C.GATE_PREVENTS

# The files a trainer WRITES. Inputs (stage1 embeddings, stage0 parquet, dataset frames) share the
# repository root, so "anything under the repo" would be the wrong rule: it would move the inputs
# into a temporary directory and the run would fail before it ever reached the mechanism.
TRAINER_WRITES = {
    "training_metrics.jsonl",
    "rqvae_best.pth",
    "sids_for_hgrec.npy",
    "sids_raw.npy",
    "item_sids.json",
    "launcher.log",
}
TRAINER_OUTPUT_DIRS = {"logs", "results"}


# --------------------------------------------------------------------------------------
# isolation and loading
# --------------------------------------------------------------------------------------
@contextlib.contextmanager
def isolated_outputs(T, tmp: Path):
    """Redirect every path the trainer WRITES into ``tmp``.

    The redirection is DISCOVERED, not listed by hand. ``METRICS_PATH`` for instance is computed
    from ``LOG_DIR`` at import time, so reassigning ``LOG_DIR`` afterwards silently leaves the
    metrics stream pointing at the real run -- which is exactly how a two-epoch verification once
    truncated a 3003-line training log. Seeds are the module paths named like something a trainer
    writes, and the redirection then expands to every module path underneath them. Read-only
    inputs are left exactly where they are.
    """
    saved: dict[str, Path] = {}
    roots: list[Path] = []

    def is_seed(value: Path) -> bool:
        return value.name in TRAINER_WRITES or value.name in TRAINER_OUTPUT_DIRS

    for name, value in list(vars(T).items()):
        if not isinstance(value, Path) or not is_seed(value):
            continue
        try:
            resolved = value.resolve()
        except OSError:
            continue
        if not str(resolved).startswith(str(REPO_ROOT.resolve()) + os.sep):
            continue
        saved[name] = value
        roots.append(resolved)
        try:
            relative = resolved.relative_to(REPO_ROOT.resolve())
            setattr(T, name, tmp / "redirected" / relative)
        except ValueError:
            setattr(T, name, tmp / "redirected" / value.name)

    if roots:
        for name, value in list(vars(T).items()):
            if name in saved or not isinstance(value, Path):
                continue
            try:
                resolved = value.resolve()
            except OSError:
                continue
            if any(resolved == root or root in resolved.parents for root in roots):
                saved[name] = value
                try:
                    relative = resolved.relative_to(REPO_ROOT.resolve())
                    setattr(T, name, tmp / "redirected" / relative)
                except ValueError:
                    setattr(T, name, tmp / "redirected" / value.name)

    for root in (tmp / "redirected", tmp / "out", tmp / "logs"):
        root.mkdir(parents=True, exist_ok=True)
    try:
        yield sorted(saved)
    finally:
        for name, value in saved.items():
            setattr(T, name, value)


def load_trainer(tree: Path, module_name: str = "cgv_trainer"):
    """Import ``tree/train_rqvae.py`` under a private name so arms cannot collide."""
    tree = Path(tree).resolve()
    entry = tree / "train_rqvae.py"
    if not entry.is_file():
        raise FileNotFoundError(f"no train_rqvae.py in {tree}")
    saved_path = list(sys.path)
    saved_mods = {k: sys.modules.get(k) for k in
                  (module_name, "model", "model.model", "model.layers")}
    for key in saved_mods:
        sys.modules.pop(key, None)
    sys.path.insert(0, str(tree))
    module = types.ModuleType(module_name)
    module.__file__ = str(entry)
    module.__dict__["__name__"] = module_name
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            spec = importlib.util.spec_from_file_location(module_name, entry)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)  # type: ignore[union-attr]
        except RuntimeError:
            # Single-arm guard: read EXPERIMENT_ARM from the source and neutralise only that
            # guard, which exists to stop an accidental rerun, not to change behaviour.
            import ast

            parsed = ast.parse(entry.read_text(encoding="utf-8"), filename=str(entry))
            parsed.body = [
                node for node in parsed.body
                if not (isinstance(node, ast.If)
                        and ast.unparse(node.test).startswith("EXPERIMENT_ARM !="))
            ]
            module = types.ModuleType(module_name)
            module.__file__ = str(entry)
            module.__dict__["__name__"] = module_name
            sys.modules[module_name] = module
            exec(compile(parsed, str(entry), "exec"), module.__dict__)  # noqa: S102
    return module, saved_path, saved_mods


def unload_trainer(saved_path, saved_mods, module_name="cgv_trainer"):
    sys.path[:] = saved_path
    for key, value in saved_mods.items():
        if value is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = value
    sys.modules.pop(module_name, None)


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------------------
# context
# --------------------------------------------------------------------------------------
class Context:
    """Everything a gate needs: the real trainer, the declaration, and the independence proof."""

    def __init__(self, tree: Path, trainer, spec: dict[str, Any]) -> None:
        self.tree = tree
        self.trainer = trainer
        self.contract = spec
        self.fingerprint = C.fingerprint(spec)
        self.independence = C.independence_report(spec)

    def gate_state(self, gate: str) -> tuple[str, str]:
        if not self.independence["independent"]:
            return C.INSUFFICIENT, (
                "the declared oracles are not independent of the mechanism: "
                + "; ".join(self.independence["problems"]))
        return C.gate_state(gate, self.contract)

    def observations(self, seed: int = 1234) -> dict[str, Any]:
        return adapters.build_observations(self.trainer, self.contract, seed=seed)

    def trace_observations(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Everything the independent trace oracles need, and nothing they may derive from the arm.

        The cotangents ``u_levels`` and the latent pullback are the chain rule factors AROUND the
        mechanism; the only thing the mechanism contributes is the Jacobian of its trace with
        respect to the latent, and that is supplied as DECLARED numbers.
        """
        mechanism = self.contract.get("mechanism") or {}
        mathematics = mechanism.get("mathematics") or {}
        gradient = self.contract["gradient_contract"]
        payload: dict[str, Any] = {
            "rule": gradient["rule"],
            "traces": obs["traces"],
            "codebooks": obs["codebooks"],
            "u_levels": adapters.level_cotangents(self.trainer, obs),
            "latent": obs["latent"],
            "encoder": obs["encoder"],
            "codebook_parameters": obs.get("codebooks_parameters", []),
            "latent_to_encoder": lambda cotangent: adapters.latent_to_encoder(obs, cotangent),
            "coefficients": mathematics.get("expected_level_to_encoder_jacobians"),
            "soft_normalizer_floor": mathematics.get("minimum_normalizer", 1e-8),
            "reference_traces": obs.get("reference_traces"),
            "reference_tokens": obs.get("reference_tokens"),
            "trace_indices": obs.get("trace_indices"),
        }
        return payload

    def oracle_observations(self, obs: dict[str, Any],
                            executed: bool = True) -> dict[str, Any]:
        """The raw mathematics the independent oracle needs for this case.

        Every value is an OBSERVATION (a gradient the real graph produced, a declared constant, a
        structural block list). No expected value is computed here -- that is the oracle's job.
        """
        gradient = self.contract["gradient_contract"]
        T = self.trainer
        terms = adapters.loss_terms(T, self.contract, obs)
        g_base = adapters.term_gradient(T, obs, terms["base_loss"])
        g_l1 = adapters.term_gradient(T, obs, terms["l1_loss"]) if "l1_loss" in terms else None
        g_other = (adapters.term_gradient(T, obs, terms["other_loss"])
                   if "other_loss" in terms else None)
        if g_l1 is None or g_other is None:
            raise ValueError("the declared assembly needs both a behaviour l1 term and other term")
        payload: dict[str, Any] = {
            "rule": gradient["rule"],
            "reference_term": gradient.get("reference_term"),
            "alpha": adapters.declared_value(T, gradient, "alpha", 1.0),
            "eps": adapters.declared_value(T, gradient, "eps", 1e-12),
            "scale": obs["scale"],
            "g_base": g_base,
            "g_l1": g_l1,
            "g_other": g_other,
            "g_recon": adapters.term_gradient(T, obs, terms["recon_loss"])
            if "recon_loss" in terms else g_base,
            "g_quant": adapters.term_gradient(T, obs, terms["quant_loss"])
            if "quant_loss" in terms else g_base,
        }
        if gradient.get("cos_threshold_flag"):
            payload["cos_threshold"] = adapters.declared_value(T, gradient, "cos_threshold", -0.2)
        if gradient.get("blocks"):
            payload["blocks"] = adapters.structural_blocks(T, self.contract, obs)
        return payload


def is_trace_mechanism(spec: dict[str, Any]) -> bool:
    """A mechanism that rewrites the residual trace has no separate projection step to execute.

    Recognised from the rule registry rather than an extra declaration key, so a specification
    that states only the mathematics is enough -- and an unrecognised rule fails closed in both
    the schema and the gradient gate.
    """
    rule = (spec.get("gradient_contract") or {}).get("rule")
    return rule in oracles.TRACE_RULES


def load_context(tree: Path, spec: dict[str, Any], trainer=None) -> Context:
    """Bind the declaration to the loaded mechanism. The declaration is data, passed in."""
    problems = C.validate(spec)
    if problems:
        raise C.ContractError("; ".join(problems))
    return Context(Path(tree).resolve(), trainer, spec)


def _result(gate: str, state: str, detail: str, measurements: dict | None = None,
            finding: str | None = None, warnings: list | None = None) -> dict[str, Any]:
    return {
        "gate": gate,
        "title": GATE_TITLES[gate],
        "state": state,
        "passed": True if state == C.PASSED else (False if state == C.FAILED else None),
        "detail": detail,
        "measurements": measurements or {},
        "finding": finding,
        "warnings": warnings or [],
    }


def _skipped(gate: str, ctx: Context) -> dict[str, Any] | None:
    state, reason = ctx.gate_state(gate)
    if state == C.PASSED:
        return None
    return _result(gate, state, reason,
                   finding=None if state == C.NOT_APPLICABLE else reason)


# --------------------------------------------------------------------------------------
# G1 -- forward contract
# --------------------------------------------------------------------------------------
def _jacobian_deviation(T, jacobian: dict[str, Any]) -> float:
    """Deprecated: a hand-built residual recurrence is not evidence about any arm.

    It used to subtract the quantiser output directly, which silently described the LEGACY trace
    while being presented as a check on whatever arm was under test. G1 now measures the arm's own
    trace; this remains only so an older declaration naming it fails loudly instead of quietly.
    """
    raise C.ContractError(
        "forward_contract.jacobian probes a hand-built recurrence, not the arm's trace; declare "
        "the level Jacobians under mechanism.mathematics and let G3 measure them on the real trace")


def gate_g1(ctx: Context) -> dict[str, Any]:
    skip = _skipped("G1", ctx)
    if skip:
        return skip
    forward = ctx.contract["forward_contract"]
    trace = is_trace_mechanism(ctx.contract)
    observations = (adapters.trace_case(ctx.trainer, ctx.contract) if trace
                    else ctx.observations())
    measurements: dict[str, Any] = {"execution": "trace" if trace else "assembly"}

    checks = oracles.check_forward_invariants(observations, forward.get("invariants", []))

    if trace and is_trace_mechanism(ctx.contract):
        checks.append(oracles.hard_assignment_algebra(
            observations, tol=float(ctx.contract["forward_contract"].get("atol", ATOL))))

    expected = None
    if forward.get("oracle"):
        oracle = C.resolve_reference(forward["oracle"])
        expected = oracle(observations)
        checks.extend(oracles.compare_forward(observations, expected, float(
            forward.get("atol", ATOL))))

    failures = [check for check in checks if not check["ok"]]
    measurements["checks"] = checks
    measurements["declared_effect"] = forward.get("declared_effect", "none")
    passed = not failures
    return _result(
        "G1", C.PASSED if passed else C.FAILED,
        (f"the forward satisfies {len(checks)} declared check(s): "
         + ", ".join(check["name"] for check in checks))
        if passed else
        ("the forward violates the declared contract: "
         + "; ".join(f"{check['name']} -> {check['detail']}" for check in failures)),
        measurements,
        finding=None if passed else
        "the mechanism's forward does not satisfy what its contract declares")


# --------------------------------------------------------------------------------------
# G2 -- gradient propagation
# --------------------------------------------------------------------------------------
def gate_g2(ctx: Context) -> dict[str, Any]:
    skip = _skipped("G2", ctx)
    if skip:
        return skip
    gradient = ctx.contract["gradient_contract"]
    if is_trace_mechanism(ctx.contract):
        return gate_g2_trace(ctx)

    expected = gradient.get("expected_paths") or {}
    declared = sorted(adapters.loss_terms(ctx.trainer, ctx.contract, ctx.observations()))

    checks, warnings = [], []
    for name in declared:
        # a fresh case per term: the mechanism's assembly frees the graph it runs on, so a shared
        # case would fail on the second term instead of reporting its gradient
        case = ctx.observations()
        term = adapters.loss_terms(ctx.trainer, ctx.contract, case)[name]
        grads = adapters.term_gradient(ctx.trainer, case, term)
        norm = oracles.flat_norm(grads)
        finite = all(bool(torch.isfinite(g).all()) for g in grads)
        read = adapters.probe_substitution(ctx.trainer, ctx.contract, name, seed=case["seed"])
        expectation = expected.get(name, "nonzero")
        if expectation == "nonzero":
            ok = finite and norm > 0.0
        elif expectation == "zero":
            ok = finite and norm == 0.0
        else:
            ok = finite
        if read <= 1e-6 and expectation != "zero":
            warnings.append(
                f"{name} reaches the declared parameters but the mechanism does not consume it "
                f"(probe substitution moved the executed gradient by {read:.3e})")
        checks.append({"term": name, "expectation": expectation, "ok": ok,
                       "gradient_norm": norm, "finite": finite,
                       "read_by_mechanism": read,
                       "detail": f"declared {expectation}, norm {norm:.6e}, finite={finite}"})

    failures = [check for check in checks if not check["ok"]]
    passed = not failures
    return _result(
        "G2", C.PASSED if passed else C.FAILED,
        (f"every declared loss path reaches the {gradient.get('target')!r} parameters with the "
         f"declared character: "
         + ", ".join(f"{c['term']}={c['expectation']}(norm {c['gradient_norm']:.3e})"
                     for c in checks))
        if passed else
        ("declared propagation does not hold: "
         + "; ".join(f"{c['term']} declared {c['expectation']} -> {c['detail']}"
                     for c in failures)),
        {"checks": checks, "target": gradient.get("target")},
        finding=None if passed else
        "a declared loss path does not deliver the gradient the contract claims",
        warnings=warnings,
    )


def gate_g2_trace(ctx: Context) -> dict[str, Any]:
    """Per-level propagation for a mechanism that rewrites the residual trace.

    Each declared behaviour level is checked separately against the declared target parameters,
    and the result is classified as disconnected / zero / non-zero / non-finite. The distinction
    matters: a level that is not in the graph at all and a level whose gradient is exactly zero
    have the same norm, and only one of them is the mathematics the declaration describes.
    """
    gradient = ctx.contract["gradient_contract"]
    observations = adapters.trace_case(ctx.trainer, ctx.contract)
    expected = gradient.get("expected_paths") or {}
    additional = gradient.get("additional_targets") or {}
    encoder_ids = {id(p) for p in observations["encoder"]}
    codebook_ids = {id(p) for p in observations.get("codebooks_parameters", [])}

    checks: list[dict[str, Any]] = []
    warnings: list[str] = []
    # A declaration names each term and says which adapter source supplies it. Levels are
    # addressed by their SOURCE, so the mapping never depends on how the terms happen to be named
    # or sorted.
    declared_terms = gradient.get("loss_terms") or {}
    by_source = {source: name for name, source in declared_terms.items()}
    level_terms = [(level, by_source.get(f"behavior_l{level + 1}"))
                   for level in range(len(observations["level_losses"]))]
    for level, name in level_terms:
        if name is None:
            continue
        source = f"behavior_l{level + 1}"
        loss = observations["level_losses"][level]
        # A detached level leaves a loss that requires no grad at all. autograd.grad would raise
        # on it, so the disconnection is classified here -- it is a reportable state, not a crash.
        if not loss.requires_grad:
            encoder_grads = [torch.zeros_like(p) for p in observations["encoder"]]
            state = "disconnected"
        else:
            encoder_grads = torch.autograd.grad(loss, observations["encoder"], retain_graph=True,
                                                allow_unused=True)
            encoder_grads = [torch.zeros_like(p) if g is None else g
                             for p, g in zip(observations["encoder"], encoder_grads)]
            state = adapters.level_path_state(encoder_grads)
        wanted = expected.get(name, "nonzero")
        if wanted == "nonzero":
            ok = state == "nonzero"
        elif wanted == "zero":
            ok = state in ("zero", "disconnected")
        else:
            ok = state in ("nonzero", "zero", "disconnected")
        entry = {"term": name, "level": level, "target": "encoder", "expectation": wanted,
                 "state": state, "ok": ok,
                 "norm": oracles.flat_norm(encoder_grads),
                 "detail": f"declared {wanted}, path is {state}"}
        if state == "zero" and wanted == "nonzero":
            entry["detail"] += " -- the level's gradient is exactly zero, not merely small"
        checks.append(entry)

        if codebook_ids:
            codebook_grads = (adapters.codebook_level_gradient(ctx.trainer, observations, level)
                              if loss.requires_grad else [])
            codebook_state = (adapters.level_path_state(codebook_grads) if codebook_grads
                              else "disconnected")
            codebook_declared = additional.get("codebooks") or {}
            wanted_codebook = codebook_declared.get(name, codebook_declared.get(source))
            if wanted_codebook is not None:
                if wanted_codebook.startswith("nonzero"):
                    codebook_ok = codebook_state == "nonzero"
                elif wanted_codebook.startswith("zero"):
                    codebook_ok = codebook_state in ("zero", "disconnected")
                else:
                    codebook_ok = True
                checks.append({"term": name, "level": level, "target": "codebooks",
                               "expectation": wanted_codebook, "state": codebook_state,
                               "ok": codebook_ok,
                               "norm": oracles.flat_norm(codebook_grads) if codebook_grads else 0.0,
                               "detail": f"declared {wanted_codebook}, path is {codebook_state}"})

    failures = [check for check in checks if not check["ok"]]
    if not checks:
        # an empty check list would pass trivially; a gate that examined nothing proves nothing
        return _result("G2", C.INSUFFICIENT,
                       "no declared behaviour level produced a check: the declaration names "
                       f"{sorted(declared_terms)} but the arm's trace has "
                       f"{len(observations['level_losses'])} level(s)",
                       finding="nothing was verified, so nothing may be certified")
    passed = not failures
    return _result(
        "G2", C.PASSED if passed else C.FAILED,
        (f"every declared level reaches its declared target with the declared character: "
         + ", ".join(f"L{check['level']}:{check['target']}={check['state']}" for check in checks))
        if passed else
        ("declared propagation does not hold: "
         + "; ".join(f"L{check['level']} {check['target']} declared {check['expectation']} but is "
                     f"{check['state']}" for check in failures)),
        {"checks": checks},
        finding=None if passed else
        "a declared behaviour level does not reach the declared parameters as declared",
        warnings=warnings,
    )


# --------------------------------------------------------------------------------------
# G3 -- gradient mathematics
# --------------------------------------------------------------------------------------
def gate_g3(ctx: Context) -> dict[str, Any]:
    skip = _skipped("G3", ctx)
    if skip:
        return skip
    gradient = ctx.contract["gradient_contract"]
    mode = gradient["mode"]
    T = ctx.trainer

    if mode == "invariants":
        observations = ctx.observations()
        executed = adapters.executed_target_gradient(T, ctx.contract, observations)
        checks = oracles.check_gradient_invariants(executed, gradient["invariants"])
        failures = [check for check in checks if not check["ok"]]
        return _result(
            "G3", C.PASSED if not failures else C.FAILED,
            (f"the executed gradient satisfies {len(checks)} declared invariant(s)")
            if not failures else
            ("the executed gradient violates the declared invariants: "
             + "; ".join(f"{c['name']} -> {c['detail']}" for c in failures)),
            {"checks": checks},
            finding=None if not failures else
            "the executed gradient does not satisfy the declared invariants",
        )

    if is_trace_mechanism(ctx.contract):
        return gate_g3_trace(ctx)

    if gradient.get("rule") == "r7_f_soft_same_code_loss":
        return gate_g3_r7_f(ctx)

    seeds = tuple(gradient.get("probe_seeds") or PROBE_SEEDS)
    activation = (ctx.contract.get("mechanism") or {}).get("activation") or {}
    epochs: list[int | None] = [None]
    if activation.get("epoch_gate"):
        gate_value = int(getattr(T, activation["epoch_gate"]))
        epochs = [gate_value, max(0, gate_value - 1)]

    ema = gradient.get("ema") or {}
    per_probe, worst, effect_probes = [], 0.0, []
    for index, seed in enumerate(seeds):
        for epoch in epochs:
            if ema:
                offset = int(ema.get("prime_seed_offset", 1))
                prime_seed = seeds[(index + offset) % len(seeds)]
                series, _primed = adapters.prime_ema(T, ctx.contract, prime_seed)
                observations = ctx.observations(seed=seed)
                payload = ctx.oracle_observations(observations)
                payload["ema_reference"] = oracles.ema_reference(
                    series + [payload["g_base"]], float(
                        adapters.declared_value(T, ema, "decay", 0.9)))
            else:
                observations = ctx.observations(seed=seed)
                payload = ctx.oracle_observations(observations)

            payload["gate_open"] = adapters.activation_open(T, ctx.contract, epoch)
            executed = adapters.executed_target_gradient(T, ctx.contract, observations,
                                                        epoch=epoch)
            expected = oracles.assemble_gradient(payload)
            unprojected = oracles.unprojected_gradient(payload)
            deviation = oracles.max_abs_deviation(executed, expected)
            effect = oracles.l2_distance(executed, unprojected)
            worst = max(worst, deviation)
            threshold = float((gradient.get("effect") or {}).get("min_deviation", 1e-6))
            if effect > threshold:
                effect_probes.append(f"seed{seed}@epoch{epoch}" if epoch is not None
                                     else f"seed{seed}")
            per_probe.append({
                "seed": seed, "epoch": epoch,
                "deviation_from_declared_rule": deviation,
                "deviation_from_unprojected": effect,
                "gate_open": payload["gate_open"],
            })

    minimum = int((gradient.get("effect") or {}).get("min_probes", 1))
    tolerance = float(gradient.get("atol", ATOL))
    formula_ok = worst <= tolerance
    effect_ok = len(effect_probes) >= minimum
    passed = formula_ok and effect_ok
    total = len(per_probe)
    detail = (f"max deviation from the declared rule {gradient['rule']!r} over {total} probe(s) "
              f"= {worst:.3e}")
    if not effect_ok:
        detail += (f"; the mechanism leaves the gradient identical to the unprojected assembly on "
                   f"all {total} probe(s), so it cannot be shown to do anything")
    elif effect_probes:
        detail += (f"; the mechanism changes the gradient on {len(effect_probes)}/{total} probe(s) "
                   f"by up to {max(p['deviation_from_unprojected'] for p in per_probe):.3e}")
    if not formula_ok:
        detail += "; the executed gradient does not match the declared rule"
    return _result(
        "G3", C.PASSED if passed else C.FAILED, detail,
        {"per_probe": per_probe,
         "max_deviation_from_declared_rule": worst,
         "probes_where_mechanism_changes_the_gradient": effect_probes,
         "rule": gradient["rule"],
         "reference_term": gradient.get("reference_term"),
         "mode": mode},
        finding=None if passed else (
            "the executed gradient contradicts the mechanism's own declared formula"
            if not formula_ok else
            "the mechanism never changes the gradient it declares to change"),
    )


def gate_g3_trace(ctx: Context) -> dict[str, Any]:
    """The executed trace gradient against the declared rule, plus a per-level Jacobian probe.

    Two independent obligations:

    * the gradient the arm's real trace produces must equal what the declared mathematics says,
      recomputed from observations by the oracle rather than read off the arm;
    * each level's measured trace-to-latent Jacobian must equal the declared coefficient, which
      is what rejects the neighbouring rules -- the legacy rule measures 0 and the plain
      straight-through measures 1, so neither can pass as 0.5.
    """
    gradient = ctx.contract["gradient_contract"]
    mathematics = (ctx.contract.get("mechanism") or {}).get("mathematics") or {}
    declared = mathematics.get("expected_level_to_encoder_jacobians")
    vjp_rule = gradient["rule"] in oracles.VJP_RULES
    if not declared and not vjp_rule:
        return _result("G3", C.INSUFFICIENT,
                       "mechanism.mathematics.expected_level_to_encoder_jacobians is required: "
                       "without declared level Jacobians there is nothing to verify the trace "
                       "against, and this rule is not one whose Jacobian the oracle derives "
                       "directly")
    coefficients = [float(value) for value in declared] if declared else []
    seeds = tuple(gradient.get("probe_seeds") or PROBE_SEEDS)
    tolerance = float(gradient.get("atol", ATOL))
    threshold = float((gradient.get("effect") or {}).get("min_deviation", 1e-6))
    minimum = int((gradient.get("effect") or {}).get("min_probes", 1))

    per_seed, worst, jacobian_rows, firing = [], 0.0, [], []
    for seed in seeds:
        observations = adapters.trace_case(ctx.trainer, ctx.contract, seed=seed)
        executed = adapters.executed_trace_gradient(ctx.trainer, observations)
        payload = ctx.trace_observations(observations)
        expectation = oracles.trace_gradient(payload)
        expected_encoder = expectation["encoder"] if isinstance(expectation, dict) else expectation
        deviation = oracles.max_abs_diff(executed, expected_encoder)
        worst = max(worst, deviation)

        # The declaration has already been matched, so "the mechanism does something" has to be
        # measured against the UNTREATED assembly, not against the declaration it agreed with.
        untreated = dict(payload)
        if coefficients:
            untreated["coefficients"] = [1.0] + [0.0] * (len(coefficients) - 1)
            untreated_expectation = oracles.trace_gradient(untreated)
            untreated_encoder = (untreated_expectation["encoder"]
                                 if isinstance(untreated_expectation, dict)
                                 else untreated_expectation)
            effect = oracles.l2_distance(executed, untreated_encoder)
        else:
            # A VJP rule has no constant coefficients to zero out; its untreated counterpart is
            # the straight-through path the arm replaced, rebuilt from the same observations.
            untreated = dict(payload)
            untreated["rule"] = "half_ste_residual"
            untreated["coefficients"] = [1.0] + [0.0] * (len(payload["traces"]) - 1)
            untreated_expectation = oracles.trace_gradient(untreated)
            untreated_encoder = (untreated_expectation["encoder"]
                                 if isinstance(untreated_expectation, dict)
                                 else untreated_expectation)
            effect = oracles.l2_distance(executed, untreated_encoder)
        codebook_deviation = 0.0
        if isinstance(expectation, dict) and expectation.get("codebooks"):
            actual_codebooks = adapters.codebook_total_gradient(ctx.trainer, observations)
            codebook_deviation = oracles.max_abs_diff(actual_codebooks,
                                                      expectation["codebooks"])
            worst = max(worst, codebook_deviation)
        per_seed.append({"seed": seed, "deviation_from_declared_rule": deviation,
                         "codebook_deviation": codebook_deviation,
                         "deviation_from_untreated": effect})
        if effect > threshold:
            firing.append(f"seed{seed}")

        if seed == seeds[0] and coefficients:
            for level, coefficient in enumerate(coefficients):
                if level >= len(observations["traces"]):
                    break
                probe = adapters.jacobian_probe(ctx.trainer, observations, level)
                measured = probe["alignment"]
                jacobian_rows.append({
                    "level": level, "declared": coefficient, "measured_alignment": measured,
                    "measured_ratio": probe["ratio"], "connected": probe["connected"],
                    "ok": probe["connected"] and abs(measured - coefficient) <= tolerance,
                })

    jacobian_failures = [row for row in jacobian_rows if not row["ok"]]
    formula_ok = worst <= tolerance
    effect_ok = len(firing) >= minimum
    passed = formula_ok and effect_ok and not jacobian_failures

    if jacobian_rows:
        detail = (f"max deviation from the declared rule {gradient['rule']!r} over "
                  f"{len(per_seed)} probe(s) = {worst:.3e}; measured level Jacobians "
                  + ", ".join(f"L{row['level']}={row['measured_alignment']:.4f}"
                              f"(declared {row['declared']})" for row in jacobian_rows))
    else:
        detail = (f"max deviation from the declared rule {gradient['rule']!r} over "
                  f"{len(per_seed)} probe(s) = {worst:.3e}, measured against the VJP the oracle "
                  "derives for the encoder and, where declared, the codebooks -- this rule's "
                  "trace-to-latent Jacobian is data dependent, so no constant level Jacobian "
                  "exists to compare")
    detail += ("; deviation from the untreated assembly on "
               + ", ".join(f"seed{entry['seed']}={entry['deviation_from_untreated']:.3e}"
                           for entry in per_seed))
    if not formula_ok:
        detail += "; the executed gradient does not match the declared rule"
    if jacobian_failures:
        detail += ("; level Jacobians disagree with the declaration: "
                   + ", ".join(f"L{row['level']} measured {row['measured_alignment']:.4f} vs "
                               f"declared {row['declared']}" for row in jacobian_failures))
    if not effect_ok:
        detail += "; the mechanism did not measurably change the gradient on any probe"
    return _result(
        "G3", C.PASSED if passed else C.FAILED, detail,
        {"per_seed": per_seed, "max_deviation_from_declared_rule": worst,
         "level_jacobians": jacobian_rows, "rule": gradient["rule"],
         "declared_level_jacobians": coefficients,
         "probes_where_mechanism_changes_the_gradient": list(firing)},
        finding=None if passed else (
            "the executed trace gradient contradicts the declared mathematics"),
    )


# --------------------------------------------------------------------------------------
# G4 -- parameter update
# --------------------------------------------------------------------------------------
def gate_g4(ctx: Context) -> dict[str, Any]:
    skip = _skipped("G4", ctx)
    if skip:
        return skip
    update = ctx.contract["update_contract"]
    T = ctx.trainer
    trace = is_trace_mechanism(ctx.contract)
    observations = (adapters.trace_case(T, ctx.contract) if trace
                    else ctx.observations())
    model = observations["model"]
    target = {id(p) for p in observations["encoder"]}

    if trace:
        adapters.apply_trace_gradient(T, observations)
    else:
        adapters.executed_target_gradient(T, ctx.contract, observations)
    clip = update.get("clip_norm")
    if clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip))
    params_before = [p.detach().clone() for p in model.parameters()]
    clipped = [None if p.grad is None else p.grad.detach().clone() for p in model.parameters()]

    lr = float(update.get("lr", 1e-3))
    weight_decay = float(update.get("weight_decay", 0.01))
    optimizer_name = update["optimizer"]
    optimizer_cls = getattr(torch.optim, optimizer_name, None)
    if optimizer_cls is None:
        raise C.ContractError(f"the declared optimizer {optimizer_name!r} does not exist")
    optimizer = optimizer_cls(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer.step()
    params_after = [p.detach().clone() for p in model.parameters()]

    oracle = C.resolve_reference(update["oracle"])
    expected = oracle({"optimizer": optimizer_name, "lr": lr, "weight_decay": weight_decay,
                       "params_before": params_before, "clipped_grads": clipped})

    worst_target, worst_other, moved_target = 0.0, 0.0, 0.0
    for parameter, before, after, want in zip(model.parameters(), params_before, params_after,
                                              expected):
        if parameter.grad is None:
            continue
        deviation = float((after - want).abs().max())
        moved = float((after - before).abs().max())
        if id(parameter) in target:
            worst_target = max(worst_target, deviation)
            moved_target = max(moved_target, moved)
        else:
            worst_other = max(worst_other, deviation)

    tolerance = float(update.get("atol", 1e-6))
    prediction_ok = worst_target <= tolerance and worst_other <= tolerance
    change_ok = (moved_target > 0.0) if update.get("requires_parameter_change", True) else True
    # The declared flags must be ANSWERED, not merely present: a declaration asking whether the
    # step reached the model is only honest if the answer is measured and reported.
    recorded_steps = {int(state["step"].item())
                      for state in optimizer.state.values() if "step" in state}
    reached_model = moved_target > 0.0
    state_recorded = bool(recorded_steps)
    passed = prediction_ok and change_ok
    if update.get("verify_step_reaches_model", False):
        passed = passed and reached_model
    if update.get("verify_optimizer_state", False):
        passed = passed and state_recorded
    return _result(
        "G4", C.PASSED if passed else C.FAILED,
        (f"the declared {optimizer_name} step (lr {lr}, weight decay {weight_decay}"
         + (f", clip {clip}" if clip else "") + f") moved the declared target parameters by "
         f"{moved_target:.3e}, within {max(worst_target, worst_other):.3e} of the independent "
         f"update oracle")
        if passed else
        (f"the declared optimizer did not produce the declared update: target deviation "
         f"{worst_target:.3e}, other-parameter deviation {worst_other:.3e}, target movement "
         f"{moved_target:.3e}"),
        {"target_max_abs_delta": moved_target,
         "target_update_deviation": worst_target,
         "other_update_deviation": worst_other,
         "optimizer": optimizer_name, "lr": lr, "weight_decay": weight_decay, "clip_norm": clip,
         "optimizer_steps_recorded": sorted(recorded_steps),
         "step_reached_model": reached_model,
         "optimizer_state_recorded": state_recorded,
         "declared_verifications": {
             "verify_step_reaches_model": bool(update.get("verify_step_reaches_model", False)),
             "verify_optimizer_state": bool(update.get("verify_optimizer_state", False))}},
        finding=None if passed else (
            "the declared optimizer and clipping do not produce the declared parameter update"),
    )


# --------------------------------------------------------------------------------------
# G5 -- distributed consistency
# --------------------------------------------------------------------------------------
def codebook_check_required(contract: dict[str, Any]) -> bool:
    """Whether the distributed check MUST cover the codebook's gradient.

    A contract whose declared gradient target is the codebook MUST have its codebook
    gradients checked: a mechanism that moves the codebook moves a gradient the encoder
    comparison cannot see. A missing measurement is a failure, never a silent pass.
    """
    distributed = contract.get("distributed_contract") or {}
    if distributed.get("verify_encoder_and_codebook_gradients", False):
        return True
    return str((contract.get("gradient_contract") or {}).get("target", "encoder")) == "codebook"


def compare_pooled_codebook(local_codebook: list[dict[str, list[torch.Tensor]]],
                            production_codebook: list[torch.Tensor],
                            aggregator, aggregation: str, weights) -> tuple[float, list[str]]:
    """The parent's INDEPENDENT pooling of the per-rank codebook gradients, and how far it is
    from what the distributed path produced.

    The local gradients were computed by the child WITHOUT DDP, on the rank's own batch. The
    pooling is done here, with the declared aggregation oracle, so the comparison is against a
    reconstruction the production path cannot influence.
    """
    names = sorted(local_codebook[0])
    pooled = {
        name: aggregator({"aggregation": aggregation,
                          "rank_grads": [local_codebook[rank][name]
                                         for rank in range(len(local_codebook))],
                          "weights": weights})
        for name in names
    }
    expected = oracles.codebook_reference(pooled)
    return oracles.max_abs_diff(production_codebook, expected), names


def gate_g5(ctx: Context) -> dict[str, Any]:
    skip = _skipped("G5", ctx)
    if skip:
        return skip
    distributed = ctx.contract["distributed_contract"]
    world_size = int(distributed.get("world_size", 2))
    aggregation = distributed.get("aggregation", "ddp_mean")
    rank_sizes = distributed.get("rank_batch_sizes")
    if rank_sizes and len(rank_sizes) != world_size:
        return _result("G5", C.INSUFFICIENT,
                       f"rank_batch_sizes lists {len(rank_sizes)} entries for "
                       f"{world_size} ranks")
    payload = CGV_DIR / f"g5_{ctx.tree.name}"
    payload.mkdir(parents=True, exist_ok=True)
    child = Path(__file__).with_name("cgv_ddp_child.py")
    proc = subprocess.run(
        [CGV_PY, str(child), str(ctx.tree), str(payload), str(world_size), str(CHILD_TIMEOUT_S)],
        input=json.dumps({"spec": ctx.contract, "rank_batch_sizes": rank_sizes}),
        capture_output=True, text=True, timeout=CHILD_TIMEOUT_S + 120)
    summary_path = payload / "summary.json"
    if not summary_path.is_file():
        return _result("G5", C.FAILED, "the multi-rank run produced no summary",
                       {"driver_rc": proc.returncode,
                        "stderr_tail": proc.stderr.strip()[-800:]},
                       finding="G5 could not execute the multi-rank path; that is a failure, not "
                               "a skip")
    summary = json.loads(summary_path.read_text())
    if not summary.get("ok"):
        return _result("G5", C.FAILED,
                       summary.get("error", "the multi-rank run failed"),
                       summary.get("measurements", {}),
                       finding="the declared multi-rank execution could not be completed")

    local = [torch.load(payload / f"rank_{rank}_local.pt", weights_only=False)
             for rank in range(world_size)]
    production = torch.load(payload / "production_pooled.pt", weights_only=False)
    oracle_inputs = json.loads((payload / "oracle_inputs.json").read_text())
    weights = summary.get("weights")
    if not is_trace_mechanism(ctx.contract):
        rule_name = oracle_inputs.get("rule")
        if not isinstance(rule_name, str) or not rule_name:
            # fail closed: the gate cannot reconstruct the declared assembly without knowing
            # which rule it is, and guessing one would verify the guess instead of the mechanism
            return _result("G5", C.FAILED,
                           "the declaration names no assembly rule (gradient_contract.rule), so "
                           "the multi-rank assembly cannot be reconstructed",
                           {"world_size": world_size, "aggregation": aggregation},
                           finding="an incomplete declaration is a blocking finding, not a pass")
        if rule_name not in oracles.RULES:
            return _result("G5", C.FAILED,
                           f"the declaration names a rule the oracles do not implement: {rule_name!r}",
                           {"world_size": world_size, "aggregation": aggregation,
                            "known_rules": list(oracles.RULES)},
                           finding="an unknown rule cannot be verified")

    aggregator = C.resolve_reference(distributed["oracle"])
    trace = is_trace_mechanism(ctx.contract)
    require_codebook = codebook_check_required(ctx.contract)
    extra: dict[str, Any] = {}

    if trace:
        # A trace mechanism has no projection step: the multi-rank question is whether DDP's own
        # averaging produced what the independent pooling of the per-rank LOCAL gradients says it
        # should. With declared rank sizes the ranks are not equally weighted, so pooling by a
        # plain mean would be the wrong reference -- and would hide exactly the bug the gate is for.
        local_encoder = [local[rank]["encoder"] for rank in range(world_size)]
        pooled_encoder = aggregator({"aggregation": aggregation, "rank_grads": local_encoder,
                                     "weights": weights})
        expected_pooled = {"encoder": pooled_encoder}
        deviation = oracles.max_abs_diff(production["encoder"], expected_pooled["encoder"])
        if distributed.get("verify_encoder_and_codebook_gradients", False) \
                and "codebooks" in production and local[0].get("codebooks"):
            pooled_codebooks = aggregator({"aggregation": aggregation,
                                           "rank_grads": [local[rank]["codebooks"]
                                                          for rank in range(world_size)],
                                           "weights": weights})
            expected_pooled["codebooks"] = pooled_codebooks
            codebook_deviation = oracles.max_abs_diff(production["codebooks"],
                                                      pooled_codebooks)
        else:
            codebook_deviation = float("nan")
    else:
        names = sorted(local[0])
        pooled_terms = {
            name: aggregator({"aggregation": aggregation,
                              "rank_grads": [local[rank][name] for rank in range(world_size)],
                              "weights": weights})
            for name in names
        }
        # the declared rule is applied to the POOLED gradients by the independent oracle, so the
        # production path is compared against a reconstruction it cannot influence
        rule_observations = {
            **oracle_inputs,
            "g_base": pooled_terms["base_loss"],
            "g_l1": pooled_terms.get("l1_loss", pooled_terms["base_loss"]),
            "g_other": pooled_terms.get("other_loss", pooled_terms["base_loss"]),
            "g_recon": pooled_terms.get("recon_loss", pooled_terms["base_loss"]),
            "g_quant": pooled_terms.get("quant_loss", pooled_terms["base_loss"]),
        }
        expected_pooled = oracles.assemble_gradient(rule_observations)
        deviation = oracles.max_abs_diff(production, expected_pooled)
        # The codebook's own gradient. A mechanism that targets the codebook moves a gradient
        # the encoder comparison cannot see, so the gate pools the per-rank codebook gradients
        # INDEPENDENTLY -- with the aggregation oracle, on the local gradients the child
        # computed without DDP -- and compares that pooling with what the distributed path
        # actually produced. Before the repair this was a NaN, which silently waived the check.
        codebook_deviation = float("nan")
        codebook_local_path = payload / "rank_0_codebook_local.pt"
        codebook_production_path = payload / "production_codebook_pooled.pt"
        if codebook_local_path.is_file() and codebook_production_path.is_file():
            local_codebook = [
                torch.load(payload / f"rank_{rank}_codebook_local.pt", weights_only=False)
                for rank in range(world_size)
            ]
            production_codebook = torch.load(codebook_production_path, weights_only=False)
            codebook_deviation, codebook_names = compare_pooled_codebook(
                local_codebook, production_codebook, aggregator, aggregation, weights)
            extra["codebook_terms"] = codebook_names
        elif require_codebook:
            return _result("G5", C.FAILED,
                           "the contract targets the codebook, but the multi-rank run produced no "
                           "codebook gradients to compare",
                           {"world_size": world_size, "aggregation": aggregation,
                            "pooled_deviation": deviation},
                           finding="the codebook half of the distributed check is missing, and a "
                                   "missing check is not a passed check")

    tolerance = float(distributed.get("atol", 1e-5))
    ranks_identical = bool(summary.get("ranks_identical"))
    passed = deviation <= tolerance and ranks_identical

    if require_codebook:
        if math.isnan(codebook_deviation):
            passed = False
            extra["codebook_pooled_deviation"] = codebook_deviation
            extra["codebook_check"] = "not measured: the multi-rank run produced no codebook gradients"
        else:
            passed = passed and codebook_deviation <= tolerance
            extra["codebook_pooled_deviation"] = codebook_deviation
    if distributed.get("verify_rank_parameter_identity", False):
        passed = passed and ranks_identical
    if rank_sizes:
        observed_rows = summary.get("local_batch_rows") or {}
        mismatched = {rank: observed_rows.get(str(rank))
                      for rank, size in enumerate(rank_sizes)
                      if observed_rows.get(str(rank)) not in (None, int(size) * 2)}
        extra["declared_rank_batch_sizes"] = rank_sizes
        extra["observed_rows"] = observed_rows
        if mismatched:
            passed = False
            extra["rank_size_mismatch"] = mismatched
    return _result(
        "G5", C.PASSED if passed else C.FAILED,
        (f"{world_size} real ranks with the declared {aggregation!r} aggregation: the production "
         f"pooled gradient matches the independent pooled reference within {deviation:.3e}, and "
         f"both ranks hold bitwise identical parameters after a step")
        if passed else
        (f"the multi-rank execution does not match the declared aggregation: deviation "
         f"{deviation:.3e} (tolerance {tolerance:.1e}), ranks identical={ranks_identical}"),
        {"world_size": world_size, "aggregation": aggregation,
         "pooled_deviation": deviation, "ranks_identical": ranks_identical,
         "local_batch_rows": summary.get("local_batch_rows"),
         "reduce_calls": summary.get("reduce_calls"),
         "elapsed_s": summary.get("elapsed_s"), **extra},
        finding=None if passed else
        "multi-rank execution disagrees with the aggregation rule the contract declares",
    )


# --------------------------------------------------------------------------------------
# G6 -- real training entry
# --------------------------------------------------------------------------------------
def gate_g6(ctx: Context, tmp: Path) -> dict[str, Any]:
    skip = _skipped("G6", ctx)
    if skip:
        return skip
    runtime = ctx.contract["runtime_contract"]
    update = ctx.contract["update_contract"]
    T = ctx.trainer

    counter: dict[str, int] = {"optimizer_steps": 0, "parameter_updates": 0,
                               "target_encoder_updates": 0}
    # The targeted parameters only exist once the trainer has built its model, so the model is
    # captured from its own construction. Observing "some parameter moved" instead would let an
    # optimizer that skipped the encoder entirely pass.
    captured_models: list[Any] = []
    model_class = getattr(ctx.trainer, "RQVAE", None)
    original_init = model_class.__init__ if model_class is not None else None

    def capturing_init(inner_self, *args, **kwargs):
        original_init(inner_self, *args, **kwargs)
        captured_models.append(inner_self)

    if model_class is not None:
        model_class.__init__ = capturing_init

    def target_parameters():
        if not captured_models:
            return []
        target = (ctx.contract.get("gradient_contract") or {}).get("target", "encoder")
        module = captured_models[-1]
        attribute = ((ctx.contract.get("model") or {}).get("parameter_groups") or {}).get(
            target, target)
        try:
            return list(adapters.resolve_dotted(module, attribute).parameters())
        except AttributeError:
            return []
    optimizer_cls = getattr(torch.optim, update["optimizer"])
    real_step = optimizer_cls.step

    def counting_step(self, *args, **kwargs):
        before = [(parameter, parameter.detach().clone())
                  for group in self.param_groups for parameter in group["params"]
                  if parameter.grad is not None]
        targets = target_parameters()
        target_before = [parameter.detach().clone() for parameter in targets]
        result = real_step(self, *args, **kwargs)
        counter["optimizer_steps"] += 1
        if any(not torch.equal(previous, parameter.detach())
               for parameter, previous in before):
            counter["parameter_updates"] += 1
        if targets and any(not torch.equal(old, parameter.detach())
                           for old, parameter in zip(target_before, targets)):
            counter["target_encoder_updates"] += 1
        return result

    optimizer_cls.step = counting_step

    observed: dict[str, int] = {}
    for item in runtime["observe"]:
        name = item["name"]
        # the optimizer-step observables are measured by the optimizer wrapper below, not by
        # instrumenting a trainer symbol
        if name in ("optimizer_steps", "parameter_updates", "target_encoder_update"):
            continue
        target = item.get("target")
        if not target:
            raise C.ContractError(f"observable {name!r} needs a target symbol to instrument")
        symbol = adapters.resolve_dotted(T, target)
        if not callable(symbol):
            raise C.ContractError(f"observable {name!r} target {target!r} is not callable")
        observed[name] = 0

        def wrapper(*args, __name=name, __original=symbol, **kwargs):
            observed[__name] += 1
            return __original(*args, **kwargs)

        _set_dotted(T, target, wrapper)

    saved: dict[str, Any] = {"EPOCHS": T.EPOCHS}
    T.EPOCHS = G6_EPOCHS
    for name, value in (runtime.get("force") or {}).items():
        saved[name] = getattr(T, name, None)
        setattr(T, name, value)
    reset = runtime.get("reset")
    if reset:
        state = getattr(T, reset, None)
        if isinstance(state, dict):
            state["buffers"] = None

    error = None
    started = time.time()
    try:
        with isolated_outputs(T, tmp):
            entrypoint = adapters.resolve_dotted(T, runtime["entrypoint"])
            try:
                entrypoint()
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
    finally:
        optimizer_cls.step = real_step
        T.EPOCHS = saved.pop("EPOCHS")
        for name, value in saved.items():
            if value is not None:
                setattr(T, name, value)

    counts = {**observed, "optimizer_steps": counter["optimizer_steps"],
              "parameter_updates": counter["parameter_updates"],
              "target_encoder_update": counter["target_encoder_updates"]}
    unmet = []
    for item in runtime["observe"]:
        minimum = int(item.get("min", 1))
        if counts.get(item["name"], 0) < minimum:
            unmet.append(f"{item['name']}: {counts.get(item['name'], 0)} < {minimum}")
    passed = error is None and not unmet
    produced = sorted(path.name for path in (tmp / "out").glob("*"))
    return _result(
        "G6", C.PASSED if passed else C.FAILED,
        (f"the real entry point {runtime['entrypoint']!r} ran in {time.time() - started:.1f}s and "
         f"satisfied every declared observable: "
         + ", ".join(f"{name}={count}" for name, count in sorted(counts.items())))
        if passed else
        (f"the real entry point did not satisfy the declared observables: {'; '.join(unmet)}"
         + (f"; entry raised {error}" if error else "")),
        {"counts": counts, "epochs_run": G6_EPOCHS, "isolated_outputs": produced,
         "elapsed_s": round(time.time() - started, 1), "error": error},
        finding=None if passed else
        "formal training does not execute the declared mechanism, or never moves a parameter",
    )


def _set_dotted(module, dotted: str, value) -> None:
    head, _, tail = dotted.partition(".")
    if not tail:
        setattr(module, head, value)
        return
    target = getattr(module, head)
    parts = tail.split(".")
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)


# --------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------
def run_cgv(tree: Path, spec: Any, *, gates: tuple[str, ...] = GATES) -> dict[str, Any]:
    """Verify the mechanism in ``tree`` against the declaration ``spec``.

    ``spec`` is the mechanism's own mathematical contract, passed as data: the seven groups in
    :mod:`contract`. Nothing is read from the tree except the mechanism's code, and the tree
    supplies no declaration of its own.
    """
    tree = Path(tree).resolve()
    trainer_path = tree / "train_rqvae.py"
    result: dict[str, Any] = {
        "arm": None,
        "tree": str(tree),
        "trainer": str(trainer_path),
        "trainer_sha256": sha256_file(trainer_path),
        "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "schema_version": C.SCHEMA_VERSION,
        "gates": {},
        "passed": False,
    }
    if result["trainer_sha256"] is None:
        result["refused"] = f"no train_rqvae.py in {tree}"
        for gate in GATES:
            result["gates"][gate] = _result(gate, C.INSUFFICIENT, result["refused"])
        result["summary"] = _summary(result, GATES)
        return result

    try:
        spec = C.normalise_spec(spec)
    except C.ContractError as exc:
        result["refused"] = str(exc)
        for gate in GATES:
            result["gates"][gate] = _result(gate, C.INSUFFICIENT, str(exc))
        result["summary"] = _summary(result, GATES)
        return result

    result["spec"] = spec
    result["spec_sha256"] = C.spec_sha256(spec)
    result["fingerprint"] = C.fingerprint(spec)
    result["declared"] = C.describe(spec)
    problems = C.validate(spec)
    if problems:
        result["refused"] = "the declaration is invalid: " + "; ".join(problems)
        for gate in GATES:
            result["gates"][gate] = _result(gate, C.INSUFFICIENT, result["refused"])
        result["summary"] = _summary(result, GATES)
        return result

    trainer, saved_path, saved_mods = load_trainer(tree)
    try:
        result["arm"] = getattr(trainer, "EXPERIMENT_ARM", None)
        ctx = load_context(tree, spec, trainer=trainer)
        result["independence"] = ctx.independence
        result["mechanism"] = (spec.get("mechanism") or {}).get("id")
        for gate in GATES:
            if gate not in gates:
                result["gates"][gate] = _result(gate, C.NOT_APPLICABLE, "not selected")
                result["gates"][gate]["passed"] = None
                result["gates"][gate]["skipped"] = True
                continue
            try:
                if gate == "G1":
                    result["gates"][gate] = gate_g1(ctx)
                elif gate == "G2":
                    result["gates"][gate] = gate_g2(ctx)
                elif gate == "G3":
                    result["gates"][gate] = gate_g3(ctx)
                elif gate == "G4":
                    result["gates"][gate] = gate_g4(ctx)
                elif gate == "G5":
                    result["gates"][gate] = gate_g5(ctx)
                elif gate == "G6":
                    with tempfile.TemporaryDirectory() as raw:
                        result["gates"][gate] = gate_g6(ctx, Path(raw))
            except Exception as exc:  # noqa: BLE001
                result["gates"][gate] = _result(
                    gate, C.FAILED, f"the gate could not be executed: {type(exc).__name__}: {exc}",
                    finding="a gate that cannot run verifies nothing")
    finally:
        unload_trainer(saved_path, saved_mods)

    result["summary"] = _summary(result, gates)
    states = [result["gates"][gate].get("state") for gate in gates]
    result["passed"] = all(state in (C.PASSED, C.NOT_APPLICABLE) for state in states) and \
        any(state == C.PASSED for state in states)
    result["gates_passed"] = result["passed"]
    return result


def _summary(result: dict[str, Any], gates: tuple[str, ...]) -> dict[str, Any]:
    return {
        gate: {
            "state": result["gates"][gate].get("state"),
            "passed": result["gates"][gate].get("passed"),
            "title": GATE_TITLES[gate],
            "prevents": GATE_PREVENTS[gate],
        }
        for gate in gates
    }


# --------------------------------------------------------------------------------------
# G3 for the R7-F soft same-code loss (issue #89)
# --------------------------------------------------------------------------------------
def gate_g3_r7_f(ctx: Context) -> dict[str, Any]:
    """Compare the mechanism's applied gradient with an INDEPENDENT analytic VJP.

    The mechanism ships its own backward. This gate recomputes dL/d(distance) by hand from the
    softmax Jacobian inside the verifier and compares. It also checks that the partner side of
    each pair contributes, because dropping it zeroes the partner's gradient while leaving an
    aggregate loss that still looks healthy.
    """
    gradient = ctx.contract["gradient_contract"]
    T = ctx.trainer
    tau = float(os.environ.get("R7_F_TAU", "0.001"))
    lam = float((ctx.contract.get("mechanism") or {}).get("mathematics", {}).get("lambda", 0.1))
    probe_seeds = tuple(gradient.get("probe_seeds") or (1, 4, 5, 12))
    model, _ = adapters.build_model(T, ctx.contract)
    layer = model.rq.vq_layers[0]
    embedding = layer.get_code_embs()

    failures: list[str] = []
    deviations: list[float] = []
    for seed in probe_seeds:
        generator = torch.Generator().manual_seed(int(seed))
        distances = torch.rand(8, int(layer.n_embed), generator=generator,
                               dtype=torch.float64, device=embedding.device) * 0.5
        pairs = torch.tensor([[0, 1], [2, 3], [4, 5]], device=embedding.device)
        analytic = oracles.r7_f_loss_vjp(distances, pairs, tau, lam)
        with torch.no_grad():
            numeric = torch.zeros_like(analytic)
            step = 1e-6
            flat = distances.reshape(-1)
            for index in range(flat.numel()):
                plus = flat.clone()
                minus = flat.clone()
                plus[index] += step
                minus[index] -= step
                numeric.reshape(-1)[index] = (
                    oracles.r7_f_forward_loss(plus.reshape(distances.shape), pairs, tau, lam)
                    - oracles.r7_f_forward_loss(minus.reshape(distances.shape), pairs, tau, lam)
                ) / (2 * step)
        deviation = float((analytic - numeric).abs().max())
        deviations.append(deviation)
        if deviation > float(gradient.get("atol", 1e-5)):
            failures.append(f"seed {seed}: analytic VJP deviates from central differences by "
                            f"{deviation:.3e}")
        # the partner side must be nonzero for every pair
        for left, right in pairs.tolist():
            if float(analytic[left].abs().max()) == 0.0:
                failures.append(f"seed {seed}: the source side of pair ({left},{right}) is zero")
            if float(analytic[right].abs().max()) == 0.0:
                failures.append(f"seed {seed}: the PARTNER side of pair ({left},{right}) is zero")
                break
    return _result(
        "G3", C.PASSED if not failures else C.FAILED,
        (f"the independent analytic VJP agrees with central differences on "
         f"{len(probe_seeds)} seeds, and both pair sides carry gradient")
        if not failures else "; ".join(failures[:4]),
        {"worst_deviation": max(deviations) if deviations else None,
         "seeds": list(probe_seeds), "tau": tau, "lambda": lam},
        finding=None if not failures else
        "the applied separation gradient does not match an independently derived VJP",
    )
