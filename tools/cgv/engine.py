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
CGV_PY = "/home/wlia0047/ar57/scratch/wenyu/genrec_env_v2/bin/python3.9"
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
    """Measure the real Jacobian of the declared residual recurrence against the declared rule.

    The probe is a Jacobian-vector product through the REAL quantizer stack: with the declared
    rule ``identity``, ``d r_(k+1) / d r_k = I`` means ``J v == v`` for every probe ``v``.
    """
    model = T.RQVAE(T._tokenizer_config(), in_dim=adapters.IN_DIM)
    embeddings = torch.randn(8, adapters.IN_DIM,
                             generator=torch.Generator().manual_seed(4242))
    residual = model.encoder(embeddings).detach().requires_grad_(True)
    layer = model.rq.vq_layers[int(jacobian.get("level", 0))]
    quant, _, _, _ = layer(residual)
    following = residual - quant
    probe = torch.randn_like(residual)
    (grad,) = torch.autograd.grad((following * probe).sum(), residual, retain_graph=False)
    return float((grad - probe).abs().max())


def gate_g1(ctx: Context) -> dict[str, Any]:
    skip = _skipped("G1", ctx)
    if skip:
        return skip
    forward = ctx.contract["forward_contract"]
    observations = ctx.observations()
    measurements: dict[str, Any] = {}

    jacobian = forward.get("jacobian")
    if jacobian:
        deviation = _jacobian_deviation(ctx.trainer, jacobian)
        observations["jacobian_identity_deviation"] = deviation
        measurements["jacobian_identity_deviation"] = deviation

    checks = oracles.check_forward_invariants(observations, forward.get("invariants", []))

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


# --------------------------------------------------------------------------------------
# G4 -- parameter update
# --------------------------------------------------------------------------------------
def gate_g4(ctx: Context) -> dict[str, Any]:
    skip = _skipped("G4", ctx)
    if skip:
        return skip
    update = ctx.contract["update_contract"]
    T = ctx.trainer
    observations = ctx.observations()
    model = observations["model"]
    target = {id(p) for p in observations["encoder"]}

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
    passed = prediction_ok and change_ok
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
         "optimizer": optimizer_name, "lr": lr, "weight_decay": weight_decay, "clip_norm": clip},
        finding=None if passed else (
            "the declared optimizer and clipping do not produce the declared parameter update"),
    )


# --------------------------------------------------------------------------------------
# G5 -- distributed consistency
# --------------------------------------------------------------------------------------
def gate_g5(ctx: Context) -> dict[str, Any]:
    skip = _skipped("G5", ctx)
    if skip:
        return skip
    distributed = ctx.contract["distributed_contract"]
    world_size = int(distributed.get("world_size", 2))
    aggregation = distributed.get("aggregation", "ddp_mean")
    payload = CGV_DIR / f"g5_{ctx.tree.name}"
    payload.mkdir(parents=True, exist_ok=True)
    child = Path(__file__).with_name("cgv_ddp_child.py")
    proc = subprocess.run(
        [CGV_PY, str(child), str(ctx.tree), str(payload), str(world_size), str(CHILD_TIMEOUT_S)],
        input=C.canonical_json(ctx.contract), capture_output=True, text=True,
        timeout=CHILD_TIMEOUT_S + 120)
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

    aggregator = C.resolve_reference(distributed["oracle"])
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

    deviation = oracles.max_abs_deviation(production, expected_pooled)
    tolerance = float(distributed.get("atol", 1e-5))
    ranks_identical = bool(summary.get("ranks_identical"))
    passed = deviation <= tolerance and ranks_identical
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
         "elapsed_s": summary.get("elapsed_s")},
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

    counter: dict[str, int] = {"optimizer_steps": 0, "parameter_updates": 0}
    optimizer_cls = getattr(torch.optim, update["optimizer"])
    real_step = optimizer_cls.step

    def counting_step(self, *args, **kwargs):
        before = [(parameter, parameter.detach().clone())
                  for group in self.param_groups for parameter in group["params"]
                  if parameter.grad is not None]
        result = real_step(self, *args, **kwargs)
        counter["optimizer_steps"] += 1
        if any(not torch.equal(previous, parameter.detach())
               for parameter, previous in before):
            counter["parameter_updates"] += 1
        return result

    optimizer_cls.step = counting_step

    observed: dict[str, int] = {}
    for item in runtime["observe"]:
        name = item["name"]
        if name in ("optimizer_steps", "parameter_updates"):
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
              "parameter_updates": counter["parameter_updates"]}
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
def run_cgv(tree: Path, spec: Any, *, gates: tuple[str, ...] = GATES,
            negative_suite: bool = False) -> dict[str, Any]:
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
                    finding="a gate that cannot run cannot certify anything")
    finally:
        unload_trainer(saved_path, saved_mods)

    result["summary"] = _summary(result, gates)
    states = [result["gates"][gate].get("state") for gate in gates]
    result["passed"] = all(state in (C.PASSED, C.NOT_APPLICABLE) for state in states) and \
        any(state == C.PASSED for state in states)
    result["gates_passed"] = result["passed"]
    if negative_suite:
        # prove the gates have discriminating power: inject the standard faults and require the
        # gate that is supposed to catch each one to fail
        import negative_suite as suite

        result["negative_suite"] = suite.run_all(tree, spec)
        # gates alone are not a verdict: a mechanism whose faults are not caught is not verified
        result["passed"] = bool(result["passed"] and result["negative_suite"]["all_ok"])
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


def write_certificate(result: dict[str, Any]) -> Path:
    CGV_DIR.mkdir(parents=True, exist_ok=True)
    mechanism = (result.get("declared") or {}).get("id") or result.get("arm") or "unknown"
    path = CGV_DIR / f"certificate_{mechanism}.json"
    path.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def load_certificate(tree: Path) -> dict[str, Any] | None:
    """The certificate for this tree, matched by contract path then by tree path."""
    tree = Path(tree).resolve()
    if not CGV_DIR.is_dir():
        return None
    candidates = sorted(CGV_DIR.glob("certificate_*.json"),
                        key=lambda path: path.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            certificate = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if Path(str(certificate.get("tree", ""))).resolve() == tree:
            return certificate
    return None


def certificate_is_valid(tree: Path, spec: Any = None) -> tuple[bool, str]:
    """A certificate is valid only if it passed, and still describes this tree and this claim.

    Four things can invalidate it, and all four must be checked or the gate is decorative: the
    trainer source changed, the DECLARATION the certificate was issued against is not the one
    supplied now, the verifier itself changed, or the certificate predates the current schema.
    Requiring the caller to pass the same declaration at training time is what stops a weak claim
    from being certified and then quietly used.
    """
    tree = Path(tree).resolve()
    certificate = load_certificate(tree)
    if certificate is None:
        return False, ("no CGV certificate. Run cgv_check(source_dir=...) and all six gates must "
                       "pass before this tree may train")
    if certificate.get("schema_version") != C.SCHEMA_VERSION:
        return False, (f"the certificate was issued under schema "
                       f"{certificate.get('schema_version')} and the current contract schema is "
                       f"{C.SCHEMA_VERSION}; re-run cgv_check")
    summary = certificate.get("summary") or {}
    blocking = [gate for gate, entry in summary.items()
                if entry.get("state") not in (C.PASSED, C.NOT_APPLICABLE)]
    if blocking:
        states = {gate: entry.get("state") for gate, entry in summary.items()}
        return False, (f"CGV gates not passing: {states}. Re-run cgv_check before training")
    independence = certificate.get("independence") or {}
    if independence and not independence.get("independent", False):
        return False, ("the certificate was issued with oracles that are not independent of the "
                       "mechanism; re-run cgv_check")
    suite = certificate.get("negative_suite")
    if not suite:
        return False, ("the certificate carries no counterexample evidence: run cgv_check, which "
                       "injects the standard faults and requires the matching gate to fail")
    if not suite.get("all_ok"):
        broken = [item.get("fault") for item in suite.get("faults", []) if not item.get("ok")]
        return False, (f"the counterexample suite could not show these faults failing: {broken}. "
                       "A gate that cannot fail proves nothing, so this mechanism may not train")
    if not certificate.get("passed"):
        return False, ("the certificate is not marked as passing even though its gates and "
                       "counterexample suite are green; re-run cgv_check")
    current_trainer = sha256_file(tree / "train_rqvae.py")
    if certificate.get("trainer_sha256") != current_trainer:
        return False, ("the trainer changed after the certificate was issued; the mechanism must "
                       "be re-verified with cgv_check before it may train")
    if spec is None:
        return False, ("no declaration was supplied: training must be authorised by the same "
                       "declaration the certificate was issued against")
    supplied = C.normalise_spec(spec)
    if C.spec_sha256(supplied) != certificate.get("spec_sha256"):
        return False, ("the supplied declaration is not the one this certificate covers "
                       f"(certificate {str(certificate.get('spec_sha256'))[:16]}, supplied "
                       f"{C.spec_sha256(supplied)[:16]}); the gates that passed judged that "
                       "declaration, so this one is unverified")
    current_fingerprint = C.fingerprint(supplied)
    recorded = certificate.get("fingerprint") or {}
    for key in ("oracles_sha256", "adapter_sha256", "negative_suite_sha256"):
        if recorded.get(key) != current_fingerprint.get(key):
            return False, (f"the verifier changed ({key}) after the certificate was issued; "
                           "re-run cgv_check")
    return True, (f"CGV certificate {certificate.get('checked_utc')} covers trainer "
                  f"{str(certificate.get('trainer_sha256'))[:16]} and declaration "
                  f"{str(certificate.get('spec_sha256'))[:16]}")
