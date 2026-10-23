"""CGV mechanism contracts -- what a mechanism declares about itself.

CGV verifies a mechanism against the mathematical behaviour the mechanism DECLARES, never
against another version of the mechanism. There is no parent tree, no baseline checkpoint and no
required on/off switch: a contract that needs one can declare one, and a contract that has none
is verified exactly the same way.

The declaration is an ARGUMENT. It arrives in the call, as data, and CGV reads no declaration
from the mechanism's tree: a mechanism hands over no files at all. That keeps the claim in the
caller's hands where it can be read, diffed and challenged, instead of buried in the code whose
correctness is in question.

The three roles are kept strictly apart:

    contract    what the mechanism claims (this module)
    adapter     how to execute the real mechanism (``adapters.py``)
    oracle      an INDEPENDENT mathematical reference (``oracles.py``)

An oracle must be able to compute the expected value from the declared mathematics alone. Every
oracle lives in the verifier's own ``oracles.py``, never in the mechanism's tree: if an oracle
could be resolved from the mechanism's own code, a wrong implementation would be compared against
the same wrong implementation and would pass. :func:`independence_report` enforces that, and the
certificate records the result.

A mechanism with no contract, or with a contract that leaves a gate without any testable
statement, is NOT passed by default. It is reported as ``insufficient_evidence``.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
import warnings
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

CGV_DIR = Path(__file__).resolve().parent
REPO_ROOT = CGV_DIR.parent.parent

# --------------------------------------------------------------------------------------
# the six gates
# --------------------------------------------------------------------------------------
GATES = ("G1", "G2", "G3", "G4", "G5", "G6")

GATE_TITLES = {
    "G1": "forward contract",
    "G2": "gradient propagation",
    "G3": "gradient mathematics",
    "G4": "parameter update",
    "G5": "distributed consistency",
    "G6": "real training entry",
}

GATE_PREVENTS = {
    "G1": "a forward that violates the declared formula, invariants or output constraints",
    "G2": "a declared loss path that does not reach the declared parameters, or a non-finite one",
    "G3": "an executed gradient that contradicts the declared gradient rule",
    "G4": "an update that the declared optimizer and clipping do not produce",
    "G5": "an aggregation rule that the multi-rank execution does not satisfy",
    "G6": "a mechanism that formal training never executes, or that never moves a parameter",
}

# gate states
PASSED = "passed"
FAILED = "failed"
NOT_APPLICABLE = "not_applicable"
INSUFFICIENT = "insufficient_evidence"

PATH_EXPECTATIONS = ("nonzero", "zero", "finite")
GRADIENT_MODES = ("formula", "invariants", "no_gradient_effect")
MECHANISM_TYPES = (
    "gradient_routing",
    "forward_transform",
    "loss_term",
    "sampling",
    "initialization",
    "quantization",
)
AGGREGATIONS = ("ddp_mean", "ddp_sum", "none")
FORWARD_EFFECTS = ("none", "changes_forward")

# Strict key sets. A declaration is a CLAIM, and a claim whose parts nobody reads is worse than
# no claim: a key nobody checks is indistinguishable from a check nobody performed. So every key
# must be either understood and acted on, or absent.
SCHEMA_KEYS = {
    "mechanism": {"id", "type", "summary", "entrypoints", "executor", "activation",
                  "mathematics", "inputs", "executor_style", "host_routing"},
    "model": {"adapter", "factory", "tokenizer_config", "forward", "parameter_groups", "probe",
              "role_inputs"},
    "forward_contract": {"observables", "declared_effect", "invariants", "oracle", "atol"},
    "gradient_contract": {
        "target", "mode", "rule", "reference_term", "oracle", "probe_weight", "probe_seeds",
        "atol", "rtol", "eps_flag", "alpha", "alpha_flag", "cos_threshold", "cos_threshold_flag",
        "blocks", "ema", "loss_terms", "expected_paths", "additional_targets",
        "jacobian_expectations", "effect", "invariants", "loss_term_callables",
    },
    "update_contract": {
        "optimizer", "lr", "weight_decay", "clip_norm", "atol", "requires_parameter_change",
        "oracle", "verify_step_reaches_model", "verify_optimizer_state",
    },
    "distributed_contract": {
        "world_size", "aggregation", "rank_batch_sizes", "atol", "oracle",
        "verify_encoder_and_codebook_gradients", "verify_rank_parameter_identity",
    },
    "runtime_contract": {
        "entrypoint", "isolation", "force", "reset", "observe", "train_entry_hash_required",
    },
}
OBSERVABLE_NAMES = ("mechanism_calls", "optimizer_steps", "parameter_updates",
                    "target_encoder_update")


class ContractError(Exception):
    """A contract that cannot be loaded, resolved or trusted."""


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------
def normalise_spec(spec: Any, *, source: str = "the declaration") -> dict[str, Any]:
    """Accept the declaration as data and validate its shape.

    The only accepted form is a mapping, which is what the ``spec`` parameter of the MCP tools
    and the CLI carries. A path or a module name is rejected on purpose: a declaration that is
    loaded from disk would be a file the mechanism tree controls.
    """
    if isinstance(spec, (str, bytes)):
        if isinstance(spec, bytes):
            try:
                spec = spec.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ContractError(f"{source} must be UTF-8 JSON") from exc
        text = spec.strip()
        if text.startswith("{") or text.startswith("["):
            try:
                spec = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ContractError(f"{source} is not valid JSON: {exc}") from exc
        else:
            raise ContractError(
                f"{source} must be the declaration itself, as an object or a JSON string; "
                f"got the string {text[:60]!r}. CGV does not load a declaration from the "
                "mechanism's tree.")
    if not isinstance(spec, dict):
        raise ContractError(f"{source} must be a mapping of the seven contract groups")
    return dict(spec)


def _load_shared(name: str) -> types.ModuleType:
    if name not in sys.modules:
        path = CGV_DIR / f"{name}.py"
        if not path.is_file():
            raise ContractError(f"the verifier has no {name} module")
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ContractError(f"cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        saved = list(sys.path)
        sys.path.insert(0, str(CGV_DIR))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                spec.loader.exec_module(module)
        finally:
            sys.path[:] = saved
    return sys.modules[name]


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(spec: dict[str, Any]) -> str:
    """The declaration in a canonical form, so its hash depends on content and not on layout."""
    return json.dumps(spec, indent=2, sort_keys=True, default=str)


def spec_sha256(spec: dict[str, Any]) -> str:
    """The identity of a declaration. This is what a certificate binds to."""
    return hashlib.sha256(canonical_json(spec).encode("utf-8")).hexdigest()


def code_tree_sha256(tree: Path) -> tuple[str | None, int]:
    """A hash over the mechanism's WHOLE active source tree, not just its entry point.

    A mechanism lives partly outside its ``train_rqvae.py`` -- the quantiser it rewrites the
    gradient of lives in ``model/layers.py``. Binding only the entry point would let a change
    there alter exactly what the certificate attests to while leaving the certificate valid.
    Every file the training entry can execute is therefore included, and only the directories
    that hold run OUTPUT are excluded, since those change on every run by design.
    """
    tree = Path(tree)
    if not tree.is_dir():
        return None, 0
    skip_dirs = {"__pycache__", "logs", "results", ".git"}
    digest = hashlib.sha256()
    count = 0
    for path in sorted(tree.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(tree)
        if any(part in skip_dirs for part in relative.parts):
            continue
        if path.suffix in (".pyc", ".pyo"):
            continue
        digest.update(str(relative).encode("utf-8"))
        digest.update(_sha256(path).encode("ascii"))
        count += 1
    return digest.hexdigest(), count


def train_config_sha256(trainer) -> str | None:
    """A hash of the training constants a certificate depends on.

    The gates reason about a declared optimiser, clipping and batch shape. If the trainer's real
    constants stopped matching that declaration, every gate would be describing a run that does
    not exist, so the constants are bound to the certificate as well.
    """
    names = (
        "EPOCHS", "LR", "WEIGHT_DECAY", "OPTIMIZER", "GRADIENT_CLIP_NORM", "BATCH_SIZE",
        "PAIR_BATCH_SIZE", "CODEBOOK_NUM", "CODEBOOK_SIZE", "CODEBOOK_DIM", "SEED",
        "BEHAVIOR_TEMPERATURE", "BEHAVIOR_WEIGHT_MAX", "RAMP_START_EPOCH", "WARMUP_EPOCHS",
        "NUM_GPUS", "ACTIVE_LEVELS", "DIRECTIONAL_MODE",
    )
    payload = {}
    for name in names:
        if hasattr(trainer, name):
            value = getattr(trainer, name)
            if isinstance(value, (str, int, float, bool, list, tuple)):
                payload[name] = value if not isinstance(value, tuple) else list(value)
    if not payload:
        return None
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str)
                          .encode("utf-8")).hexdigest()


def fingerprint(spec: dict[str, Any]) -> dict[str, Any]:
    """Everything a certificate binds to: the declaration and the verifier that judged it."""
    return {
        "schema_version": SCHEMA_VERSION,
        "spec_sha256": spec_sha256(spec),
        "oracles_sha256": _sha256(CGV_DIR / "oracles.py"),
        "adapter_sha256": _sha256(CGV_DIR / "adapters.py"),
    }


# --------------------------------------------------------------------------------------
# reference resolution
# --------------------------------------------------------------------------------------
def resolve_reference(reference: str) -> Any:
    """Resolve ``"module:attribute"`` to a live object inside the verifier.

    Only ``oracles`` and ``adapters`` are reachable, both under ``tools/cgv``. The mechanism's
    own module is not, and neither is any module the mechanism supplies: an oracle that lives with
    the code it judges could repeat the mechanism's mistake, which is the one thing an oracle must
    never do.
    """
    if ":" not in reference:
        raise ContractError(f"reference {reference!r} must be written as 'module:attribute'")
    module_name, attribute = reference.split(":", 1)
    if module_name not in ("oracles", "adapters"):
        raise ContractError(
            f"reference {reference!r} must use 'oracles:' or 'adapters:': the verifier only "
            "resolves its own modules, never anything the mechanism supplies")
    module = _load_shared(module_name)
    if not hasattr(module, attribute):
        raise ContractError(f"{reference!r}: {module_name} defines no {attribute!r}")
    return getattr(module, attribute)


# --------------------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------------------
def _require(mapping: Any, key: str, where: str, problems: list[str]) -> Any:
    if not isinstance(mapping, dict):
        problems.append(f"{where} must be a mapping")
        return None
    if key not in mapping:
        problems.append(f"{where}.{key} is required")
        return None
    return mapping[key]


def validate(contract: dict[str, Any]) -> list[str]:
    """Structural validation. Returns a list of problems; empty means the contract is usable.

    Unknown keys are problems, not trivia. A declaration that carries a check nobody performs
    reads exactly like a declaration whose checks all passed, which is the failure mode this
    whole tool exists to prevent.
    """
    problems: list[str] = []
    version = contract.get("schema_version")
    if version != SCHEMA_VERSION:
        problems.append(f"schema_version must be {SCHEMA_VERSION}, found {version!r}")
    for group, allowed in SCHEMA_KEYS.items():
        section = contract.get(group)
        if isinstance(section, dict):
            unknown = sorted(set(section) - allowed)
            if unknown:
                problems.append(
                    f"{group} declares keys CGV does not act on: {unknown}. Either the "
                    "declaration means something CGV cannot check, or the key is a typo; both "
                    "must be resolved before this mechanism may be verified")

    mechanism = _require(contract, "mechanism", "contract", problems)
    if mechanism is not None:
        _require(mechanism, "id", "mechanism", problems)
        kind = _require(mechanism, "type", "mechanism", problems)
        if kind is not None and kind not in MECHANISM_TYPES:
            problems.append(f"mechanism.type {kind!r} not in {MECHANISM_TYPES}")
        entry = mechanism.get("entrypoints", [])
        if not isinstance(entry, list):
            problems.append("mechanism.entrypoints must be a list of trainer symbol names")
        activation = mechanism.get("activation")
        if activation is not None and not isinstance(activation, dict):
            problems.append("mechanism.activation must be a mapping when present")

    model = _require(contract, "model", "contract", problems)
    if model is not None:
        adapter = _require(model, "adapter", "model", problems)
        if adapter is not None and adapter != "rqvae":
            problems.append(f"model.adapter {adapter!r} is not a known adapter")
        groups = _require(model, "parameter_groups", "model", problems)
        if groups is not None and not isinstance(groups, dict):
            problems.append("model.parameter_groups must map a group name to an attribute path")

    forward = _require(contract, "forward_contract", "contract", problems)
    if forward is not None:
        observables = _require(forward, "observables", "forward_contract", problems)
        if observables is not None and (not isinstance(observables, list) or not observables):
            problems.append("forward_contract.observables must be a non-empty list")
        invariants = forward.get("invariants", [])
        oracle = forward.get("oracle")
        if not invariants and not oracle:
            problems.append(
                "forward_contract declares neither invariants nor an oracle: the forward cannot "
                "be verified against anything")
        effect = forward.get("declared_effect")
        if effect is not None and effect not in FORWARD_EFFECTS:
            problems.append(f"forward_contract.declared_effect {effect!r} not in {FORWARD_EFFECTS}")

    gradient = _require(contract, "gradient_contract", "contract", problems)
    if gradient is not None:
        mode = _require(gradient, "mode", "gradient_contract", problems)
        if mode is not None and mode not in GRADIENT_MODES:
            problems.append(f"gradient_contract.mode {mode!r} not in {GRADIENT_MODES}")
        if mode in ("formula", "invariants"):
            _require(gradient, "target", "gradient_contract", problems)
            terms = _require(gradient, "loss_terms", "gradient_contract", problems)
            if terms is not None and not isinstance(terms, dict):
                problems.append("gradient_contract.loss_terms must map a term name to an adapter "
                                "source")
            paths = _require(gradient, "expected_paths", "gradient_contract", problems)
            if isinstance(paths, dict):
                for name, expectation in paths.items():
                    if expectation not in PATH_EXPECTATIONS:
                        problems.append(
                            f"gradient_contract.expected_paths[{name!r}] = {expectation!r} not in "
                            f"{PATH_EXPECTATIONS}")
            elif paths is not None:
                problems.append("gradient_contract.expected_paths must be a mapping")
            if mode == "formula" and not gradient.get("oracle"):
                problems.append(
                    "gradient_contract.mode 'formula' requires an independent oracle; without one "
                    "the executed gradient cannot be checked against the declared rule")
            if mode == "invariants" and not gradient.get("invariants"):
                problems.append("gradient_contract.mode 'invariants' requires an invariants list")
        if mode == "no_gradient_effect":
            reason = gradient.get("reason")
            if not reason:
                problems.append(
                    "gradient_contract.mode 'no_gradient_effect' must state a reason, so that the "
                    "missing gradient check is an explicit claim instead of an omission")

    update = _require(contract, "update_contract", "contract", problems)
    if update is not None:
        _require(update, "optimizer", "update_contract", problems)
        if not update.get("oracle"):
            problems.append("update_contract requires an independent oracle")

    distributed = _require(contract, "distributed_contract", "contract", problems)
    if distributed is not None:
        world = _require(distributed, "world_size", "distributed_contract", problems)
        if world is not None and (not isinstance(world, int) or world < 1):
            problems.append("distributed_contract.world_size must be a positive integer")
        aggregation = distributed.get("aggregation")
        if aggregation is not None and aggregation not in AGGREGATIONS:
            problems.append(f"distributed_contract.aggregation {aggregation!r} not in "
                            f"{AGGREGATIONS}")
        if aggregation != "none" and not distributed.get("oracle"):
            problems.append(
                "distributed_contract requires an independent oracle unless aggregation is 'none'")

    runtime = _require(contract, "runtime_contract", "contract", problems)
    if runtime is not None:
        _require(runtime, "entrypoint", "runtime_contract", problems)
        observe = _require(runtime, "observe", "runtime_contract", problems)
        if observe is not None and (not isinstance(observe, list) or not observe):
            problems.append("runtime_contract.observe must be a non-empty list of observables")
        elif isinstance(observe, list):
            for item in observe:
                if not isinstance(item, dict) or "name" not in item:
                    problems.append("each runtime_contract.observe entry needs a name")
                    continue
                if item["name"] not in OBSERVABLE_NAMES:
                    problems.append(
                        f"runtime_contract.observe names {item['name']!r}, which CGV does not "
                        f"measure; known observables are {list(OBSERVABLE_NAMES)}")
                if item["name"] not in ("optimizer_steps", "parameter_updates",
                                        "target_encoder_update") \
                        and not item.get("target"):
                    problems.append(
                        f"observable {item['name']!r} needs a 'target' naming the trainer symbol "
                        "to instrument")
    return problems


# --------------------------------------------------------------------------------------
# oracle independence
# --------------------------------------------------------------------------------------
def independence_report(spec: dict[str, Any]) -> dict[str, Any]:
    """Prove that no oracle is the mechanism itself.

    Two checks, both able to fail loudly:

    1. every oracle reference resolves into the verifier's own ``oracles`` module, never into
       anything the mechanism supplies;
    2. no oracle is one of the mechanism's own declared entrypoints.
    """
    problems: list[str] = []
    oracle_refs: list[str] = []
    for group in ("forward_contract", "gradient_contract", "update_contract",
                  "distributed_contract"):
        section = spec.get(group) or {}
        if isinstance(section, dict) and section.get("oracle"):
            oracle_refs.append(str(section["oracle"]))

    for reference in oracle_refs:
        module_name = reference.split(":", 1)[0] if ":" in reference else ""
        if module_name != "oracles":
            problems.append(
                f"oracle {reference!r} is not an independent oracle from the verifier's own "
                "oracles module")

    entrypoints = {name.split(".")[-1]
                   for name in (spec.get("mechanism") or {}).get("entrypoints", []) or []}
    for reference in oracle_refs:
        attribute = reference.split(":", 1)[1] if ":" in reference else ""
        if attribute in entrypoints:
            problems.append(
                f"oracle {reference!r} is one of the mechanism's own entrypoints: a mechanism may "
                "not be its own oracle")

    return {"independent": not problems, "oracles": sorted(set(oracle_refs)),
            "problems": problems}


def gate_state(gate: str, contract: dict[str, Any]) -> tuple[str, str]:
    """What a gate can do with this contract, before any execution.

    Returns ``(state, reason)`` where state is ``PASSED`` when the gate has something testable to
    check. ``NOT_APPLICABLE`` is an explicit, declared statement; ``INSUFFICIENT`` means the
    contract leaves the gate with nothing to verify, which never counts as success.
    """
    if gate == "G1":
        forward = contract.get("forward_contract") or {}
        if forward.get("invariants") or forward.get("oracle"):
            return PASSED, ""
        return INSUFFICIENT, "the contract declares no forward invariant and no forward oracle"
    if gate == "G2":
        gradient = contract.get("gradient_contract") or {}
        if gradient.get("mode") == "no_gradient_effect":
            return NOT_APPLICABLE, (
                "the contract declares the mechanism has no gradient effect: "
                f"{gradient.get('reason', 'no reason given')}")
        if gradient.get("loss_terms") and gradient.get("expected_paths"):
            return PASSED, ""
        return INSUFFICIENT, (
            "the contract declares neither loss_terms nor expected_paths, so no propagation can "
            "be checked")
    if gate == "G3":
        gradient = contract.get("gradient_contract") or {}
        mode = gradient.get("mode")
        if mode == "no_gradient_effect":
            return NOT_APPLICABLE, (
                "the contract declares the mechanism has no gradient effect: "
                f"{gradient.get('reason', 'no reason given')}")
        if mode == "formula" and gradient.get("oracle"):
            return PASSED, ""
        if mode == "invariants" and gradient.get("invariants"):
            return PASSED, ""
        return INSUFFICIENT, (
            f"gradient_contract.mode {mode!r} provides no independent oracle and no invariants")
    if gate == "G4":
        update = contract.get("update_contract") or {}
        if update.get("oracle"):
            return PASSED, ""
        return INSUFFICIENT, "update_contract declares no independent update oracle"
    if gate == "G5":
        distributed = contract.get("distributed_contract") or {}
        if distributed.get("aggregation") == "none":
            return NOT_APPLICABLE, "the contract declares aggregation 'none'"
        if distributed.get("oracle"):
            return PASSED, ""
        return INSUFFICIENT, "distributed_contract declares no independent aggregation oracle"
    if gate == "G6":
        runtime = contract.get("runtime_contract") or {}
        if runtime.get("entrypoint") and runtime.get("observe"):
            return PASSED, ""
        return INSUFFICIENT, (
            "runtime_contract declares no entrypoint or no observables, so the real training path "
            "cannot be checked")
    raise ContractError(f"unknown gate {gate!r}")


def describe(contract: dict[str, Any]) -> dict[str, Any]:
    mechanism = contract.get("mechanism") or {}
    return {
        "id": mechanism.get("id"),
        "type": mechanism.get("type"),
        "entrypoints": list(mechanism.get("entrypoints", []) or []),
        "activation": mechanism.get("activation"),
        "forward": {
            "observables": list((contract.get("forward_contract") or {}).get("observables", [])),
            "invariants": [item.get("name") if isinstance(item, dict) else str(item)
                           for item in (contract.get("forward_contract") or {})
                           .get("invariants", []) or []],
            "declared_effect": (contract.get("forward_contract") or {}).get("declared_effect"),
        },
        "gradient": {
            "mode": (contract.get("gradient_contract") or {}).get("mode"),
            "target": (contract.get("gradient_contract") or {}).get("target"),
            "rule": (contract.get("gradient_contract") or {}).get("rule"),
            "expected_paths": (contract.get("gradient_contract") or {}).get("expected_paths"),
        },
        "update": {"optimizer": (contract.get("update_contract") or {}).get("optimizer")},
        "distributed": dict(contract.get("distributed_contract") or {}),
        "runtime": {"entrypoint": (contract.get("runtime_contract") or {}).get("entrypoint"),
                    "observe": [item.get("name") for item in
                                (contract.get("runtime_contract") or {}).get("observe", []) or []]},
    }

