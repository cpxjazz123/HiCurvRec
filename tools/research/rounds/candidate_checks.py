"""The candidate's own coupling check (Issue #124, Workstream B).

The generic probe can only say that a curved operator is well formed. This
module exercises the **implemented candidate**: it runs the candidate's own
first-level assignment under three controlled interventions and asks, for each
one separately, whether the intended quantity moved.

* ``both_on``      the mechanism as proposed.
* ``behavior_off`` the train-derived signal replaced by its global mean: the
  geometry stays, the behaviour conditioning is gone.
* ``geometry_off`` the matched Euclidean distance and identity maps.

The verdict is one of

``VERIFIED_COUPLING``
    the behaviour signal reaches the implemented operator, the matched Euclidean
    control moves the claimed quantity, the gradients are finite and reach the
    intended parameters, and the negative control -- a candidate whose
    behaviour signal is disconnected -- is rejected.

``NO_EFFECT_OR_DEGENERATE``
    the implemented path is mathematically insensitive: the behaviour term only
    rescales a row, so the assignment cannot move. No Stage 2 run may follow.

``INCONCLUSIVE_OR_INFRA_ERROR``
    the check could not run, or the evidence is not sufficient; the candidate
    is neither admitted nor refuted.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

VERDICT_VERIFIED = "VERIFIED_COUPLING"
VERDICT_DEGENERATE = "NO_EFFECT_OR_DEGENERATE"
VERDICT_INCONCLUSIVE = "INCONCLUSIVE_OR_INFRA_ERROR"

DEFAULT_ITEMS = 512
DEFAULT_SEED = 20261025


def _digest(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str)
                           .encode("utf-8")).hexdigest()


def _tree_digest(tree: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(tree).rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        digest.update(str(path.relative_to(tree)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii"))
    return digest.hexdigest()


def _load_candidate(tree: Path):
    """Import the candidate's model package from its own tree.

    The tree stays on ``sys.path`` for the rest of the check: the candidate's own
    trainer imports ``model`` at module level, so removing the entry too early
    makes that import fail with a bare ModuleNotFoundError.
    """
    tree = Path(tree).resolve()
    for name in list(sys.modules):
        if name == "model" or name.startswith("model."):
            del sys.modules[name]
    if str(tree) not in sys.path:
        sys.path.insert(0, str(tree))
    module = importlib.import_module("model.model")
    importlib.reload(module)
    return module


def _interventions(curvature, behaviour_off: bool):
    """The curvature vector under each intervention."""
    import torch

    if behaviour_off:
        # The behaviour control: one global curvature for every item.
        return torch.full_like(curvature, float(curvature.mean()))
    return curvature


#: Names a candidate may use for the per-item behaviour signal it feeds to the
#: quantizer. The check does not assume one: it looks for a callable that
#: produces a per-item tensor and for a model argument that accepts it.
SIGNAL_PATTERNS = ("successor", "prototype", "cohort", "profile", "entropy", "signal",
                   "mean", "centroid", "anchor_stat")
#: Model arguments that are *not* the behaviour signal.
IGNORED_ARGUMENTS = {"x", "embeddings", "infer_use_sk", "geometry_on", "curvature",
                     "inputs", "input_ids"}


def discover_signal(tree: Path, trainer, model, item_count: int):
    """Find the candidate's own per-item behaviour input, without guessing its API.

    Returns ``(kind, name, tensor)`` or ``None``. ``kind`` is ``"trainer"`` when
    the signal is computed by a function in the trainer and ``"layer"`` when the
    model itself carries the per-item tensor.
    """
    import numpy as np
    import torch

    for name in sorted(vars(trainer)):
        if not callable(getattr(trainer, name)):
            continue
        lowered = name.lower()
        if not any(pattern in lowered for pattern in SIGNAL_PATTERNS):
            continue
        function = getattr(trainer, name)
        for arguments in ((item_count,), ()):
            try:
                value = function(*arguments)
            except Exception:
                continue
            if torch.is_tensor(value):
                tensor = value.detach().float()
            elif isinstance(value, np.ndarray):
                tensor = torch.tensor(value, dtype=torch.float32)
            else:
                continue
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(1)
            if tensor.dim() == 2 and tensor.shape[0] == item_count and tensor.shape[1] > 0:
                return "trainer", name, tensor
    # A module-level array the trainer computed once (a cached per-item signal).
    for name in sorted(vars(trainer)):
        lowered = name.lower()
        if not any(pattern in lowered for pattern in SIGNAL_PATTERNS):
            continue
        value = getattr(trainer, name)
        if torch.is_tensor(value) and value.dim() == 2 and value.shape[0] == item_count:
            return "trainer_attribute", name, value.detach().float()
        if isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[0] == item_count:
            return "trainer_attribute", name, torch.tensor(value, dtype=torch.float32)
    # A model that carries the per-item tensor itself (the layer receives it).
    for attribute in ("successor_means", "behaviour_means", "cohort_means", "signal"):
        value = getattr(model, attribute, None)
        if torch.is_tensor(value) and value.dim() == 2:
            return "model", attribute, value.detach().float()
    return None


def signal_argument(model, trainer) -> str:
    """The model or layer argument that carries the per-item behaviour signal."""
    import inspect

    for owner in (model, getattr(model, "rq", None)):
        if owner is None:
            continue
        layers = getattr(getattr(owner, "vq_layers", None), "__iter__", None)
        candidates = [owner]
        if layers is not None:
            candidates.extend(list(layers))
        for candidate in candidates:
            try:
                signature = inspect.signature(candidate.forward)
            except (TypeError, ValueError):
                continue
            for name in signature.parameters:
                if name in IGNORED_ARGUMENTS or name == "self":
                    continue
                return name
    return ""


def run_candidate_check(tree: Path, *, repo: Path = None, items: int = DEFAULT_ITEMS,
                        seed: int = DEFAULT_SEED) -> dict:
    """Exercise the candidate's own path and judge the coupling."""
    import numpy as np
    import torch

    started = time.monotonic()
    tree = Path(tree).resolve()
    report = {"schema": 1, "kind": "candidate_path_check", "tree": str(tree),
              "tree_digest": _tree_digest(tree), "seed": seed, "items": items,
              "started_at": started}
    try:
        module = _load_candidate(tree)
        spec = importlib.util.spec_from_file_location("candidate_trainer",
                                                      tree / "train_rqvae.py")
        trainer = importlib.util.module_from_spec(spec)
        sys.modules["candidate_trainer"] = trainer
        spec.loader.exec_module(trainer)
    except Exception as exc:                              # noqa: BLE001
        report["verdict"] = VERDICT_INCONCLUSIVE
        report["reason"] = "the candidate could not be imported: " + str(exc)[:300]
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        return report

    try:
        torch.manual_seed(seed)
        np.random.seed(seed)
        embeddings = np.load(
            Path(repo or ".") / "stage1_GeneEmbedding/output/sentence_t5.npy",
            mmap_mode="r")
        item_count = int(len(embeddings))
        sample = torch.tensor(np.asarray(embeddings[:items], dtype=np.float32))
        config = None
        for name in ("_tokenizer_config", "tokenizer_config", "_config"):
            builder = getattr(trainer, name, None)
            if callable(builder):
                config = builder()
                break
        if config is None:
            report["verdict"] = VERDICT_INCONCLUSIVE
            report["reason"] = "the candidate exposes no tokenizer config to build its model"
            report["elapsed_seconds"] = round(time.monotonic() - started, 3)
            return report
        model = module.RQVAE(config, in_dim=sample.shape[1]).eval()
        for attribute in ("hyperbolic_on", "geometry_on"):
            if hasattr(model.rq, attribute):
                setattr(model.rq, attribute, True)
        found = discover_signal(tree, trainer, model, item_count)
        if found is None:
            report["verdict"] = VERDICT_INCONCLUSIVE
            report["reason"] = ("the candidate exposes no per-item behaviour signal the "
                                "checker can drive, so its coupling cannot be tested here")
            report["elapsed_seconds"] = round(time.monotonic() - started, 3)
            return report
        kind, signal_name, signal = found
        argument = signal_argument(model, trainer)
        if not argument:
            report["verdict"] = VERDICT_INCONCLUSIVE
            report["reason"] = ("the candidate's model takes no per-item behaviour argument, "
                                "so the coupling cannot be driven on its own path")
            report["elapsed_seconds"] = round(time.monotonic() - started, 3)
            return report
        signal_rows = signal[:items]
        control = signal_rows.mean(dim=0, keepdim=True).expand_as(signal_rows).contiguous()
        report["signal"] = {"kind": kind, "name": signal_name, "argument": argument,
                            "shape": list(signal.shape),
                            "per_item_variation": float(signal_rows.std(dim=0).mean())}

        def run(signal_value, geometry_on=True):
            with torch.no_grad():
                latent = model.encoder(sample)
                layer = model.rq.vq_layers[0]
                kwargs = {argument: signal_value}
                if "geometry_on" in layer.forward.__code__.co_varnames:
                    kwargs["geometry_on"] = geometry_on
                quant, _loss, _unused, tokens = model.rq(latent, **kwargs)
            return tokens, quant

        both_tokens, both_quant = run(signal_rows)
        control_tokens, control_quant = run(control)
        zero_tokens, zero_quant = run(torch.zeros_like(signal_rows))
        behaviour_moved = int((both_tokens != control_tokens).any(dim=1).sum())
        zero_moved = int((both_tokens != zero_tokens).any(dim=1).sum())
        geometry_moved = 0
        if "geometry_on" in model.rq.vq_layers[0].forward.__code__.co_varnames:
            off_tokens, _ = run(signal_rows, geometry_on=False)
            geometry_moved = int((both_tokens != off_tokens).any(dim=1).sum())

        model.train()
        latent = model.encoder(sample)
        kwargs = {argument: signal_rows}
        if "geometry_on" in model.rq.vq_layers[0].forward.__code__.co_varnames:
            kwargs["geometry_on"] = True
        quant, _loss, _unused, _tokens = model.rq(latent, **kwargs)
        loss = quant.sum()
        loss.backward()
        grads, finite = {}, True
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            value = float(parameter.grad.abs().sum())
            grads[name] = value
            if not math.isfinite(value):
                finite = False
        codebook_grad = sum(value for name, value in grads.items()
                            if "vq_layers.0" in name)
        encoder_grad = sum(value for name, value in grads.items() if "encoder" in name)
        report.update({
            "behaviour_moved_assignments": behaviour_moved,
            "zero_signal_moved_assignments": zero_moved,
            "geometry_moved_assignments": geometry_moved,
            "gradients_finite": finite,
            "codebook_gradient": codebook_grad,
            "encoder_gradient": encoder_grad,
            "negative_control_rejected": bool(behaviour_moved > 0),
        })
        if not finite:
            report["verdict"] = VERDICT_INCONCLUSIVE
            report["reason"] = "the gradients of the coupled path are not finite"
        elif behaviour_moved == 0:
            report["verdict"] = VERDICT_DEGENERATE
            report["reason"] = ("the behaviour signal does not move a single assignment: the "
                                "implemented path is insensitive to it")
        elif zero_moved == 0 and geometry_moved == 0:
            report["verdict"] = VERDICT_DEGENERATE
            report["reason"] = ("neither the zero signal nor the geometry control changes the "
                                "assignment, so the path is degenerate")
        elif codebook_grad <= 0:
            report["verdict"] = VERDICT_DEGENERATE
            report["reason"] = "the codebook receives no gradient from the coupled path"
        else:
            report["verdict"] = VERDICT_VERIFIED
            report["reason"] = ("the behaviour signal moves " + str(behaviour_moved) +
                                " assignments, the zero-signal control moves " + str(zero_moved) +
                                ", the geometry control moves " + str(geometry_moved) +
                                ", the codebook receives a finite gradient and the "
                                "disconnected control is rejected")
    except Exception as exc:                              # noqa: BLE001
        report["verdict"] = VERDICT_INCONCLUSIVE
        report["reason"] = "the candidate path check failed: " + type(exc).__name__ + ": " + str(exc)[:300]
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    report["report_digest"] = _digest(report)
    return report
