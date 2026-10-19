"""Replay one frozen step in a FRESH model/optimizer, using the ARM's own update code.

The separation from ``r7_a_frozen_replay`` is the whole point of track A: the capture module only
copies out of a run, and this module never runs the training loop. It reconstructs the exact objects
the loop had at the fork -- the same ``RQVAE``, the same optimizer class and hyperparameters, the
same scheduler -- loads the frozen tensors into them, and then calls the ARM's own trace, loss
assembly, projection and step.

Because every numeric step below is the trainer's function, the FULL branch is not "a loss that
looks like the arm's": it is the arm's loss. Any disagreement with a golden replay is a real
disagreement, not a second implementation.
"""
from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
for _entry in (str(REPO_ROOT / "tools/cgv"), str(REPO_ROOT / "tools")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

import contract as C  # noqa: E402
import engine as E  # noqa: E402

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def tensor_digest(tensor: torch.Tensor) -> str:
    with torch.no_grad():
        value = tensor.detach().to(torch.float64).cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def parameter_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor_digest(value).encode("ascii"))
    return digest.hexdigest()


class ReplayError(RuntimeError):
    """Raised when a branch cannot be taken. Never swallowed into a number."""


def build_frozen_branch(tree: Path, bundle: dict[str, Any], *, module_name: str) -> dict[str, Any]:
    """Construct model + optimizer + scheduler that hold EXACTLY the frozen state.

    Nothing here shares storage with a live run: every tensor is copied, so a branch can never write
    through to the training model it came from.
    """
    tree = Path(tree).resolve()
    trainer, saved_path, saved_mods = E.load_trainer(tree, module_name=module_name)
    trainer.SEED = bundle.get("seed", 42)

    model = _construct_model(trainer, bundle).to(DEVICE)
    model.load_state_dict({name: value.to(DEVICE) for name, value in bundle["model_state"].items()})
    model.train(mode=bool(bundle["training_mode"]))

    optimizer_cls = getattr(torch.optim, trainer.OPTIMIZER)
    # exactly the arm's construction: every parameter, the arm's lr and weight decay
    optimizer = optimizer_cls(list(model.parameters()), lr=trainer.LR,
                              weight_decay=trainer.WEIGHT_DECAY)
    frozen_optimizer = bundle.get("optimizer_state") or {}
    if not frozen_optimizer.get("param_groups"):
        # an INITIALISATION snapshot: no optimizer has run yet, so there are no moments to load.
        # A replay is impossible from here and ``assemble_and_step`` says so, but a read-only
        # observer only needs the model, so the branch is still built.
        if bundle.get("step") == 0:
            return {"trainer": trainer, "model": model, "optimizer": None,
                    "saved_path": saved_path, "saved_mods": saved_mods,
                    "module_name": module_name, "tree": tree,
                    "initialisation_only": True}
        raise ReplayError("the frozen bundle carries no optimizer state but it is not the "
                          "initialisation snapshot, so this is not a replayable step")
    optimizer.load_state_dict(copy.deepcopy(frozen_optimizer))
    return {"trainer": trainer, "model": model, "optimizer": optimizer,
            "saved_path": saved_path, "saved_mods": saved_mods, "module_name": module_name,
            "tree": tree}


def optimizer_hyperparameter_names(trainer) -> tuple[str, ...]:
    names = ("LR", "BETAS", "EPS", "WEIGHT_DECAY", "AMSGRAD")
    return tuple(name for name in names if hasattr(trainer, name))


def _construct_model(trainer, bundle: dict[str, Any]):
    """Build the same object the loop built, from the captured tensors' own shapes.

    Shapes come from the frozen state, not from constants retyped here, so a branch can never be
    constructed at a different geometry than the run it replays.
    """
    state = bundle["model_state"]
    encoder_key = next(name for name in sorted(state) if name.startswith("encoder.") and
                       name.endswith(".weight"))
    in_dim = int(state[encoder_key].shape[1])
    if hasattr(trainer, "_tokenizer_config"):
        return trainer.RQVAE(trainer._tokenizer_config(), in_dim=in_dim)
    raise ReplayError(f"cannot build the arm's model: {trainer.tree} exposes no _tokenizer_config")


def release_branch(branch: dict[str, Any]) -> None:
    E.unload_trainer(branch["saved_path"], branch["saved_mods"], branch["module_name"])


def hard_assignments(model, embeddings: torch.Tensor) -> dict[str, Any]:
    """The model's OWN inference path, so 'switching' means what the trained decoder will see."""
    was_training = model.training
    model.eval()
    layer = model.rq.vq_layers[0]
    device = embeddings.device
    with torch.no_grad():
        trace = model.encoder(embeddings.to(device))
        if isinstance(trace, tuple):
            trace = trace[0]
        # the quantizer's own distance, including any hyperbolic assignment hook it installs
        winner = torch.argmin(layer._distances(trace.reshape(-1, layer.dim)), dim=-1)
    model.train(was_training)
    return {"indices": winner.detach().cpu(), "used_codes": int(torch.unique(winner).numel())}


def margin_profile(model, embeddings: torch.Tensor) -> dict[str, float]:
    """Top1/top2 gap in the quantizer's own distance, reported as the population it is."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        encoder = model.encoder(embeddings)
        if isinstance(encoder, tuple):
            encoder = encoder[0]
        trace = model.rq(encoder, mode="train")
        if isinstance(trace, tuple):
            trace = trace[0]
        layer = model.rq.vq_layers[0]
        _, indices, _, _ = layer.get_indices(trace[:, :layer.latent_dim].detach())
        codebook = layer.codebook.t()
        flat = trace[:, :layer.latent_dim]
        distance = torch.cdist(flat, codebook, p=2) / (temp_of(model) ** 2)
        ordered = distance.sort(dim=1).values
        margin = (ordered[:, 1] - ordered[:, 0])
    model.train(was_training)
    margin = margin.float()
    return {
        "l1_margin_mean": float(margin.mean()),
        "l1_margin_p05": float(margin.quantile(0.05)),
        "l1_margin_median": float(margin.median()),
        "l1_collision_rate": float((indices[:, 0] == indices[:, 1]).float().mean()),
    }


def temp_of(model) -> float:
    for layer in model.rq.vq_layers:
        if hasattr(layer, "temp"):
            return float(layer.temp)
    return 1.0


# --------------------------------------------------------------------------------------
# interventions
# --------------------------------------------------------------------------------------
INTERVENTIONS = ("FULL", "NO-DEEP", "BASE-ONLY")
NEGATIVE_CONTROL = {"r4_08": "NO-DEEP"}


def assemble_and_step(branch: dict[str, Any], bundle: dict[str, Any], *,
                      intervention: str) -> dict[str, Any]:
    """Take exactly one optimizer step from the frozen state, through the ARM's own code.

    ``FULL`` removes nothing and is the control the whole track rests on: it must land on the same
    parameters the real loop reached. ``NO-DEEP`` and ``BASE-ONLY`` remove their loss contribution
    BEFORE the arm assembles and projects, so R4-08's PCGrad sees the reduced gradient set rather
    than a difference of two finished gradients -- a gradient difference is not a counterfactual
    for a nonlinear projection.
    """
    if intervention not in INTERVENTIONS:
        raise ReplayError(f"unknown intervention {intervention!r}")
    if branch.get("initialisation_only"):
        raise ReplayError("this bundle is the initialisation snapshot: no optimizer step has run, "
                          "so there is nothing to replay")
    trainer, model, optimizer = branch["trainer"], branch["model"], branch["optimizer"]
    model.train()
    optimizer.zero_grad(set_to_none=True)

    device = DEVICE
    item_batch = bundle["item_batch"].to(device)
    # an inert step has no pair minibatch and no labels at all: the deep path was never taken
    pair_embeddings = bundle.get("pair_embeddings")
    source_ids = bundle.get("source_ids")
    target_ids = bundle.get("target_ids")
    if source_ids is not None:
        source_ids = source_ids.to(device)
    if target_ids is not None:
        target_ids = target_ids.to(device)
    successor = bundle.get("successor")
    successor = None if successor is None else successor.to(device)
    weight = float(bundle["applied_weight"])
    keep_base = intervention != "BASE-ONLY"
    keep_deep = intervention != "NO-DEEP"
    deep_possible = (weight > 0.0 and pair_embeddings is not None and source_ids is not None)

    reconstructed, quant_loss, _, _ = model(item_batch)
    base_loss, _ = model.compute_loss(item_batch, reconstructed, quant_loss)

    behavior_loss = torch.zeros((), device=device)
    behavior_l1 = behavior_other = None
    valid_levels = 0
    deep_present = False
    if deep_possible and keep_deep:
        residuals = trainer._residuals_with_trace(model, pair_embeddings.to(device))
        if getattr(trainer, "R4_08_PCGRAD_ENC", False):
            behavior_l1, behavior_other, valid_levels = trainer._r4_08_split_behavior_loss(
                residuals, source_ids, target_ids, successor)
            if valid_levels > 0:
                total_beh = ((behavior_l1 if behavior_l1 is not None else 0.0)
                             + (behavior_other if behavior_other is not None else 0.0))
                behavior_loss = total_beh / float(valid_levels)
                deep_present = True
        else:
            if getattr(trainer, "DIRECTIONAL_MODE", "none") != "none":
                behavior_loss = trainer.directional_residual_loss(
                    residuals, source_ids, target_ids, trainer.directional_heads,
                    trainer.DIRECTIONAL_MODE == "sym", successor)
            else:
                behavior_loss = trainer.behavior_contrastive_loss(
                    residuals, source_ids, target_ids, successor)
            deep_present = True

    if not keep_base:
        base_loss = torch.zeros((), device=device)
        behavior_l1 = behavior_other = None
    if not keep_deep:
        behavior_loss = torch.zeros((), device=device)

    loss = trainer.assemble_losses(base_loss, behavior_loss, weight,
                                   torch.zeros((), device=device), 0.0, False)

    model.zero_grad(set_to_none=True)
    projected = False
    no_signal = False
    # The projection resolves a conflict BETWEEN the base and the deep gradient. With either term
    # removed there is no conflict to resolve, and the trainer itself only projects on the full
    # path -- so BASE-ONLY and NO-DEEP must not be projected, exactly as the production loop would.
    if (deep_present and keep_base and keep_deep
            and getattr(trainer, "R4_08_PCGRAD_ENC", False)):
        # the arm replaces the encoder gradient with its own PCGrad assembly; only that call, with
        # the arm's own arguments, can produce what the real loop produced
        loss.backward(retain_graph=True)
        trainer._r4_08_apply_projection(
            base_loss, behavior_l1, behavior_other,
            list(model.encoder.parameters()), weight, valid_levels=valid_levels)
        projected = True
    elif loss.requires_grad:
        loss.backward()
    else:
        # Every loss term was removed, so there is genuinely no learning signal left. That is a
        # RESULT, not an error: the branch must report a no-op update rather than crash or invent
        # a zero-gradient graph just to keep DDP happy.
        no_signal = True

    clip = float(getattr(trainer, "GRADIENT_CLIP_NORM", 0.0) or 0.0)
    if clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    if bundle.get("lr_at_step") is not None:
        for group in optimizer.param_groups:
            group["lr"] = float(bundle["lr_at_step"])
    optimizer.step()

    return {
        "intervention": intervention,
        "deep_present": deep_present,
        "applied_weight": weight,
        "lr": float(bundle.get("lr_at_step") or 0.0),
        "projection_applied": projected,
        "no_learning_signal": no_signal,
        "base_loss": float(base_loss.detach()),
        "behavior_loss": float(behavior_loss.detach()),
        "parameter_sha256": parameter_digest(model),
        "update_sha256": update_digest(bundle["model_state"], model),
    }


def update_digest(before: dict[str, Any], model: torch.nn.Module) -> str:
    """Hash the ACTUAL update delta, so 'the update was applied twice' cannot hide."""
    digest = hashlib.sha256()
    state = model.state_dict()
    for name in sorted(before):
        if name not in state:
            continue
        delta = state[name].detach().to(torch.float64).cpu() - before[name].to(torch.float64)
        digest.update(name.encode("utf-8"))
        digest.update(tensor_digest(delta).encode("ascii"))
    return digest.hexdigest()


def max_parameter_difference(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    worst = 0.0
    worst_name = None
    for name in sorted(left):
        if name not in right:
            return {"identical": False, "max_abs_diff": None, "tensor": name,
                    "reason": "the reference has no such tensor"}
        delta = (right[name].detach().to(torch.float64).cpu()
                 - left[name].to(torch.float64)).abs().max().item()
        if delta > worst:
            worst, worst_name = delta, name
    return {"identical": worst == 0.0, "max_abs_diff": worst, "tensor": worst_name}
