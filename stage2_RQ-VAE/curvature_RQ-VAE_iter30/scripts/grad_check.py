"""Shared one-checkpoint/one-batch FCCR-1 gradient gate for fixed routes."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from modules.warm_start import load_warm_start
from profile_routes import (
    STAGE0_TRAIN,
    STAGE0_TRAIN_SHA256,
    STAGE1_EMBEDDING,
    STAGE1_EMBEDDING_SHA256,
    STAGE1_ITEM_IDS,
    STAGE1_ITEM_IDS_SHA256,
    WARMSTART_PATH,
    WARMSTART_SHA256,
    _require_digest,
    _validate_stage2_identity,
    get_profile,
)

GRADIENT_EPSILON = 1e-12
REQUIRED_LOSS_FIELDS = ("reconstruction_loss", "rqvae_loss", "behavior_loss")


def _load_training_module():
    trainer_path = SOURCE_ROOT / "curvature_RQ-VAE.py"
    spec = importlib.util.spec_from_file_location("iter30_gradient_trainer", trainer_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Stage2 trainer: {trainer_path}")
    trainer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = trainer
    spec.loader.exec_module(trainer)
    return trainer


def _gradient_summary(loss: torch.Tensor, named_parameters, name: str) -> dict:
    if not loss.requires_grad or loss.grad_fn is None:
        raise RuntimeError(f"gradient gate: {name} is detached")
    parameters = [parameter for _, parameter in named_parameters]
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=True, allow_unused=True
    )
    nonzero_names = []
    for (parameter_name, _parameter), gradient in zip(named_parameters, gradients):
        if gradient is None:
            continue
        if not torch.isfinite(gradient).all().item():
            raise RuntimeError(f"gradient gate: {name} has non-finite gradient")
        if gradient.abs().sum().item() > GRADIENT_EPSILON:
            nonzero_names.append(parameter_name)
    if not nonzero_names:
        raise RuntimeError(f"gradient gate: {name} reaches no nonzero model gradient")
    return {
        "requires_grad": loss.requires_grad,
        "grad_fn": type(loss.grad_fn).__name__,
        "value": float(loss.detach().item()),
        "nonzero_parameter_names": nonzero_names,
    }


def _make_batch(trainer, device):
    dataset = trainer.TransitionDataset(
        trainer.EMB_NPY, trainer.ITEM_IDS_JSON, trainer.TRAIN_PARQUET
    )
    active_indices = [
        index for index, targets in enumerate(dataset.next_items) if targets
    ]
    if len(active_indices) < trainer.BATCH_SIZE:
        raise RuntimeError(
            f"gradient gate needs {trainer.BATCH_SIZE} active sources, "
            f"found {len(active_indices)}"
        )
    raw_batch = trainer.collate_items(
        [dataset[index] for index in active_indices[: trainer.BATCH_SIZE]]
    )
    return trainer._build_seq_batch(*(value.to(device) for value in raw_batch))


def _verify_dispatch_identity(profile, label):
    if profile["label"] != label:
        raise RuntimeError(f"gradient gate route mismatch: {label}")
    identity = _validate_stage2_identity(profile)
    for path, digest, name in (
        (STAGE1_EMBEDDING, STAGE1_EMBEDDING_SHA256, "Stage1 embedding"),
        (STAGE1_ITEM_IDS, STAGE1_ITEM_IDS_SHA256, "Stage1 item IDs"),
        (STAGE0_TRAIN, STAGE0_TRAIN_SHA256, "Stage0 train parquet"),
        (WARMSTART_PATH, WARMSTART_SHA256, "iter8 warm-start"),
    ):
        _require_digest(path, digest, name)
    return identity


def run_gradient_check(label: str, entrypoint: str) -> None:
    profile = get_profile(label)
    wrapper_path = Path(entrypoint).resolve(strict=True)
    if wrapper_path != profile["grad_entrypoint"].resolve(strict=True):
        raise RuntimeError(f"gradient wrapper/profile mismatch: {wrapper_path} != {label}")
    if not torch.cuda.is_available():
        raise RuntimeError("iter30 Stage2 gradient gate requires CUDA")

    identity = _verify_dispatch_identity(profile, label)
    trainer = _load_training_module()
    trainer._apply_profile(profile)
    mapping_details = trainer._load_closed_form_mapping(profile["mapping_id"])
    fixed_curvatures = mapping_details["closed_form_c_l"]
    if not np.allclose(
        fixed_curvatures,
        profile["curvature"],
        rtol=0.0,
        atol=trainer.FORMULA_COMPARISON_TOL,
    ):
        raise RuntimeError("gradient gate mapping differs from fixed route")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(profile["seed"])
    np.random.seed(profile["seed"])
    model = trainer.RqVae(
        input_dim=trainer.INPUT_DIM,
        embed_dim=trainer.EMBED_DIM,
        hidden_dims=trainer.HIDDEN_DIMS,
        codebook_size=trainer.CODEBOOK_SIZE,
        codebook_kmeans_init=True,
        n_layers=trainer.N_LAYERS,
        commitment_weight=trainer.COMMITMENT_WEIGHT,
        sk_eps=0.05,
        sk_iters=3,
        c_cyclic_min=trainer.C_CYCLIC_MIN,
        c_cyclic_max=trainer.C_CYCLIC_MAX,
        c_cyclic_period=trainer.C_CYCLIC_PERIOD,
        midpoint_layer_mask=trainer.MIDPOINT_LAYER_MASK,
        residual_layer_norms=[1.0] * trainer.N_LAYERS,
        freeze_layer_scale=False,
        fixed_layer_curvatures=fixed_curvatures,
    ).to(device)
    warm_start = load_warm_start(model, fixed_curvatures)
    batch = _make_batch(trainer, device)
    model.train()
    model.zero_grad(set_to_none=True)
    output = model(batch)
    total_loss = output.loss
    if not total_loss.requires_grad or total_loss.grad_fn is None:
        raise RuntimeError("gradient gate: total_loss requires_grad/grad_fn invariant failed")
    if not torch.isfinite(total_loss).all().item():
        raise RuntimeError("gradient gate: total_loss is non-finite")
    if float(output.curvature_regularization.detach().item()) != 0.0:
        raise RuntimeError("gradient gate: FCCR-1 curvature regularization is nonzero")

    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not named_parameters:
        raise RuntimeError("gradient gate: model has no trainable parameters")
    loss_checks = {
        name: _gradient_summary(getattr(output, name), named_parameters, name)
        for name in REQUIRED_LOSS_FIELDS
    }
    total_loss.backward()

    nonzero_gradients = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all().item():
            raise RuntimeError(f"gradient gate: non-finite backward gradient on {name}")
        if parameter.grad.abs().sum().item() > GRADIENT_EPSILON:
            nonzero_gradients.append(name)
    if not nonzero_gradients:
        raise RuntimeError("gradient gate: total_loss backward has no nonzero model gradient")

    parameter_ids = {id(parameter) for parameter in model.parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    fixed_curvature = []
    for index, layer in enumerate(model.layers):
        buffer = layer._fixed_c
        if buffer.requires_grad or id(buffer) in parameter_ids or id(buffer) in optimizer_ids:
            raise RuntimeError(f"gradient gate: fixed curvature L{index} is trainable")
        if buffer.grad is not None:
            raise RuntimeError(f"gradient gate: fixed curvature L{index} received a gradient")
        value = float(buffer.detach().cpu().item())
        if abs(value - fixed_curvatures[index]) > trainer.FIXED_CURVATURE_INVARIANCE_TOL:
            raise RuntimeError(f"gradient gate: fixed curvature L{index} changed")
        fixed_curvature.append(value)

    last_fields = []
    for module_name, module in model.named_modules():
        last_fields.extend(
            f"{module_name}.{attribute}" if module_name else attribute
            for attribute in vars(module)
            if attribute.startswith("_last_")
        )
    if last_fields:
        raise RuntimeError(f"gradient gate found unreviewed _last_* state: {last_fields}")

    report = {
        "status": "PASS",
        "profile": label,
        "mapping_id": profile["mapping_id"],
        "seed": profile["seed"],
        "batch_size": int(batch.x.shape[0]),
        "warm_start": warm_start,
        "fixed_curvature": fixed_curvature,
        "total_loss": float(total_loss.detach().item()),
        "loss_gradients": loss_checks,
        "total_loss_nonzero_gradient_parameters": nonzero_gradients,
        "curvature_regularization": float(output.curvature_regularization.detach().item()),
        "last_detach_state_fields": last_fields,
        "dispatcher_identity": {
            "input_sha256": identity["input_sha256"],
            "source_sha256": identity["source_sha256"],
        },
    }
    print("STAGE2_GRADIENT_GATE_PASS")
    print(json.dumps(report, indent=2, sort_keys=True))
