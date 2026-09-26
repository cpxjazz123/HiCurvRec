"""Four-layer mechanism verification for the Iter25 curvature prior."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = SCRIPT_DIR.parent / "curvature_RQ-VAE.py"
REFERENCE_CKPT = Path(
    "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/"
    "curvature_RQ-VAE_iter8/out/rqvae/instruments/rqvae_best.pth"
)
OFF_PRIOR = [0.001, 0.932889, 1.0]
EXPECTED_PRIOR = [0.99, 0.1262333365, 0.0581856194]
GRAD_EPSILON = 1e-12
RELATIVE_UPDATE_EPSILON = 1e-7
BEHAVIOR_STEPS = 200
BEHAVIOR_LOSS_DELTA = 1e-6


def _load_training_module():
    spec = importlib.util.spec_from_file_location("rqtrain_iter25", TRAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load training entry: {TRAIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_batch(rqtrain, device: torch.device):
    dataset = rqtrain.TransitionDataset(
        rqtrain.EMB_NPY, rqtrain.ITEM_IDS_JSON, rqtrain.TRAIN_PARQUET
    )
    active_indices = [
        index for index, targets in enumerate(dataset.next_items) if targets
    ]
    if len(active_indices) < rqtrain.BATCH_SIZE:
        raise RuntimeError(
            f"Only {len(active_indices)} train-transition sources; need "
            f"{rqtrain.BATCH_SIZE} for the MVG batch"
        )
    raw_batch = rqtrain.collate_items(
        [dataset[index] for index in active_indices[: rqtrain.BATCH_SIZE]]
    )
    batch = rqtrain._build_seq_batch(
        *(value.to(device) for value in raw_batch)
    )
    return batch


def _load_checkpoint_state(device: torch.device):
    if not REFERENCE_CKPT.is_file():
        raise FileNotFoundError(f"Warm-start checkpoint missing: {REFERENCE_CKPT}")
    state = torch.load(REFERENCE_CKPT, map_location=device, weights_only=False)
    if not isinstance(state.get("model"), dict):
        raise RuntimeError("Warm-start checkpoint has no model state_dict")
    return state


def _build_model(rqtrain, device: torch.device, prior, checkpoint_state):
    model = rqtrain.RqVae(
        input_dim=rqtrain.INPUT_DIM,
        embed_dim=rqtrain.EMBED_DIM,
        hidden_dims=rqtrain.HIDDEN_DIMS,
        codebook_size=rqtrain.CODEBOOK_SIZE,
        codebook_kmeans_init=False,
        n_layers=rqtrain.N_LAYERS,
        commitment_weight=rqtrain.COMMITMENT_WEIGHT,
        sk_eps=0.05,
        sk_iters=3,
        c_cyclic_min=rqtrain.C_CYCLIC_MIN,
        c_cyclic_max=rqtrain.C_CYCLIC_MAX,
        c_cyclic_period=rqtrain.C_CYCLIC_PERIOD,
        midpoint_layer_mask=rqtrain.MIDPOINT_LAYER_MASK,
        residual_layer_norms=prior,
    ).to(device)
    scale_keys = {
        f"layers.{index}.c_layer_scale" for index in range(rqtrain.N_LAYERS)
    }
    transfer = {
        name: value
        for name, value in checkpoint_state["model"].items()
        if name not in scale_keys
    }
    missing, unexpected = model.load_state_dict(transfer, strict=False)
    if set(missing) != scale_keys or unexpected:
        raise RuntimeError(
            "MVG warm-start mismatch: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    return model


def _require_finite_scalar(value: torch.Tensor, label: str) -> None:
    if not value.requires_grad or value.grad_fn is None:
        raise RuntimeError(f"MVG FAIL: {label} is detached")
    if not torch.isfinite(value).all().item():
        raise RuntimeError(f"MVG FAIL: {label} is non-finite")


def _gradient_norms(loss: torch.Tensor, parameters, label: str):
    if not loss.requires_grad:
        raise RuntimeError(f"MVG FAIL: {label} does not require gradients")
    gradients = torch.autograd.grad(
        loss,
        tuple(parameters),
        retain_graph=True,
        allow_unused=True,
    )
    norms = []
    for gradient in gradients:
        if gradient is not None:
            if not torch.isfinite(gradient).all().item():
                raise RuntimeError(f"MVG FAIL: {label} has a non-finite gradient")
            norms.append(float(gradient.detach().abs().sum().item()))
    if not any(norm > GRAD_EPSILON for norm in norms):
        raise RuntimeError(f"MVG FAIL: {label} has no non-zero parameter gradient")
    return norms


def _verify_graph_and_gradients(rqtrain, model, batch):
    prior = rqtrain._build_layer_scale_prior()
    initial_scales = [float(layer.c_layer_scale.detach().item()) for layer in model.layers]
    if not np.allclose(initial_scales, prior, rtol=0.0, atol=1e-6):
        raise RuntimeError(
            f"MVG FAIL: initialized scales {initial_scales} != prior {prior}"
        )
    if not np.allclose(prior, EXPECTED_PRIOR, rtol=0.0, atol=1e-8):
        raise RuntimeError(f"MVG FAIL: derived prior changed unexpectedly: {prior}")
    if not np.allclose(
        [layer.c_layer_norm for layer in model.layers], prior, rtol=0.0, atol=1e-7
    ):
        raise RuntimeError("MVG FAIL: curvature regularization centers differ from prior")

    curvature_evidence = {}
    for step, phase in (
        (0, 0.0),
        (rqtrain.C_CYCLIC_PERIOD // 2, 1.0),
    ):
        model.set_curriculum_step(step)
        actual = [float(layer.get_c().detach().item()) for layer in model.layers]
        expected = [
            rqtrain.C_CYCLIC_MIN
            * math.exp(
                ((value + phase) / 2.0)
                * math.log(rqtrain.C_CYCLIC_MAX / rqtrain.C_CYCLIC_MIN)
            )
            for value in prior
        ]
        if not np.allclose(actual, expected, rtol=0.0, atol=1e-5):
            raise RuntimeError(
                f"MVG FAIL: cyclic curvature mismatch at step {step}: "
                f"actual={actual}, expected={expected}"
            )
        if any(
            value < rqtrain.C_CYCLIC_MIN or value > rqtrain.C_CYCLIC_MAX
            for value in actual
        ):
            raise RuntimeError(f"MVG FAIL: curvature escaped bounds: {actual}")
        if not (actual[0] > actual[1] > actual[2]):
            raise RuntimeError(
                f"MVG FAIL: expected fixed-phase layer ordering, got {actual}"
            )
        curvature_evidence[step] = actual

    # Move each scale a small amount away from its anchor so the anchor's
    # derivative is observable; the five-step check uses an unperturbed model.
    with torch.no_grad():
        for layer in model.layers:
            delta = -0.02 if layer.c_layer_scale.item() > 0.95 else 0.02
            layer.c_layer_scale.add_(delta)
    model.train()
    model.set_curriculum_step(0)
    model.zero_grad(set_to_none=True)
    output = model(batch)
    _require_finite_scalar(output.loss, "total_loss")
    components = {
        "reconstruction_loss": output.reconstruction_loss,
        "rqvae_loss": output.rqvae_loss,
        "behavior_loss": output.behavior_loss,
        "curvature_regularization": output.curvature_regularization,
    }
    component_norms = {
        name: _gradient_norms(value.mean(), model.parameters(), name)
        for name, value in components.items()
    }
    if output.behavior_loss.detach().item() <= 0.0:
        raise RuntimeError("MVG FAIL: behavior contrastive loss is inactive")

    scales = tuple(layer.c_layer_scale for layer in model.layers)
    for name, loss in (
        ("behavior_loss", output.behavior_loss),
        ("curvature_regularization", output.curvature_regularization),
    ):
        gradients = torch.autograd.grad(
            loss,
            scales,
            retain_graph=True,
            allow_unused=True,
        )
        norms = []
        for layer_index, gradient in enumerate(gradients):
            if gradient is None or not torch.isfinite(gradient).all().item():
                raise RuntimeError(
                    f"MVG FAIL: {name} has no finite scale gradient at layer {layer_index}"
                )
            norm = float(gradient.detach().abs().sum().item())
            if norm <= GRAD_EPSILON:
                raise RuntimeError(
                    f"MVG FAIL: {name} scale gradient is zero at layer {layer_index}"
                )
            norms.append(norm)
        component_norms[f"{name}_per_layer_scale"] = norms

    output.loss.backward()
    total_scale_gradients = []
    for layer_index, parameter in enumerate(scales):
        gradient = parameter.grad
        if (
            gradient is None
            or not torch.isfinite(gradient).all().item()
            or gradient.abs().sum().item() <= GRAD_EPSILON
        ):
            raise RuntimeError(
                f"MVG FAIL: total loss has no finite non-zero scale gradient "
                f"at layer {layer_index}"
            )
        total_scale_gradients.append(float(gradient.detach().abs().sum().item()))
    return {
        "prior": prior,
        "initial_scales": initial_scales,
        "curvature": curvature_evidence,
        "component_grad_norms": component_norms,
        "total_scale_grad_norms": total_scale_gradients,
        "behavior_loss": float(output.behavior_loss.detach().item()),
    }


def _verify_five_step_updates(rqtrain, device, batch, checkpoint_state):
    prior = rqtrain._build_layer_scale_prior()
    model = _build_model(rqtrain, device, prior, checkpoint_state)
    model.train()
    before = [layer.c_layer_scale.detach().clone() for layer in model.layers]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=rqtrain.OPTIMIZER_LR,
        weight_decay=rqtrain.OPTIMIZER_WEIGHT_DECAY,
    )
    for step in range(5):
        model.set_curriculum_step(step)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        _require_finite_scalar(output.loss, f"five-step loss {step}")
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), rqtrain.GRAD_CLIP_NORM)
        optimizer.step()
    relative_updates = [
        float(
            (layer.c_layer_scale.detach() - old).abs().item()
            / max(abs(float(old.item())), 1e-12)
        )
        for layer, old in zip(model.layers, before)
    ]
    if any(value <= RELATIVE_UPDATE_EPSILON for value in relative_updates):
        raise RuntimeError(
            f"MVG FAIL: five-step relative scale updates too small: {relative_updates}"
        )
    return relative_updates


def _train_behavior_arm(rqtrain, device, batch, checkpoint_state, prior):
    torch.manual_seed(rqtrain.SEED)
    np.random.seed(rqtrain.SEED)
    model = _build_model(rqtrain, device, prior, checkpoint_state)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=rqtrain.OPTIMIZER_LR,
        weight_decay=rqtrain.OPTIMIZER_WEIGHT_DECAY,
    )
    for step in range(BEHAVIOR_STEPS):
        model.set_curriculum_step(step)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        _require_finite_scalar(output.loss, f"behavior arm loss {step}")
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), rqtrain.GRAD_CLIP_NORM)
        optimizer.step()
    model.set_curriculum_step(BEHAVIOR_STEPS)
    model.eval()
    with torch.no_grad():
        terminal = model(batch)
    return {
        "loss": float(terminal.loss.item()),
        "behavior_loss": float(terminal.behavior_loss.item()),
        "scales": [float(layer.c_layer_scale.item()) for layer in model.layers],
        "curvature": [float(layer.get_c().item()) for layer in model.layers],
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("MVG requires CUDA")
    rqtrain = _load_training_module()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(rqtrain.SEED)
    np.random.seed(rqtrain.SEED)
    batch = _load_batch(rqtrain, device)
    checkpoint_state = _load_checkpoint_state(device)

    model = _build_model(
        rqtrain, device, rqtrain._build_layer_scale_prior(), checkpoint_state
    )
    graph_gradients = _verify_graph_and_gradients(rqtrain, model, batch)
    updates = _verify_five_step_updates(rqtrain, device, batch, checkpoint_state)
    on = _train_behavior_arm(
        rqtrain,
        device,
        batch,
        checkpoint_state,
        rqtrain._build_layer_scale_prior(),
    )
    off = _train_behavior_arm(
        rqtrain, device, batch, checkpoint_state, OFF_PRIOR
    )
    loss_delta = abs(on["loss"] - off["loss"])
    if not math.isfinite(loss_delta) or loss_delta <= BEHAVIOR_LOSS_DELTA:
        raise RuntimeError(
            f"MVG FAIL: 200-step ON/OFF loss delta {loss_delta:.9g} "
            f"<= {BEHAVIOR_LOSS_DELTA}"
        )

    print("MVG PASS")
    print(f"warm_start=PASS checkpoint={REFERENCE_CKPT}")
    print(f"batch_shape={tuple(batch.x.shape)}")
    print(f"prior={graph_gradients['prior']}")
    print(f"initial_scales={graph_gradients['initial_scales']}")
    print(f"curvature_by_step={graph_gradients['curvature']}")
    print(f"component_gradient_norms={graph_gradients['component_grad_norms']}")
    print(f"total_scale_grad_norms={graph_gradients['total_scale_grad_norms']}")
    print(f"five_step_relative_scale_updates={updates}")
    print(f"200_step_on={on}")
    print(f"200_step_off={off}")
    print(f"200_step_abs_total_loss_delta={loss_delta:.9g}")


if __name__ == "__main__":
    main()
