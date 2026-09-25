"""Gradient-path preflight for behavior-conditioned Stage2 curvature."""
from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = SCRIPT_DIR.parent / "curvature_RQ-VAE.py"
# iter9: warm-start from iter8's checkpoint (the most recent Stage-2 run).
REFERENCE_CKPT = (
    SCRIPT_DIR.parents[1]
    / "curvature_RQ-VAE_iter8"
    / "results"
    / "out"
    / "rqvae"
    / "instruments"
    / "rqvae_best.pth"
)
GRAD_EPSILON = 1e-12


def _load_training_module():
    spec = importlib.util.spec_from_file_location("rqtrain_iter9", TRAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load training entry: {TRAIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("MVG requires CUDA")
    rqtrain = _load_training_module()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(rqtrain.SEED)

    dataset = rqtrain.TransitionDataset(
        rqtrain.EMB_NPY, rqtrain.ITEM_IDS_JSON, rqtrain.TRAIN_PARQUET
    )
    active_indices = [i for i, targets in enumerate(dataset.next_items) if targets]
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
        residual_layer_norms=rqtrain._load_layer_norms(),
    ).to(device)

    state = torch.load(REFERENCE_CKPT, map_location=device, weights_only=False)
    checkpoint_weights = state["model"]
    # iter7 changes the cyclic-curvature range, so we re-initialize the learned
    # layer-scales to the calibrated baseline (interior of the new range) while
    # keeping iter4's embeddings & codes. iter4's scales were all clamped to 1e-4,
    # which would freeze behavior gradients at the boundary in iter7's wider range.
    trainable_curvature_keys = {
        f"layers.{index}.c_layer_scale" for index in range(rqtrain.N_LAYERS)
    }
    transferable_weights = {
        name: value
        for name, value in checkpoint_weights.items()
        if name not in trainable_curvature_keys
    }
    missing, unexpected = model.load_state_dict(transferable_weights, strict=False)
    if set(missing) != trainable_curvature_keys or unexpected:
        raise RuntimeError(
            f"MVG FAIL: unexpected checkpoint migration keys: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    # Set curriculum to mid-cycle so curvature is ~geometric-mean of the wider range,
    # which avoids vanishing behavior-loss gradients at the very low-c start.
    model.set_curriculum_step(rqtrain.C_CYCLIC_PERIOD // 2)
    with tempfile.NamedTemporaryFile(
        prefix="rqvae_iter9_mvg_", suffix=".pt", dir=SCRIPT_DIR, delete=False
    ) as handle:
        checkpoint = Path(handle.name)
    try:
        torch.save({"model": model.state_dict()}, checkpoint)
        roundtrip = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(roundtrip["model"], strict=True)
    finally:
        checkpoint.unlink(missing_ok=True)

    model.train()
    output = model(batch)
    if not output.loss.requires_grad or output.loss.grad_fn is None:
        raise RuntimeError("MVG FAIL: total loss is detached")
    if not torch.isfinite(output.loss).item():
        raise RuntimeError("MVG FAIL: total loss is non-finite")
    if not torch.isfinite(output.behavior_loss).item() or output.behavior_loss.item() <= 0:
        raise RuntimeError("MVG FAIL: behavior contrastive loss is not active")

    curvature_parameters = [layer.c_layer_scale for layer in model.layers]
    behavior_gradients = torch.autograd.grad(
        output.behavior_loss,
        curvature_parameters,
        retain_graph=True,
        allow_unused=True,
    )
    behavior_grad_norms = []
    for layer_index, gradient in enumerate(behavior_gradients):
        if gradient is None or not torch.isfinite(gradient).all().item():
            raise RuntimeError(
                f"MVG FAIL: behavior loss has no finite curvature gradient at layer {layer_index}"
            )
        norm = float(gradient.detach().abs().sum().item())
        if norm <= GRAD_EPSILON:
            raise RuntimeError(
                f"MVG FAIL: behavior curvature gradient is zero at layer {layer_index}"
            )
        behavior_grad_norms.append(norm)

    output.loss.backward()
    for layer_index, parameter in enumerate(curvature_parameters):
        if parameter.grad is None or not torch.isfinite(parameter.grad).all().item():
            raise RuntimeError(f"MVG FAIL: total-loss curvature gradient invalid at layer {layer_index}")
        if parameter.grad.abs().sum().item() <= GRAD_EPSILON:
            raise RuntimeError(f"MVG FAIL: total-loss curvature gradient zero at layer {layer_index}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    before = [parameter.detach().clone() for parameter in curvature_parameters]
    optimizer.step()
    updates = [
        float((parameter.detach() - old).abs().item())
        for old, parameter in zip(before, curvature_parameters)
    ]
    if not any(value > 0 for value in updates):
        raise RuntimeError("MVG FAIL: optimizer did not update learned curvature")
    curvature = [float(layer.get_c().detach().item()) for layer in model.layers]
    if any(not (rqtrain.C_CYCLIC_MIN <= value <= rqtrain.C_CYCLIC_MAX) for value in curvature):
        raise RuntimeError(f"MVG FAIL: learned curvature escaped configured bounds: {curvature}")

    print("MVG PASS")
    print("warm_start=PASS (iter8 weights loaded)")
    print(f"batch_shape={tuple(batch.x.shape)} active_pairs={int(batch.ids.ne(batch.ids_fut).sum().item())}")
    print(f"behavior_loss={float(output.behavior_loss.detach().item()):.6f}")
    print(f"behavior_curvature_grad_norms={[round(value, 8) for value in behavior_grad_norms]}")
    print(f"curvature_updates={[f'{value:.3e}' for value in updates]}")
    print(f"curvature_after_step={curvature}")


if __name__ == "__main__":
    main()
