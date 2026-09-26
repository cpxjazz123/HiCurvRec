"""MVG preflight for Iter26 fixed closed-form curvature (raw residual medians)."""
from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = SCRIPT_DIR.parent / "curvature_RQ-VAE.py"
REFERENCE_CKPT = Path(
    "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/"
    "curvature_RQ-VAE_iter8/out/rqvae/instruments/rqvae_best.pth"
)
GRAD_EPSILON = 1e-12
RELATIVE_UPDATE_EPSILON = 1e-7
INVARIANCE_TOL = 1e-6
INVARIANCE_STEPS = (0, 25_000, 50_000, 100_000)
EXPECTED_FIXED_C = [0.6145357379232853, 0.5333020920777128, 0.3814078098431606]


def _load_training_module():
    spec = importlib.util.spec_from_file_location("rqtrain_iter26", TRAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load training entry: {TRAIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_batch(rqtrain, device):
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
    return rqtrain._build_seq_batch(*(value.to(device) for value in raw_batch))


def _load_checkpoint(device):
    if not REFERENCE_CKPT.is_file():
        raise FileNotFoundError(f"Warm-start checkpoint missing: {REFERENCE_CKPT}")
    state = torch.load(REFERENCE_CKPT, map_location=device, weights_only=False)
    if not isinstance(state.get("model"), dict):
        raise RuntimeError("Warm-start checkpoint has no model state_dict")
    return state


def _build_model(rqtrain, device, fixed_c):
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
        residual_layer_norms=[1.0] * rqtrain.N_LAYERS,
        fixed_layer_curvatures=fixed_c,
    ).to(device)
    expected_missing = {f"layers.{index}._fixed_c" for index in range(rqtrain.N_LAYERS)}
    transfer = {
        name: value
        for name, value in _load_checkpoint(device)["model"].items()
        if not name.endswith(".c_layer_scale")
    }
    missing, unexpected = model.load_state_dict(transfer, strict=False)
    if set(missing) != expected_missing or unexpected:
        raise RuntimeError(
            f"MVG warm-start mismatch: missing={sorted(missing)} "
            f"expected_missing={sorted(expected_missing)} "
            f"unexpected={sorted(unexpected)}"
        )
    return model


def _fixed_c_values(model):
    return [float(layer._fixed_c.detach().cpu().item()) for layer in model.layers]


def _check_layer_buffers(model, fixed_c):
    """Hard contract — c_layer_scale must be absent; _fixed_c present."""
    for layer_index, layer in enumerate(model.layers):
        if not hasattr(layer, "_fixed_c"):
            raise RuntimeError(
                f"MVG FAIL: layer {layer_index} missing _fixed_c buffer"
            )
        if hasattr(layer, "c_layer_scale") and isinstance(
            layer.c_layer_scale, torch.nn.Parameter
        ):
            raise RuntimeError(
                f"MVG FAIL: layer {layer_index} c_layer_scale is a learnable Parameter"
            )
    if not np.allclose(_fixed_c_values(model), fixed_c, rtol=0.0, atol=INVARIANCE_TOL):
        raise RuntimeError(
            f"MVG FAIL: _fixed_c buffers differ from closed-form target "
            f"{fixed_c} vs {_fixed_c_values(model)}"
        )
    return _fixed_c_values(model)


def _check_invariance_across_steps(model, fixed_c, fixed_state_dict):
    """Snapshot state, set step in {0, 25k, 50k, 100k}, verify c is identical."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as handle:
        ckpt = Path(handle.name)
    try:
        torch.save({"model": fixed_state_dict}, ckpt)
        roundtrip = torch.load(ckpt, map_location=next(model.parameters()).device, weights_only=False)
        model.load_state_dict(roundtrip["model"], strict=True)
    finally:
        ckpt.unlink(missing_ok=True)
    snapshots = {}
    for step in INVARIANCE_STEPS:
        model.set_curriculum_step(step)
        cs = _fixed_c_values(model)
        if not np.allclose(cs, fixed_c, rtol=0.0, atol=INVARIANCE_TOL):
            raise RuntimeError(
                f"MVG FAIL: step={step} c drifted: live={cs} target={fixed_c} "
                f"tol={INVARIANCE_TOL}"
            )
        snapshots[step] = cs
    return snapshots


def _check_gradients(model, batch):
    model.train()
    output = model(batch)
    if not output.loss.requires_grad or output.loss.grad_fn is None:
        raise RuntimeError("MVG FAIL: total loss detached")
    if not torch.isfinite(output.loss).all().item():
        raise RuntimeError("MVG FAIL: total loss non-finite")
    if output.behavior_loss.detach().item() <= 0.0:
        raise RuntimeError("MVG FAIL: behavior loss inactive")
    if output.curvature_regularization.detach().item() != 0.0:
        raise RuntimeError(
            f"MVG FAIL: curvature_reg must be 0 for fixed c, got "
            f"{float(output.curvature_regularization):.6f}"
        )
    grads = torch.autograd.grad(
        output.loss,
        tuple(model.parameters()),
        retain_graph=True,
        allow_unused=True,
    )
    if not any(
        gradient is not None and gradient.abs().sum().item() > GRAD_EPSILON
        for gradient in grads
    ):
        raise RuntimeError("MVG FAIL: no parameter receives a non-zero gradient")


def _check_optimizer_updates(model):
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    before = [parameter.detach().clone() for parameter in trainable]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=1e-4)
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        output = model(_SEEN_BATCH)
        output.loss.backward()
        optimizer.step()
    rel_updates = [
        float((parameter.detach() - old).norm().item() / max(old.norm().item(), 1e-12))
        for parameter, old in zip(trainable, before)
    ]
    if any(value <= RELATIVE_UPDATE_EPSILON for value in rel_updates):
        raise RuntimeError(
            f"MVG FAIL: 5-step relative L2 updates too small: {rel_updates}"
        )
    return rel_updates


def _check_finite_baseline(rqtrain, fixed_c):
    torch.manual_seed(rqtrain.SEED)
    np.random.seed(rqtrain.SEED)
    model = _build_model(rqtrain, torch.device("cuda:0"), fixed_c)
    _check_invariance_across_steps(
        model, fixed_c, model.state_dict()
    )
    return _fixed_c_values(model)


_SEEN_BATCH = None


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("MVG requires CUDA")
    rqtrain = _load_training_module()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(rqtrain.SEED)
    np.random.seed(rqtrain.SEED)

    fixed_c = rqtrain._load_closed_form_curvatures()
    if not np.allclose(fixed_c, EXPECTED_FIXED_C, rtol=0.0, atol=1e-9):
        raise RuntimeError(
            f"MVG FAIL: closed_form_c_l changed unexpectedly: "
            f"live={fixed_c} expected={EXPECTED_FIXED_C}"
        )

    batch = _load_batch(rqtrain, device)
    global _SEEN_BATCH
    _SEEN_BATCH = batch

    model = _build_model(rqtrain, device, fixed_c)
    initial_c = _check_layer_buffers(model, fixed_c)
    snapshots = _check_invariance_across_steps(model, fixed_c, model.state_dict())
    _check_gradients(model, batch)
    rel_updates = _check_optimizer_updates(model)
    final_c = _check_finite_baseline(rqtrain, fixed_c)

    print("MVG PASS")
    print(f"warm_start=PASS checkpoint={REFERENCE_CKPT}")
    print(f"closed_form_c_l={fixed_c}")
    print(f"initial_c_live={initial_c}")
    print(f"invariance_snapshots={snapshots}")
    print(f"final_c_live={final_c}")
    print(f"five_step_relative_updates={rel_updates}")


if __name__ == "__main__":
    main()