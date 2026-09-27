"""One-checkpoint, one-real-batch MVG for Iter36 assignment energy."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = SCRIPT_DIR.parent / "curvature_RQ-VAE.py"
REFERENCE_CKPT = Path(
    "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/"
    "curvature_RQ-VAE_iter8/out/rqvae/instruments/rqvae_best.pth"
)
EXPECTED_FIXED_C = [1.3660953164241916, 0.7347829661951981, 0.6439958072706683]
GRAD_EPSILON = 1e-12


def _load_training_module():
    spec = importlib.util.spec_from_file_location("rqtrain_iter36", TRAIN_SCRIPT)
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
            f"Only {len(active_indices)} transition sources; "
            f"need {rqtrain.BATCH_SIZE}"
        )
    raw_batch = rqtrain.collate_items(
        [dataset[index] for index in active_indices[: rqtrain.BATCH_SIZE]]
    )
    return rqtrain._build_seq_batch(*(value.to(device) for value in raw_batch))


def _build_model(rqtrain, device, fixed_c, state):
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
    missing, unexpected = model.load_state_dict(
        {
            key: value
            for key, value in state["model"].items()
            if not key.endswith((".c_layer_scale", "._fixed_c"))
        },
        strict=False,
    )
    expected_missing = {f"layers.{i}._fixed_c" for i in range(rqtrain.N_LAYERS)}
    if set(missing) != expected_missing or unexpected:
        raise RuntimeError(
            f"warm-start mismatch: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    return model


def _assignment_distances(layer, x):
    from modules.hyperbolic import _expmap0_t, _poincare_distance_t

    curvature = layer.get_c()
    curvature_3d = curvature.view(1, 1, 1)
    codebook = layer.embedding.weight
    latent_h = _expmap0_t(x.unsqueeze(1), curvature_3d)
    codebook_h = _expmap0_t(
        codebook.unsqueeze(0).expand(x.shape[0], codebook.shape[0], -1),
        curvature_3d,
    )
    distances = _poincare_distance_t(
        latent_h.expand(x.shape[0], codebook.shape[0], -1),
        codebook_h,
        curvature_3d,
    ).squeeze(-1)
    if (distances.max() - distances.min()).item() <= 1e-6:
        x64 = x.to(dtype=torch.float64)
        codebook64 = codebook.to(dtype=torch.float64)
        x_sq = (x64 * x64).sum(dim=1, keepdim=True)
        codebook_sq = (codebook64 * codebook64).sum(dim=1).unsqueeze(0)
        distances = (x_sq + codebook_sq - 2.0 * (x64 @ codebook64.t())).clamp_min(0.0)
    return distances


def _effective_epsilon(layer, centered):
    current_c = layer.get_c().view(1, 1, 1)
    c_min = torch.tensor(
        layer.c_cyclic_min, device=centered.device, dtype=centered.dtype
    )
    c_max = torch.tensor(
        layer.c_cyclic_max, device=centered.device, dtype=centered.dtype
    )
    c_scale = (current_c / c_max.clamp_min(1e-12)).view(1, 1)
    return (layer.sk_eps * c_scale).clamp_min(1e-4).item()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("MVG requires CUDA")
    rqtrain = _load_training_module()
    from modules.hyperbolic import _sinkhorn_algorithm

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(rqtrain.SEED)
    np.random.seed(rqtrain.SEED)

    fixed_c = rqtrain._load_closed_form_curvatures()
    if not np.allclose(fixed_c, EXPECTED_FIXED_C, rtol=0.0, atol=1e-9):
        raise RuntimeError(f"fixed curvature changed: {fixed_c}")
    if not REFERENCE_CKPT.is_file():
        raise FileNotFoundError(REFERENCE_CKPT)
    checkpoint = torch.load(REFERENCE_CKPT, map_location=device, weights_only=False)
    if not isinstance(checkpoint.get("model"), dict):
        raise RuntimeError("warm-start checkpoint has no model state")
    model = _build_model(rqtrain, device, fixed_c, checkpoint)
    batch = _load_batch(rqtrain, device)
    model.train()
    before_curvature = [float(layer.get_c().item()) for layer in model.layers]

    paired_x = torch.cat((batch.x, batch.x_fut), dim=0)
    quantized = model.get_semantic_ids(paired_x)
    layer_evidence = []
    total_changed_ids = 0
    max_assignment_delta = 0.0
    for layer_index, layer in enumerate(model.layers):
        x = quantized.residuals[layer_index].transpose(0, 1)
        distances = _assignment_distances(layer, x).detach()
        centered_linear = layer._center_distance_for_constraint(distances)
        centered_squared = layer._center_distance_for_constraint(distances.square())
        epsilon = _effective_epsilon(layer, centered_squared)
        linear_assignments = _sinkhorn_algorithm(
            centered_linear.double(), epsilon, layer.sk_iters
        )
        squared_assignments = _sinkhorn_algorithm(
            centered_squared.double(), epsilon, layer.sk_iters
        )
        actual = layer(x)
        expected_squared_ids = torch.argmax(squared_assignments, dim=-1)
        if not torch.equal(actual.ids, expected_squared_ids):
            raise RuntimeError(f"Iter36 forward assignment mismatch at layer {layer_index}")
        changed_ids = int(
            (torch.argmax(linear_assignments, dim=-1) != actual.ids).sum().item()
        )
        assignment_delta = float(
            (linear_assignments - squared_assignments).abs().max().item()
        )
        if not np.isfinite(assignment_delta):
            raise RuntimeError(f"non-finite assignment delta at layer {layer_index}")
        total_changed_ids += changed_ids
        max_assignment_delta = max(max_assignment_delta, assignment_delta)
        layer_evidence.append(
            {"layer": layer_index, "changed_ids": changed_ids, "max_assignment_delta": assignment_delta}
        )
    if total_changed_ids == 0 or max_assignment_delta <= 1e-8:
        raise RuntimeError(
            "squared energy produced no measurable hard-assignment effect on the batch"
        )

    output = model(batch)
    if not output.loss.requires_grad or output.loss.grad_fn is None:
        raise RuntimeError("total loss has no autograd path")
    if not torch.isfinite(output.loss).item():
        raise RuntimeError("total loss is non-finite")
    output.loss.backward()
    nonzero_gradients = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all().item():
            raise RuntimeError(f"non-finite gradient: {name}")
        if parameter.grad.abs().sum().item() > GRAD_EPSILON:
            nonzero_gradients.append(name)
    if not nonzero_gradients:
        raise RuntimeError("no trainable parameter received a non-zero gradient")

    after_curvature = [float(layer.get_c().item()) for layer in model.layers]
    if before_curvature != after_curvature or not np.allclose(
        after_curvature, fixed_c, rtol=0.0, atol=1e-6
    ):
        raise RuntimeError(
            f"fixed curvature changed: before={before_curvature}, after={after_curvature}"
        )

    print("MVG PASS")
    print(f"checkpoint={REFERENCE_CKPT}")
    print(f"closed_form_c_l={fixed_c}; fixed_after_forward={after_curvature}")
    print(f"squared_energy_layers={layer_evidence}")
    print(f"total_changed_ids={total_changed_ids}; max_assignment_delta={max_assignment_delta:.8g}")
    print(f"loss={float(output.loss.detach().item()):.8g}")
    print(f"nonzero_gradient_parameters={len(nonzero_gradients)}")


if __name__ == "__main__":
    main()
