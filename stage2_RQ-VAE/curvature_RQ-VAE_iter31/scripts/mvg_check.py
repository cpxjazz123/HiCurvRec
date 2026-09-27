"""Single-checkpoint, single-batch HRA Step6 effect and gradient check."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = SCRIPT_DIR.parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))
TRAIN_SCRIPT = SOURCE_DIR / "curvature_RQ-VAE.py"
REFERENCE_CKPT = Path(
    "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/"
    "curvature_RQ-VAE_iter8/out/rqvae/instruments/rqvae_best.pth"
)
GRAD_EPSILON = 1e-12
INVARIANCE_TOL = 1e-6
INVARIANCE_STEPS = (0, 25_000, 50_000, 100_000)
EXPECTED_FIXED_C = [1.3660953164241916, 0.7347829661951981, 0.6439958072706683]


def _load_training_module():
    spec = importlib.util.spec_from_file_location("rqtrain_iter31", TRAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Iter31 training entry: {TRAIN_SCRIPT}")
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


def _build_model(rqtrain, device, fixed_c, checkpoint_state):
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
    expected_missing = {
        f"layers.{index}._fixed_c" for index in range(rqtrain.N_LAYERS)
    }
    transfer = {
        name: value
        for name, value in checkpoint_state["model"].items()
        if not name.endswith((".c_layer_scale", "._fixed_c"))
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
    """Verify FCCR values are fixed buffers, not model parameters."""
    parameter_ids = {id(parameter) for parameter in model.parameters()}
    buffer_names = {name for name, _ in model.named_buffers()}
    for layer_index, layer in enumerate(model.layers):
        name = f"layers.{layer_index}._fixed_c"
        if not hasattr(layer, "_fixed_c") or name not in buffer_names:
            raise RuntimeError(f"MVG FAIL: layer {layer_index} lacks its _fixed_c buffer")
        if layer._fixed_c.requires_grad or id(layer._fixed_c) in parameter_ids:
            raise RuntimeError(
                f"MVG FAIL: layer {layer_index} fixed curvature is trainable"
            )
    live = _fixed_c_values(model)
    if not np.allclose(live, fixed_c, rtol=0.0, atol=INVARIANCE_TOL):
        raise RuntimeError(
            f"MVG FAIL: fixed curvature differs from contract: {live} vs {fixed_c}"
        )
    return live


def _check_optimizer_exclusion(model):
    """Mirror Stage2's AdamW(model.parameters()) and check fixed buffers are absent."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    for layer_index, layer in enumerate(model.layers):
        if id(layer._fixed_c) in optimizer_parameter_ids:
            raise RuntimeError(
                f"MVG FAIL: layer {layer_index} fixed curvature entered AdamW"
            )
    return {
        "optimizer": "AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)",
        "fixed_curvature_excluded": True,
        "optimizer_step_performed": False,
    }


def _check_invariance_across_steps(model, fixed_c):
    """Check fixed buffers and get_c() across train/eval and curriculum probes."""
    snapshots = {}
    for mode in ("train", "eval"):
        getattr(model, mode)()
        for step in INVARIANCE_STEPS:
            model.set_curriculum_step(step)
            buffers = _fixed_c_values(model)
            live = [float(layer.get_c().detach().cpu().item()) for layer in model.layers]
            if (
                not np.allclose(buffers, fixed_c, rtol=0.0, atol=INVARIANCE_TOL)
                or not np.allclose(live, fixed_c, rtol=0.0, atol=INVARIANCE_TOL)
            ):
                raise RuntimeError(
                    f"MVG FAIL: mode={mode} step={step} "
                    f"buffers={buffers} get_c={live} target={fixed_c}"
                )
            snapshots[f"{mode}:{step}"] = {"buffer": buffers, "get_c": live}
    return snapshots


def _exp_projection_stats(tangent, curvature):
    sqrt_c = curvature.sqrt()
    norm = tangent.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(tangent.dtype).eps)
    scaled_norm = sqrt_c * norm
    raw_exp = (torch.tanh(scaled_norm) / scaled_norm) * tangent
    max_norm = (1.0 - 1e-6) / sqrt_c
    raw_norm = raw_exp.norm(dim=-1, keepdim=True)
    return {
        "projected_count": int((raw_norm > max_norm).sum().item()),
        "sample_count": int(raw_norm.numel()),
        "max_preprojection_radius_fraction": float((raw_norm / max_norm).max().item()),
    }


def _log_clamp_stats(point, curvature):
    sqrt_c = curvature.sqrt()
    norm = point.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(point.dtype).eps)
    scaled_norm = sqrt_c * norm
    return {
        "clamped_count": int((scaled_norm > 1.0 - 1e-5).sum().item()),
        "sample_count": int(scaled_norm.numel()),
        "max_scaled_norm": float(scaled_norm.max().item()),
    }


def _mobius_denominator_stats(x, y, curvature):
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    denominator = 1 + 2 * curvature * xy + (curvature**2) * x2 * y2
    floor = torch.finfo(denominator.dtype).eps
    return {
        "clamped_count": int((denominator < floor).sum().item()),
        "sample_count": int(denominator.numel()),
        "minimum_raw_denominator": float(denominator.min().item()),
    }


def _check_direct_step6_effect(model, batch):
    """Compare both Step6 outputs on one identical quantized tensor and record domains."""
    from modules.hyperbolic import _expmap0_t, _logmap0_t, _mobius_add_t

    model.eval()
    model.set_curriculum_step(0)
    with torch.no_grad():
        quantized = model.get_semantic_ids(batch.x)
        embeddings = quantized.embeddings
        if embeddings.shape != (3, model.embed_dim, batch.x.shape[0]):
            raise RuntimeError(
                f"MVG FAIL: quantized embeddings have unexpected shape {tuple(embeddings.shape)}"
            )
        z_hra = model._step6_sum_embeddings(quantized)
        z_euclidean = embeddings.sum(dim=0).transpose(0, 1)

        curvatures = [layer.get_c().view(1, 1) for layer in model.layers]
        c0 = curvatures[0]
        common_points = []
        exp_projection = {"source": [], "common_reference": []}
        log_clamp = {"source_curvature": [], "common_reference": []}
        for layer_index in (0, 1, 2):
            e_l = embeddings[layer_index].transpose(0, 1)
            exp_projection["source"].append(
                _exp_projection_stats(e_l, curvatures[layer_index])
            )
            q_l = _expmap0_t(e_l, curvatures[layer_index])
            log_clamp["source_curvature"].append(
                _log_clamp_stats(q_l, curvatures[layer_index])
            )
            source_tangent = _logmap0_t(q_l, curvatures[layer_index])
            exp_projection["common_reference"].append(
                _exp_projection_stats(source_tangent, c0)
            )
            common_points.append(_expmap0_t(source_tangent, c0))
        if any(not torch.isfinite(point).all().item() for point in common_points):
            raise RuntimeError("MVG FAIL: common-reference point is non-finite")

        inner_denominator = _mobius_denominator_stats(
            common_points[1], common_points[2], c0
        )
        inner = _mobius_add_t(common_points[1], common_points[2], c0)
        outer_denominator = _mobius_denominator_stats(
            common_points[0], inner, c0
        )
        h = _mobius_add_t(common_points[0], inner, c0)
        if not torch.isfinite(inner).all().item():
            raise RuntimeError("MVG FAIL: inner Möbius aggregate is non-finite")
        log_clamp["common_reference"].append(_log_clamp_stats(h, c0))
        z_reconstructed = _logmap0_t(h, c0)
        if not torch.equal(z_hra, z_reconstructed):
            raise RuntimeError("MVG FAIL: observed helper path differs from Step6 output")

        for label, value in (("HRA", z_hra), ("Euclidean", z_euclidean), ("common_ball", h)):
            if not torch.isfinite(value).all().item():
                raise RuntimeError(f"MVG FAIL: {label} output is non-finite")
        if z_hra.shape != z_euclidean.shape or z_hra.shape != (
            batch.x.shape[0], model.embed_dim
        ):
            raise RuntimeError("MVG FAIL: Step6 output shape differs from the decoder contract")

        delta = z_hra - z_euclidean
        sqrt_c0 = c0.sqrt()
        scaled_h_norm = sqrt_c0 * h.norm(dim=-1)
        scaled_inner_norm = sqrt_c0 * inner.norm(dim=-1)
        inner_domain_valid = bool((scaled_inner_norm < 1.0).all().item())
        common_ball_domain_valid = bool((scaled_h_norm < 1.0).all().item())
        if not inner_domain_valid or not common_ball_domain_valid:
            raise RuntimeError(
                "MVG FAIL: Möbius aggregate lies outside the common-curvature ball"
            )
        direct_effect = {
            "same_quantized_embeddings": True,
            "shape": list(z_hra.shape),
            "hra_finite": bool(torch.isfinite(z_hra).all().item()),
            "euclidean_finite": bool(torch.isfinite(z_euclidean).all().item()),
            "hra_norm": float(z_hra.norm().item()),
            "euclidean_norm": float(z_euclidean.norm().item()),
            "max_abs_difference": float(delta.abs().max().item()),
            "l2_difference": float(delta.norm().item()),
            "relative_l2_difference": float(
                (delta.norm() / z_euclidean.norm().clamp_min(torch.finfo(z_euclidean.dtype).eps)).item()
            ),
            "direct_effect_threshold": "none_preregistered; report_only",
            "common_ball_max_sqrt_c_norm": float(scaled_h_norm.max().item()),
            "inner_ball_domain_valid": inner_domain_valid,
            "common_ball_domain_valid": common_ball_domain_valid,
            "exp_projection_incidence": exp_projection,
            "log_clamp_incidence": log_clamp,
            "mobius_denominator_incidence": {
                "inner_l1_l2": inner_denominator,
                "outer_l0_inner": outer_denominator,
            },
        }
    return direct_effect


def _check_gradients(model, batch):
    model.train()
    model.zero_grad(set_to_none=True)
    output = model(batch)
    loss = output.loss
    if not loss.requires_grad or loss.grad_fn is None:
        raise RuntimeError("MVG FAIL: total loss is detached")
    if not torch.isfinite(loss).all().item():
        raise RuntimeError("MVG FAIL: total loss is non-finite")
    if output.curvature_regularization.detach().item() != 0.0:
        raise RuntimeError(
            f"MVG FAIL: fixed-curvature regularization must be zero, got "
            f"{float(output.curvature_regularization):.6f}"
        )

    named_parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    parameters = [parameter for _, parameter in named_parameters]
    components = {
        "reconstruction": (output.reconstruction_loss.mean(), "encoder."),
        "quantizer": (output.rqvae_loss.mean(), "layers."),
        "behavior_loss": (output.behavior_loss, None),
    }
    component_gradients = {}
    for label, (component, expected_prefix) in components.items():
        if not component.requires_grad or component.grad_fn is None:
            raise RuntimeError(f"MVG FAIL: {label} loss is detached")
        if not torch.isfinite(component).all().item():
            raise RuntimeError(f"MVG FAIL: {label} loss is non-finite")
        if label == "behavior_loss" and component.ndim != 0:
            raise RuntimeError("MVG FAIL: behavior_loss must be a scalar")
        gradients = torch.autograd.grad(
            component, parameters, retain_graph=True, allow_unused=True
        )
        nonzero = []
        for (name, _), gradient in zip(named_parameters, gradients):
            if gradient is None:
                continue
            if not torch.isfinite(gradient).all().item():
                raise RuntimeError(f"MVG FAIL: non-finite {label} gradient on {name}")
            if gradient.abs().sum().item() > GRAD_EPSILON:
                nonzero.append(name)
        if expected_prefix is None:
            has_expected_gradient = bool(nonzero)
        else:
            has_expected_gradient = any(
                name.startswith(expected_prefix) for name in nonzero
            )
        if not has_expected_gradient:
            target = (
                "any trainable model parameter"
                if expected_prefix is None
                else f"{expected_prefix} parameters"
            )
            raise RuntimeError(
                f"MVG FAIL: {label} loss has no nonzero gradient in {target}"
            )
        component_gradients[label] = nonzero

    loss.backward()
    nonzero_names = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all().item():
            raise RuntimeError(f"MVG FAIL: non-finite total-loss gradient on {name}")
        if parameter.grad.abs().sum().item() > GRAD_EPSILON:
            nonzero_names.append(name)
    if not any(name.startswith("encoder.") for name in nonzero_names):
        raise RuntimeError("MVG FAIL: total loss has no nonzero encoder gradient")
    if not any(name.startswith("layers.") and ".embedding.weight" in name for name in nonzero_names):
        raise RuntimeError("MVG FAIL: total loss has no nonzero codebook gradient")
    return {
        "total_loss": float(loss.detach().item()),
        "nonzero_total_gradient_parameters": nonzero_names,
        "component_nonzero_gradient_parameters": component_gradients,
        "hra_specific_auxiliary_loss": False,
    }


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
            f"MVG FAIL: closed_form_c_l changed: live={fixed_c} expected={EXPECTED_FIXED_C}"
        )
    checkpoint_state = _load_checkpoint(device)
    batch = _load_batch(rqtrain, device)
    model = _build_model(rqtrain, device, fixed_c, checkpoint_state)
    initial_c = _check_layer_buffers(model, fixed_c)
    snapshots = _check_invariance_across_steps(model, fixed_c)
    direct_effect = _check_direct_step6_effect(model, batch)
    gradients = _check_gradients(model, batch)
    after_gradients_c = _check_layer_buffers(model, fixed_c)
    optimizer_exclusion = _check_optimizer_exclusion(model)

    print("MVG PASS")
    print(f"warm_start=PASS checkpoint={REFERENCE_CKPT}")
    print(f"closed_form_c_l={fixed_c}")
    print(f"initial_c_live={initial_c}")
    print(f"invariance_snapshots={snapshots}")
    print(f"direct_step6_effect={direct_effect}")
    print(f"gradients={gradients}")
    print(f"after_gradient_c={after_gradients_c}")
    print(f"optimizer_exclusion={optimizer_exclusion}")


if __name__ == "__main__":
    main()
