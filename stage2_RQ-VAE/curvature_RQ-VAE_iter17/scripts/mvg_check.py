"""Four-layer Mechanism Verification Gate for iter17's curvature-metric AdamW."""
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
RIEMANNIAN_EPSILON = 1e-3
BASE_LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
COUNTERFACTUAL_STEPS = 200
GRAD_EPSILON = 1e-12


def _load_training_module():
    spec = importlib.util.spec_from_file_location("rqtrain_iter17", TRAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load training entry: {TRAIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_optimizer(model, n_layers):
    layer_parameters = [[] for _ in range(n_layers)]
    other_parameters = []
    for name, parameter in model.named_parameters():
        if name.startswith("layers."):
            layer_index = int(name.split(".")[1])
            layer_parameters[layer_index].append(parameter)
        else:
            other_parameters.append(parameter)
    if any(not parameters for parameters in layer_parameters):
        raise RuntimeError("MVG FAIL: a quantizer layer has no optimizer parameters")

    groups = [
        {
            "params": parameters,
            "lr": BASE_LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "riemannian_layer_index": layer_index,
        }
        for layer_index, parameters in enumerate(layer_parameters)
    ]
    if other_parameters:
        groups.append(
            {
                "params": other_parameters,
                "lr": BASE_LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "riemannian_layer_index": None,
            }
        )
    return torch.optim.AdamW(groups)


def _curvature_scales(model, optimizer, reference_inverse, step, use_metric=True):
    model.set_curriculum_step(step)
    curvatures = [float(layer.get_c().detach().item()) for layer in model.layers]
    inverse = [1.0 / (value + RIEMANNIAN_EPSILON) for value in curvatures]
    multipliers = [value / reference_inverse for value in inverse]
    for group in optimizer.param_groups:
        layer_index = group["riemannian_layer_index"]
        multiplier = (
            multipliers[layer_index]
            if use_metric and layer_index is not None
            else 1.0
        )
        group["lr"] = BASE_LEARNING_RATE * multiplier
    return curvatures, multipliers


def _layer_relative_updates(model, before, n_layers):
    results = []
    named_parameters = dict(model.named_parameters())
    for layer_index in range(n_layers):
        names = [name for name in before if name.startswith(f"layers.{layer_index}.")]
        before_norm_sq = sum(before[name].double().square().sum() for name in names)
        delta_norm_sq = sum(
            (named_parameters[name].detach() - before[name]).double().square().sum()
            for name in names
        )
        before_norm = float(torch.sqrt(before_norm_sq).item())
        delta_norm = float(torch.sqrt(delta_norm_sq).item())
        relative = delta_norm / max(before_norm, GRAD_EPSILON)
        results.append((layer_index, relative))
    return results


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("MVG requires CUDA")
    rqtrain = _load_training_module()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(rqtrain.SEED)
    np.random.seed(rqtrain.SEED)

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
    batch = rqtrain._build_seq_batch(*(value.to(device) for value in raw_batch))

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
    expected_missing = {
        f"layers.{layer_index}.c_layer_scale"
        for layer_index in range(rqtrain.N_LAYERS)
    }
    transferable = {
        name: value
        for name, value in state["model"].items()
        if name not in expected_missing
    }
    missing, unexpected = model.load_state_dict(transferable, strict=False)
    if set(missing) != expected_missing or unexpected:
        raise RuntimeError(
            "MVG FAIL: warm-start mismatch "
            f"(missing={sorted(missing)}, unexpected={sorted(unexpected)})"
        )

    model.set_curriculum_step(0)
    initial_curvatures = [float(layer.get_c().detach().item()) for layer in model.layers]
    initial_inverse = [
        1.0 / (value + RIEMANNIAN_EPSILON) for value in initial_curvatures
    ]
    reference_inverse = sum(initial_inverse) / len(initial_inverse)
    if not np.isfinite(reference_inverse) or reference_inverse <= 0:
        raise RuntimeError("MVG FAIL: invalid initial metric reference")

    initial_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    with tempfile.NamedTemporaryFile(
        prefix="rqvae_iter17_mvg_", suffix=".pt", dir=SCRIPT_DIR, delete=False
    ) as handle:
        checkpoint = Path(handle.name)
    try:
        torch.save({"model": initial_state}, checkpoint)
        roundtrip = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(roundtrip["model"], strict=True)
    finally:
        checkpoint.unlink(missing_ok=True)

    # Gate 1 — total-loss computation graph.
    model.train()
    output = model(batch)
    if not output.loss.requires_grad or output.loss.grad_fn is None:
        raise RuntimeError("MVG FAIL: total loss is detached")
    if not torch.isfinite(output.loss).item():
        raise RuntimeError("MVG FAIL: total loss is non-finite")
    if not torch.isfinite(output.behavior_loss).item() or output.behavior_loss.item() <= 0:
        raise RuntimeError("MVG FAIL: behavior contrastive loss is inactive")

    # Gate 2 — nonzero gradients through every trainable quantizer-layer tensor.
    layer_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.startswith("layers.") and not name.endswith(".c_layer_scale")
    ]
    if not layer_parameters:
        raise RuntimeError("MVG FAIL: no quantizer-layer parameters found")
    gradients = torch.autograd.grad(
        output.loss,
        [parameter for _, parameter in layer_parameters],
        retain_graph=True,
        allow_unused=True,
    )
    gradient_norms = {}
    for (name, _), gradient in zip(layer_parameters, gradients):
        if gradient is None or not torch.isfinite(gradient).all().item():
            raise RuntimeError(f"MVG FAIL: {name} gradient is missing or non-finite")
        norm = float(torch.linalg.vector_norm(gradient).item())
        if norm <= GRAD_EPSILON:
            raise RuntimeError(f"MVG FAIL: {name} gradient is zero")
        gradient_norms[name] = norm

    # Gate 3 — five optimizer steps must update every quantizer layer materially.
    model.load_state_dict(initial_state, strict=True)
    model.train()
    optimizer = _build_optimizer(model, rqtrain.N_LAYERS)
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name.startswith("layers.")
    }
    for step in range(5):
        model.set_curriculum_step(step)
        optimizer.zero_grad(set_to_none=True)
        step_output = model(batch)
        step_output.loss.backward()
        if rqtrain.GRAD_CLIP_NORM > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), rqtrain.GRAD_CLIP_NORM
            )
        _curvature_scales(model, optimizer, reference_inverse, step)
        optimizer.step()
    update_ratios = _layer_relative_updates(model, before, rqtrain.N_LAYERS)
    for layer_index, relative in update_ratios:
        if not np.isfinite(relative) or relative <= 1e-7:
            raise RuntimeError(
                f"MVG FAIL: layer{layer_index} five-step relative update "
                f"{relative:.3e} is not > 1e-7"
            )

    # Gate 4 — same-seed 200-step ON/OFF counterfactual must change total loss.
    def run_counterfactual(use_metric):
        model.load_state_dict(initial_state, strict=True)
        model.train()
        optimizer = _build_optimizer(model, rqtrain.N_LAYERS)
        torch.manual_seed(rqtrain.SEED)
        np.random.seed(rqtrain.SEED)
        for step in range(COUNTERFACTUAL_STEPS):
            model.set_curriculum_step(step)
            optimizer.zero_grad(set_to_none=True)
            step_output = model(batch)
            if not torch.isfinite(step_output.loss).item():
                raise RuntimeError("MVG FAIL: counterfactual loss is non-finite")
            step_output.loss.backward()
            if rqtrain.GRAD_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), rqtrain.GRAD_CLIP_NORM
                )
            _curvature_scales(
                model, optimizer, reference_inverse, step, use_metric=use_metric
            )
            optimizer.step()
        model.set_curriculum_step(COUNTERFACTUAL_STEPS)
        model.eval()
        with torch.no_grad():
            final_output = model(batch)
        if not torch.isfinite(final_output.loss).item():
            raise RuntimeError("MVG FAIL: final counterfactual loss is non-finite")
        return (
            float(final_output.loss.item()),
            float(final_output.behavior_loss.item()),
        )

    loss_on, behavior_on = run_counterfactual(use_metric=True)
    loss_off, behavior_off = run_counterfactual(use_metric=False)
    loss_delta = abs(loss_on - loss_off)
    if loss_delta <= 1e-6:
        raise RuntimeError(
            f"MVG FAIL: 200-step ON/OFF total-loss delta {loss_delta:.3e} "
            "does not exceed 1e-6"
        )
    model.load_state_dict(initial_state, strict=True)

    # Verify direct curvature effect at both ends and midpoint of the cycle.
    multiplier_snapshots = {}
    curvature_snapshots = {}
    for step in (0, rqtrain.C_CYCLIC_PERIOD // 2, rqtrain.C_CYCLIC_PERIOD):
        model.set_curriculum_step(step)
        curvatures = [float(layer.get_c().detach().item()) for layer in model.layers]
        multipliers = [
            (1.0 / (value + RIEMANNIAN_EPSILON)) / reference_inverse
            for value in curvatures
        ]
        if not all(np.isfinite(value) and value > 0 for value in multipliers):
            raise RuntimeError(f"MVG FAIL: invalid metric scales at step {step}")
        curvature_snapshots[step] = curvatures
        multiplier_snapshots[step] = multipliers
    initial_mean = sum(multiplier_snapshots[0]) / rqtrain.N_LAYERS
    if abs(initial_mean - 1.0) > 1e-6:
        raise RuntimeError("MVG FAIL: step-0 multipliers are not normalized to mean 1")
    midpoint = rqtrain.C_CYCLIC_PERIOD // 2
    if not any(
        abs(a - b) > 1e-3
        for a, b in zip(multiplier_snapshots[0], multiplier_snapshots[midpoint])
    ):
        raise RuntimeError("MVG FAIL: metric multipliers ignore the cyclic schedule")

    print("MVG PASS")
    print(
        f"batch_shape={tuple(batch.x.shape)} "
        f"active_pairs={int(batch.ids.ne(batch.ids_fut).sum().item())}"
    )
    print(f"initial_c_l={[round(value, 6) for value in initial_curvatures]}")
    print(f"reference_mean_inverse_c={reference_inverse:.6f}")
    print("DE-1 c_l / m_l by curriculum step:")
    for step in (0, midpoint, rqtrain.C_CYCLIC_PERIOD):
        print(
            f"  step={step}: c={curvature_snapshots[step]} "
            f"m={multiplier_snapshots[step]}"
        )
    print("DE-2 gradient L2 norms by layer:")
    for layer_index in range(rqtrain.N_LAYERS):
        values = [
            value
            for name, value in gradient_norms.items()
            if name.startswith(f"layers.{layer_index}.")
        ]
        print(f"  layer{layer_index}: {values}")
    print(f"DE-3 five-step relative updates={update_ratios}")
    print(
        f"DE-4 200-step counterfactual: L_on={loss_on:.8f} "
        f"L_off={loss_off:.8f} delta={loss_delta:.3e} "
        f"behavior_on={behavior_on:.8f} behavior_off={behavior_off:.8f}"
    )


if __name__ == "__main__":
    main()
