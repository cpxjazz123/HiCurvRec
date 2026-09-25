"""Mechanism verification for iter18's curvature-conditioned AdamW beta2."""
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
COUNTERFACTUAL_STEPS = 200
GRAD_EPSILON = 1e-12


def _load_training_module():
    spec = importlib.util.spec_from_file_location("rqtrain_iter18", TRAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load training entry: {TRAIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_optimizer_contract(optimizer, model, rqtrain, expected_layer_beta2):
    model_parameter_ids = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    grouped_parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
        if parameter.requires_grad
    ]
    grouped_ids = [id(parameter) for parameter in grouped_parameters]
    if len(grouped_ids) != len(set(grouped_ids)):
        raise RuntimeError("MVG FAIL: optimizer parameters occur in multiple groups")
    if set(grouped_ids) != model_parameter_ids:
        raise RuntimeError("MVG FAIL: optimizer does not exactly cover trainable parameters")

    observed_layers = set()
    for group in optimizer.param_groups:
        layer_index = group["curvature_layer_index"]
        beta1, beta2 = group["betas"]
        expected = (
            expected_layer_beta2[layer_index]
            if layer_index is not None
            else rqtrain.ADAMW_BASE_BETA2
        )
        if layer_index is not None:
            if layer_index in observed_layers:
                raise RuntimeError("MVG FAIL: duplicate quantizer optimizer group")
            observed_layers.add(layer_index)
        if abs(beta1 - rqtrain.ADAMW_BETA1) > 1e-12:
            raise RuntimeError("MVG FAIL: unexpected AdamW beta1")
        if abs(beta2 - expected) > 1e-12:
            raise RuntimeError(
                f"MVG FAIL: layer {layer_index} beta2={beta2} expected {expected}"
            )
        if abs(group["lr"] - rqtrain.ADAMW_BASE_LR) > 1e-12:
            raise RuntimeError("MVG FAIL: learning rate is not the fixed base rate")
        if abs(group["eps"] - rqtrain.ADAMW_EPS) > 1e-18:
            raise RuntimeError("MVG FAIL: unexpected AdamW epsilon")
        if abs(group["weight_decay"] - rqtrain.ADAMW_WEIGHT_DECAY) > 1e-12:
            raise RuntimeError("MVG FAIL: unexpected AdamW weight decay")
    if observed_layers != set(range(len(expected_layer_beta2))):
        raise RuntimeError("MVG FAIL: quantizer optimizer groups do not cover every layer")


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
        results.append((layer_index, delta_norm / max(before_norm, GRAD_EPSILON)))
    return results


def _loss_gradient_norm(loss_item, parameters):
    gradients = torch.autograd.grad(
        loss_item, parameters, retain_graph=True, allow_unused=True
    )
    squared_norm = sum(
        float(gradient.detach().double().square().sum().item())
        for gradient in gradients
        if gradient is not None
    )
    return float(np.sqrt(squared_norm))


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

    optimizer_template, initial_curvatures, initial_u, initial_beta2 = (
        rqtrain.build_curvature_conditioned_adamw(model)
    )
    _assert_optimizer_contract(
        optimizer_template, model, rqtrain, initial_beta2
    )
    if not all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in initial_u):
        raise RuntimeError("MVG FAIL: initial normalized curvature is out of range")
    if not all(np.isfinite(value) and 0.990 <= value <= 0.999 for value in initial_beta2):
        raise RuntimeError("MVG FAIL: initial beta2 is out of range")
    for i, curvature_i in enumerate(initial_curvatures):
        for j, curvature_j in enumerate(initial_curvatures):
            if curvature_i > curvature_j and initial_beta2[i] > initial_beta2[j] + 1e-12:
                raise RuntimeError("MVG FAIL: beta2 is not monotone with curvature")

    initial_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    with tempfile.NamedTemporaryFile(
        prefix="rqvae_iter18_mvg_", suffix=".pt", dir=SCRIPT_DIR, delete=False
    ) as handle:
        checkpoint = Path(handle.name)
    try:
        torch.save({"model": initial_state}, checkpoint)
        roundtrip = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(roundtrip["model"], strict=True)
    finally:
        checkpoint.unlink(missing_ok=True)

    # One checkpoint / one batch: total and component loss gradient paths.
    model.train()
    output = model(batch)
    if not output.loss.requires_grad or output.loss.grad_fn is None:
        raise RuntimeError("MVG FAIL: total loss is detached")
    if not torch.isfinite(output.loss).item():
        raise RuntimeError("MVG FAIL: total loss is non-finite")
    if not torch.isfinite(output.behavior_loss).item() or output.behavior_loss.item() <= 0:
        raise RuntimeError("MVG FAIL: behavior contrastive loss is inactive")

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    component_norms = {}
    for name, loss_item, require_nonzero in (
        ("reconstruction_loss", output.reconstruction_loss, True),
        ("rqvae_loss", output.rqvae_loss, True),
        ("behavior_loss", output.behavior_loss, True),
        ("curvature_regularization", output.curvature_regularization, False),
    ):
        if (
            not loss_item.requires_grad
            or loss_item.grad_fn is None
            or not torch.isfinite(loss_item).all().item()
        ):
            raise RuntimeError(f"MVG FAIL: {name} is detached or non-finite")
        norm = _loss_gradient_norm(loss_item, parameters)
        if not np.isfinite(norm) or (require_nonzero and norm <= GRAD_EPSILON):
            raise RuntimeError(
                f"MVG FAIL: {name} gradient norm {norm:.3e}; "
                f"nonzero required={require_nonzero}"
            )
        component_norms[name] = norm

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
    output.loss.backward()
    rqtrain.check_step9_backward(model)
    for layer_index in range(rqtrain.N_LAYERS):
        layer_norm_sq = sum(
            float(parameter.grad.detach().double().square().sum().item())
            for name, parameter in model.named_parameters()
            if name.startswith(f"layers.{layer_index}.") and parameter.grad is not None
        )
        if not np.isfinite(layer_norm_sq) or layer_norm_sq <= GRAD_EPSILON:
            raise RuntimeError(f"MVG FAIL: layer{layer_index} total-loss gradient is zero")

    # Five optimizer steps: every quantizer layer must move.
    model.load_state_dict(initial_state, strict=True)
    model.train()
    optimizer, _, _, _ = rqtrain.build_curvature_conditioned_adamw(model)
    _assert_optimizer_contract(optimizer, model, rqtrain, initial_beta2)
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), rqtrain.GRAD_CLIP_NORM)
        optimizer.step()
    update_ratios = _layer_relative_updates(model, before, rqtrain.N_LAYERS)
    for layer_index, relative in update_ratios:
        if not np.isfinite(relative) or relative <= 1e-7:
            raise RuntimeError(
                f"MVG FAIL: layer{layer_index} five-step relative update "
                f"{relative:.3e} is not > 1e-7"
            )

    def run_counterfactual(use_conditioned_beta2):
        model.load_state_dict(initial_state, strict=True)
        model.train()
        optimizer, _, _, _ = rqtrain.build_curvature_conditioned_adamw(
            model, use_conditioned_beta2=use_conditioned_beta2
        )
        expected_beta2 = (
            initial_beta2
            if use_conditioned_beta2
            else [rqtrain.ADAMW_BASE_BETA2] * rqtrain.N_LAYERS
        )
        _assert_optimizer_contract(optimizer, model, rqtrain, expected_beta2)
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
            optimizer.step()
        model.set_curriculum_step(COUNTERFACTUAL_STEPS)
        model.eval()
        with torch.no_grad():
            final_output = model(batch)
        if not torch.isfinite(final_output.loss).item():
            raise RuntimeError("MVG FAIL: final counterfactual loss is non-finite")
        return float(final_output.loss.item()), float(final_output.behavior_loss.item())

    loss_on, behavior_on = run_counterfactual(use_conditioned_beta2=True)
    loss_off, behavior_off = run_counterfactual(use_conditioned_beta2=False)
    loss_delta = abs(loss_on - loss_off)
    if loss_delta <= 1e-6:
        raise RuntimeError(
            f"MVG FAIL: 200-step ON/OFF total-loss delta {loss_delta:.3e} "
            "does not exceed 1e-6"
        )

    # Confirm that the original curvature curriculum remains cyclic and finite.
    model.load_state_dict(initial_state, strict=True)
    curvature_snapshots = {}
    for step in (0, rqtrain.C_CYCLIC_PERIOD // 2, rqtrain.C_CYCLIC_PERIOD):
        model.set_curriculum_step(step)
        values = [float(layer.get_c().detach().item()) for layer in model.layers]
        if not all(np.isfinite(value) and value > 0 for value in values):
            raise RuntimeError(f"MVG FAIL: invalid curvature at step {step}")
        curvature_snapshots[step] = values
    midpoint = rqtrain.C_CYCLIC_PERIOD // 2
    if not any(
        abs(initial - middle) > 1e-3
        for initial, middle in zip(curvature_snapshots[0], curvature_snapshots[midpoint])
    ):
        raise RuntimeError("MVG FAIL: cyclic curvature is inactive")
    if not all(
        abs(initial - end) <= 1e-6
        for initial, end in zip(
            curvature_snapshots[0], curvature_snapshots[rqtrain.C_CYCLIC_PERIOD]
        )
    ):
        raise RuntimeError("MVG FAIL: curvature does not return at cycle boundary")

    print("MVG PASS")
    print(
        f"batch_shape={tuple(batch.x.shape)} "
        f"active_pairs={int(batch.ids.ne(batch.ids_fut).sum().item())}"
    )
    print(f"initial_c_l={[round(value, 6) for value in initial_curvatures]}")
    print(f"initial_u_l={[round(value, 6) for value in initial_u]}")
    print(f"beta2_l={[round(value, 6) for value in initial_beta2]}")
    print(f"DE-1 fixed group betas={[(rqtrain.ADAMW_BETA1, value) for value in initial_beta2]}")
    print(f"DE-2 component gradient norms={component_norms}")
    print(f"DE-2 quantizer parameter gradient L2 norms={gradient_norms}")
    print(f"DE-3 five-step relative updates={update_ratios}")
    print(
        f"DE-4 200-step counterfactual: L_on={loss_on:.8f} "
        f"L_off={loss_off:.8f} delta={loss_delta:.3e} "
        f"behavior_on={behavior_on:.8f} behavior_off={behavior_off:.8f}"
    )
    print(f"curvature cycle snapshots={curvature_snapshots}")


if __name__ == "__main__":
    main()
