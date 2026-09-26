"""Single-checkpoint FCCR-1 formula, gradient, invariance, and activation check."""
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
GRAD_EPSILON = 1e-12
RELATIVE_UPDATE_EPSILON = 1e-7
INVARIANCE_TOL = 1e-6
COUNTERFACTUAL_ATOL = 1e-8
INVARIANCE_STEPS = (0, 25_000, 50_000, 100_000)
EXPECTED_FIXED_C = [1.3660953164241916, 0.7347829661951981, 0.6439958072706683]
CONTROL_FIXED_C = [0.6145357379232853, 0.5333020920777128, 0.3814078098431606]


def _load_training_module():
    spec = importlib.util.spec_from_file_location("rqtrain_iter29", TRAIN_SCRIPT)
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
    expected_missing = {f"layers.{index}._fixed_c" for index in range(rqtrain.N_LAYERS)}
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
    """Verify fixed curvature is a non-trainable buffer, never a parameter."""
    parameter_ids = {id(parameter) for parameter in model.parameters()}
    for layer_index, layer in enumerate(model.layers):
        if not hasattr(layer, "_fixed_c"):
            raise RuntimeError(
                f"MVG FAIL: layer {layer_index} missing _fixed_c buffer"
            )
        if layer._fixed_c.requires_grad:
            raise RuntimeError(
                f"MVG FAIL: layer {layer_index} _fixed_c requires gradients"
            )
        if id(layer._fixed_c) in parameter_ids:
            raise RuntimeError(
                f"MVG FAIL: layer {layer_index} _fixed_c is exposed as a parameter"
            )
        if hasattr(layer, "c_layer_scale") and isinstance(
            layer.c_layer_scale, torch.nn.Parameter
        ):
            raise RuntimeError(
                f"MVG FAIL: layer {layer_index} c_layer_scale is a learnable Parameter"
            )
    if not np.allclose(_fixed_c_values(model), fixed_c, rtol=0.0, atol=INVARIANCE_TOL):
        raise RuntimeError(
            f"MVG FAIL: _fixed_c buffers differ from target "
            f"{fixed_c} vs {_fixed_c_values(model)}"
        )
    return _fixed_c_values(model)


def _check_invariance_across_steps(model, fixed_c):
    """Check get_c() and fixed buffers across modes and curriculum steps."""
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


def _check_gradients(model, batch):
    model.train()
    model.zero_grad(set_to_none=True)
    output = model(batch)
    loss = output.loss
    if not loss.requires_grad or loss.grad_fn is None:
        raise RuntimeError("MVG FAIL: total loss detached")
    if not torch.isfinite(loss).all().item():
        raise RuntimeError("MVG FAIL: total loss non-finite")
    if output.curvature_regularization.detach().item() != 0.0:
        raise RuntimeError(
            f"MVG FAIL: curvature_reg must be 0 for fixed c, got "
            f"{float(output.curvature_regularization):.6f}"
        )
    loss.backward()
    nonzero_names = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all().item():
            raise RuntimeError(f"MVG FAIL: non-finite gradient on {name}")
        if parameter.grad.abs().sum().item() > GRAD_EPSILON:
            nonzero_names.append(name)
    if not nonzero_names:
        raise RuntimeError("MVG FAIL: no trainable model parameter has a non-zero gradient")
    return {
        "loss": float(loss.detach().item()),
        "nonzero_gradient_parameters": nonzero_names,
    }


def _check_counterfactual(candidate, control, batch):
    candidate_state = candidate.state_dict()
    control_state = control.state_dict()
    if candidate_state.keys() != control_state.keys():
        raise RuntimeError("MVG FAIL: candidate/control state keys differ")
    for name in candidate_state:
        if name.endswith("._fixed_c"):
            continue
        if not torch.equal(candidate_state[name], control_state[name]):
            raise RuntimeError(
                f"MVG FAIL: non-curvature state differs between models at {name}"
            )

    candidate.eval()
    control.eval()
    candidate.set_curriculum_step(0)
    control.set_curriculum_step(0)
    with torch.no_grad():
        candidate_output = candidate.get_semantic_ids(batch.x)
        control_output = control.get_semantic_ids(batch.x)
    candidate_loss = float(candidate_output.quantize_loss.mean().item())
    control_loss = float(control_output.quantize_loss.mean().item())
    if not np.isfinite([candidate_loss, control_loss]).all():
        raise RuntimeError("MVG FAIL: counterfactual quantizer loss is non-finite")
    loss_delta = abs(candidate_loss - control_loss)
    assignment_fraction = float(
        (candidate_output.sem_ids != control_output.sem_ids).float().mean().item()
    )
    if not np.isfinite(assignment_fraction):
        raise RuntimeError("MVG FAIL: counterfactual assignment difference is non-finite")
    if assignment_fraction == 0.0 and loss_delta <= COUNTERFACTUAL_ATOL:
        raise RuntimeError(
            "MVG FAIL: candidate/control curvature produced no measurable "
            f"quantizer effect (assignment_fraction={assignment_fraction}, "
            f"quantize_loss_delta={loss_delta}, atol={COUNTERFACTUAL_ATOL})"
        )
    return {
        "candidate_quantize_loss": candidate_loss,
        "control_quantize_loss": control_loss,
        "quantize_loss_delta": loss_delta,
        "assignment_fraction_changed": assignment_fraction,
        "atol": COUNTERFACTUAL_ATOL,
    }


def _check_optimizer_updates(model, batch, fixed_c):
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    before = [parameter.detach().clone() for parameter in trainable]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=1e-4)
    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    for layer_index, layer in enumerate(model.layers):
        if id(layer._fixed_c) in optimizer_parameter_ids:
            raise RuntimeError(
                f"MVG FAIL: layer {layer_index} fixed curvature is in the optimizer"
            )
    fixed_before = _fixed_c_values(model)
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        output.loss.backward()
        optimizer.step()
        if not np.allclose(
            _fixed_c_values(model), fixed_c, rtol=0.0, atol=INVARIANCE_TOL
        ):
            raise RuntimeError("MVG FAIL: fixed curvature changed after optimizer.step()")
    rel_updates = [
        float((parameter.detach() - old).norm().item() / max(old.norm().item(), 1e-12))
        for parameter, old in zip(trainable, before)
    ]
    if any(value <= RELATIVE_UPDATE_EPSILON for value in rel_updates):
        raise RuntimeError(
            f"MVG FAIL: 5-step relative L2 updates too small: {rel_updates}"
        )
    if not np.allclose(
        _fixed_c_values(model), fixed_before, rtol=0.0, atol=INVARIANCE_TOL
    ):
        raise RuntimeError("MVG FAIL: fixed curvature changed during optimizer updates")
    return rel_updates


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
    checkpoint_state = _load_checkpoint(device)
    batch = _load_batch(rqtrain, device)
    candidate = _build_model(rqtrain, device, fixed_c, checkpoint_state)
    control = _build_model(rqtrain, device, CONTROL_FIXED_C, checkpoint_state)
    initial_c = _check_layer_buffers(candidate, fixed_c)
    control_c = _check_layer_buffers(control, CONTROL_FIXED_C)
    snapshots = _check_invariance_across_steps(candidate, fixed_c)
    counterfactual = _check_counterfactual(candidate, control, batch)
    gradients = _check_gradients(candidate, batch)
    rel_updates = _check_optimizer_updates(candidate, batch, fixed_c)
    after_updates = _check_invariance_across_steps(candidate, fixed_c)

    print("MVG PASS")
    print(f"warm_start=PASS checkpoint={REFERENCE_CKPT}")
    print(f"closed_form_c_l={fixed_c}")
    print(f"control_c_l={CONTROL_FIXED_C}")
    print(f"initial_c_live={initial_c}")
    print(f"control_c_live={control_c}")
    print(f"invariance_snapshots={snapshots}")
    print(f"counterfactual={counterfactual}")
    print(f"gradients={gradients}")
    print(f"five_step_relative_updates={rel_updates}")
    print(f"after_update_invariance={after_updates}")


if __name__ == "__main__":
    main()
