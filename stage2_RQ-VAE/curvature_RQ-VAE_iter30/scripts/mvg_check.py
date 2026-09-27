"""Iter30 FCCR-1 invariance and same-batch mapping activation verification."""
from __future__ import annotations

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
    CONTROL_CURVATURE,
    CANDIDATE_CURVATURE,
    ITER26_MAPPING,
    ITER29_MAPPING,
    STAGE0_TRAIN,
    STAGE0_TRAIN_SHA256,
    STAGE1_EMBEDDING,
    STAGE1_EMBEDDING_SHA256,
    STAGE1_ITEM_IDS,
    STAGE1_ITEM_IDS_SHA256,
    WARMSTART_PATH,
    WARMSTART_SHA256,
    _require_digest,
    _load_training_module,
    get_profile,
)

INVARIANCE_STEPS = (0, 25_000, 50_000, 100_000)
INVARIANCE_TOL = 1e-6
GRADIENT_EPSILON = 1e-12


def _make_model(trainer, device, fixed_curvature):
    model = trainer.RqVae(
        input_dim=trainer.INPUT_DIM,
        embed_dim=trainer.EMBED_DIM,
        hidden_dims=trainer.HIDDEN_DIMS,
        codebook_size=trainer.CODEBOOK_SIZE,
        codebook_kmeans_init=False,
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
        fixed_layer_curvatures=list(fixed_curvature),
    ).to(device)
    warmstart = load_warm_start(model, fixed_curvature)
    return model, warmstart


def _load_batch(trainer, device):
    dataset = trainer.TransitionDataset(
        trainer.EMB_NPY, trainer.ITEM_IDS_JSON, trainer.TRAIN_PARQUET
    )
    active_indices = [
        index for index, targets in enumerate(dataset.next_items) if targets
    ]
    if len(active_indices) < trainer.BATCH_SIZE:
        raise RuntimeError(
            f"MVG requires {trainer.BATCH_SIZE} active sources, found {len(active_indices)}"
        )
    raw_batch = trainer.collate_items(
        [dataset[index] for index in active_indices[: trainer.BATCH_SIZE]]
    )
    return trainer._build_seq_batch(*(value.to(device) for value in raw_batch))


def _fixed_values(model):
    return [float(layer._fixed_c.detach().cpu().item()) for layer in model.layers]


def _assert_fixed(model, target, where):
    parameter_ids = {id(parameter) for parameter in model.parameters()}
    actual = []
    for index, layer in enumerate(model.layers):
        buffer = layer._fixed_c
        if buffer.requires_grad or id(buffer) in parameter_ids:
            raise RuntimeError(f"MVG FAIL: fixed curvature L{index} is trainable at {where}")
        if buffer.grad is not None:
            raise RuntimeError(f"MVG FAIL: fixed curvature L{index} received a gradient")
        value = float(buffer.detach().cpu().item())
        live = float(layer.get_c().detach().cpu().item())
        if abs(value - target[index]) > INVARIANCE_TOL or abs(live - target[index]) > INVARIANCE_TOL:
            raise RuntimeError(
                f"MVG FAIL: fixed curvature changed at {where}, L{index}: "
                f"buffer={value} get_c={live} target={target[index]}"
            )
        actual.append(value)
    return actual


def _check_invariance(model, target):
    snapshots = {}
    for mode in ("train", "eval"):
        getattr(model, mode)()
        for step in INVARIANCE_STEPS:
            model.set_curriculum_step(step)
            snapshots[f"{mode}:{step}"] = _assert_fixed(model, target, f"{mode}:{step}")
    return snapshots


def _capture_l0_distance(trainer, model, batch):
    quantize_type = type(model.layers[0])
    original = quantize_type._center_distance_for_constraint
    captured = []

    def capture(distances):
        captured.append(distances.detach().clone())
        return original(distances)

    quantize_type._center_distance_for_constraint = staticmethod(capture)
    try:
        model.eval()
        model.set_curriculum_step(0)
        with torch.no_grad():
            output = model.get_semantic_ids(batch.x)
    finally:
        quantize_type._center_distance_for_constraint = staticmethod(original)
    if len(captured) != trainer.N_LAYERS:
        raise RuntimeError(
            f"MVG expected {trainer.N_LAYERS} pre-centering tensors, got {len(captured)}"
        )
    return captured[0], output


def _check_counterfactual(trainer, control, candidate, batch):
    control_state = control.state_dict()
    candidate_state = candidate.state_dict()
    if control_state.keys() != candidate_state.keys():
        raise RuntimeError("MVG FAIL: control/candidate state keys differ")
    for name in control_state:
        if name.endswith("._fixed_c"):
            continue
        if not torch.equal(control_state[name], candidate_state[name]):
            raise RuntimeError(f"MVG FAIL: non-curvature state differs at {name}")

    control.eval()
    candidate.eval()
    with torch.no_grad():
        control_latent = control.encode(batch.x)
        candidate_latent = candidate.encode(batch.x)
    if (
        control_latent.shape != candidate_latent.shape
        or control_latent.dtype != candidate_latent.dtype
        or control_latent.device != candidate_latent.device
        or not torch.equal(control_latent, candidate_latent)
    ):
        raise RuntimeError("MVG FAIL: encoded L0 inputs differ between mapping arms")
    encoded_l0_max_abs_difference = float(
        (control_latent - candidate_latent).abs().max().item()
    )
    control_first, control_output = _capture_l0_distance(trainer, control, batch)
    control_repeat, _ = _capture_l0_distance(trainer, control, batch)
    candidate_first, candidate_output = _capture_l0_distance(trainer, candidate, batch)
    candidate_repeat, _ = _capture_l0_distance(trainer, candidate, batch)
    tensors = (control_first, control_repeat, candidate_first, candidate_repeat)
    if any(tensor.shape != tensors[0].shape for tensor in tensors):
        raise RuntimeError("MVG FAIL: L0 distance tensor shapes differ")
    if any(tensor.dtype != tensors[0].dtype for tensor in tensors):
        raise RuntimeError("MVG FAIL: L0 distance tensor dtypes differ")
    if any(tensor.device != tensors[0].device for tensor in tensors):
        raise RuntimeError("MVG FAIL: L0 distance tensor devices differ")
    if any(not torch.isfinite(tensor).all().item() for tensor in tensors):
        raise RuntimeError("MVG FAIL: L0 distance tensor is non-finite")

    repeat_control = float((control_first - control_repeat).abs().max().item())
    repeat_candidate = float((candidate_first - candidate_repeat).abs().max().item())
    repeat_tolerance = max(repeat_control, repeat_candidate)
    mapping_distance = float((candidate_first - control_first).abs().max().item())
    if not mapping_distance > repeat_tolerance:
        raise RuntimeError(
            "MVG FAIL: registered L0 mapping did not exceed deterministic repeat "
            f"tolerance: M_distance={mapping_distance}, tolerance={repeat_tolerance}"
        )

    control_loss = float(control_output.quantize_loss.mean().item())
    candidate_loss = float(candidate_output.quantize_loss.mean().item())
    assignment_fraction = float(
        (control_output.sem_ids != candidate_output.sem_ids).float().mean().item()
    )
    return {
        "signal": "L0 per-example-by-code Poincare distances immediately before centering/Sinkhorn",
        "shape": list(control_first.shape),
        "dtype": str(control_first.dtype),
        "device": str(control_first.device),
        "M_distance_candidate_minus_control_max_abs": mapping_distance,
        "same_arm_repeat_max_abs_control": repeat_control,
        "same_arm_repeat_max_abs_candidate": repeat_candidate,
        "deterministic_repeat_tolerance": repeat_tolerance,
        "control_quantize_loss_secondary": control_loss,
        "candidate_quantize_loss_secondary": candidate_loss,
        "assignment_fraction_changed_secondary": assignment_fraction,
        "encoded_l0_shape": list(control_latent.shape),
        "encoded_l0_dtype": str(control_latent.dtype),
        "encoded_l0_device": str(control_latent.device),
        "encoded_l0_max_abs_difference": encoded_l0_max_abs_difference,
    }


def _check_optimizer_invariance(model, target, batch):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    for index, layer in enumerate(model.layers):
        if id(layer._fixed_c) in optimizer_ids:
            raise RuntimeError(f"MVG FAIL: fixed curvature L{index} is in optimizer")
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        if not output.loss.requires_grad or output.loss.grad_fn is None:
            raise RuntimeError("MVG FAIL: total loss detached during optimizer check")
        if not torch.isfinite(output.loss).all().item():
            raise RuntimeError("MVG FAIL: total loss is non-finite during optimizer check")
        if float(output.curvature_regularization.detach().item()) != 0.0:
            raise RuntimeError("MVG FAIL: fixed FCCR-1 curvature regularization is nonzero")
        output.loss.backward()
        if not any(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all().item()
            and parameter.grad.abs().sum().item() > GRADIENT_EPSILON
            for parameter in model.parameters()
            if parameter.requires_grad
        ):
            raise RuntimeError("MVG FAIL: model receives no finite nonzero gradients")
        optimizer.step()
        _assert_fixed(model, target, "after optimizer.step")
    return _check_invariance(model, target)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("iter30 MVG requires CUDA")
    for path, digest, label in (
        (STAGE1_EMBEDDING, STAGE1_EMBEDDING_SHA256, "Stage1 embedding"),
        (STAGE1_ITEM_IDS, STAGE1_ITEM_IDS_SHA256, "Stage1 item IDs"),
        (STAGE0_TRAIN, STAGE0_TRAIN_SHA256, "Stage0 train parquet"),
        (WARMSTART_PATH, WARMSTART_SHA256, "iter8 warm-start"),
    ):
        _require_digest(path, digest, label)

    control_profile = get_profile("iter26_mapping_seed43")
    candidate_profile = get_profile("iter29_mapping_seed43")
    if control_profile["mapping_id"] != ITER26_MAPPING:
        raise RuntimeError("MVG control profile is not the registered iter26 mapping")
    if candidate_profile["mapping_id"] != ITER29_MAPPING:
        raise RuntimeError("MVG candidate profile is not the registered iter29 mapping")
    trainer = _load_training_module()
    trainer._apply_profile(candidate_profile)
    control_mapping = trainer._load_closed_form_mapping(ITER26_MAPPING)
    candidate_mapping = trainer._load_closed_form_mapping(ITER29_MAPPING)
    if not np.allclose(
        control_mapping["closed_form_c_l"], CONTROL_CURVATURE, rtol=0.0,
        atol=trainer.FORMULA_COMPARISON_TOL,
    ):
        raise RuntimeError("MVG control formula differs from the registered vector")
    if not np.allclose(
        candidate_mapping["closed_form_c_l"], CANDIDATE_CURVATURE, rtol=0.0,
        atol=trainer.FORMULA_COMPARISON_TOL,
    ):
        raise RuntimeError("MVG candidate formula differs from the S04 contract")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(43)
    np.random.seed(43)
    batch = _load_batch(trainer, device)
    control, control_warmstart = _make_model(
        trainer, device, control_mapping["closed_form_c_l"]
    )
    candidate, candidate_warmstart = _make_model(
        trainer, device, candidate_mapping["closed_form_c_l"]
    )
    control_invariance = _check_invariance(
        control, control_mapping["closed_form_c_l"]
    )
    candidate_invariance = _check_invariance(
        candidate, candidate_mapping["closed_form_c_l"]
    )
    counterfactual = _check_counterfactual(trainer, control, candidate, batch)
    control_post_update = _check_optimizer_invariance(
        control, control_mapping["closed_form_c_l"], batch
    )
    candidate_post_update = _check_optimizer_invariance(
        candidate, candidate_mapping["closed_form_c_l"], batch
    )
    report = {
        "status": "PASS",
        "checkpoint": str(WARMSTART_PATH),
        "checkpoint_sha256": WARMSTART_SHA256,
        "batch_size": int(batch.x.shape[0]),
        "control": {
            "mapping_id": ITER26_MAPPING,
            "fixed_curvature": CONTROL_CURVATURE,
            "warm_start": control_warmstart,
            "invariance_before_update": control_invariance,
            "invariance_after_five_updates": control_post_update,
        },
        "candidate": {
            "mapping_id": ITER29_MAPPING,
            "fixed_curvature": CANDIDATE_CURVATURE,
            "warm_start": candidate_warmstart,
            "invariance_before_update": candidate_invariance,
            "invariance_after_five_updates": candidate_post_update,
        },
        "counterfactual": counterfactual,
    }
    print("ITER30_MVG_PASS")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
