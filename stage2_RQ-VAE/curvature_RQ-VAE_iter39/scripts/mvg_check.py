"""One-checkpoint, one-real-batch MVG for Iter39 code behavior contrast."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = SCRIPT_DIR.parent / "curvature_RQ-VAE.py"
REFERENCE_CKPT = Path(
    "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/"
    "curvature_RQ-VAE_iter8/out/rqvae/instruments/rqvae_best.pth"
)
EXPECTED_FIXED_C = [1.3660953164241916, 0.7347829661951981, 0.6439958072706683]
GRAD_EPSILON = 1e-12
CONTRAST_GRAD_EPSILON = 1e-12


def _load_training_module():
    spec = importlib.util.spec_from_file_location("rqtrain_iter39", TRAIN_SCRIPT)
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


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("MVG requires CUDA")
    rqtrain = _load_training_module()
    from modules.hyperbolic import _expmap0_t, _poincare_distance_t
    from modules.rqvae import BEHAVIOR_TEMPERATURE

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
    captured = []
    original_get_semantic_ids = model.get_semantic_ids

    def capture_quantized(inputs):
        quantized = original_get_semantic_ids(inputs)
        captured.append(quantized)
        return quantized

    model.get_semantic_ids = capture_quantized
    output = model(batch)
    if len(captured) != 1:
        raise RuntimeError(f"expected one quantizer output, got {len(captured)}")
    quantized = captured.pop()
    code_behavior_loss = model._behavior_contrastive_loss(
        quantized, batch, batch.x.shape[0]
    )
    if not torch.allclose(
        output.behavior_loss, code_behavior_loss, rtol=1e-6, atol=1e-7
    ):
        raise RuntimeError("forward behavior loss did not use quantized code vectors")

    duplicate_targets = batch.ids_fut[:, None].eq(batch.ids_fut[None, :])
    duplicate_targets.fill_diagonal_(False)
    valid_rows = batch.ids.ne(batch.ids_fut)
    residual_losses = []
    batch_size = batch.x.shape[0]
    for layer_index, layer in enumerate(model.layers):
        source = quantized.residuals[
            layer_index, :, :batch_size
        ].transpose(0, 1)
        candidates = quantized.residuals[
            layer_index, :, batch_size:
        ].transpose(0, 1)
        curvature = layer.get_c().reshape(1, 1, 1)
        source_ball = _expmap0_t(source.unsqueeze(1), curvature)
        candidates_ball = _expmap0_t(candidates.unsqueeze(0), curvature)
        distances = _poincare_distance_t(
            source_ball, candidates_ball, curvature
        ).squeeze(-1)
        logits = (-distances / BEHAVIOR_TEMPERATURE).masked_fill(
            duplicate_targets, float("-inf")
        )
        if bool(valid_rows.any().item()):
            labels = torch.arange(batch_size, device=batch.x.device)
            residual_losses.append(
                F.cross_entropy(logits[valid_rows], labels[valid_rows])
            )
        else:
            residual_losses.append(distances.sum() * 0.0)
    legacy_residual_loss = torch.stack(residual_losses).mean()
    behavior_delta = float(
        (output.behavior_loss.detach() - legacy_residual_loss.detach())
        .abs()
        .item()
    )
    if not np.isfinite(behavior_delta) or behavior_delta <= 1e-8:
        raise RuntimeError(
            f"code-vector behavior loss has no measurable change: {behavior_delta}"
        )

    behavior_gradients = torch.autograd.grad(
        output.behavior_loss,
        tuple(model.encoder.parameters()),
        retain_graph=True,
        allow_unused=True,
    )
    if not any(
        gradient is not None
        and torch.isfinite(gradient).all().item()
        and gradient.abs().sum().item() > CONTRAST_GRAD_EPSILON
        for gradient in behavior_gradients
    ):
        raise RuntimeError("code-vector behavior loss has no finite encoder gradient")
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
    print(f"code_vs_residual_behavior_loss_delta={behavior_delta:.8g}")
    print(f"loss={float(output.loss.detach().item()):.8g}")
    print(f"nonzero_gradient_parameters={len(nonzero_gradients)}")


if __name__ == "__main__":
    main()
