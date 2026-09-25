"""Four-layer preflight for the hard-coded curvature RQ-VAE training run."""
from __future__ import annotations

import copy
import importlib.util
import tempfile
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = SCRIPT_DIR.parent / "curvature_RQ-VAE.py"
EPSILON_GRADIENT = 1e-12
EPSILON_UPDATE = 1e-7
EPSILON_BEHAVIOR = 1e-6
UPDATE_STEPS = 5
BEHAVIOR_STEPS = 200


def _load_training_module():
    spec = importlib.util.spec_from_file_location("rqtrain", TRAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载训练入口: {TRAIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _new_model(rqtrain, device):
    return rqtrain.RqVae(
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
    ).to(device)


def _train_steps(model, batch, rqtrain, device, *, curvature_start, fixed_curvature):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    losses = []
    seq_batch = rqtrain._build_seq_batch(batch, device)
    for step in range(BEHAVIOR_STEPS):
        model.set_curriculum_step(
            fixed_curvature if fixed_curvature is not None else curvature_start + step
        )
        optimizer.zero_grad(set_to_none=True)
        output = model(seq_batch)
        if not torch.isfinite(output.loss).item():
            raise RuntimeError(f"behavior step={step}: total loss 含 NaN/Inf")
        output.loss.backward()
        optimizer.step()
        losses.append(float(output.loss.detach().item()))
    return losses


def main() -> None:
    if not TRAIN_SCRIPT.is_file():
        raise FileNotFoundError(f"训练入口不存在: {TRAIN_SCRIPT}")
    if not torch.cuda.is_available():
        raise RuntimeError("MVG 需要 CUDA")

    rqtrain = _load_training_module()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(rqtrain.SEED)
    np.random.seed(rqtrain.SEED)

    embeddings = np.load(rqtrain.EMB_NPY, allow_pickle=False)
    if embeddings.ndim != 2 or embeddings.shape[1] != rqtrain.INPUT_DIM:
        raise ValueError(
            f"embedding shape={embeddings.shape}, expected (N, {rqtrain.INPUT_DIM})"
        )
    if embeddings.shape[0] < rqtrain.BATCH_SIZE:
        raise ValueError("embedding 样本数小于单卡 batch size")
    batch = torch.from_numpy(
        embeddings[: rqtrain.BATCH_SIZE].astype(np.float32, copy=False)
    ).to(device)

    model = _new_model(rqtrain, device)
    encoder_norms = model.encode(batch).norm(dim=-1)
    if not torch.isfinite(encoder_norms).all().item() or (
        encoder_norms > 1.0 + 1e-6
    ).any().item():
        raise RuntimeError("MVG latent bound FAIL: encoder norm exceeds unit radius")
    from modules.hyperbolic import _expmap0_t, _poincare_distance_t

    flat_probe = copy.deepcopy(model.layers[0])
    flat_probe.eval()
    with torch.no_grad():
        flat_probe.weight.zero_()
        flat_probe.weight[0, 0] = 0.1
        flat_probe.weight[1, 1] = 0.1
        flat_probe.weight[2, 0] = -0.1
        flat_probe.weight[3, 1] = -0.1
        large_probe = torch.zeros((4, rqtrain.EMBED_DIM), device=device)
        large_probe[:, 0] = torch.tensor([1e11, 0, -1e11, 0], device=device)
        large_probe[:, 1] = torch.tensor([0, 1e11, 0, -1e11], device=device)
        curvature = flat_probe.get_c()
        c3 = curvature.view(1, 1, 1)
        probe_query_h = _expmap0_t(large_probe.unsqueeze(1), c3)
        probe_codebook_h = _expmap0_t(
            flat_probe.weight.unsqueeze(0).expand(4, -1, -1), c3
        )
        probe_distances = _poincare_distance_t(
            probe_query_h.expand_as(probe_codebook_h), probe_codebook_h, c3
        ).squeeze(-1)
        if (probe_distances.max() - probe_distances.min()).item() > 1e-6:
            raise RuntimeError("MVG flat-fallback setup FAIL: distances not saturated")
        probe_output = flat_probe(large_probe)
        if probe_output.ids.unique().numel() != 4:
            raise RuntimeError(
                "MVG flat-fallback FAIL: large finite latents lost code ordering"
            )
    # Exercise the same serialized state-dict contract before any GPU training.
    with tempfile.NamedTemporaryFile(
        prefix="rqvae_mvg_", suffix=".pt", dir=SCRIPT_DIR, delete=False
    ) as handle:
        checkpoint = Path(handle.name)
    try:
        torch.save({"model": model.state_dict()}, checkpoint)
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model"], strict=True)
    finally:
        checkpoint.unlink(missing_ok=True)

    # Regression for the saturated-distance case that previously aborted one rank.
    flat_distances = torch.full((2, 3), 12.0, device=device)
    flat_normalized = model.layers[0]._center_distance_for_constraint(flat_distances)
    if not torch.equal(flat_normalized, torch.zeros_like(flat_normalized)):
        raise RuntimeError("MVG distance normalization FAIL: flat distances not finite zero")
    ranged_distances = torch.tensor([[2.0, 3.0], [4.0, 10.0]], device=device)
    ranged_normalized = model.layers[0]._center_distance_for_constraint(ranged_distances)
    if (
        not torch.isfinite(ranged_normalized).all().item()
        or abs(float(ranged_normalized.min().item()) + 1.0) > 1e-6
        or abs(float(ranged_normalized.max().item()) - 1.0) > 1e-6
    ):
        raise RuntimeError("MVG distance normalization FAIL: range scaling is incorrect")


    # Layer 1: total loss remains attached to autograd.
    model.train()
    model.set_curriculum_step(0)
    seq_batch = rqtrain._build_seq_batch(batch, device)
    output = model(seq_batch)
    if not output.loss.requires_grad or output.loss.grad_fn is None:
        raise RuntimeError("MVG Layer 1 FAIL: total_loss 未连接 autograd graph")
    if not torch.isfinite(output.loss).item():
        raise RuntimeError("MVG Layer 1 FAIL: total_loss 含 NaN/Inf")

    # Layer 2: every residual-quantization layer has a finite, nonzero path.
    residual = model.encode(batch)
    layer_grad_norms = []
    parameters = tuple(model.parameters())
    for layer_index, layer in enumerate(model.layers):
        quantized = layer(residual)
        loss_item = quantized.loss.mean()
        gradients = torch.autograd.grad(
            loss_item,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        for gradient in gradients:
            if gradient is not None and not torch.isfinite(gradient).all().item():
                raise RuntimeError(
                    f"MVG Layer 2 FAIL: layer={layer_index} gradient 含 NaN/Inf"
                )
        norm = sum(
            float(gradient.detach().norm().item())
            for gradient in gradients
            if gradient is not None
        )
        if norm <= EPSILON_GRADIENT:
            raise RuntimeError(
                f"MVG Layer 2 FAIL: layer={layer_index} 无非零梯度, norm={norm:.3e}"
            )
        layer_grad_norms.append(norm)
        residual = model._step5_transport(
            model._step4_m2_residual(residual, quantized.embeddings, layer_index),
            layer_index,
        )
    output.loss.backward()

    # Layer 3: five optimizer steps update the codebook parameters materially.
    update_model = copy.deepcopy(model)
    update_model.train()
    update_model.set_curriculum_step(0)
    update_optimizer = torch.optim.AdamW(
        update_model.parameters(), lr=1e-3, weight_decay=1e-4
    )
    codebooks = [layer.embedding.weight for layer in update_model.layers]
    before = [parameter.detach().clone() for parameter in codebooks]
    for _ in range(UPDATE_STEPS):
        update_optimizer.zero_grad(set_to_none=True)
        update_model(rqtrain._build_seq_batch(batch, device)).loss.backward()
        update_optimizer.step()
    update_ratios = [
        float((after.detach() - old).norm().item() / old.norm().clamp_min(1e-12).item())
        for old, after in zip(before, codebooks)
    ]
    if not any(ratio > EPSILON_UPDATE for ratio in update_ratios):
        raise RuntimeError(
            f"MVG Layer 3 FAIL: codebook update ratios={update_ratios}"
        )

    # Layer 4: compare 200 identical-batch steps at high cyclic curvature vs c_min.
    on_model = copy.deepcopy(model)
    off_model = copy.deepcopy(model)
    on_losses = _train_steps(
        on_model,
        batch,
        rqtrain,
        device,
        curvature_start=rqtrain.C_CYCLIC_PERIOD // 2,
        fixed_curvature=None,
    )
    off_losses = _train_steps(
        off_model,
        batch,
        rqtrain,
        device,
        curvature_start=0,
        fixed_curvature=0,
    )
    c_on = float(on_model.layers[0].get_c().item())
    c_off = float(off_model.layers[0].get_c().item())
    loss_delta = abs(on_losses[0] - off_losses[0])
    if abs(c_on - c_off) <= EPSILON_BEHAVIOR or loss_delta <= EPSILON_BEHAVIOR:
        raise RuntimeError(
            "MVG Layer 4 FAIL: cyclic-curvature ON/OFF did not change behavior "
            f"(c_on={c_on:.6g}, c_off={c_off:.6g}, loss_delta={loss_delta:.6g})"
        )

    print("MVG PASS")
    print("checkpoint_contract=temporary state_dict save/reload PASS")
    print(f"batch_shape={tuple(batch.shape)}")
    print(f"layer_gradient_norms={[round(value, 6) for value in layer_grad_norms]}")
    print(f"codebook_update_ratios={[f'{value:.3e}' for value in update_ratios]}")
    print(
        f"behavior_steps={BEHAVIOR_STEPS} c_on={c_on:.6f} c_off={c_off:.6f} "
        f"first_loss_delta={loss_delta:.6g}"
    )


if __name__ == "__main__":
    main()
