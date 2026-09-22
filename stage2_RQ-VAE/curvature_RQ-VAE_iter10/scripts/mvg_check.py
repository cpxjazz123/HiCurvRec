"""iter10 partial-reconstruction 机制的四层 MVG 检查。

所有路径和超参从 iter10 训练入口硬编码读取；本脚本无 CLI 参数。
任一层失败直接 raise，只有四层全部通过才输出 ``MVG PASS``。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch


ITER_DIR = Path(
    "/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter10"
)
TRAIN_SCRIPT = ITER_DIR / "curvature_RQ-VAE.py"
EPS_GRAD = 1e-12
EPS_UPDATE = 1e-7
EPS_BEHAVIOR = 1e-6
UPDATE_STEPS = 5
BEHAVIOR_STEPS = 200


def _load_training_module():
    if not TRAIN_SCRIPT.is_file():
        raise FileNotFoundError(f"训练入口不存在: {TRAIN_SCRIPT}")
    spec = importlib.util.spec_from_file_location("iter10_rqtrain", TRAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载训练入口: {TRAIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _new_model(rqtrain, device: torch.device):
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
    ).to(device)
    return model


def _finite_positive(value: torch.Tensor, name: str) -> None:
    if value.ndim != 0 or not torch.isfinite(value).item() or value.item() <= 1e-12:
        raise RuntimeError(
            f"{name} 必须是有限正标量，实际 shape={tuple(value.shape)} "
            f"value={float(value.item()) if value.numel() == 1 else '非标量'}"
        )


def _nonzero_gradients(loss: torch.Tensor, model: torch.nn.Module, name: str):
    gradients = torch.autograd.grad(
        loss,
        tuple(model.parameters()),
        retain_graph=True,
        allow_unused=True,
    )
    finite_nonzero = [
        (parameter_name, float(gradient.norm().item()))
        for (parameter_name, parameter), gradient in zip(
            model.named_parameters(), gradients
        )
        if parameter.requires_grad
        and gradient is not None
        and torch.isfinite(gradient).all().item()
        and gradient.abs().sum().item() > EPS_GRAD
    ]
    if not finite_nonzero:
        raise RuntimeError(f"MVG Layer2 FAIL: {name} 没有非零有限梯度")
    return finite_nonzero


def _prefix_displacements(model, quantized, name: str) -> float:
    embeddings = quantized.embeddings.detach().clone()
    if name not in ("l1", "l2"):
        raise ValueError(f"未知 codeword displacement 层: {name}")
    layer_index = 1 if name == "l1" else 2
    baseline = embeddings.sum(dim=0).transpose(0, 1)
    changed = embeddings.clone()
    changed[layer_index] = 0.0
    replaced = changed.sum(dim=0).transpose(0, 1)
    baseline_output = model.decode(baseline)
    replaced_output = model.decode(replaced)
    displacement = (baseline_output - replaced_output).norm(dim=-1).mean()
    value = float(displacement.item())
    if not np.isfinite(value) or value <= 1e-6:
        raise RuntimeError(
            f"MVG Layer2 FAIL: {name} codeword decoder displacement={value:.6e} <= 1e-6"
        )
    return value


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("MVG 需要 CUDA，但当前 CUDA 不可用")
    rqtrain = _load_training_module()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    emb_path = Path(rqtrain.EMB_NPY)
    if not emb_path.is_file():
        raise FileNotFoundError(f"embedding 不存在: {emb_path}")
    embeddings = np.load(emb_path, allow_pickle=False).astype(np.float32)
    if embeddings.ndim != 2 or embeddings.shape[1] != rqtrain.INPUT_DIM:
        raise ValueError(
            f"embedding shape={embeddings.shape}，期望第二维={rqtrain.INPUT_DIM}"
        )
    if embeddings.shape[0] < rqtrain.BATCH_SIZE:
        raise ValueError("embedding 样本数小于一个 batch")
    batch = torch.from_numpy(embeddings[: rqtrain.BATCH_SIZE]).to(device)
    seq_batch = rqtrain._build_seq_batch(batch, device)

    torch.manual_seed(rqtrain.SEED)
    np.random.seed(rqtrain.SEED)
    model = _new_model(rqtrain, device)
    model.train()
    model.set_curriculum_step(0)

    # Layer 1: 三项 partial loss 都必须接入总计算图。
    output = model(seq_batch)
    if not output.loss.requires_grad or output.loss.grad_fn is None:
        raise RuntimeError("MVG Layer1 FAIL: total loss 未连接计算图")
    partial_names = (
        "partial_reconstruction_loss_l0",
        "partial_reconstruction_loss_l01",
        "partial_reconstruction_loss_l012",
    )
    partial_values = {}
    for name in partial_names:
        value = getattr(output, name)
        _finite_positive(value, name)
        if not value.requires_grad or value.grad_fn is None:
            raise RuntimeError(f"MVG Layer1 FAIL: {name} 未连接计算图")
        partial_values[name] = float(value.item())
    print(
        "[MVG/L1] graph PASS "
        f"total={float(output.loss.item()):.6f} partial={partial_values}"
    )

    # Layer 2: L01/L012 partial loss 各自必须有非零梯度，并核验单层 codeword 可辨识。
    l01_grad = _nonzero_gradients(
        output.partial_reconstruction_loss_l01, model, "partial_l01"
    )
    l012_grad = _nonzero_gradients(
        output.partial_reconstruction_loss_l012, model, "partial_l012"
    )
    l1_displacement = _prefix_displacements(model, model.get_semantic_ids(batch), "l1")
    l2_displacement = _prefix_displacements(model, model.get_semantic_ids(batch), "l2")
    print(
        "[MVG/L2] gradient PASS "
        f"l01_nonzero={len(l01_grad)} l012_nonzero={len(l012_grad)} "
        f"l1_displacement={l1_displacement:.6e} "
        f"l2_displacement={l2_displacement:.6e}"
    )

    # Layer 3: 5 步 optimizer 更新，至少一个 encoder/codebook 参数真实变化。
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    for _ in range(UPDATE_STEPS):
        optimizer.zero_grad(set_to_none=True)
        step_output = model(seq_batch)
        step_output.loss.backward()
        optimizer.step()
    ratios = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        delta = (parameter.detach() - before[name]).norm().item()
        base = max(before[name].norm().item(), 1e-12)
        ratios[name] = delta / base
    updated = {name: value for name, value in ratios.items() if value > EPS_UPDATE}
    if not updated:
        raise RuntimeError("MVG Layer3 FAIL: 5 步后没有参数达到更新阈值")
    print(
        "[MVG/L3] update PASS "
        f"steps={UPDATE_STEPS} max_relative_update={max(updated.values()):.6e}"
    )

    # Layer 4: 只关闭 partial supervision；最终 L012 保持不变，ON/OFF 差异应来自 L0/L01。
    torch.manual_seed(rqtrain.SEED)
    np.random.seed(rqtrain.SEED)
    on_model = _new_model(rqtrain, device)
    torch.manual_seed(rqtrain.SEED)
    np.random.seed(rqtrain.SEED)
    off_model = _new_model(rqtrain, device)
    on_model.train()
    off_model.train()
    off_optimizer = torch.optim.AdamW(off_model.parameters(), lr=1e-3, weight_decay=1e-4)
    on_optimizer = torch.optim.AdamW(on_model.parameters(), lr=1e-3, weight_decay=1e-4)
    off_model.partial_reconstruction_weight_l0 = 0.0
    off_model.partial_reconstruction_weight_l01 = 0.0
    off_model.partial_reconstruction_weight_l012 = 1.0
    for step in range(BEHAVIOR_STEPS):
        on_model.set_curriculum_step(step)
        off_model.set_curriculum_step(step)
        on_optimizer.zero_grad(set_to_none=True)
        on_output = on_model(seq_batch)
        on_output.loss.backward()
        on_optimizer.step()
        off_optimizer.zero_grad(set_to_none=True)
        off_output = off_model(seq_batch)
        off_output.loss.backward()
        off_optimizer.step()
    loss_diff = abs(float(on_output.loss.item()) - float(off_output.loss.item()))
    if not np.isfinite(loss_diff) or loss_diff <= EPS_BEHAVIOR:
        raise RuntimeError(
            f"MVG Layer4 FAIL: partial ON/OFF loss_diff={loss_diff:.6e} <= {EPS_BEHAVIOR:.1e}"
        )
    print(
        "[MVG/L4] behavior PASS "
        f"loss_diff={loss_diff:.6e} steps={BEHAVIOR_STEPS}"
    )
    print("MVG PASS")
    print(f"workdir={ITER_DIR}")
    print(f"embedding={emb_path}")


if __name__ == "__main__":
    main()
