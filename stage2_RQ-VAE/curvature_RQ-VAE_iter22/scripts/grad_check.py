"""RQ-VAE 训练前梯度通路检查。

从 curvature_RQ-VAE.py 动态读取配置，使用一个真实 embedding batch 和一个
临时 checkpoint，验证 total loss、动态曲率 quantize loss 以及 encoder/codebook
的反向梯度。检查失败直接抛出异常，不提供 fallback。
"""
from __future__ import annotations

import importlib.util
import os
import tempfile
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = SCRIPT_DIR.parent / "curvature_RQ-VAE.py"


if not TRAIN_SCRIPT.is_file():
    raise FileNotFoundError(f"训练入口不存在: {TRAIN_SCRIPT}")

spec = importlib.util.spec_from_file_location("rqtrain", TRAIN_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"无法加载训练入口模块: {TRAIN_SCRIPT}")
rqtrain = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rqtrain)

if not torch.cuda.is_available():
    raise RuntimeError("grad_check 需要 CUDA，但当前 CUDA 不可用")

device = torch.device("cuda:0")
torch.cuda.set_device(device)
torch.manual_seed(rqtrain.SEED)
np.random.seed(rqtrain.SEED)

emb_path = Path(rqtrain.EMB_NPY)
if not emb_path.is_file():
    raise FileNotFoundError(f"embedding 文件不存在: {emb_path}")
arr = np.load(emb_path).astype(np.float32)
if arr.ndim != 2 or arr.shape[1] != rqtrain.INPUT_DIM:
    raise ValueError(
        f"embedding shape={arr.shape}，期望第二维为 {rqtrain.INPUT_DIM}"
    )
if arr.shape[0] < rqtrain.BATCH_SIZE:
    raise ValueError(
        f"embedding 样本数 {arr.shape[0]} 小于 batch size {rqtrain.BATCH_SIZE}"
    )

batch = torch.from_numpy(arr[: rqtrain.BATCH_SIZE]).to(device)
model = rqtrain.RqVae(
    input_dim=rqtrain.INPUT_DIM,
    embed_dim=rqtrain.EMBED_DIM,
    hidden_dims=rqtrain.HIDDEN_DIMS,
    codebook_size=rqtrain.CODEBOOK_SIZE,
    codebook_kmeans_init=True,
    n_layers=rqtrain.N_LAYERS,
    commitment_weight=rqtrain.COMMITMENT_WEIGHT,
    sk_eps=0.05,
    sk_iters=3,
    c_cyclic_min=rqtrain.C_CYCLIC_MIN,
    c_cyclic_max=rqtrain.C_CYCLIC_MAX,
    c_cyclic_period=rqtrain.C_CYCLIC_PERIOD,
    midpoint_layer_mask=rqtrain.MIDPOINT_LAYER_MASK,
).to(device)

# 用显式的临时 checkpoint 验证 state_dict 恢复路径，再从恢复后的模型做检查。
with tempfile.NamedTemporaryFile(
    prefix="rqvae_grad_check_", suffix=".pt", dir=SCRIPT_DIR, delete=False
) as handle:
    checkpoint_path = Path(handle.name)
try:
    torch.save({"model": model.state_dict()}, checkpoint_path)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model" not in state:
        raise KeyError("梯度检查 checkpoint 缺少 model 字段")
    model.load_state_dict(state["model"], strict=True)
finally:
    if checkpoint_path.exists():
        checkpoint_path.unlink()

model.train()
seq_batch = rqtrain._build_seq_batch(batch, device)

# 在周期起点和中点各做一次，确保 c(t) 真正进入 assignment 与 quantize loss。
losses = []
for step in (0, rqtrain.C_CYCLIC_PERIOD // 2):
    model.set_curriculum_step(step)
    model.zero_grad(set_to_none=True)
    output = model(seq_batch)
    total_loss = output.loss
    if not total_loss.requires_grad or total_loss.grad_fn is None:
        raise RuntimeError(f"step={step}: total loss 未连接计算图")
    if not torch.isfinite(total_loss).item():
        raise RuntimeError(f"step={step}: total loss 含 NaN/Inf")

    layer_losses = [
        layer.quantize_loss(
            query=model.encoder(batch),
            value=layer.get_item_embeddings(
                model.get_semantic_ids(batch).sem_ids[:, layer_index]
            ),
            c=layer.get_c().view(1, 1),
        )
        for layer_index, layer in enumerate(model.layers)
    ]
    for layer_index, layer_loss in enumerate(layer_losses):
        gradients = torch.autograd.grad(
            layer_loss.mean(),
            tuple(model.parameters()),
            retain_graph=True,
            allow_unused=True,
        )
        if not any(
            gradient is not None and gradient.abs().sum().item() > 0
            for gradient in gradients
        ):
            raise RuntimeError(
                f"step={step} layer={layer_index}: quantize loss 无非零梯度"
            )

    total_loss.backward()
    encoder_grad = [
        parameter.grad
        for parameter in model.encoder.parameters()
        if parameter.requires_grad
    ]
    codebook_grad = [layer.embedding.weight.grad for layer in model.layers]
    if not any(
        gradient is not None and gradient.abs().sum().item() > 0
        for gradient in encoder_grad
    ):
        raise RuntimeError(f"step={step}: encoder 梯度为零")
    if not any(
        gradient is not None and gradient.abs().sum().item() > 0
        for gradient in codebook_grad
    ):
        raise RuntimeError(f"step={step}: codebook 梯度为零")
    if not all(
        gradient is None or torch.isfinite(gradient).all().item()
        for gradient in encoder_grad + codebook_grad
    ):
        raise RuntimeError(f"step={step}: 梯度含 NaN/Inf")

    losses.append(float(output.rqvae_loss.detach().item()))

if losses[0] == losses[1]:
    raise RuntimeError("c(t) 改变后 quantize loss 数值未变化")

print("GRAD_CHECK PASS")
print(f"embedding={emb_path}")
print(f"batch_shape={tuple(batch.shape)}")
print(f"curvature_steps=(0,{rqtrain.C_CYCLIC_PERIOD // 2})")
print(f"rqvae_loss_values={losses}")
