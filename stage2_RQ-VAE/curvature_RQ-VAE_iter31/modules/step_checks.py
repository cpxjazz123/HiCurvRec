"""RQ-VAE 分步执行检查。

每个 Step 只验证该步骤的输入/输出契约；检查失败直接抛出异常，
不通过默认值或降级路径掩盖错误。
"""
from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import Iterable

import torch
from torch import nn, Tensor


def _require(condition: bool, message: str, error_type=ValueError) -> None:
    if not condition:
        raise error_type(message)


def _check_tensor(
    value: Tensor,
    name: str,
    *,
    ndim: int | None = None,
    shape: tuple[int | None, ...] | None = None,
    device: torch.device | None = None,
    finite: bool = True,
) -> None:
    _require(torch.is_tensor(value), f"{name} 必须是 Tensor")
    if ndim is not None:
        _require(value.ndim == ndim, f"{name} ndim={value.ndim}，期望 {ndim}")
    if shape is not None:
        _require(len(shape) == value.ndim, f"{name} shape 约束维度不匹配")
        for index, expected in enumerate(shape):
            if expected is not None:
                _require(
                    value.shape[index] == expected,
                    f"{name} shape={tuple(value.shape)}，第 {index} 维期望 {expected}",
                )
    if device is not None:
        _require(value.device == device, f"{name} 位于 {value.device}，期望 {device}")
    if finite:
        _require(torch.isfinite(value).all().item(), f"{name} 含 NaN 或 Inf")


def check_step1_config(
    input_dim: int,
    embed_dim: int,
    hidden_dims: list[int],
    codebook_size: int,
    n_layers: int,
    c_min: float,
    c_max: float,
    c_period: int,
    sk_eps: float,
    sk_iters: int,
) -> None:
    """Step1：检查硬编码配置的基本不变量。"""
    _require(input_dim > 0, "Step1: input_dim 必须为正数")
    _require(embed_dim > 0, "Step1: embed_dim 必须为正数")
    _require(hidden_dims and all(value > 0 for value in hidden_dims), "Step1: hidden_dims 无效")
    _require(codebook_size > 1, "Step1: codebook_size 必须大于 1")
    _require(n_layers > 0, "Step1: n_layers 必须为正数")
    _require(0 < c_min <= c_max, "Step1: cyclic curvature 范围无效")
    _require(c_period > 0, "Step1: cyclic curvature period 必须为正数")
    _require(sk_eps > 0, "Step1: Sinkhorn epsilon 必须为正数")
    _require(sk_iters > 0, "Step1: Sinkhorn iterations 必须为正数")


def check_step2_dataset(dataset, input_dim: int) -> None:
    """Step2：检查数据集非空、维度正确且全为有限值。"""
    _require(len(dataset) > 0, "Step2: 数据集为空")
    embeddings = dataset.embeddings
    _check_tensor(embeddings, "Step2 embeddings", ndim=2, finite=True)
    _require(embeddings.shape[1] == input_dim, "Step2: embedding 维度与 input_dim 不一致")


def check_step3_distributed(rank: int, world_size: int, local_rank: int) -> None:
    """Step3：检查 DDP 环境已经正确初始化。"""
    _require(dist_is_initialized(), "Step3: distributed process group 未初始化", RuntimeError)
    _require(rank >= 0, "Step3: rank 必须非负")
    _require(world_size > 0, "Step3: world_size 必须为正数")
    _require(0 <= local_rank < torch.cuda.device_count(), "Step3: local_rank 不在 CUDA 设备范围内")
    _require(torch.distributed.get_rank() == rank, "Step3: rank 与 process group 不一致", RuntimeError)
    _require(torch.distributed.get_world_size() == world_size, "Step3: world_size 与 process group 不一致", RuntimeError)


def dist_is_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


# === Step0: 每次训练前清空旧产物的 glob 模式 (R42: 防止上一次 baseline lock 残留) ===
_STEP0_CLEAN_PATTERNS = (
    "rqvae_step*.pt",
    "rqvae_final.pt",
    "sids_step*.npy",
    "sids_for_hgrec*.npy",
    "sids_raw.npy",
    "quality_step*.json",
)


def check_step0_clean_output(
    out_dir: str,
    *,
    rank: int = 0,
    barrier: bool = True,
) -> int:
    """Step0: rank 0 删除 out_dir 下所有历史产物, 其他 rank 通过 barrier 等待.

    只删匹配 _STEP0_CLEAN_PATTERNS 的文件, 不动其他文件 (例如 logs/, configs/).
    删除失败 / 路径不是文件 / 目录不存在 必须直接 raise, 不允许 fallback.

    Returns: rank 0 删除的文件数 (其他 rank 返回 0).
    """
    _require(
        isinstance(out_dir, str) and out_dir,
        "Step0: out_dir 必须为非空字符串",
    )
    out_path = Path(out_dir)
    _require(
        out_path.is_absolute(),
        f"Step0: out_dir 必须是绝对路径, 实际 {out_dir}",
    )

    # 非 rank 0 先 barrier 等待 (rank 0 删除完后再唤醒)
    if rank != 0 and barrier and dist_is_initialized():
        torch.distributed.barrier()
        return 0

    # rank 0 删除流程
    if not out_path.exists():
        # 目录不存在, 创建出来即可 (后续 save_ckpt 还会再创建一次)
        out_path.mkdir(parents=True, exist_ok=True)
        if barrier and dist_is_initialized():
            torch.distributed.barrier()
        return 0

    _require(
        out_path.is_dir(),
        f"Step0: out_dir 不是目录: {out_path}",
    )

    deleted = 0
    for pattern in _STEP0_CLEAN_PATTERNS:
        for candidate in out_path.glob(pattern):
            _require(
                candidate.is_file(),
                f"Step0: 匹配 {pattern} 的 {candidate} 不是普通文件",
            )
            try:
                candidate.unlink()
                deleted += 1
            except OSError as exc:
                raise RuntimeError(
                    f"Step0: 删除历史产物失败 {candidate}: {exc}"
                ) from exc

    if barrier and dist_is_initialized():
        torch.distributed.barrier()
    return deleted


def check_step4_model(
    model: nn.Module,
    *,
    input_dim: int,
    embed_dim: int,
    codebook_size: int,
    n_layers: int,
    device: torch.device,
) -> None:
    """Step4：检查 Encoder、Decoder 和所有 codebook 的结构/数值。"""
    _require(isinstance(model, nn.Module), "Step4: model 不是 nn.Module")
    _require(len(model.layers) == n_layers, "Step4: 量化层数不一致")
    _require(model.encoder.input_dim == input_dim, "Step4: Encoder input_dim 不一致")
    _require(model.encoder.out_dim == embed_dim, "Step4: Encoder out_dim 不一致")
    _require(model.decoder.input_dim == embed_dim, "Step4: Decoder input_dim 不一致")
    _require(model.decoder.out_dim == input_dim, "Step4: Decoder out_dim 不一致")
    for layer_index, layer in enumerate(model.layers):
        _check_tensor(
            layer.embedding.weight,
            f"Step4 layer {layer_index} codebook",
            ndim=2,
            shape=(codebook_size, embed_dim),
            device=device,
        )
        curvature = layer.get_c()
        _check_tensor(curvature, f"Step4 layer {layer_index} curvature", ndim=0)
        _require(curvature.item() > 0, f"Step4 layer {layer_index} curvature 必须为正数")
    for name, parameter in model.named_parameters():
        _require(torch.isfinite(parameter).all().item(), f"Step4 参数 {name} 含 NaN 或 Inf")


def check_step5_optimizer(optimizer: torch.optim.Optimizer, model: nn.Module) -> None:
    """Step5：检查优化器覆盖全部可训练参数。"""
    model_parameters = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    optimizer_parameters = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
        if parameter.requires_grad
    }
    _require(model_parameters == optimizer_parameters, "Step5: optimizer 未精确覆盖可训练参数")
    _require(len(optimizer.param_groups) > 0, "Step5: optimizer 没有参数组")


def check_step6_batch(batch: Tensor, input_dim: int, device: torch.device) -> None:
    """Step6：检查一个 batch 的形状、设备和数值。"""
    _check_tensor(batch, "Step6 batch", ndim=2, device=device, finite=True)
    _require(batch.shape[0] > 0, "Step6: batch 为空")
    _require(batch.shape[1] == input_dim, "Step6: batch 输入维度不一致")


def check_step7_forward(
    output,
    batch: Tensor,
    *,
    n_layers: int,
    codebook_size: int,
    embed_dim: int,
    device: torch.device,
) -> None:
    """Step7：检查量化前向输出和每层 semantic IDs。"""
    _check_tensor(output.embeddings, "Step7 embeddings", ndim=3, device=device)
    _check_tensor(output.residuals, "Step7 residuals", ndim=3, device=device)
    _check_tensor(output.sem_ids, "Step7 semantic ids", ndim=2, device=device)
    _check_tensor(output.quantize_loss, "Step7 quantize loss", ndim=1, device=device)
    _require(output.embeddings.shape == (n_layers, embed_dim, batch.shape[0]), f"Step7 embeddings shape 错误 actual={tuple(output.embeddings.shape)} expected={(n_layers, embed_dim, batch.shape[0])}")
    _require(output.residuals.shape == (n_layers, embed_dim, batch.shape[0]), f"Step7 residuals shape 错误 actual={tuple(output.residuals.shape)} expected={(n_layers, embed_dim, batch.shape[0])}")
    _require(output.sem_ids.shape == (batch.shape[0], n_layers), "Step7 semantic ids shape 错误")
    _require(output.quantize_loss.shape == (batch.shape[0],), "Step7 quantize loss shape 错误")
    _require(
        bool(((output.sem_ids >= 0) & (output.sem_ids < codebook_size)).all().item()),
        "Step7 semantic ids 超出 codebook 范围",
    )


def check_step8_loss(losses, training: bool) -> None:
    """Step8：检查 reconstruction、quantize 和 total loss。"""
    for name in ("loss", "reconstruction_loss", "rqvae_loss"):
        value = getattr(losses, name)
        _check_tensor(value, f"Step8 {name}", ndim=0)
        _require(value.item() >= 0, f"Step8 {name} 不应为负数")
    if training:
        _require(losses.loss.requires_grad, "Step8 total loss 未连接计算图", RuntimeError)
        _require(losses.loss.grad_fn is not None, "Step8 total loss 缺少 grad_fn", RuntimeError)


def check_step9_backward(model: nn.Module) -> None:
    """Step9：检查反向传播产生有限且非零的梯度。"""
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    _require(gradients, "Step9: 没有可训练参数的梯度", RuntimeError)
    nonzero = False
    for index, gradient in enumerate(gradients):
        _check_tensor(gradient, f"Step9 gradient {index}", device=gradient.device)
        if gradient.abs().sum().item() > 0:
            nonzero = True
    _require(nonzero, "Step9: 所有梯度均为零", RuntimeError)


def check_step10_update(before: Iterable[Tensor], model: nn.Module) -> None:
    """Step10：检查 optimizer.step() 确实更新了至少一个参数。"""
    after = [parameter.detach() for parameter in model.parameters() if parameter.requires_grad]
    before_values = list(before)
    _require(len(before_values) == len(after), "Step10: 参数快照数量发生变化", RuntimeError)
    changed = any(not torch.equal(old, new) for old, new in zip(before_values, after))
    _require(changed, "Step10: optimizer.step() 未更新任何参数", RuntimeError)
    for index, parameter in enumerate(after):
        _require(torch.isfinite(parameter).all().item(), f"Step10 参数 {index} 更新后含 NaN 或 Inf", RuntimeError)


def check_step11_curvature(
    model: nn.Module,
    step: int,
    c_min: float,
    c_max: float,
    c_min_per_layer: list | None = None,
    c_max_per_layer: list | None = None,
) -> None:
    """Step11：检查曲率同步更新后仍在 cyclic curvature 范围内。

    支持 per-layer 异质 c_min/c_max (R36n b / iter31+). 若提供 per-layer 列表,
    使用每层自己的范围; 否则回退到全局 [c_min, c_max] (旧版 scalar 形式).
    """
    _require(step >= 0, "Step11: global step 必须非负")
    for layer_index, layer in enumerate(model.layers):
        curvature = layer.get_c()
        _check_tensor(curvature, f"Step11 layer {layer_index} curvature", ndim=0)
        value = curvature.item()
        if c_min_per_layer is not None and c_max_per_layer is not None:
            layer_min = float(c_min_per_layer[layer_index])
            layer_max = float(c_max_per_layer[layer_index])
        else:
            layer_min = float(c_min)
            layer_max = float(c_max)
        _require(
            layer_min - 1e-6 <= value <= layer_max + 1e-6,
            f"Step11 layer {layer_index} curvature={value} 超出 per-layer 范围 [{layer_min}, {layer_max}]",
        )


def check_step12_checkpoint(path: str) -> None:
    """Step12：检查 checkpoint 已写入且非空。"""
    checkpoint = Path(path)
    _require(checkpoint.is_file(), f"Step12 checkpoint 不存在：{checkpoint}", FileNotFoundError)
    _require(checkpoint.stat().st_size > 0, f"Step12 checkpoint 为空：{checkpoint}", RuntimeError)
