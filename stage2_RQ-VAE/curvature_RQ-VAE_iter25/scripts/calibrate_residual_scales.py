"""离线测算 v318 baseline 的层间残差尺度 → per-layer c_layer_norm 归一化值.

iter1 (residual-scaled cyclic curvature) 的层间曲率依赖 baseline checkpoint
的 encoder + residual 中位尺度。本脚本：

  1. 加载 baseline ckpt (固定路径，与 iter1 自己的 ckpt 无关);
  2. 在 `ITEM_EMB_NPY` 上做无梯度前向（与 sid_quality.build_sids_for_corpus
     一致的 1024 分块），逐层收集 residual 范数；
  3. 计算对数归一化尺度 u_l = log(s_max/s_l) / log(s_max/s_min)，并把
     u_l 写到 `out/decoder/instruments_hgrec_configs/<mech>/layer_norms.json`。

`layer_norms.json` 由 `curvature_RQ-VAE.py` 在 `main()` 启动前读取，作为
`RqVae(..., residual_layer_norms=...)` 的输入。

用法:
    /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 \
        scripts/calibrate_residual_scales.py

硬编码路径, 0 CLI 参数 (Project Rules §1).
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
STAGE2_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(STAGE2_DIR))

import curvature_config  # noqa: E402


def _load_rqvae_module():
    """加载 curvature_RQ-VAE.py 当作模块（文件名含 `-`，必须 importlib）。"""
    spec = importlib.util.spec_from_file_location(
        "rqtrain", STAGE2_DIR / "curvature_RQ-VAE.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"无法加载训练入口: {STAGE2_DIR / 'curvature_RQ-VAE.py'}"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _measure_layer_norms(rqtrain, ckpt_path: Path, emb_path: Path, device):
    """加载 ckpt 在 corpus 上做无梯度前向，返回每层 residual 范数中位数。"""
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"ckpt 不存在: {ckpt_path}")
    if not emb_path.is_file():
        raise FileNotFoundError(f"embedding 不存在: {emb_path}")

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

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model" not in checkpoint:
        raise KeyError(f"checkpoint 缺少 model 字段: {ckpt_path}")
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"checkpoint/model mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    step = checkpoint.get("global_step", checkpoint.get("iter", 0))
    model.set_curriculum_step(step)
    model.eval()

    arr = np.load(emb_path, allow_pickle=False).astype(np.float32, copy=False)
    if arr.ndim != 2 or arr.shape[1] != rqtrain.INPUT_DIM:
        raise ValueError(
            f"embedding shape {arr.shape}, expected (*, {rqtrain.INPUT_DIM})"
        )

    chunk_size = 1024
    norms_per_layer = []
    with torch.inference_mode():
        for start in range(0, len(arr), chunk_size):
            chunk = torch.from_numpy(arr[start:start + chunk_size]).to(device)
            result = model.get_semantic_ids(chunk)
            # RqVaeOutput.residuals: (n_layers, embed_dim, batch) → norm over embed_dim
            norms_per_layer.append(result.residuals.norm(dim=1).cpu())
    stacked = torch.cat(norms_per_layer, dim=1) if norms_per_layer else torch.empty(0)
    if stacked.numel() == 0 or stacked.shape[0] != rqtrain.N_LAYERS:
        raise RuntimeError(f"残差范数矩阵形状异常: {tuple(stacked.shape)}")
    if not torch.isfinite(stacked).all():
        raise RuntimeError("残差范数含 NaN 或 Inf")

    medians = stacked.median(dim=1).values.tolist()
    return medians, int(step)


def _normalize(medians: list[float]) -> list[float]:
    """对数归一化 u_l = log(s_max/s_l) / log(s_max/s_min).

    s_max 自身（u_l=0）会触发 c_l ≡ c_min (常数)，导致与纯静态 curvature 等价；
    因此把 u_l ∈ (0, 1] 缩到 [EPS_LAYER, 1]（EPS_LAYER = 1e-3），让 L0 仍偏向
    c_min 但仍保有 cyclic 时间相位（u_l 接近 0 时 c_l(t) = c_min * r^{g(t)/2}，
    g=0 时 c = c_min，g=1 时 c = sqrt(c_min * c_max) ≈ 0.55）。
    """
    s_max = max(medians)
    s_min = min(medians)
    if not (s_max > 0 and s_min > 0):
        raise ValueError(f"残差范数必须为正: medians={medians}")
    if s_max == s_min:
        return [1.0] * len(medians)
    log_span = math.log(s_max / s_min)
    raw = [
        math.log(s_max / s) / log_span
        for s in medians
    ]
    eps_layer = 1e-3
    return [
        max(value, eps_layer) for value in raw
    ]


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("calibrate_residual_scales 需要 CUDA")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    rqtrain = _load_rqvae_module()
    # Calibration MUST read the promoted baseline ckpt, not this iter's empty
    # output dir; ignore RQVAE_CKPT_PATH (which iter1 redirects to itself).
    baseline_ckpt = (
        "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE/rqvae_best.pth"
    )
    ckpt_path = Path(baseline_ckpt)
    emb_path = Path(curvature_config.ITEM_EMB_NPY)
    print(
        f"[calib] device={device} | ckpt={ckpt_path} | emb={emb_path}",
        flush=True,
    )

    medians, step = _measure_layer_norms(rqtrain, ckpt_path, emb_path, device)
    layer_norms = _normalize(medians)

    out_dir = Path(curvature_config._CONFIG_DIR) / "out/decoder/instruments_hgrec_configs" / f"hgrec_{curvature_config.MECHANISM_NAME}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "layer_norms.json"
    payload = {
        "checkpoint_step": step,
        "checkpoint_path": str(ckpt_path),
        "medians": [float(x) for x in medians],
        "layer_norms": [float(x) for x in layer_norms],
        "c_cyclic_min": rqtrain.C_CYCLIC_MIN,
        "c_cyclic_max": rqtrain.C_CYCLIC_MAX,
        "c_cyclic_period": rqtrain.C_CYCLIC_PERIOD,
        "source": "scripts/calibrate_residual_scales.py:2026-09-24 iter1",
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(
        f"[calib] step={step} medians={[round(x, 6) for x in medians]} "
        f"layer_norms={[round(x, 6) for x in layer_norms]} -> {out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()