#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Issue #120 — Stage2 提取 pre-quantization residual.

R34 严格: 4 脚本必须存在. Issue #120 仅 Stage2 几何诊断, 不重训 Stage2.

对 Issue #119 8 个 Stage2 ckpt, 提取每层 (L0, L1, L2) 的 pre-quantization residual:
  - L0 residual = encoder(item_emb)  (32-dim latent)
  - L1 residual = L0_residual - L0_quantized
  - L2 residual = L1_residual - L1_quantized

下游 stage4_beam20.py 用这些 residual pairwise Euclidean distance 作为 D_ref,
评估量化后 codebook 距离在不同 c 下能否保留 pre-quantization 关系.

R30: 全超参硬编码 (8 κ, 3 层, ITEM_EMB_NPY, FIX_C=True).
R31: 单脚本.
R32: 直接 python3 执行.
"""
import os
import json
import numpy as np
import torch
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# 顶部硬编码 (R30 严格)
# ──────────────────────────────────────────────────────────────
REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON_MODULE_SRC = "/home/wlia0047/ar57/wenyu/GeneRec/taskA/stage2"
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"

# 8 κ 网格 (与 Issue #119 一致)
STAGE2_KAPPAS = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
STAGE2_KAPPA_TAGS = {0.01: "0p01", 0.1: "0p1", 0.5: "0p5", 1.0: "1", 2.0: "2", 5.0: "5", 10.0: "10", 20.0: "20"}

# Stage2 模型配置 (与 Issue #119 / #210 一致)
EMB_DIM = 768
E_DIM = 32
CODEBOOK_SIZES = [64, 128, 256]
ENCODER_LAYERS = [512, 256, 128, 64]
BETA = 0.25
SK_EPSILONS = [0.0, 0.0, 0.0]
SK_ITERS = 3
FIX_C = True  # Issue #119 FIXED_CURV 模式

# Item embedding (Issue #119 复用的 issue210 Stage1 产物)
ITEM_EMB_NPY = "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history_v77_backup/taskA_stage1_hyp_v2/item_emb_u32.npy"

# Issue #119 8 ckpt 根目录
ISSUE119_CKPT_ROOT = "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/issue219_per_curvature_retrain"

# Issue #120 产物 (per-κ, 3 层 pre-quantization residual)
OUTPUT_ROOT = "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/issue120_pre_quant_residual_ref"


def load_stage2_model(ckpt_path: str, device: str = "cuda"):
    """加载 Stage2 KappaAwareHRQVAE 模型 + ckpt 权重."""
    import sys
    sys.path.insert(0, PYTHON_MODULE_SRC)
    from taskA_stage2 import KappaAwareHRQVAE  # noqa: E402

    model = KappaAwareHRQVAE(
        in_dim=EMB_DIM,
        num_emb_list=CODEBOOK_SIZES,
        e_dim=E_DIM,
        layers=ENCODER_LAYERS,
        beta=BETA,
        kmeans_init=False,
        kmeans_iters=0,
        sk_eps=SK_EPSILONS,
        sk_iters=SK_ITERS,
        fix_c=FIX_C,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]
    # strict=False 容忍 DDP module. 前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[len("module."):]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    model.to(device)
    return model


def extract_pre_quant_residual(model, item_emb: torch.Tensor, device: str = "cuda"):
    """对 item_emb 跑 encoder + vq_layers, 返回 3 层 pre-quantization residual (latent 32-dim).

    L0 = encoder(item_emb)
    L1 = residual_after_L0 = L0 - L0_quantized
    L2 = residual_after_L1 = L1 - L1_quantized
    """
    item_emb = item_emb.to(device)
    with torch.no_grad():
        z = model.encoder(item_emb)  # (N, 32)
        residuals = [z.cpu().numpy()]
        residual = z
        for q in model.vq_layers:
            x_res, _, _ = q(residual, use_sk=True)
            residual = residual - x_res
            residuals.append(residual.cpu().numpy())
    return residuals  # 3 层, 每层 (N, 32)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)

    # 加载 item_emb
    item_emb = torch.from_numpy(np.load(ITEM_EMB_NPY)).float()
    print(f"[issue120-stage2] item_emb shape: {item_emb.shape}, dtype: {item_emb.dtype}")

    # 8 κ 循环
    for kappa in STAGE2_KAPPAS:
        tag = STAGE2_KAPPA_TAGS[kappa]
        ckpt_path = f"{ISSUE119_CKPT_ROOT}/c_fixed_k{tag}/hrqvae_kappa_sync.ckpt"
        if not Path(ckpt_path).exists():
            print(f"[issue120-stage2] ⚠ missing ckpt: {ckpt_path}, skip")
            continue
        out_dir = f"{OUTPUT_ROOT}/pre_quant_residual_k{tag}"
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        print(f"[issue120-stage2] === κ={kappa} (ckpt: {ckpt_path}) ===")

        model = load_stage2_model(ckpt_path, device=device)
        residuals = extract_pre_quant_residual(model, item_emb, device=device)

        # 保存 3 层 pre-quantization residual
        for layer_idx, resid in enumerate(residuals):
            out_path = f"{out_dir}/L{layer_idx}_pre_quant_residual.npy"
            np.save(out_path, resid)
            print(f"[issue120-stage2]   L{layer_idx} shape={resid.shape} → {out_path}")

        # 释放
        del model
        torch.cuda.empty_cache()

    # 写 manifest
    manifest = {
        "issue": "#120",
        "date": "2026-08-11",
        "method": "extract pre-quantization residual from 8 κ × 100ep Stage2 ckpts (Issue #119)",
        "kappas": STAGE2_KAPPAS,
        "layers": ["L0", "L1", "L2"],
        "shape_per_layer": [list(item_emb.shape[0:1]) + [E_DIM]],
        "output_root": OUTPUT_ROOT,
    }
    with open(f"{OUTPUT_ROOT}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[issue120-stage2] wrote {OUTPUT_ROOT}/manifest.json")


if __name__ == "__main__":
    main()
