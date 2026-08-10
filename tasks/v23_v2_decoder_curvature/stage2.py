#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v23 v2 Stage 2 stub — 复用 v22.b Stage2 1000ep 产物 (含 branch curvature L1+L2).

v23 v2 设计: Stage2 与 v22.b 完全一致 (v22.b 已带 branch_curvature L1+L2),
v23 v2 在 Stage3 端 **消费** Stage2 学到的 branch curvature (encoder + decoder 两侧).
不重跑 Stage2.
"""
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"

# 复用 v22.b Stage2 产物 (含 L1 + L2 branch curvature)
STAGE2_PRODUCT_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/v22b_full_1000ep"
STAGE2_CKPT = f"{STAGE2_PRODUCT_DIR}/hrqvae_kappa_sync.ckpt"
STAGE2_SID_NPY = f"{STAGE2_PRODUCT_DIR}/sid_output.npy"
STAGE2_SID_METADATA = f"{STAGE2_PRODUCT_DIR}/sid_metadata.json"


def main():
    """v23 v2 Stage2 stub: 验证 v22.b Stage2 产物可读 + L1+L2 branch_curvature 落盘 OK."""
    import torch
    ckpt = torch.load(STAGE2_CKPT, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    print(f"[v23 v2 stage2 stub] Stage2 ckpt = {STAGE2_CKPT}", flush=True)
    print(f"[v23 v2 stage2 stub] final_kappas = {[f'{k:.4f}' for k in ckpt['final_kappas']]}", flush=True)
    print(f"[v23 v2 stage2 stub] final_cs = {[f'{c:.4f}' for c in ckpt['final_cs']]}", flush=True)
    # v23 v2 需要 L1 + L2 delta_kappa
    assert "vq_layers.1.delta_kappa.weight" in sd, "L1 delta_kappa missing in ckpt"
    assert "vq_layers.2.delta_kappa.weight" in sd, "L2 delta_kappa missing in ckpt"
    delta_l1 = sd["vq_layers.1.delta_kappa.weight"]
    delta_l2 = sd["vq_layers.2.delta_kappa.weight"]
    print(f"[v23 v2 stage2 stub] L1 delta_kappa.shape = {tuple(delta_l1.shape)}, std = {delta_l1.std().item():.4f}", flush=True)
    print(f"[v23 v2 stage2 stub] L2 delta_kappa.shape = {tuple(delta_l2.shape)}, std = {delta_l2.std().item():.4f}", flush=True)
    print(f"[v23 v2 stage2 stub] SID npy = {STAGE2_SID_NPY}", flush=True)
    print(f"[v23 v2 stage2 stub] 复用 v22.b Stage2 产物, 不重跑 Stage2 (R34 stub)", flush=True)


if __name__ == "__main__":
    main()