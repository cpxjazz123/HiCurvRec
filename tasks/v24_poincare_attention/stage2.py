#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v24 Stage 2 stub — 复用 v22.b Stage2 ckpt + SID (R34 允许 stub if 产物已存在).

v24 显式消费 Stage2 branch curvature (per-L0 for L1, per-(L0,L1) for L2). 直接复用 v22.b 产物:
  - hrqvae_kappa_sync.ckpt (含 final_kappas + vq_layers.{1,2}.delta_kappa.weight)
  - sid_output.npy (sha=332c948cce32...)
  - sid_metadata.json

R30 严格: 硬编码路径常量, 不读 os.environ.
R34 合规: 单脚本入口, 不 fork 版本.
"""
import hashlib
import os
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
STAGE2_PRODUCT_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/v22b_full_1000ep")
STAGE2_CKPT = STAGE2_PRODUCT_DIR / "hrqvae_kappa_sync.ckpt"
SID_NPY = STAGE2_PRODUCT_DIR / "sid_output.npy"
SID_METADATA = STAGE2_PRODUCT_DIR / "sid_metadata.json"
SID_SHA = "332c948cce329944ee6a879636e91d261377b04766eec47df5b1fff9edc6eec5"  # v22.b file SHA


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    # 验证 Stage2 ckpt 存在 + SID sha 匹配 (R18 + R5 baseline 一致性)
    if not STAGE2_CKPT.exists():
        raise FileNotFoundError(f"Stage2 ckpt 不存在: {STAGE2_CKPT}")
    if not SID_NPY.exists():
        raise FileNotFoundError(f"SID npy 不存在: {SID_NPY}")
    if not SID_METADATA.exists():
        raise FileNotFoundError(f"SID metadata 不存在: {SID_METADATA}")

    actual_sha = _sha256(SID_NPY)
    if actual_sha != SID_SHA:
        raise ValueError(
            f"SID sha mismatch: got {actual_sha}, expected {SID_SHA} ({SID_NPY})")

    # 验证 Stage2 ckpt 包含 curvature 所需字段
    import torch
    ckpt = torch.load(str(STAGE2_CKPT), map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    required = [
        "vq_layers.1.delta_kappa.weight",  # (64, 1)
        "vq_layers.2.delta_kappa.weight",  # (8192, 1)
    ]
    for k in required:
        if k not in sd:
            raise KeyError(f"Stage2 ckpt 缺 curvature 字段: {k}")
    final_kappas = ckpt.get("final_kappas", None)
    if final_kappas is None or len(final_kappas) != 3:
        raise ValueError(f"Stage2 ckpt final_kappas 不合法: {final_kappas}")

    print(f"[v24-stage2] reuse Stage2 ckpt: {STAGE2_CKPT}")
    print(f"[v24-stage2] reuse SID: {SID_NPY} (sha={SID_SHA[:16]}...)")
    print(f"[v24-stage2] final_kappas = {[round(float(k), 4) for k in final_kappas]}")
    print(f"[v24-stage2] L1 delta_kappa std = {sd['vq_layers.1.delta_kappa.weight'].std().item():.4f}")
    print(f"[v24-stage2] L2 delta_kappa std = {sd['vq_layers.2.delta_kappa.weight'].std().item():.4f}")


if __name__ == "__main__":
    main()