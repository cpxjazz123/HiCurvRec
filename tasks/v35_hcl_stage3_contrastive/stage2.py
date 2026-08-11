#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v35 Stage 2 stub — 复用 v22.b Stage2 ckpt + SID (R34 允许 stub if 产物已存在).

v35 HCL 显式消费 Stage2 c_per_layer = final_cs (3-element, 单曲率 c_avg = mean(c_per_layer))
作 d_P 度量. 直接复用 v22.b 产物:
  - hrqvae_kappa_sync.ckpt (含 final_cs)
  - sid_output.npy (sha=332c948cce32...)
  - sid_metadata.json

R30 严格: 硬编码路径常量, 不读 os.environ.
R34 合规: 单脚本入口, 不 fork 版本.
"""
import hashlib
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

    # 验证 Stage2 ckpt 包含 HCL 所需字段 (final_cs)
    import torch
    ckpt = torch.load(str(STAGE2_CKPT), map_location="cpu", weights_only=False)
    final_cs = ckpt.get("final_cs", None)
    if final_cs is None or len(final_cs) != 3:
        raise ValueError(f"Stage2 ckpt final_cs 不合法: {final_cs}")

    c_avg = sum(float(c) for c in final_cs) / 3
    print(f"[v35-stage2] reuse Stage2 ckpt: {STAGE2_CKPT}")
    print(f"[v35-stage2] reuse SID: {SID_NPY} (sha={SID_SHA[:16]}...)")
    print(f"[v35-stage2] final_cs = {[round(float(c), 4) for c in final_cs]}")
    print(f"[v35-stage2] c_avg (单曲率 for HCL d_P) = {c_avg:.4f}")


if __name__ == "__main__":
    main()