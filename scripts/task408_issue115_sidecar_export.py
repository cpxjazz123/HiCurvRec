#!/usr/bin/env python3
"""Task #408 / Issue #115 [方向A Gate1] 独立导出 #100 Stage 1 sidecar

Per Issue #115 spec §Gate1:
- 加载 task84 HG-Rec baseline (200 epoch best_collision_model.pth, = #100 PASS 等价产物)
- 导出每层: θ (learnable curvature parameter), κ=κ_max·tanh(θ) (effective curvature),
            scale (per-layer), codebook norm, K 数量, checkpoint SHA256
- 三层 = L0 K64 / L1 K128 / L2 K256 (固定)
- 若 checkpoint 实际缺失或三层不符 → Gate 1 FAIL

实施: per-component kappa + per-layer scale + per-layer codebook
"""
from __future__ import annotations

import sys
import os
import json
import hashlib
from pathlib import Path

import torch
import numpy as np

HGREC_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec").resolve()
sys.path.insert(0, str(HGREC_ROOT))

# Issue #97 patch (跟 task84/task394 一致)
from model import utils as hgrec_utils  # type: ignore


def patched_poincare_distance(x, y, c):
    diff = hgrec_utils.mobius_add(-x, y, c)
    diff = hgrec_utils.proj_to_ball(diff, c)
    sqrt_c = c ** 0.5
    norm = diff.norm(dim=-1, keepdim=True).clamp_min(hgrec_utils._eps(diff))
    return (2.0 / sqrt_c) * hgrec_utils.artanh(sqrt_c * norm)


hgrec_utils.poincare_distance = patched_poincare_distance

from model.hrqvae_free_curv import FreeCurvVectorQuantization  # type: ignore


#===========================================================================================
# Configuration
#===========================================================================================
CKPT_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/products/task84/ckpt/Instruments/Jul-23-2026_20-08-06_beta_1.000_codebook_[64,128,256]_sk_0.000/best_collision_model.pth"
EXPECTED_K = [64, 128, 256]
EXPECTED_E_DIM = 32
EXPECTED_M = 3
EXPECTED_KAPPA_MAX = 2.0

SID_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_task396.npy"

OUT_JSON = Path("/home/wlia0047/ar57/wenyu/GeneRec/verdicts/task408_issue115_sidecar.json")
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)


#===========================================================================================
# Helpers
#===========================================================================================
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_state_dict(ckpt_path):
    """Load state_dict from FreeCurvHRQVAE checkpoint.

    task84 HG-Rec baseline ckpt 格式: dict wrapper with keys 'args', 'epoch', 'best_loss',
    'best_collision_rate', 'state_dict', 'optimizer'. Real state_dict is in 'state_dict' field.
    """
    # weights_only=False for trusted HG-Rec task84 ckpt (R2 + R12 安全: 自己训练产物)
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "state_dict" in raw:
        return raw["state_dict"]
    return raw


def parse_free_curv_state(state_dict):
    """Parse FreeCurvHRQVAE state_dict: 'layers.{l}.theta_m' + 'layers.{l}.embeddings.weight' + 'encoder.*' + 'decoder.*'"""
    # Find per-layer keys
    layer_keys = {}
    for key in state_dict.keys():
        if key.startswith("layers."):
            parts = key.split(".")
            layer_idx = int(parts[1])
            field = ".".join(parts[2:])
            if layer_idx not in layer_keys:
                layer_keys[layer_idx] = {}
            layer_keys[layer_idx][field] = state_dict[key]

    return layer_keys


def extract_layer_sidecar(layer_state, layer_idx, expected_k, expected_e_dim, kappa_max):
    """Extract sidecar from one layer's state."""
    # θ_m (learnable per-component curvature parameter)
    theta_m = layer_state.get("theta_m", None)
    if theta_m is None:
        return {
            "layer": layer_idx,
            "error": f"Layer {layer_idx} missing theta_m",
            "gate1_pass": False,
        }

    # Compute effective κ_m = κ_max · tanh(θ_m)
    theta_np = theta_m.detach().cpu().numpy()
    kappa_np = kappa_max * np.tanh(theta_np)

    # Codebook embeddings
    embeddings = layer_state.get("embeddings.weight", None)
    if embeddings is None:
        return {
            "layer": layer_idx,
            "error": f"Layer {layer_idx} missing embeddings.weight",
            "gate1_pass": False,
        }

    K, e_dim = embeddings.shape

    # Validate K and e_dim
    if K != expected_k:
        return {
            "layer": layer_idx,
            "error": f"Layer {layer_idx} K={K} != expected {expected_k}",
            "gate1_pass": False,
        }
    if e_dim != expected_e_dim:
        return {
            "layer": layer_idx,
            "error": f"Layer {layer_idx} e_dim={e_dim} != expected {expected_e_dim}",
            "gate1_pass": False,
        }

    # Codebook stats
    emb_np = embeddings.detach().cpu().numpy()
    codebook_norm_per_entry = np.linalg.norm(emb_np, axis=1)
    codebook_norm_mean = float(codebook_norm_per_entry.mean())
    codebook_norm_std = float(codebook_norm_per_entry.std())
    codebook_norm_min = float(codebook_norm_per_entry.min())
    codebook_norm_max = float(codebook_norm_per_entry.max())

    # Per-component codebook norms (split by block_dims)
    # FreeCurvVQ with M=3, e_dim=32 → block_dims = [11, 11, 10]
    if expected_e_dim == 32 and len(theta_np) == 3:
        block_dims = [11, 11, 10]
        per_comp_norms = []
        start = 0
        for m, bd in enumerate(block_dims):
            comp = emb_np[:, start:start+bd]
            per_comp_norms.append(float(np.linalg.norm(comp, axis=1).mean()))
            start += bd
    else:
        per_comp_norms = None

    # κ stats
    kappa_min = float(kappa_np.min())
    kappa_max_actual = float(kappa_np.max())
    kappa_mean = float(kappa_np.mean())
    kappa_abs_mean = float(np.abs(kappa_np).mean())

    # scale (issue spec §Gate1: "scale" 是 per-layer 的标量; HG-Rec 中用 β)
    # 注: HG-Rec baseline 没有显式 scale param, 但 issue spec 要求
    # 这里用 per-layer codebook norm mean 作为 scale proxy (per-layer 总尺度)
    scale = codebook_norm_mean

    return {
        "layer": layer_idx,
        "K": int(K),
        "e_dim": int(e_dim),
        "theta_m": theta_np.tolist(),
        "kappa_m_effective": kappa_np.tolist(),
        "kappa_min": kappa_min,
        "kappa_max": kappa_max_actual,
        "kappa_mean": kappa_mean,
        "kappa_abs_mean": kappa_abs_mean,
        "scale": scale,
        "codebook_norm_mean": codebook_norm_mean,
        "codebook_norm_std": codebook_norm_std,
        "codebook_norm_min": codebook_norm_min,
        "codebook_norm_max": codebook_norm_max,
        "per_component_norm_mean": per_comp_norms,
        "gate1_pass": True,
    }


def main():
    print("=" * 80, flush=True)
    print("[Task #408 / Issue #115 Gate 1] 独立导出 #100 Stage 1 sidecar", flush=True)
    print("=" * 80, flush=True)

    # === Stage 1 ckpt SHA256 + size ===
    print(f"\n[Ckpt] {CKPT_PATH}", flush=True)
    ckpt_sha256 = sha256_file(CKPT_PATH)
    ckpt_size = os.path.getsize(CKPT_PATH)
    print(f"  SHA256: {ckpt_sha256}", flush=True)
    print(f"  Size: {ckpt_size} bytes", flush=True)

    # === Load state_dict ===
    print(f"\n[Load] state_dict...", flush=True)
    state = load_state_dict(CKPT_PATH)
    print(f"  Total keys: {len(state)}", flush=True)

    # === Parse per-layer ===
    layer_keys = parse_free_curv_state(state)
    print(f"  Found {len(layer_keys)} layers in state_dict: {sorted(layer_keys.keys())}", flush=True)

    # === Extract sidecar per layer ===
    sidecar = {
        "ckpt_path": CKPT_PATH,
        "ckpt_sha256": ckpt_sha256,
        "ckpt_size_bytes": ckpt_size,
        "expected_K": EXPECTED_K,
        "expected_e_dim": EXPECTED_E_DIM,
        "expected_M": EXPECTED_M,
        "expected_kappa_max": EXPECTED_KAPPA_MAX,
        "layers": [],
    }

    all_pass = True
    for l in sorted(layer_keys.keys()):
        layer_sidecar = extract_layer_sidecar(
            layer_keys[l], l,
            expected_k=EXPECTED_K[l],
            expected_e_dim=EXPECTED_E_DIM,
            kappa_max=EXPECTED_KAPPA_MAX,
        )
        sidecar["layers"].append(layer_sidecar)
        if not layer_sidecar["gate1_pass"]:
            all_pass = False
        print(f"\n[Layer {l}]", flush=True)
        print(f"  K={layer_sidecar.get('K')}, e_dim={layer_sidecar.get('e_dim')}", flush=True)
        print(f"  theta_m={layer_sidecar.get('theta_m')}", flush=True)
        print(f"  kappa_m_effective={layer_sidecar.get('kappa_m_effective')}", flush=True)
        print(f"  scale (codebook norm mean)={layer_sidecar.get('scale'):.4f}" if layer_sidecar.get('scale') else "  scale=N/A", flush=True)
        print(f"  gate1_pass={layer_sidecar['gate1_pass']}", flush=True)

    # === SID SHA256 ===
    print(f"\n[SID] {SID_PATH}", flush=True)
    sid_sha256 = sha256_file(SID_PATH)
    sid_size = os.path.getsize(SID_PATH)
    sid_array = np.load(SID_PATH)
    print(f"  SHA256: {sid_sha256}", flush=True)
    print(f"  Size: {sid_size} bytes", flush=True)
    print(f"  Shape: {sid_array.shape}", flush=True)
    print(f"  Unique SID: {len(np.unique(sid_array, axis=0))}/{len(sid_array)}", flush=True)

    # === Save audit JSON ===
    sidecar["sid_path"] = SID_PATH
    sidecar["sid_sha256"] = sid_sha256
    sidecar["sid_size_bytes"] = sid_size
    sidecar["sid_shape"] = list(sid_array.shape)
    sidecar["sid_unique"] = int(len(np.unique(sid_array, axis=0)))
    sidecar["sid_unique_ratio"] = float(len(np.unique(sid_array, axis=0)) / len(sid_array))
    sidecar["gate1_pass"] = all_pass and sidecar["sid_unique_ratio"] == 1.0

    with open(OUT_JSON, "w") as f:
        json.dump(sidecar, f, indent=2)

    print(f"\n{'=' * 80}", flush=True)
    print(f"GATE 1 {'✅ PASS' if sidecar['gate1_pass'] else '❌ FAIL'}", flush=True)
    print(f"{'=' * 80}", flush=True)
    print(f"\nAudit saved: {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()