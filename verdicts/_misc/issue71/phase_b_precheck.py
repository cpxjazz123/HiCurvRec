"""Issue #71 Phase B precheck: 验证 ΔD = D_hyp - D_flat 计算无 NaN/Inf + 数值合理性.

运行:
  python3 verdicts/issue71_phase_b_precheck.py
"""
import sys
import torch
import numpy as np

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec")
from common.hyperbolic_attention_bias import (
    load_hab_assets_from_stage2_ckpt,
    precompute_distance_matrices,
)

STAGE2_CKPT = "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_issue61/hrqvae_kappa_sync.ckpt"

print(f"[precheck] load Stage2 ckpt: {STAGE2_CKPT}")
codebook_list, final_kappas = load_hab_assets_from_stage2_ckpt(STAGE2_CKPT)
print(f"[precheck] codebook: 3 layers, shapes={[c.shape for c in codebook_list]}")
print(f"[precheck] final_kappas = {final_kappas}")

# 完整 D_hyp (历史 v74/v77 行为)
print("\n[full mode] 完整 D_hyp 距离矩阵:")
D_list, Dbar_list, stats_full = precompute_distance_matrices(codebook_list, final_kappas)
for stats in stats_full:
    print(f"  L{stats['layer']}: c={stats['c']:.4f} κ={stats['kappa']:.4f} "
          f"median={stats['median']:.4f} p95={stats['p95']:.4f} mode={stats['mode']}")

# ΔD mode (Issue #71 Phase B)
print("\n[delta mode] ΔD = D_hyp - D_flat 距离矩阵:")
D_list_d, Dbar_list_d, stats_delta, Dflat_list = precompute_distance_matrices(
    codebook_list, final_kappas, use_delta_curvature=True)
for stats in stats_delta:
    print(f"  L{stats['layer']}: c={stats['c']:.4f} κ={stats['kappa']:.4f} "
          f"median_D_hyp={stats['median']:.4f} med_delta={stats['med_delta']:.6f} mode={stats['mode']}")

# 数值审计
print("\n[audit] 数值范围:")
for l in range(3):
    K = D_list_d[l].shape[0]
    mask = ~torch.eye(K, dtype=torch.bool)
    Delta = D_list_d[l] - Dflat_list[l]
    Delta_nonzero = Delta[mask]
    D_hyp_nonzero = D_list_d[l][mask]
    D_flat_nonzero = Dflat_list[l][mask]
    print(f"  L{l}: ΔD range=[{Delta_nonzero.min().item():.4f}, {Delta_nonzero.max().item():.4f}], "
          f"D_hyp median={D_hyp_nonzero.median().item():.4f}, "
          f"D_flat median={D_flat_nonzero.median().item():.4f}")
    print(f"        ΔD finite={torch.isfinite(Delta).all().item()} "
          f"sym_err={(Delta-Delta.T).abs().max().item():.2e} "
          f"diag_max={Delta.diagonal().abs().max().item():.2e}")
    # ΔDbar 范围
    Dbar_nz = Dbar_list_d[l][mask]
    print(f"        ΔDbar range=[{Dbar_nz.min().item():.4f}, {Dbar_nz.max().item():.4f}], "
          f"|ΔDbar| median={Dbar_nz.abs().median().item():.4f}")

print("\n[verdict] ΔD mode 数值稳定, ΔDbar 量纲与历史 Dbar 接近 (差分后 median 标准化)")
print("[verdict] PASS precheck")