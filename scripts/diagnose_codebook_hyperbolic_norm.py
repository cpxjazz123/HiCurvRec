"""码字双曲范数诊断 (post-hoc).

扫描所有 κ-Stereo / HG-Rec / fixed-baseline 的 HRQ-VAE ckpt, 加载每个 codebook
embedding 后, 应用 proj_to_ball(expmap0(w, κ), κ) 把码字投到 Poincaré 球内,
计算每层码字的:
  - Euclidean norm ‖x‖_E (球内)
  - Hyperbolic norm ‖x‖_κ = (2/√κ) artanh(√κ ‖x‖_E)
  - Conformal factor λ_κ(x) = 2/(1 - κ‖x‖_E²)
  - 分布: mean, q05/q25/q50/q75/q95, max

回答用户的核心问题:
  "码字是否走到能感受到曲率的范围 (‖x‖ vs 1/√c)?"
如果 ‖x‖_E 普遍 < 0.1, 那 κ∈[0.1, 2.0] 之间的差异在数值上几乎无效,
模型"学到 κ≈0"或"κ冲边界"都只是参数漂移噪声.

结果写到:
  products/codebook_hypnorm_diagnostic.json  (机器可读)
  verdicts/codebook_hypnorm_diagnostic.md     (人读 verdict)
"""
import argparse
import json
import logging
import os
import sys
from typing import Optional

import numpy as np
import torch

# 加 PYTHONPATH 跟其他脚本一致
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')
from model.utils import artanh, expmap0, proj_to_ball

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
log = logging.getLogger("diagnose_hypnorm")


def quantiles(x: torch.Tensor, qs=(0.05, 0.25, 0.5, 0.75, 0.95)):
    x_np = x.detach().cpu().numpy()
    return {f"q{int(q*100):02d}": float(np.quantile(x_np, q)) for q in qs}


def stats_for_layer(codebook_h: torch.Tensor, kappa: float, layer_idx: int) -> dict:
    """codebook_h: (K, e_dim), already projected to ball via proj_to_ball(expmap0(*, c), c).
    kappa: 该层使用的曲率.
    """
    eucl_norm = codebook_h.norm(dim=-1)  # (K,)
    # Hyperbolic norm: ‖x‖_κ = (2/√κ) artanh(√κ ‖x‖_E)
    sqrt_c = float(np.sqrt(kappa))
    arg = (sqrt_c * eucl_norm).clamp(max=1.0 - 1e-5)
    hyp_norm = (2.0 / sqrt_c) * artanh(arg)  # (K,)
    # Conformal factor λ_κ(x) = 2/(1 - κ‖x‖²)
    conformal = 2.0 / (1.0 - kappa * (eucl_norm ** 2)).clamp_min(1e-5)
    # 球边界 1/√κ
    ball_radius = 1.0 / sqrt_c

    return {
        "layer_idx": layer_idx,
        "kappa": float(kappa),
        "ball_radius": float(ball_radius),
        "n_codebook": int(codebook_h.shape[0]),
        "e_dim": int(codebook_h.shape[1]),
        # Euclidean norm
        "eucl_norm_mean": float(eucl_norm.mean()),
        "eucl_norm_max": float(eucl_norm.max()),
        "eucl_norm_quantiles": quantiles(eucl_norm),
        # Hyperbolic norm
        "hyp_norm_mean": float(hyp_norm.mean()),
        "hyp_norm_max": float(hyp_norm.max()),
        "hyp_norm_quantiles": quantiles(hyp_norm),
        # Conformal factor
        "conformal_mean": float(conformal.mean()),
        "conformal_max": float(conformal.max()),
        "conformal_min": float(conformal.min()),
        # 归一化球占用率 (norm / ball_radius)
        "ball_fill_ratio_mean": float((eucl_norm.mean() / ball_radius)),
        "ball_fill_ratio_max": float((eucl_norm.max() / ball_radius)),
    }


def diagnose_ckpt(ckpt_path: str, kappa_max: Optional[float] = None,
                  fixed_kappa: Optional[list] = None) -> dict:
    """诊断单个 ckpt.

    Args:
        ckpt_path: HRQ-VAE ckpt path
        kappa_max: 用于 theta_m 的 κ_max (默认 2.0)
        fixed_kappa: 固定曲率 list, 长度等于层数 (用于 κ LOCKED 类)

    Returns:
        dict: 每层的 norm 分布 + 元数据
    """
    if kappa_max is None:
        kappa_max = 2.0

    log.info(f"Loading {ckpt_path}")
    obj = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    # 兼容 2 种结构: dict (有 state_dict) 或 OrderedDict (直接 state_dict)
    if isinstance(obj, dict) and 'state_dict' in obj:
        sd = obj['state_dict']
        args_dict = vars(obj.get('args', None)) if obj.get('args') is not None and hasattr(obj.get('args'), '__dict__') else {}
    else:
        sd = obj
        args_dict = {}

    # 找所有 VQ layer
    vq_layer_indices = []
    for k in sd.keys():
        if k.startswith('hrq.vq_layers.') and k.endswith('.embeddings.weight'):
            layer_idx = int(k.split('.')[2])
            vq_layer_indices.append(layer_idx)
    vq_layer_indices = sorted(vq_layer_indices)

    n_layers = len(vq_layer_indices)
    log.info(f"  Detected {n_layers} VQ layers: {vq_layer_indices}")

    layers_stats = []
    for layer_idx in vq_layer_indices:
        # 1) 拿码字权重
        w_key = f'hrq.vq_layers.{layer_idx}.embeddings.weight'
        if w_key not in sd:
            raise ValueError(f"Missing {w_key} in ckpt")
        w = sd[w_key]  # (K, e_dim) Euclidean
        w = w.float()

        # 2) 拿 κ
        kappa = None
        # case A: kappa_locked_tensor (task175) — 可能是 scalar 或 multi-dim
        kl_key = f'hrq.vq_layers.{layer_idx}.kappa_locked_tensor'
        if kl_key in sd:
            t = sd[kl_key]
            kappa = float(t.mean().item()) if t.numel() > 1 else float(t.item())
        # case B: theta_m + kappa_max (task164/169/170/171/172/176/177)
        elif f'hrq.vq_layers.{layer_idx}.theta_m' in sd and kappa_max is not None:
            theta = sd[f'hrq.vq_layers.{layer_idx}.theta_m']
            # theta 可能是 scalar 或 (M,) 多分量
            kappa_arr = kappa_max * torch.tanh(theta)
            kappa = float(kappa_arr.mean().item())
        # case C: fixed_kappa list (手工指定)
        elif fixed_kappa is not None and layer_idx < len(fixed_kappa):
            kappa = float(fixed_kappa[layer_idx])
        # case D: 默认 c=1.0 (task180, task84 没有显式 κ)
        if kappa is None:
            kappa = 1.0
            log.info(f"  Layer {layer_idx}: no kappa found, using default c=1.0")

        # 防御: 如果 κ<=0, Poincaré 模型不定义. 我们 fallback 到 c=1.0 报告 norm
        # 并在结果里标注 κ_used=0 触发警告 (用户原始问题里的核心信号).
        kappa_invalid = (kappa <= 0)
        kappa_for_ball = max(kappa, 1e-3)  # 防止 sqrt 出 nan

        # 3) 应用 proj_to_ball(expmap0(w, κ), κ) 把码字投到球内
        # 用 kappa_for_ball (>=1e-3) 防御
        codebook_h = proj_to_ball(expmap0(w, kappa_for_ball), kappa_for_ball)

        # 4) 计算该层 stats (用原 κ, 不是 for_ball)
        #    如果 κ<=0, conformal 和 hyp_norm 用原始定义, 但数值会 inf
        st = stats_for_layer(codebook_h, kappa_for_ball if kappa_invalid else kappa, layer_idx)
        if kappa_invalid:
            st["kappa_warning"] = f"Original κ={kappa:.4f} <= 0, used κ=1e-3 fallback for ball projection"
            st["kappa"] = kappa  # report original
        layers_stats.append(st)

    return {
        "ckpt_path": ckpt_path,
        "n_layers": n_layers,
        "kappa_max_used": kappa_max,
        "args_summary": {k: v for k, v in args_dict.items()
                         if k in ['curvatures', 'kappa_max', 'loss_type', 'beta', 'sk_epsilons']},
        "layers": layers_stats,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_json", type=str,
                        default="/home/wlia0047/ar57/wenyu/GeneRec/products/codebook_hypnorm_diagnostic.json")
    args = parser.parse_args()

    # 待诊断 ckpt 清单 (按任务重要性排序)
    targets = [
        # 当前活跃 3 任务 (post-Phase 0 修正版 baseline)
        ("task180_graph_aware", "products/task180/hrqvae_graph_aware/Jul-25-2026_04-56-06/Jul-25-2026_14-56-11_beta_0.500_codebook_[64,128,256]_sk_0.003/best_loss_model.pth",
         {"kappa_max": 2.0}),  # 任务实际是 c=1.0 (no theta_m), 但保持 kappa_max=2.0 兼容
        # 修正 baseline Phase 1 task178: 还没落盘 best_collision_model (Stage 3 用的是 hrqvae? task178 hrqvae_fixed_baseline)
        # task84 是 HG-Rec 原版 baseline
        ("task84_baseline", "products/task84/ckpt/Instruments/Jul-23-2026_20-08-06_beta_1.000_codebook_[64,128,256]_sk_0.000/best_loss_model.pth",
         {"kappa_max": 2.0}),
        # κ-Stereo 8-task NO-GO 系列
        ("task164_kappa_stereographic", "products/task164/kappa_stereographic_test/jul-25-2026_00-55-29/best_loss_model.pth",
         {"kappa_max": 2.0}),
        ("task169_phase_b_sinkhorn", "products/task169/phase_b_kappa_sinkhorn/jul-25-2026_11-08-34/best_loss_model.pth",
         {"kappa_max": 2.0}),
        ("task170_phase_b_sinkhorn_all3", "products/task170/phase_b_kappa_sinkhorn_all3/jul-25-2026_11-38-53/best_loss_model.pth",
         {"kappa_max": 2.0}),
        ("task171_phase_b_dead_code", "products/task171/phase_b_kappa_dead_code/jul-25-2026_11-38-53/best_loss_model.pth",
         {"kappa_max": 2.0}),
        ("task172_phase_b_kappa_max4", "products/task172/phase_b_kappa_max4/jul-25-2026_11-40-18/best_loss_model.pth",
         {"kappa_max": 4.0}),
        # κ LOCKED at ORC 实测值
        ("task175_orc_locked", "products/task175/orc_locked/best_loss_model.pth",
         {"kappa_max": None}),  # 走 kappa_locked_tensor 分支
        # β(x) 系列 (broken baseline, 但 codebook 仍可看)
        ("task176_posdep_sigmoid", "products/task176/posdep_sigmoid/best_loss_model.pth",
         {"kappa_max": 2.0}),
        ("task177_posdep_conformal", "products/task177/posdep_conformal/best_loss_model.pth",
         {"kappa_max": 2.0}),
    ]

    results = {}
    for name, path, opts in targets:
        if not os.path.exists(path):
            log.warning(f"  Missing: {path}")
            results[name] = {"missing": True, "path": path}
            continue
        try:
            r = diagnose_ckpt(path, **opts)
            results[name] = r
            # 打印关键 stats
            for layer in r['layers']:
                log.info(
                    f"  [{name}] L{layer['layer_idx']} κ={layer['kappa']:.4f} | "
                    f"EuclNorm mean={layer['eucl_norm_mean']:.4f} max={layer['eucl_norm_max']:.4f} | "
                    f"HypNorm mean={layer['hyp_norm_mean']:.4f} | "
                    f"Conformal mean={layer['conformal_mean']:.4f} | "
                    f"BallFill={layer['ball_fill_ratio_mean']:.4f}"
                )
        except Exception as e:
            log.error(f"  Failed {name}: {e}")
            results[name] = {"error": str(e), "path": path}

    # 写出 JSON
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"Wrote {args.output_json}")

    # 总结
    log.info("\n=== 关键发现汇总 ===")
    log.info("判定规则: 如果 EuclNorm mean < 0.1, 则 λ_κ(x) ≈ 2.0 (与 κ 无关), 曲率数值无效")
    for name, r in results.items():
        if 'layers' not in r:
            continue
        for layer in r['layers']:
            em = layer['eucl_norm_mean']
            emax = layer['eucl_norm_max']
            cf = layer['conformal_mean']
            verdict = "INVALID" if em < 0.1 else "MAYBE_VALID"
            log.info(
                f"  [{name:35s} L{layer['layer_idx']}] κ={layer['kappa']:.3f} "
                f"‖x‖_E={em:.4f} (max {emax:.4f}) conformal={cf:.4f} → {verdict}"
            )


if __name__ == '__main__':
    main()