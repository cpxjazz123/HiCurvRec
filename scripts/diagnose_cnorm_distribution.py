"""
诊断: c‖x‖² 分布 + κ 实际学习轨迹 (用户 2026-07-25 跟进请求)

目的:
1. 报告 c‖x‖² 分布, 不要只报 ‖x‖_E 均值
   - 答: 有多少比例的码字满足 c‖x‖² > 1-ε (即被 clamp 到边界)
2. 确认 κ 实际学习轨迹
   - 答: 跨 epoch 看 κ 是否变化, 范数是否变化
   - 对 fixed-κ 任务 (task181), 用 6 个 epoch ckpts 看 ‖x‖² 演化
   - 对 learnable-κ 任务 (task164/169/170/171/172), 看最终 κ 值范围

不依赖任何外部 LLM, 纯 numpy 加载 + 统计.
"""

import json
import os
import sys
import numpy as np
import torch

# 把 repo root 加到 path, import hyperbolic utils
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')

from model.utils import proj_to_ball, expmap0  # noqa


def load_hrqvae_ckpt(ckpt_path: str, fixed_kappa: float = None, learnable_kappa: bool = True):
    """Load HRQ-VAE ckpt and extract per-layer codebook + κ.

    Args:
        ckpt_path: path to .pth
        fixed_kappa: if not None, use this κ for all layers (overrides learned)
        learnable_kappa: if True, try to read θ_m from state_dict and compute κ = κ_max * tanh(θ_m)
    """
    raw = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    # Ckpt 可能是 {'state_dict': ..., 'args': ..., 'kappa_history': ...} 嵌套结构
    if isinstance(raw, dict) and 'state_dict' in raw:
        sd = raw['state_dict']
        args_obj = raw.get('args', None) or {}
        kappa_history = raw.get('kappa_history', None)
        epoch = raw.get('epoch', None)
        # args 可能是 dict 或 Namespace
        if hasattr(args_obj, '__dict__'):
            args = vars(args_obj)
        else:
            args = args_obj
    else:
        sd = raw
        args = {}
        kappa_history = None
        epoch = None

    # Get κ_max from args (for learnable κ tasks)
    kappa_max = args.get('kappa_max', None)
    if isinstance(kappa_max, list):
        kappa_max = kappa_max[0] if len(kappa_max) > 0 else None

    layers_info = []
    layer_keys = sorted([k for k in sd.keys() if 'embedding' in k.lower() and 'weight' in k.lower()])

    print(f'  ckpt keys (sample): {list(sd.keys())[:8]} ...', flush=True)
    print(f'  embedding-related keys: {layer_keys}', flush=True)
    print(f'  args: kappa_max={kappa_max}, epoch={epoch}')
    if kappa_history is not None:
        print(f'  kappa_history: type={type(kappa_history).__name__}', flush=True)
        if isinstance(kappa_history, dict):
            print(f'    keys: {list(kappa_history.keys())}')
            print(f'    epochs: {kappa_history.get("epochs")}')
            print(f'    per_layer_per_m (first): {kappa_history.get("per_layer_per_m", [None])[0] if kappa_history.get("per_layer_per_m") else "empty"}')

    for li, k in enumerate(layer_keys):
        w = sd[k]  # (n_codebook, e_dim)

        # Extract κ for this layer
        if fixed_kappa is not None:
            kappa_l = fixed_kappa
        elif learnable_kappa and kappa_max is not None:
            # Try to read theta_m from state_dict
            theta_key = f'hrq.vq_layers.{li}.theta_m'
            if theta_key in sd:
                theta_m = sd[theta_key]
                if theta_m.numel() == 1:
                    theta_val = float(theta_m.item())
                else:
                    theta_val = float(theta_m.mean().item())
                kappa_l = float(kappa_max) * float(np.tanh(theta_val))
            else:
                kappa_l = float(kappa_max)
        else:
            kappa_l = 1.0  # HG-Rec baseline default

        if kappa_l <= 0:
            print(f'  L{li} WARNING: κ={kappa_l:.4f} (negative — spherical or trivial regime)')
            # Apply expmap0 with κ<0 — it should still work for spherical
            try:
                x_ball = proj_to_ball(expmap0(w, kappa_l), kappa_l)
            except Exception as e:
                print(f'  L{li} expmap failed: {e}, using raw')
                x_ball = w
            eucl_norm = x_ball.norm(dim=-1).numpy()
            # For κ<0 the ball condition is ‖x‖² < 1/|κ| (i.e., c‖x‖² < 1 with c=-κ)
            # So compute c‖x‖² where c is the *magnitude* of curvature (positive number)
            cn_abs = abs(kappa_l) * (eucl_norm ** 2)
            # boundary check: cn_abs > 1 means projected to boundary
            # but we use cn_sq = κ * ‖x‖² as signed indicator
            cn_signed = kappa_l * (eucl_norm ** 2)  # negative, so > -1+ε means "near boundary"
            layers_info.append({
                'layer_idx': li,
                'kappa': kappa_l,
                'kappa_signed_for_log': True,
                'n_codebook': int(w.shape[0]),
                'e_dim': int(w.shape[1]),
                'eucl_norm': eucl_norm,
                'cnorm_sq_signed': cn_signed,  # for κ<0 tasks, negative
                'cnorm_sq_abs': cn_abs,  # for boundary check
            })
        else:
            try:
                x_ball = proj_to_ball(expmap0(w, kappa_l), kappa_l)
            except Exception as e:
                print(f'  L{li} expmap failed: {e}, using raw')
                x_ball = w
            eucl_norm = x_ball.norm(dim=-1).numpy()
            c_norm_sq = kappa_l * (eucl_norm ** 2)  # ‖x‖² since c=κ, both positive
            layers_info.append({
                'layer_idx': li,
                'kappa': kappa_l,
                'kappa_signed_for_log': False,
                'n_codebook': int(w.shape[0]),
                'e_dim': int(w.shape[1]),
                'eucl_norm': eucl_norm,
                'cnorm_sq': c_norm_sq,
            })

    return {
        'kappa_max': kappa_max,
        'layers': layers_info,
        'kappa_history': _serialize_kappa_history(kappa_history),
    }


def _serialize_kappa_history(kh):
    """Convert kappa_history to JSON-serializable form."""
    if kh is None:
        return None
    if isinstance(kh, dict):
        return {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in kh.items()}
    return str(kh)


def stats_for_cnorm_sq(csq: np.ndarray, sign_convention: str = 'positive') -> dict:
    """Returns the c‖x‖² distribution stats.

    For κ>0 (hyperbolic), csq should be in [0, 1). For κ<0 (spherical), csq is negative.
    """
    if sign_convention == 'positive':
        q = np.quantile(csq, [0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999])
        return {
            'mean': float(csq.mean()),
            'max': float(csq.max()),
            'q05': float(q[0]),
            'q25': float(q[1]),
            'q50': float(q[2]),
            'q75': float(q[3]),
            'q90': float(q[4]),
            'q95': float(q[5]),
            'q99': float(q[6]),
            'q999': float(q[7]),
            'frac_gt_1_minus_1e-3': float((csq > 1 - 1e-3).mean()),
            'frac_gt_1_minus_1e-4': float((csq > 1 - 1e-4).mean()),
            'frac_gt_1_minus_1e-5': float((csq > 1 - 1e-5).mean()),
            'frac_gt_1_minus_1e-6': float((csq > 1 - 1e-6).mean()),
            'frac_exactly_1_or_above': float((csq >= 1.0).mean()),
        }
    else:
        # For κ<0: csq is negative. Boundary = 0 (i.e., > 0 means projected to boundary)
        # Actually for κ<0 (spherical), expmap maps to sphere of radius 1/sqrt(|κ|).
        # Boundary check: |κ| * ‖x‖² < 1 strictly inside.
        abs_csq = np.abs(csq)
        q = np.quantile(abs_csq, [0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999])
        return {
            'signed_mean': float(csq.mean()),
            'abs_mean': float(abs_csq.mean()),
            'abs_max': float(abs_csq.max()),
            'abs_q05': float(q[0]),
            'abs_q25': float(q[1]),
            'abs_q50': float(q[2]),
            'abs_q75': float(q[3]),
            'abs_q95': float(q[5]),
            'frac_gt_1_minus_1e-3': float((abs_csq > 1 - 1e-3).mean()),
            'frac_gt_1_minus_1e-4': float((abs_csq > 1 - 1e-4).mean()),
            'frac_gt_1_minus_1e-5': float((abs_csq > 1 - 1e-5).mean()),
            'frac_gt_1_minus_1e-6': float((abs_csq > 1 - 1e-6).mean()),
        }


def summarize_layer(l, kappa_max=None):
    """Build per-layer JSON entry."""
    is_signed = l.get('kappa_signed_for_log', False)
    if is_signed:
        cn_stats = stats_for_cnorm_sq(l['cnorm_sq_signed'], sign_convention='negative')
    else:
        cn_stats = stats_for_cnorm_sq(l['cnorm_sq'], sign_convention='positive')
    return {
        'layer_idx': l['layer_idx'],
        'kappa': l['kappa'],
        'n_codebook': l['n_codebook'],
        'e_dim': l['e_dim'],
        'eucl_norm': {
            'mean': float(l['eucl_norm'].mean()),
            'max': float(l['eucl_norm'].max()),
            'q05': float(np.quantile(l['eucl_norm'], 0.05)),
            'q50': float(np.quantile(l['eucl_norm'], 0.50)),
            'q95': float(np.quantile(l['eucl_norm'], 0.95)),
        },
        'cnorm_sq_distribution': cn_stats,
    }


def main():
    out = {}

    # === Part 1: task181 6 epoch ckpts — norm trajectory + c‖x‖² distribution ===
    print('=' * 70)
    print('Part 1: task181 norm evolution (50 epoch training, fixed κ=1)')
    print('=' * 70)
    task181_dir = '/home/wlia0047/ar57/wenyu/GeneRec/products/task181/hrqvae_norm_evolution/Jul-25-2026_15-15-18_beta_0.500_codebook_[64,128,256]_sk_0.003'
    ckpt_files = [
        ('epoch_29_collision_0.0466_model.pth', 29),
        ('epoch_34_collision_0.0463_model.pth', 34),
        ('epoch_44_collision_0.0532_model.pth', 44),
        ('epoch_49_collision_0.0539_model.pth', 49),
    ]
    for fname, epoch in ckpt_files:
        path = os.path.join(task181_dir, fname)
        if not os.path.exists(path):
            print(f'  ⚠ missing: {path}')
            continue
        print(f'\n[Task181 epoch {epoch}] loading {fname}...')
        # task181 uses fixed κ=1 (HG-Rec baseline, no learnable curvature)
        ckpt_data = load_hrqvae_ckpt(path, fixed_kappa=1.0, learnable_kappa=False)
        out[f'task181_epoch_{epoch}'] = {
            'epoch': epoch,
            'recipe': 'HG-Rec baseline (Phase 0 fixed, β=0.5, codebook=[64,128,256], κ=1 fixed)',
            'kappa_history': ckpt_data['kappa_history'],
            'layers': [summarize_layer(l) for l in ckpt_data['layers']],
        }

    # === Part 2: learnable-κ tasks — final κ values + c‖x‖² stats ===
    print('\n' + '=' * 70)
    print('Part 2: learnable-κ tasks — final ckpt cross-section')
    print('=' * 70)
    learnable_kappa_tasks = [
        ('task164_kappa_stereographic', 'products/task164/kappa_stereographic_test/jul-25-2026_00-55-29/best_loss_model.pth'),
        ('task169_phase_b_sinkhorn', 'products/task169/phase_b_kappa_sinkhorn/jul-25-2026_11-08-34/best_loss_model.pth'),
        ('task170_phase_b_sinkhorn_all3', 'products/task170/phase_b_kappa_sinkhorn_all3/jul-25-2026_11-38-53/best_loss_model.pth'),
        ('task171_phase_b_dead_code', 'products/task171/phase_b_kappa_dead_code/jul-25-2026_11-38-53/best_loss_model.pth'),
        ('task172_phase_b_kappa_max4', 'products/task172/phase_b_kappa_max4/jul-25-2026_11-40-18/best_loss_model.pth'),
    ]
    base = '/home/wlia0047/ar57/wenyu/GeneRec'
    for name, rel_path in learnable_kappa_tasks:
        path = os.path.join(base, rel_path)
        if not os.path.exists(path):
            print(f'  ⚠ missing: {path}')
            continue
        print(f'\n[{name}] loading...')
        ckpt_data = load_hrqvae_ckpt(path, learnable_kappa=True)
        out[name] = {
            'kappa_max': ckpt_data['kappa_max'],
            'kappa_history': ckpt_data['kappa_history'],
            'layers': [summarize_layer(l) for l in ckpt_data['layers']],
        }

    # === Save ===
    out_path = '/home/wlia0047/ar57/wenyu/GeneRec/products/codebook_cnorm_distribution.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\n✅ Saved to {out_path}')
    print(f'  size: {os.path.getsize(out_path) / 1024:.1f} KB')

    # === Console summary ===
    print('\n' + '=' * 70)
    print('Console summary — Task181 c‖x‖² evolution per epoch per layer (κ=1 fixed)')
    print('=' * 70)
    print(f'{"epoch":>5} {"layer":>5} {"κ":>6} {"mean":>8} {"q95":>8} {"max":>8}  '
          f'{"frac>1-1e-3":>12} {"frac>1-1e-4":>12} {"frac>1-1e-5":>12}')
    for epoch_key in sorted(k for k in out if k.startswith('task181_epoch_')):
        d = out[epoch_key]
        for l in d['layers']:
            cn = l['cnorm_sq_distribution']
            print(f"{d['epoch']:>5} {l['layer_idx']:>5} {l['kappa']:>6.3f} "
                  f"{cn['mean']:>8.4f} {cn['q95']:>8.4f} {cn['max']:>8.4f}  "
                  f"{cn['frac_gt_1_minus_1e-3']*100:>11.3f}% {cn['frac_gt_1_minus_1e-4']*100:>11.3f}% "
                  f"{cn['frac_gt_1_minus_1e-5']*100:>11.3f}%")

    print('\n' + '=' * 70)
    print('Console summary — learnable-κ tasks final κ + c‖x‖² distribution')
    print('=' * 70)
    print(f'{"task":<35} {"κ_final":>9} {"layer":>5} {"mean":>9} {"max":>9}  '
          f'{"frac>1-1e-3":>12} {"frac>1-1e-5":>12}')
    for name in [n for n, _ in learnable_kappa_tasks if n in out]:
        d = out[name]
        for l in d['layers']:
            cn = l['cnorm_sq_distribution']
            print(f"{name:<35} {l['kappa']:>9.4f} {l['layer_idx']:>5} "
                  f"{cn.get('mean', cn.get('abs_mean', -1)):>9.4f} {cn.get('max', cn.get('abs_max', -1)):>9.4f}  "
                  f"{cn['frac_gt_1_minus_1e-3']*100:>11.3f}% {cn['frac_gt_1_minus_1e-5']*100:>11.3f}%")


if __name__ == '__main__':
    main()