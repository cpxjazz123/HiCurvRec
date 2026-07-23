#!/usr/bin/env python3
# task7_controllable_whitening.py — α-mix whitening 扫描
#
# 思路 (post-hoc):
#   取 RQ Group A 的 r_lst, 每 α 对各层残差做:
#       r_l_alpha = (1-α) * r_l + α * whiten(r_l)
#   用 **同一码本 C_l** 重分配 codes:
#       c_l_alpha = argmin_k ||r_l_alpha - C_l[k]||²
#   记录:
#     - new reconstruction QErr
#     - new 序列 Δ_1, Δ_2, Δ_3
#     - 利用率 util
#     - 与原 RQ Group A 的 codes 变化率 (code_drift)
#
# 输出: 6-α JSON table + 6-α SID tensor (for futher TIGER training, optional)
# α ∈ {0.0, 0.1, 0.25, 0.5, 0.75, 1.0}

import sys, os, json
sys.path.insert(0, '/fs04/ar57/wenyu/GeneRec/GRID')
import numpy as np
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ALPHAS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/results/exp22'
os.makedirs(OUT_DIR, exist_ok=True)

GROUP_A_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/A_baseline_rqidx.pt'


def whiten(r, eps=1e-6):
    """逐 batch/sample 标准化: per-sample z-norm + per-dim standardization
       简单实现: per-sample L2 norm → unit norm; 然后 per-dim z-score across batch.
       target: 让 isotropic 各向同性. D=2048 so 简单 per-sample L2-norm + per-feature center.
    """
    # per-sample L2 normalize
    norm = r.norm(dim=-1, keepdim=True).clamp(min=eps)
    r_unit = r / norm
    # per-feature z-score across batch
    mean = r_unit.mean(dim=0, keepdim=True)
    std = r_unit.std(dim=0, keepdim=True).clamp(min=eps)
    r_z = (r_unit - mean) / std
    # Final rescale: 让 r_z 保持 r 的 magnitude
    r_white = r_z * (norm / r_z.norm(dim=-1, keepdim=True).clamp(min=eps))
    return r_white


def reconstruct(x_init, codebooks, codes):
    """对 RQ 风格 nested 还原: r_0 = x; r_l = r_{l-1} - C_l[codes_l]
       最终重建: x_init - sum_{l} C_l[codes_l]
    """
    x = x_init
    for C, c in zip(codebooks, codes.T):  # codes.T shape (L, N)
        x = x - C[c]
    recon = x_init - x
    return recon


def reconstruct_with_alpha(x_init, codebooks, codes_orig, alpha):
    """post-hoc whitening on r_l 然后用 同一码本 重分配 codes_alpha:
       对每 l, 先 compute r_l_orig = r_{l-1} - C_l[codes_orig_l]
       r_l_alpha = (1-α) * r_l_orig + α * whiten(r_l_orig)
       codes_l_alpha = argmin_k ||r_l_alpha - C_l[k]||²
       重建: x_init - sum_l C_l[codes_alpha_l]
    """
    L = len(codebooks)
    x = x_init.clone()
    codes_alpha = torch.zeros_like(codes_orig)
    for l in range(L):
        r_l = x  # current residual
        r_l_w = whiten(r_l)
        r_l_alpha = (1 - alpha) * r_l + alpha * r_l_w
        # Re-assign using same codebook C_l
        C = codebooks[l]
        rC = r_l_alpha @ C.T
        Cn2 = (C ** 2).sum(-1)
        scores = 2 * rC - Cn2.unsqueeze(0)
        c_a = scores.argmax(dim=-1)
        codes_alpha[:, l] = c_a
        x = x - C[c_a]   # update residual for next layer
    return codes_alpha


def main():
    print('=' * 70)
    print('task22 — α-scan controllable whitening (post-hoc)')
    print('=' * 70)

    bundle = torch.load(GROUP_A_PATH, weights_only=False, map_location='cpu')
    codebooks = bundle['codebooks']           # list of (K, D)
    idx_lst = bundle['idx_lst']               # list of (N,) per layer
    r_lst = bundle['r_lst']                   # list of (N, D) per layer, r_lst[0] = x input
    codes_orig = torch.stack(idx_lst, dim=1).long()   # (N, L)
    x_init = r_lst[0].float()
    print(f'codebooks: {len(codebooks)} × {codebooks[0].shape}')
    print(f'codes_orig: {codes_orig.shape}')
    print(f'x_init: {x_init.shape}')

    codebooks_gpu = [c.to(DEVICE).float() for c in codebooks]
    codes_orig_gpu = codes_orig.to(DEVICE)
    x_init_gpu = x_init.to(DEVICE).float()

    # Baseline metrics
    recon_orig = reconstruct_with_alpha(x_init_gpu, codebooks_gpu, codes_orig_gpu, 0.0)
    # Actually alpha=0 → no whitening → codes_alp = codes_orig. qerr = baseline qerr
    # 但 our reconstruct_with_alpha 算的是 re-assigned codes, alpha=0 gives same codes
    # So baseline QErr (alpha=0) = orig QErr

    results = {'alphas': ALPHAS, 'per_alpha': {}}
    for alpha in ALPHAS:
        print(f'\n--- α={alpha} ---')
        codes_alpha = reconstruct_with_alpha(x_init_gpu, codebooks_gpu, codes_orig_gpu, alpha)
        codes_alpha_cpu = codes_alpha.cpu()

        # QErr: 还原 x 用 codes_alpha 通过原来 RQ 嵌套方式: r = x - sum C_l[c_alpha_l]
        recon = x_init_gpu.clone()
        for l, C in enumerate(codebooks_gpu):
            recon = recon - C[codes_alpha[:, l]]
        qerr = ((recon ** 2).sum(-1)).mean().item()

        # Delta_l = avg ||C_l[k_new] - C_l[k_orig]||
        delta_l = []
        for l in range(len(codebooks)):
            C = codebooks_gpu[l]
            d = (C[codes_alpha[:, l]] - C[codes_orig_gpu[:, l]]).norm(dim=-1).mean().item()
            delta_l.append(d)

        # Code change rate
        code_drift = (codes_alpha_cpu != codes_orig).float().mean().item()

        # Per-layer util
        util_l = [codes_alpha_cpu[:, l].unique().shape[0] for l in range(len(codebooks))]

        info = {
            'alpha': alpha,
            'qerr': qerr,
            'delta_l_per_layer': delta_l,
            'code_drift_total': code_drift,
            'util_per_layer': util_l,
        }
        results['per_alpha'][str(alpha)] = info
        print(f'  QErr={qerr:.4f}, Δ_l={[round(v, 4) for v in delta_l]}, '
              f'code_drift={code_drift:.4f}, util={util_l}')

    # JSON output
    out_path = os.path.join(OUT_DIR, 'task7_alpha_scan.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n=== JSON → {out_path} ===')

    # Summary table
    summary = ['# task22 — α-scan summary', '', '| α | QErr | Δ_1 | Δ_2 | Δ_3 | code_drift | util |', '|---|------|-----|-----|-----|-----------|-----|']
    for alpha in ALPHAS:
        info = results['per_alpha'][str(alpha)]
        s = (f"| {alpha} | {info['qerr']:.4f} | {info['delta_l_per_layer'][0]:.4f} | "
             f"{info['delta_l_per_layer'][1]:.4f} | {info['delta_l_per_layer'][2]:.4f} | "
             f"{info['code_drift_total']:.4f} | {info['util_per_layer']} |")
        summary.append(s)
    md_path = os.path.join(OUT_DIR, 'task7_alpha_scan_summary.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(summary))
    print(f'\n=== Summary → {md_path} ===')
    print('\n'.join(summary))


if __name__ == '__main__':
    main()
