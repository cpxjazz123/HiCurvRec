#!/usr/bin/env python3
# task19_additive_quantization.py — AQ 加性量化
#
# 3 层加性量化: x ≈ C_1[c_1] + C_2[c_2] + C_3[c_3] (无嵌套)
# 关键: 没有 "first-layer dictatorship" 因为每层独立贡献
#
# 算法:
#   1. 初始化码本 C_l ∈ R^(K, D), from RQ C_l (Group A codebook warmup)
#   2. Repeat until converge:
#       a) Assignment step: 对每个 x 找最优 (c_1, c_2, c_3) ∈ [K]^3
#          最小化 ||x - C_1[c_1] - C_2[c_2] - C_3[c_3]||²
#          → 暴力 beam search: c_1 = argmin_k ||x - C_1[k]||² (init)
#          → c_2 = argmin_k ||x - C_1[c_1] - C_2[k]||² (fix c_1)
#          → c_3 = argmin_k ||x - C_1[c_1] - C_2[c_2] - C_3[k]||²
#       b) Codebook update: 对每个 (c_1, c_2, c_3) 组合 + 边界情况:
#          加权中心: C_l[c_l] = mean(assigned x - 其他层贡献) (条件均值)
#   3. dedup digit 加入 (4th)
#   4. 输出 SID tensor

import sys, os, json, time
sys.path.insert(0, '/fs04/ar57/wenyu/GeneRec/GRID')
import numpy as np
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_LAYERS = 3
N_CLUSTERS = 256
MAX_ITER = 20
BATCH = 1024
SEED = 42

EMBED_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt'
GROUP_A_CODEBOOKS = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/A_baseline_rqidx.pt'
SID_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/task19_aq_s22/pickle/merged_predictions_tensor.pt'
CKPT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/results/exp21'
os.makedirs(CKPT_DIR, exist_ok=True)


# ----------- Joint AQ via conditional assignment -----------

def aq_assign_batched(x, codebooks, batch=BATCH):
    """对每个 x 找最优 (c_1, c_2, c_3) via greedy conditional assignment.
       x: (N, D) input
       codebooks: list of n_layers tensors, each (K, D)
       Returns: codes (N, n_layers)

    简化策略: 不枚举 (256^3=16.7M) 组合, 用条件贪心:
       c_1 = argmin_k ||x - C_1[k]||²
       c_2 = argmin_k ||x - C_1[c_1] - C_2[k]||²
       c_3 = argmin_k ||x - C_1[c_1] - C_2[c_2] - C_3[k]||²
    """
    N, D = x.shape
    L = len(codebooks)
    codes = torch.zeros(N, L, dtype=torch.long, device=x.device)

    # Compute reconstruction sum (cumulative)
    recon = torch.zeros_like(x)

    for l, C in enumerate(codebooks):
        # Residual = x - recon_so_far
        residual = x - recon                                            # (N, D)
        # Best c_l: argmin_k ||residual - C[k]||²
        # = argmax_k 2 * residual · C[k] - ||C[k]||²
        rC = residual @ C.T                                             # (N, K)
        Cn2 = (C ** 2).sum(-1)                                          # (K,)
        score = 2 * rC - Cn2.unsqueeze(0)                              # (N, K)
        c_l = score.argmax(dim=-1)                                      # (N,)
        codes[:, l] = c_l
        # Update reconstruction
        recon = recon + C[c_l]                                          # (N, D)
    return codes


def aq_update_codebooks(x, codes, codebooks, n_clusters, batch=BATCH):
    """对加性量化正确的条件 K-means 更新:
       C_l[k] = mean over mask(=k) of ( x - sum_{j≠l} C_j[c_j] )

       这才是 AQ 的真正 K-means 更新 (additive conditional mean).
       每个 layer l 用 fixed 其他层 contribution 作为 baseline, 重算 residual.
    """
    N, D = x.shape
    L = codes.shape[1]
    new_codebooks = []

    # Pre-compute assigned contributions per layer
    assigned_recon_per_layer = []
    for li in range(L):
        assigned_recon_per_layer.append(codebooks[li][codes[:, li]])   # (N, D)

    for li in range(L):
        # Residual for layer l: x - sum of OTHER layers' assigned contributions
        other_contribs = torch.zeros_like(x)
        for jj in range(L):
            if jj != li:
                other_contribs = other_contribs + assigned_recon_per_layer[jj]
        residual_for_l = x - other_contribs                              # (N, D)

        new_C = torch.zeros(n_clusters, D, device=x.device)
        counts = torch.zeros(n_clusters, device=x.device)
        c_l = codes[:, li]
        for k in range(n_clusters):
            mask = (c_l == k)
            if mask.any():
                new_C[k] = residual_for_l[mask].mean(0)
                counts[k] = mask.sum().float()
            else:
                # Empty cluster: keep old centroid
                new_C[k] = codebooks[li][k].clone()
        new_codebooks.append(new_C)
    return new_codebooks


def aq_run(x, codebooks_init, n_clusters=N_CLUSTERS, max_iter=MAX_ITER, batch=BATCH,
           seed=SEED, verbose=True):
    """加性量化主循环
       x: (N, D)
       codebooks_init: list of (K, D) initial codebooks (e.g. RQ warmup)
       Returns: codes (N, L), codebooks (L, K, D), history
    """
    codebooks = [c.clone() for c in codebooks_init]
    L = len(codebooks)
    history = []
    best_qerr = float('inf')
    patience_count = 0

    for it in range(max_iter):
        t0 = time.time()
        # Assign
        codes = aq_assign_batched(x, codebooks, batch=batch)

        # QErr
        recon = torch.zeros_like(x)
        for l, C in enumerate(codebooks):
            recon = recon + C[codes[:, l]]
        q_err = ((x - recon) ** 2).sum(-1).mean().item()
        history.append(q_err)
        if verbose:
            print(f'  iter {it}: QErr={q_err:.4f}, util={[codes[:,l].unique().shape[0] for l in range(L)]} ({time.time()-t0:.1f}s)')

        # Update (conditional mean for additive: needs old codebooks for fallback)
        new_codebooks = aq_update_codebooks(x, codes, codebooks, batch=batch, n_clusters=n_clusters)

        # Convergence check (sum-of-clusters Frobenius change)
        diff = sum((nc - c).norm() for nc, c in zip(new_codebooks, codebooks)).item()
        codebooks = new_codebooks

        if q_err < best_qerr - 1e-3:
            best_qerr = q_err
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= 5:
                if verbose:
                    print(f'  early stop at iter {it}')
                break
    return codes.cpu(), codebooks, history


# ----------- Main -----------

def main():
    print('=' * 70)
    print('task21 — AQ 加性量化 (joint codebook + conditional assignment)')
    print('=' * 70)

    # Load full embedding
    t0 = time.time()
    x = torch.load(EMBED_PATH, weights_only=False, map_location='cpu').float().to(DEVICE)
    print(f'Loaded x: {x.shape}, time={time.time()-t0:.1f}s')
    N, D = x.shape

    # Initialize AQ codebooks from RQ Group A codebooks (warmup)
    bundle = torch.load(GROUP_A_CODEBOOKS, weights_only=False, map_location='cpu')
    rq_codebooks = bundle['codebooks']   # list of (K, D) RQ codebooks
    print(f'RQ warmup codebooks: {len(rq_codebooks)} × {rq_codebooks[0].shape}')

    # Use RQ codebooks directly as AQ init (pragmatic warmup)
    aq_codebooks_init = [c.to(DEVICE).float() for c in rq_codebooks[:N_LAYERS]]
    print(f'AQ init codebooks: {len(aq_codebooks_init)} × {aq_codebooks_init[0].shape}')

    # Run AQ
    print('\n=== AQ joint codebook learning ===')
    t0_aq = time.time()
    codes, codebooks, history = aq_run(x, aq_codebooks_init, n_clusters=N_CLUSTERS,
                                       max_iter=MAX_ITER, batch=BATCH, seed=SEED, verbose=True)
    print(f'\nAQ done in {time.time()-t0_aq:.1f}s')

    codes_cpu = codes
    print(f'\nCodes shape: {codes_cpu.shape}')
    for l in range(N_LAYERS):
        n_unique = codes_cpu[:, l].unique().shape[0]
        print(f'  L{l+1}: {n_unique}/{N_CLUSTERS} unique')
    unique_tuples = torch.unique(codes_cpu, dim=0).shape[0]
    print(f'  Unique code tuples: {unique_tuples}/{N} (collision rate {1-unique_tuples/N:.4f})')

    # Per-layer drop diagnostics: 模拟 drop 1 layer 的影响 (in quant space)
    # 重建 (using full codes)
    recon = torch.zeros_like(x)
    for l, C in enumerate(codebooks):
        recon = recon + C[codes_cpu[:, l]]
    full_qerr = ((x - recon) ** 2).sum(-1).mean().item()
    print(f'\nFull reconstruction QErr: {full_qerr:.4f}')

    # Drop L1: only use L2 + L3
    recon_drop1 = torch.zeros_like(x)
    for l in [1, 2]:
        recon_drop1 = recon_drop1 + codebooks[l][codes_cpu[:, l]]
    drop1_qerr = ((x - recon_drop1) ** 2).sum(-1).mean().item()
    drop1_ratio = drop1_qerr / full_qerr
    print(f'  Drop L1:  QErr={drop1_qerr:.4f}  ratio={drop1_ratio:.2f}×')
    # Drop L2
    recon_drop2 = codebooks[0][codes_cpu[:, 0]] + codebooks[2][codes_cpu[:, 2]]
    drop2_qerr = ((x - recon_drop2) ** 2).sum(-1).mean().item()
    drop2_ratio = drop2_qerr / full_qerr
    print(f'  Drop L2:  QErr={drop2_qerr:.4f}  ratio={drop2_ratio:.2f}×')
    # Drop L3
    recon_drop3 = codebooks[0][codes_cpu[:, 0]] + codebooks[1][codes_cpu[:, 1]]
    drop3_qerr = ((x - recon_drop3) ** 2).sum(-1).mean().item()
    drop3_ratio = drop3_qerr / full_qerr
    print(f'  Drop L3:  QErr={drop3_qerr:.4f}  ratio={drop3_ratio:.2f}×')

    # Add dedup digit
    full_codes = torch.cat([codes_cpu, torch.zeros(N, 1, dtype=torch.long)], dim=1)
    seen = set()
    for i in range(N):
        key = tuple(codes_cpu[i].cpu().tolist())
        if key in seen:
            full_codes[i, N_LAYERS] = 1
        else:
            seen.add(key)
    sid = full_codes.t().cpu()                                          # (4, 11924)
    os.makedirs(os.path.dirname(SID_PATH), exist_ok=True)
    torch.save(sid, SID_PATH)
    print(f'\nSID saved: {sid.shape} → {SID_PATH}')

    # Save info + drop diagnostics
    info = {
        'algorithm': 'task19_AQ_additive_quantization',
        'version': 'joint codebook + conditional assignment',
        'warmup': 'Group A RQ codebooks (initial)',
        'n_layers': N_LAYERS,
        'n_clusters': N_CLUSTERS,
        'aq_time_sec': time.time() - t0_aq,
        'collision_rate': 1 - unique_tuples / N,
        'l1_unique': int(codes_cpu[:, 0].unique().shape[0]),
        'l2_unique': int(codes_cpu[:, 1].unique().shape[0]),
        'l3_unique': int(codes_cpu[:, 2].unique().shape[0]),
        'sid_path': SID_PATH,
        'sid_shape': list(sid.shape),
        'full_qerr': full_qerr,
        'drop_l1_ratio': drop1_ratio,
        'drop_l2_ratio': drop2_ratio,
        'drop_l3_ratio': drop3_ratio,
        'qerr_history': history,
        'kill_lines': {
            'structure_A_drop_l1_should_be_lt_0.5': drop1_ratio < 0.50,
            'structure_A_max_minus_min_should_be_lt_0.08':
                abs(drop1_ratio - drop3_ratio) < 0.08,
            'structure_B_delta_ratio_using_drop_qerrs':
                f'drop1/drop3 = {drop1_ratio / drop3_ratio:.2f}'
        }
    }

    info_path = os.path.join(CKPT_DIR, 'task19_aq_run_info.json')
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f'\n=== info → {info_path} ===')
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
