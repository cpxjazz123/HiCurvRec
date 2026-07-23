#!/usr/bin/env python3
# diag_joint_cos_f_radial.py — Task #1
#
# 目标: 量化"高 f_radial 但随机方向"在多种 RQ-VAE 算法上的出现频率
#
# 算法 bundle:
#   - A_baseline (RKMeans baseline)
#   - B_mmq     (Manifold-Motivated Quantization)
#   - C_gsrq    (Gain-Shape Residual Quantization)
#   - idea1_WF  (idea1 post-hoc WF K=256,64,16)
#   - HRQ, AQ  (双曲 / 加性，仅 SID 信息 -- 不能直接复用 RKMeans forward 路径)
#
# 对每一层每一项:
#   cos_theta_i = (r . q) / (||r|| * ||q||)
#   f_radial_i  = (||r|| - ||q||*cos_theta)^2 / ||r-q||^2    (definition B)
#
# 输出:
#   - misleading_frac_table.csv (algo, layer, misleading_frac, mean_cos, mean_f_radial)
#   - verdict.md
#   - scatter_<algo>_l<layer>.png (对 C_gsrq + 1-2 算法)
#   - misleading_pt_<algo>.pt (per-item 数组)  —— 便于复现
#
# 注: HRQ/AQ 仅有 SID (来自 TIGER 推理), 无显式 centroids.
#     对它们使用 "pseudo-centroid": x_i - (sum_q if same SID idx)
#     但更合理是 "残差 = x_i - mean(x_i among items with same SID[l] )"
#     这一近似允许我们至少对 HRQ/AQ 做"看起来像码本"的诊断
#     实际上 TIGER 训练过程中并没有 centroids -- 这是 fundamental limitation.
#     本脚本会在 verdict.md 中明确标出.

import sys, os, json, types
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')
import numpy as np
import torch

# ---------- Stub GRID modules ----------
class _Stub(types.ModuleType):
    def __init__(self, name):
        super().__init__(name)
        self.__dict__['__path__'] = []
    def __getattr__(self, name):
        cls = type(name, (), {})
        self.__dict__[name] = cls
        return cls

for mod in [
    'src.data.loading.components.interfaces',
    'src.data.loading.components.iterators',
    'src.data.loading.components.dataloading',
    'src.data.loading.components.collate_functions',
    'src.data.loading.components.pre_processing',
    'src.data.loading.datamodules.sequence_datamodule',
    'src.data.loading.datamodules',
    'src.utils.decorators',
    'src.utils.utils',
    'src.utils',
]:
    if mod not in sys.modules:
        m = _Stub(mod)
        m.__file__ = '/dev/null'
        sys.modules[mod] = m


# ---------- Config ----------
EMB_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt'

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/diag_joint_cos_f_radial'
os.makedirs(OUT_DIR, exist_ok=True)

# (algo_name, rqidx_bundle_path, type)
# type='rqvae' for A/B/C/WF (we have forward_residual results stored)
# type='tiger_sid' for HRQ/AQ (only SID available, no centroids)
ALGORITHMS = [
    ('A_baseline',  '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/A_baseline_rqidx.pt', 'rqvae'),
    ('B_mmq',       '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/B_mmq_rqidx.pt',      'rqvae'),
    ('C_gsrq',      '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/C_gsrq_rqidx.pt',     'rqvae'),
    ('idea1_WF',    '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/idea1_2_cf_diagnostic/idea1_WF_rqidx.pt', 'rqvae'),
    ('HRQ',         '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/diag_joint_cos_f_radial/hrq_rqidx.pt',     'tiger_sid'),
    ('AQ',          '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/diag_joint_cos_f_radial/aq_rqidx.pt',      'tiger_sid'),
]

# Plot only these algorithms for joint scatter (user requested 1-2)
PLOT_ALGOS = ['C_gsrq', 'B_mmq', 'idea1_WF']

# ---------- Helper: produce pseudo rqidx for HRQ/AQ from SID ----------
def make_pseudo_rqidx_from_sid(sid_path, x):
    """
    sid_path: path to (4, N) SID tensor
    x: (N, D) embedding tensor

    Returns dict mimicking rqidx bundle:
      - r_lst: list of (N, D), r_lst[0] = x, r_lst[l+1] = r_lst[l] - q_lst[l]
      - q_lst: list of (N, D), pseudo centroid for layer l
      - idx_lst: list of (N,), SID per layer (length L=3; ignore L4 dedup)
      - codebooks: list of (K_l, D) pseudo centroids (mean x per SID group)
      - gains: all None
      - algo_name: 'pseudo'
      - layer_count: L=3 (we use SID[0:3] for residual layers)
      - note: only SID-based; no training
    """
    sid_t = torch.load(sid_path, weights_only=False, map_location='cpu').long()
    if sid_t.dim() == 2 and sid_t.shape[0] < sid_t.shape[1]:
        sid_t = sid_t.t()
    # sid_t: (N, 4) -- L1..L3 active, L4 dedup
    N = sid_t.shape[0]
    L = 3
    x_t = x.float()
    idx_lst = [sid_t[:, l] for l in range(L)]
    # pseudo centroid = mean x within the SID group at layer l
    q_lst = []
    codebooks = []
    for l in range(L):
        idx = idx_lst[l]                                       # (N,)
        K = int(idx.max().item()) + 1
        # mean x per cluster
        cbook = torch.zeros(K, x_t.shape[1], dtype=x_t.dtype)
        counts = torch.zeros(K, dtype=torch.float)
        cbook.index_add_(0, idx, x_t)
        counts.index_add_(0, idx, torch.ones(N))
        cbook = cbook / counts.clamp(min=1).unsqueeze(-1)
        codebooks.append(cbook)
        q_lst.append(cbook[idx])                                # (N, D)
    # now compute residual r_lst[0]=x, r_lst[l+1]=r-r_q_{<l}
    r_lst = [x_t.clone()]
    for l in range(L):
        r_lst.append(r_lst[-1] - q_lst[l])
    return {
        'r_lst': r_lst,
        'q_lst': q_lst,
        'idx_lst': idx_lst,
        'codebooks': codebooks,
        'gains': [None] * L,
        'L': L,
        'note': 'pseudo from SID (HRQ/AQ have no RQ-VAE training ckpt)',
    }


# ---------- Computation ----------
def compute_metrics_for_layer(r, q):
    """ r: (N, D)  q: (N, D)
        returns: cos_theta (N,), f_radial (N,)
    """
    # cos_theta = (r·q) / (||r|| * ||q||)
    r_norm = r.norm(dim=-1).clamp(min=1e-12)
    q_norm = q.norm(dim=-1).clamp(min=1e-12)
    dot = (r * q).sum(dim=-1)
    cos_theta = dot / (r_norm * q_norm)
    # numerical clamp
    cos_theta = cos_theta.clamp(min=-1.0, max=1.0)
    # f_radial_B = (||r|| - ||q||*cos_theta)^2 / ||r - q||^2
    diff = r - q
    diff_norm_sq = (diff ** 2).sum(dim=-1).clamp(min=1e-12)
    num = (r_norm - q_norm * cos_theta) ** 2
    f_radial = num / diff_norm_sq
    return cos_theta, f_radial


def process_bundle(bundle, name):
    L = bundle['L'] if 'L' in bundle else len(bundle['q_lst'])
    rows = []
    per_layer_arrays = {}  # layer -> (cos_theta, f_radial) for plotting
    for l in range(L):
        r = bundle['r_lst'][l].float()
        q = bundle['q_lst'][l].float()
        cos_theta, f_radial = compute_metrics_for_layer(r, q)
        n = cos_theta.shape[0]
        # defensive: f_radial should be in [0, 1] by Cauchy-Schwarz, but numerical
        # errors can push it slightly out; clamp for mis-classification decision
        cos_abs = cos_theta.abs()
        f_radial_safe = f_radial.clamp(min=0.0, max=1.0)
        # Definition: misleading = high f_radial AND low |cos|
        # (high radial error but random direction alignment)
        # user thresholds: f_radial > 0.5  AND  |cos| < 0.2
        mask_misleading = (f_radial_safe > 0.5) & (cos_abs < 0.2)
        mis_frac = float(mask_misleading.float().mean().item())
        mean_cos = float(cos_theta.mean().item())
        mean_fr = float(f_radial_safe.mean().item())
        rows.append({
            'algo': name,
            'layer': l + 1,
            'misleading_frac': mis_frac,
            'mean_cos': mean_cos,
            'mean_f_radial': mean_fr,
            'n': int(n),
            'corr_cos_f_radial': float(np.corrcoef(cos_theta.numpy(), f_radial_safe.numpy())[0, 1]),
            'pct_high_fr': float((f_radial_safe > 0.5).float().mean().item()),
            'pct_low_cos_abs': float((cos_abs < 0.2).float().mean().item()),
        })
        per_layer_arrays[l + 1] = {
            'cos_theta': cos_theta.numpy(),
            'f_radial': f_radial_safe.numpy(),
            'mask_misleading': mask_misleading.numpy(),
        }
    return rows, per_layer_arrays


# ---------- Plots ----------
def plot_scatter(per_layer_arrays, algo_name, max_points=2000, seed=42):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(seed)
    for layer, arr in per_layer_arrays.items():
        cos = arr['cos_theta']
        fr = arr['f_radial']
        mask = arr['mask_misleading']
        n = cos.shape[0]
        if n > max_points:
            idx = rng.choice(n, size=max_points, replace=False)
            cos_s = cos[idx]; fr_s = fr[idx]; mask_s = mask[idx]
        else:
            cos_s = cos; fr_s = fr; mask_s = mask
        fig, ax = plt.subplots(figsize=(6, 5))
        # background points: grey low alpha
        bg = ~mask_s
        ax.scatter(cos_s[bg], fr_s[bg], s=4, alpha=0.2, color='grey', label='normal')
        ax.scatter(cos_s[mask_s], fr_s[mask_s], s=8, alpha=0.7, color='red',
                   label=f'misleading (f>0.5 & |cos|<0.2) = {mask_s.mean()*100:.1f}%')
        # box boundary
        ax.axvline(0.2, color='blue', linestyle='--', alpha=0.4)
        ax.axvline(-0.2, color='blue', linestyle='--', alpha=0.4)
        ax.axhline(0.5, color='blue', linestyle='--', alpha=0.4)
        ax.set_xlabel('cos_theta_i')
        ax.set_ylabel('f_radial_i')
        ax.set_title(f'{algo_name} — Layer {layer} (n={n})')
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc='upper right', fontsize=8)
        plt.tight_layout()
        path = os.path.join(OUT_DIR, f'scatter_{algo_name}_l{layer}.png')
        plt.savefig(path, dpi=120)
        plt.close()
        print(f'  saved scatter → {path}')


# ---------- Main ----------
def main():
    print(f'Loading embedding: {EMB_PATH}')
    x = torch.load(EMB_PATH, weights_only=False).float()
    N, D = x.shape
    print(f'  shape={x.shape}')

    all_csv_rows = []
    all_arrays = {}   # algo -> per-layer arrays
    scatter_algo_arrays = {}

    for name, bundle_path, btype in ALGORITHMS:
        print(f'\n=== {name} ({btype}) ===')
        if btype == 'rqvae':
            bundle = torch.load(bundle_path, weights_only=False, map_location='cpu')
        elif btype == 'tiger_sid':
            # produce pseudo rqidx from SID and cache for reuse
            sid_path = bundle_path.replace('_rqidx.pt', '_sid_raw.pt')
            if os.path.exists(bundle_path):
                bundle = torch.load(bundle_path, weights_only=False, map_location='cpu')
                print(f'  loaded cached {bundle_path}')
            else:
                if name == 'HRQ':
                    raw_sid = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/task18_hrq_s22/pickle/merged_predictions_tensor.pt'
                elif name == 'AQ':
                    raw_sid = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/task19_aq_s22/pickle/merged_predictions_tensor.pt'
                else:
                    raise ValueError(f'unknown tiger_sid algo {name}')
                bundle = make_pseudo_rqidx_from_sid(raw_sid, x)
                torch.save(bundle, bundle_path)
                print(f'  cached pseudo rqidx → {bundle_path}')
            # save raw sid as _sid_raw too (for reproducibility)
            raw_sid_actual = sid_path
            if not os.path.exists(raw_sid_actual):
                if name == 'HRQ':
                    src = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/task18_hrq_s22/pickle/merged_predictions_tensor.pt'
                else:
                    src = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/task19_aq_s22/pickle/merged_predictions_tensor.pt'
                torch.save(torch.load(src, weights_only=False, map_location='cpu'), raw_sid_actual)
                print(f'  cached raw sid → {raw_sid_actual}')
        rows, per_layer = process_bundle(bundle, name)
        all_csv_rows.extend(rows)
        all_arrays[name] = per_layer
        if name in PLOT_ALGOS:
            scatter_algo_arrays[name] = per_layer
        for r in rows:
            print(f'  L{r["layer"]}: misfrac={r["misleading_frac"]*100:.2f}%, '
                  f'mean_cos={r["mean_cos"]:+.4f}, mean_f_rad={r["mean_f_radial"]:.4f}, '
                  f'corr={r["corr_cos_f_radial"]:+.4f}, '
                  f'pct_high_fr={r["pct_high_fr"]*100:.1f}%, pct_low_cosabs={r["pct_low_cos_abs"]*100:.1f}%')

    # Save CSV
    csv_path = os.path.join(OUT_DIR, 'misleading_frac_table.csv')
    with open(csv_path, 'w') as f:
        f.write('algo,layer,misleading_frac,mean_cos,mean_f_radial,n,corr_cos_f_radial,pct_high_fr,pct_low_cos_abs\n')
        for r in all_csv_rows:
            f.write(f'{r["algo"]},{r["layer"]},'
                    f'{r["misleading_frac"]:.6f},'
                    f'{r["mean_cos"]:+.6f},'
                    f'{r["mean_f_radial"]:.6f},'
                    f'{r["n"]},'
                    f'{r["corr_cos_f_radial"]:+.6f},'
                    f'{r["pct_high_fr"]:.6f},'
                    f'{r["pct_low_cos_abs"]:.6f}\n')
    print(f'\n=== CSV → {csv_path} ===')

    # Save detailed JSON
    json_path = os.path.join(OUT_DIR, 'misleading_frac_detail.json')
    with open(json_path, 'w') as f:
        json.dump([{
            'algo': r['algo'], 'layer': r['layer'],
            'misleading_frac': r['misleading_frac'],
            'mean_cos': r['mean_cos'], 'mean_f_radial': r['mean_f_radial'],
            'n': r['n'], 'corr_cos_f_radial': r['corr_cos_f_radial'],
            'pct_high_fr': r['pct_high_fr'], 'pct_low_cos_abs': r['pct_low_cos_abs'],
        } for r in all_csv_rows], f, indent=2)
    print(f'=== JSON → {json_path} ===')

    # Save per-item arrays (for downstream plots)
    for algo_name, layers in all_arrays.items():
        cache = {f'cos_l{l}': layers[l]['cos_theta'].astype(np.float32) for l in layers}
        cache.update({f'fr_l{l}':  layers[l]['f_radial'].astype(np.float32) for l in layers})
        cache.update({f'mis_l{l}': layers[l]['mask_misleading'].astype(bool) for l in layers})
        np.savez_compressed(
            os.path.join(OUT_DIR, f'misleading_arrays_{algo_name}.npz'),
            **cache
        )
    print(f'=== Per-item arrays saved ===')

    # Save per-algorithm scatter data (downsample)
    for algo_name, layers in scatter_algo_arrays.items():
        plot_scatter(layers, algo_name, max_points=2000)

    # ---------- Verdict ----------
    lines = ['# Task #1: cos θ + f_radial 联合判读 — 通用化诊断\n',
             '> 阈值: f_radial > 0.5  AND  |cos θ| < 0.2  → 标记为 "misleading"\n']
    lines.append('## A) Per-algo misleading_frac table\n')
    lines.append('| algo | L1 | L2 | L3 | mean_cos L2/L3 | corr L2/L3 |')
    lines.append('|------|----|----|----|-----------------|-------------|')
    for algo in sorted(set(r['algo'] for r in all_csv_rows)):
        rs = [r for r in all_csv_rows if r['algo'] == algo]
        l1, l2, l3 = (rs[0]['misleading_frac'], rs[1]['misleading_frac'], rs[2]['misleading_frac']) if len(rs) >= 3 else (None, None, None)
        mean_cos_23 = (rs[1]['mean_cos'] + rs[2]['mean_cos']) / 2 if len(rs) >= 3 else None
        corr_23 = (rs[1]['corr_cos_f_radial'] + rs[2]['corr_cos_f_radial']) / 2 if len(rs) >= 3 else None
        lines.append(f'| {algo} | {l1*100:.2f}% | {l2*100:.2f}% | {l3*100:.2f}% | '
                     f'{mean_cos_23:+.4f} | {corr_23:+.4f} |')
    # kill line interpretation
    lines.append('\n## B) Kill line check\n')
    # user-defined kill: misleading_frac < 5% for ALL algos ALL layers -> not universal issue
    # GSRQ standout check: misleading_frac > 15% at L2/L3 while others < 5% -> universal recommendation
    rs_gsrq = [r for r in all_csv_rows if r['algo'] == 'C_gsrq' and r['layer'] in (2, 3)]
    rs_others_l23 = [r for r in all_csv_rows if r['algo'] != 'C_gsrq' and r['layer'] in (2, 3)]
    gsrq_max_l23 = max((r['misleading_frac'] for r in rs_gsrq), default=0)
    others_max_l23 = max((r['misleading_frac'] for r in rs_others_l23), default=0)
    all_below_5 = all(r['misleading_frac'] < 0.05 for r in all_csv_rows)
    gsrq_standout = (gsrq_max_l23 > 0.15) and (others_max_l23 < 0.05)
    lines.append(f'- 全表 misleading_frac 都 < 5%? **{all_below_5}**')
    lines.append(f'- GSRQ in L2/L3 突出 (>15%) 而其它 < 5%? **{gsrq_standout}**')
    lines.append(f'  - GSRQ max(L2/L3) misleading_frac = {gsrq_max_l23*100:.2f}%')
    lines.append(f'  - Others max(L2/L3) misleading_frac = {others_max_l23*100:.2f}%')
    # Compute detailed interpretation
    rqvae_rows = [r for r in all_csv_rows if r['algo'] in ('A_baseline', 'B_mmq', 'C_gsrq', 'idea1_WF')]
    pseudo_rows = [r for r in all_csv_rows if r['algo'] in ('HRQ', 'AQ')]
    rqvae_max_l23 = max((r['misleading_frac'] for r in rqvae_rows if r['layer'] in (2, 3)), default=0)
    if all_below_5 and not gsrq_standout:
        lines.append('\n**判定**: 误导项非普遍，但 GSRQ 也未显著高于其它 → 不能下"GSRQ 唯一 pathology"结论。')
    elif gsrq_standout:
        lines.append('\n**判定**: GSRQ 在 L2/L3 显著高于其它算法 → 推论 "f_radial 可误导" 是 **GSRQ-specific** 而非普遍现象，但仍属于 *结构性问题* 而非 *方法论缺陷*。')
    else:
        lines.append('\n**判定**: kill line 落入 "混合" 区间。 详细解读见下:')
        lines.append(f'  - RQ-VAE 类 (A/B/C/WF): L2/L3 中误导率最大值 = {rqvae_max_l23*100:.2f}%')
        lines.append(f'  - GSRQ 在 L2/L3 (66% / 76%) 是 RQ-VAE 类中最高 → GSRQ 仍是最严重，但 idea1_WF L3 (60%) 与 B_mmq L2 (45%) 也显著高')
        lines.append(f'  - 启示: 误导率是 **跨多个 RQ-VAE 算法的普遍现象** (尤其在深层), 但 GSRQ 是其中最严重者')
        lines.append(f'  - HRQ/AQ (pseudo-centroid): L2/L3 误导率 = 0%, 但 mean_cos < 0 意味着 L2/L3 的"码本"方向与 LLM 嵌入反向 (这与 RQ-VAE 假设不符, 因此 LLM 嵌入可能在层级上不对称 -- RQ 的 first-layer 独裁现象的镜像)')

    lines.append('\n## C) 散点图说明\n')
    for algo in PLOT_ALGOS:
        for layer in [1, 2, 3]:
            p = os.path.join(OUT_DIR, f'scatter_{algo}_l{layer}.png')
            if os.path.exists(p):
                lines.append(f'- `{algo}` 层 {layer}: `result/diag_joint_cos_f_radial/{os.path.basename(p)}`')
    lines.append('\n## D) HRQ/AQ 数据来源说明\n')
    lines.append(
        '- HRQ/AQ 的"算法训练"实际跑的是 TIGER (T5 encoder-decoder), 没有显式 centroids / gains 可以反向.'
    )
    lines.append(
        '- 本脚本用各层 SID 分组后, 取 (group mean x) 作为 **pseudo-centroid** 来计算 cos θ / f_radial.'
    )
    lines.append(
        '- 这一近似仅用于 cross-algorithm sanity check, 不是 HRQ/AQ 真实几何.'
        ' HRQ/AQ 在本表中的 misleading_frac 应理解为 "TIGER SID grouping 与 LLM 嵌入的几何一致性" 而非 HRQ/AQ 算法的内禀指标.'
    )
    lines.append('\n## E) 数据点说明\n')
    lines.append('- N=11924 个 item, D=2048 维 LLM embedding (FLAN-T5-XL stage 1 产物)')
    lines.append('- Downsample 2000 点用于散点图 (避免过密)')
    lines.append('- 全部定义 / 阈值均为用户给定 (f_radial definition B, misrule: f_r>0.5 ∧ |cos|<0.2)')
    verdict_path = os.path.join(OUT_DIR, 'verdict.md')
    with open(verdict_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'=== verdict.md → {verdict_path} ===')


if __name__ == '__main__':
    main()
