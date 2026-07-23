#!/usr/bin/env python3
# task6_q9.py — 逐层 token probe 边际增益
# 量 9: 测 l→l+1 信息增量
# 设计：训 Linear probe 从 SID[:l] 预测 flan-t5 embedding，测 cosine sim 增量
# 替代方案（更纯）：用同一序列预测 next token，测 top-1 acc delta

import sys, os, json
sys.path.insert(0, '/fs04/ar57/wenyu/GeneRec/GRID')
import numpy as np
import torch
import torch.nn as nn

OUT18 = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6'
TASK20_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task18'
os.makedirs(TASK20_DIR, exist_ok=True)

EMB_PATH = '/fs04/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt'
SID_PATHS = {
    'A_baseline': '/fs04/ar57/wenyu/GeneRec/GRID/logs/inference/runs/task13_group_a_s22/pickle/merged_predictions_tensor.pt',
    'B_mmq':      '/fs04/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-08/00-32-52/pickle/merged_predictions_tensor.pt',  # 复用 task1 baseline
    'C_gsrq':     '/fs04/ar57/wenyu/GeneRec/GRID/logs/inference/runs/task13_group_c_s22/pickle/merged_predictions_tensor.pt',
}

V = 257   # vocab = W + 1 (含 dedup digit padding)
D_EMB = 64   # token embedding dim for probe

class LinearProbe(nn.Module):
    def __init__(self, vocab=V, d_emb=D_EMB, d_out=2048):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_emb)
        self.probe = nn.Sequential(
            nn.Linear(d_emb, 512),
            nn.ReLU(),
            nn.Linear(512, d_out),
        )
    def forward(self, x):                       # x: (B, l) long
        if x.numel() == 0:
            # l=0: no input, return zeros
            return torch.zeros(x.shape[0], 2048, device=x.device)
        e = self.embed(x).mean(dim=1)          # (B, d_emb)
        return self.probe(e)

def train_probe(sid_prefix, target_emb, n_epochs=300, lr=1e-3, batch_size=512):
    """sid_prefix: (N, l) long
       target_emb: (N, 2048) float
    """
    device = 'cpu'
    model = LinearProbe().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    N = sid_prefix.shape[0]
    target_norm = target_emb / (target_emb.norm(dim=-1, keepdim=True) + 1e-8)

    losses = []
    for epoch in range(n_epochs):
        perm = torch.randperm(N)
        total = 0.0
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            x = sid_prefix[idx].to(device)
            y = target_emb[idx].to(device)
            pred = model(x)
            loss = crit(pred, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += loss.item() * x.shape[0]
        avg = total / N
        losses.append(avg)
        if epoch % 50 == 0:
            print(f'    epoch {epoch}: mse={avg:.4f}')

    # Eval cosine sim on full set
    model.eval()
    with torch.no_grad():
        pred = model(sid_prefix.to(device))
        pred_norm = pred / (pred.norm(dim=-1, keepdim=True) + 1e-8)
        cos = (pred_norm * target_norm).sum(-1).mean().item()
    return cos, losses

def main():
    print('=' * 60)
    print('task16_q9 — 逐层 token probe 边际增益')
    print('=' * 60)

    # Load flan-t5 embedding
    print(f'\nLoading flan-t5 embedding from {EMB_PATH}')
    emb = torch.load(EMB_PATH, weights_only=False)
    N, D = emb.shape
    print(f'  shape={emb.shape}, dtype={emb.dtype}')

    all_results = {}
    for name, sid_path in SID_PATHS.items():
        print(f'\n=== {name}: loading SID from {sid_path} ===')
        if not os.path.exists(sid_path):
            print(f'  SID missing, skipping')
            continue
        sid = torch.load(sid_path, weights_only=False)
        if sid.shape[0] != N:
            print(f'  SID shape {sid.shape} mismatch with emb {emb.shape}, transposing')
            sid = sid.T
        L = sid.shape[1]
        print(f'  SID shape={sid.shape}')

        # Probe per l
        cos_per_l = []
        delta_per_l = []
        prev_cos = 0.0
        for l in range(L + 1):
            if l == 0:
                # l=0 baseline: no info, predict zero vector → cosine = 0
                cos_l = 0.0
            else:
                prefix = sid[:, :l]                # (N, l)
                print(f'  l={l}: training probe on prefix shape {prefix.shape}')
                cos_l, _ = train_probe(prefix, emb, n_epochs=200)
            delta = cos_l - prev_cos
            cos_per_l.append(cos_l)
            delta_per_l.append(delta)
            print(f'    l={l}: cos={cos_l:.4f}, Δ={delta:.4f}')
            prev_cos = cos_l

        all_results[name] = {
            'algorithm': name,
            'L': L,
            'cos_per_l': cos_per_l,
            'delta_per_l': delta_per_l,
        }

    # Save
    combined_path = os.path.join(TASK20_DIR, 'task6_q9_all.json')
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\n=== JSON → {combined_path} ===')

    for name, result in all_results.items():
        p = os.path.join(TASK20_DIR, f'task16_q9_{name}.json')
        with open(p, 'w') as f:
            json.dump(result, f, indent=2)
        print(f'  → {p}')

    # ---------- Plot ----------
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        COLORS = {'A_baseline': '#1f77b4', 'B_mmq': '#ff7f0e', 'C_gsrq': '#2ca02c'}
        LABELS = {
            'A_baseline': 'A baseline (Euclid, no normalize)',
            'B_mmq':      'B MMQ (cosine, normalize=True)',
            'C_gsrq':     'C GSRQ (gain-shape, no normalize)',
        }

        fig, ax = plt.subplots(1, 2, figsize=(12, 4))

        # 左图: cos sim per l
        for name, result in all_results.items():
            ls = list(range(result['L'] + 1))
            ax[0].plot(ls, result['cos_per_l'], marker='o',
                       color=COLORS[name], label=LABELS[name], linewidth=2)
        ax[0].set_xlabel('prefix length l')
        ax[0].set_ylabel(r'cos(probe(SID$_{<l}$), flan-t5)')
        ax[0].set_title('Quantity 9: probe cosine similarity')
        ax[0].legend(fontsize=8)
        ax[0].grid(alpha=0.3)

        # 右图: delta cos per l
        for name, result in all_results.items():
            ls = list(range(result['L'] + 1))
            ax[1].bar([l + (0.27 * list(all_results.keys()).index(name) - 0.27) for l in ls],
                       result['delta_per_l'], width=0.27,
                       color=COLORS[name], label=LABELS[name], alpha=0.85)
        ax[1].set_xlabel('transition l → l+1')
        ax[1].set_ylabel(r'$\Delta$ cosine')
        ax[1].set_title('Quantity 9: marginal gain  $\\Delta$ cos')
        ax[1].legend(fontsize=8)
        ax[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(TASK20_DIR, 'task16_q9_probe_marginal.png'),
                    dpi=120, bbox_inches='tight')
        plt.close()
        print(f'\n=== PNG → {TASK20_DIR}/task16_q9_probe_marginal.png ===')
    except Exception as e:
        print(f'plot failed: {e}')

    # ---------- Verdict ----------
    verdict_lines = ['# Task 20 量 9 — 逐层 token probe 边际增益\n']
    verdict_lines.append('\n## 量化结果（probe cos sim 越高 = prefix 信息量越大）\n')
    for name, result in all_results.items():
        verdict_lines.append(f'\n### {name}\n')
        verdict_lines.append('| l | cos sim | Δ (l→l+1) |')
        verdict_lines.append('|---|---------|-----------|')
        prev = 0.0
        for l in range(result['L'] + 1):
            verdict_lines.append(f'| {l} | {result["cos_per_l"][l]:.4f} | {result["delta_per_l"][l]:.4f} |')
            prev = result['cos_per_l'][l]

    verdict_lines.append('\n## 判所有 idea\n')
    for name, result in all_results.items():
        deltas = result['delta_per_l']
        # delta_3_to_4 是关键（深层是否还带推荐信息）
        if len(deltas) >= 4:
            deep_delta = deltas[3]
        else:
            deep_delta = deltas[-1]
        if deep_delta > 0.01:
            verdict_lines.append(f'- **{name}**: Δ_3→4 = {deep_delta:.4f} > 0.01 → 深层有边际推荐信息，所有 idea 成立')
        else:
            verdict_lines.append(f'- **{name}**: Δ_3→4 = {deep_delta:.4f} ≤ 0.01 → 深层几乎无边际信息，方案价值减半')

    verdict_lines.append('\n## 整体判定\n')
    verdict_lines.append('（结合 task16+19+20 综合分析）')
    verdict_path = os.path.join(TASK20_DIR, 'task6_q9_verdict.md')
    with open(verdict_path, 'w') as f:
        f.write('\n'.join(verdict_lines))
    print(f'\n=== Verdict → {verdict_path} ===')

if __name__ == '__main__':
    main()