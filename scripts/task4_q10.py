#!/usr/bin/env python3
# task6_q10.py — 前缀条件熵 H(c_l | C_<l)
# 量 10: 测 prefix 已知后 c_l 还剩多少不确定性（ReSID 的核心量）
# 实现：训 probe (context → predict c_l) → softmax → 计算平均熵
# 与 task18 q9 完全对称：q9 算 cos sim，q10 算 entropy

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

V = 257       # vocab = 256 codes + 1 dedup digit (256 也用作 padding)
D_EMB = 64
MAX_VOCAB = 257   # output classes

class PrefixProbe(nn.Module):
    """Predict next token given prefix."""
    def __init__(self, vocab=V, d_emb=D_EMB, d_out=MAX_VOCAB):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_emb)
        self.probe = nn.Sequential(
            nn.Linear(d_emb, 256),
            nn.ReLU(),
            nn.Linear(256, d_out),
        )
    def forward(self, x):
        if x.numel() == 0:
            # l=0: predict uniform over vocab
            B = x.shape[0] if x.dim() > 0 else 1
            return torch.zeros(B, MAX_VOCAB, device=x.device if isinstance(x, torch.Tensor) else 'cpu')
        e = self.embed(x).mean(dim=1)
        return self.probe(e)

def train_probe_entropy(sid_full, l, n_epochs=200, lr=1e-3, batch_size=512):
    """Train probe from SID[:, :l] to predict SID[:, l]
       Returns mean entropy over validation set (lower = more predictable)
    """
    device = 'cpu'
    N = sid_full.shape[0]
    target = sid_full[:, l].clamp(max=MAX_VOCAB - 1)        # (N,) long
    prefix = sid_full[:, :l]                                # (N, l) long

    uniform_entropy = float(np.log(MAX_VOCAB))

    # l=0: no input, probe = uniform prediction → entropy ≈ log(V)
    if l == 0:
        return uniform_entropy, uniform_entropy

    model = PrefixProbe().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    # Split 90/10
    n_train = int(N * 0.9)
    perm = torch.randperm(N)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    for epoch in range(n_epochs):
        perm_t = train_idx[torch.randperm(n_train)]
        for i in range(0, n_train, batch_size):
            idx = perm_t[i:i+batch_size]
            x = prefix[idx]
            y = target[idx]
            pred = model(x)
            loss = crit(pred, y)
            optim.zero_grad()
            loss.backward()
            optim.step()

    # Eval entropy on val set
    model.eval()
    with torch.no_grad():
        all_entropies = []
        for i in range(0, len(val_idx), batch_size):
            idx = val_idx[i:i+batch_size]
            x = prefix[idx]
            y = target[idx]
            logits = model(x)
            p = torch.softmax(logits, dim=-1)
            ent = -(p * p.clamp(min=1e-10).log()).sum(-1)
            all_entropies.append(ent)
        entropies = torch.cat(all_entropies)
        mean_entropy = entropies.mean().item()

    return mean_entropy, uniform_entropy

def main():
    print('=' * 60)
    print('task16_q10 — 前缀条件熵 H(c_l | C_<l)')
    print('=' * 60)

    all_results = {}
    for name, sid_path in SID_PATHS.items():
        print(f'\n=== {name}: loading SID from {sid_path} ===')
        if not os.path.exists(sid_path):
            print(f'  SID missing, skipping')
            continue
        sid = torch.load(sid_path, weights_only=False)
        if sid.shape[0] == 4 and sid.shape[1] == 11924:
            sid = sid.T   # (4, 11924) -> (11924, 4)
        L = sid.shape[1]
        print(f'  SID shape={sid.shape}, L={L}')

        # Per l
        entropy_per_l = []
        uniform_per_l = []
        delta_per_l = []
        prev_ent = float(np.log(MAX_VOCAB))   # uniform entropy baseline
        for l in range(L):
            print(f'  l={l+1}: training probe to predict c_{l+1}')
            ent, uni = train_probe_entropy(sid, l)
            delta = prev_ent - ent          # reduction in entropy
            entropy_per_l.append(ent)
            uniform_per_l.append(uni)
            delta_per_l.append(delta)
            print(f'    H(c_{l+1}|C_<{l+1})={ent:.4f} nat (uniform={uni:.4f}), Δ_entropy={delta:.4f}')
            prev_ent = ent

        all_results[name] = {
            'algorithm': name,
            'L': L,
            'entropy_per_l': entropy_per_l,
            'uniform_entropy': uniform_per_l,
            'delta_entropy_per_l': delta_per_l,
        }

    # Save
    combined_path = os.path.join(TASK20_DIR, 'task6_q10_all.json')
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\n=== JSON → {combined_path} ===')

    for name, result in all_results.items():
        p = os.path.join(TASK20_DIR, f'task16_q10_{name}.json')
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

        # 左: H(c_l | C_<l) per l
        for name, result in all_results.items():
            ls = list(range(1, result['L'] + 1))
            ax[0].plot(ls, result['entropy_per_l'], marker='o',
                       color=COLORS[name], label=LABELS[name], linewidth=2)
        ax[0].axhline(np.log(MAX_VOCAB), color='gray', linestyle='--', label='uniform')
        ax[0].set_xlabel('layer l')
        ax[0].set_ylabel(r'$H(c_\ell \mid C_{<\ell})$ (nats)')
        ax[0].set_title('Quantity 10: prefix conditional entropy')
        ax[0].legend(fontsize=8)
        ax[0].grid(alpha=0.3)

        # 右: delta H
        for name, result in all_results.items():
            ls = list(range(1, result['L'] + 1))
            ax[1].bar([l + (0.27 * list(all_results.keys()).index(name) - 0.27) for l in ls],
                       result['delta_entropy_per_l'], width=0.27,
                       color=COLORS[name], label=LABELS[name], alpha=0.85)
        ax[1].set_xlabel('layer l')
        ax[1].set_ylabel(r'$\Delta H$ (entropy reduction)')
        ax[1].set_title('Quantity 10: entropy reduction per layer')
        ax[1].legend(fontsize=8)
        ax[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(TASK20_DIR, 'task16_q10_prefix_entropy.png'),
                    dpi=120, bbox_inches='tight')
        plt.close()
        print(f'\n=== PNG → {TASK20_DIR}/task16_q10_prefix_entropy.png ===')
    except Exception as e:
        print(f'plot failed: {e}')

    # ---------- Verdict ----------
    verdict_lines = ['# Task 20 量 10 — 前缀条件熵 H(c_l | C_<l)\n']
    verdict_lines.append('\n## 量化结果\n')
    verdict_lines.append(f'(uniform random baseline: H = log({MAX_VOCAB}) ≈ {np.log(MAX_VOCAB):.3f} nats)\n')
    verdict_lines.append('| Algorithm | H(c_1) | H(c_2) | H(c_3) | H(c_4) |')
    verdict_lines.append('|-----------|--------|--------|--------|--------|')
    for name, result in all_results.items():
        ents = [f'{e:.3f}' for e in result['entropy_per_l']]
        while len(ents) < 4:
            ents.append('-')
        verdict_lines.append(f'| {name} | {ents[0]} | {ents[1]} | {ents[2]} | {ents[3]} |')

    verdict_lines.append('\n## 判 D 命门（prefix 可预测 vs 可精化）\n')
    verdict_lines.append('| Algorithm | 低熵层 (H < 4.0) | 该层 H | 含义 |')
    verdict_lines.append('|-----------|-------------------|--------|------|')
    for name, result in all_results.items():
        for l, h in enumerate(result['entropy_per_l']):
            if h < 4.0:
                verdict_lines.append(f'| {name} | L{l+1} | {h:.3f} | prefix 可预测（ReSID 眼里"好的"）|')

    verdict_lines.append('\n## 整体判定\n')
    verdict_lines.append('（结合 task16+19+20 综合）')
    verdict_path = os.path.join(TASK20_DIR, 'task6_q10_verdict.md')
    with open(verdict_path, 'w') as f:
        f.write('\n'.join(verdict_lines))
    print(f'\n=== Verdict → {verdict_path} ===')

if __name__ == '__main__':
    main()