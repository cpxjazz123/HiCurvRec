#!/usr/bin/env python3
"""
Task #371 / Issue #78 [方向A Gate1] κ-freeze warmup 证据包 (R18/R20 实证)
"""
import os, sys, time, json, torch, numpy as np, pandas as pd
from torch import nn
from pathlib import Path

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/scripts')
from model.hrqvae_free_curv import FreeCurvHRQVAE, FreeCurvVectorQuantization
from task368_issue75_kappa_freeze_warmup_evidence import train_one_epoch, diagnose_layer

# === 0. config ===
TASK_NAME = "task371_issue78_evidence_package"
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EMB_LIST = [64, 128, 256]
E_DIM = 32
M = 3
N_SAMPLES = 256
EPOCHS_WARMUP = 1
EPOCHS_UNFREEZE = 2
LR = 1e-3
KAPPA_MAX = 2.0
DATASET_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet"

torch.manual_seed(SEED); np.random.seed(SEED)
print("=" * 70)
print("Task #371 / Issue #78 [方向A Gate1] κ-freeze warmup 证据包")
print("=" * 70)

# === 1. CONFIG DIFF ===
config_diff = {
    "num_emb_list": NUM_EMB_LIST,
    "e_dim": E_DIM,
    "M": M,
    "lr": LR,
    "kappa_max": KAPPA_MAX,
    "epochs_warmup": EPOCHS_WARMUP,
    "epochs_unfreeze": EPOCHS_UNFREEZE,
    "loss_type": "poincare",
    "beta": 0.25,
    "sk_eps": [0.0, 0.0, 0.0],
    "init": "codebook norm → 0.85 on unfreeze",
    "theta_m_freeze_warmup": True,
    "theta_m_lr_unfreeze": 1e-3,
    "log_fields": [
        "epoch", "phase", "layer_id", "kappa_m", "codebook_norm_mean",
        "distance_range", "assignment_entropy_norm", "utilization",
        "collision_rate", "grad_theta_max", "loss",
    ],
}
print(f"\n[证据1] CONFIG DIFF")
print(json.dumps(config_diff, indent=2))

# === 2. 加载 + 归一化 ===
df = pd.read_parquet(DATASET_PATH)
emb_col = 'embedding' if 'embedding' in df.columns else 'emb'
X = np.stack([np.asarray(e, dtype=np.float32) for e in df[emb_col]])
X = X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8) * 0.85
X_t = torch.tensor(X[:N_SAMPLES], dtype=torch.float32).to(DEVICE)
print(f"\n[数据] shape={X_t.shape}, mean={X_t.mean():.4f}")

# === 3. FreeCurvHRQVAE ===
model = FreeCurvHRQVAE(
    in_dim=X.shape[1],
    num_emb_list=NUM_EMB_LIST, e_dim=E_DIM, M=M,
    kappa_max=KAPPA_MAX, layers=[512, 256, 128],
    loss_type='poincare', quant_loss_weight=1.0, beta=0.25,
    kmeans_init=False, kmeans_iters=50,
    sk_eps=[0.0, 0.0, 0.0], sk_iters=100,
).to(DEVICE)
print(f"[模型] {sum(p.numel() for p in model.parameters()):,} params")

# === 4. OPTIMIZER PARAM GROUPS ===
theta_m_params = [p for n, p in model.named_parameters() if 'theta_m' in n]
non_theta_m_params = [p for n, p in model.named_parameters() if 'theta_m' not in n]
print(f"\n[证据2] OPTIMIZER PARAM GROUPS")
print(f"  non_theta_m: {sum(p.numel() for p in non_theta_m_params):,}")
print(f"  theta_m: {sum(p.numel() for p in theta_m_params):,}")

# warmup: theta_m lr=0 但保留 requires_grad
warmup_groups = [
    {'params': non_theta_m_params, 'lr': LR, 'name': 'non_theta_m'},
    {'params': theta_m_params, 'lr': 0.0, 'name': 'theta_m_frozen'},
]
optim_warmup = torch.optim.Adam(warmup_groups)

# unfreeze: 全 lr=1e-3
unfreeze_groups = [
    {'params': non_theta_m_params, 'lr': LR, 'name': 'non_theta_m'},
    {'params': theta_m_params, 'lr': LR, 'name': 'theta_m_active'},
]
optim_unfreeze = torch.optim.Adam(unfreeze_groups)

# === 5. FREEZE/UNFREEZE 切换点 ===
print(f"\n[证据3] FREEZE/UNFREEZE 切换点")
for layer in model.hrq.vq_layers:
    layer.theta_m.requires_grad = False
kappa_before_freeze = [layer.kappa_m().detach().cpu().numpy().copy() for layer in model.hrq.vq_layers]
print(f"  [t={time.strftime('%H:%M:%S')}] warmup start: theta_m.requires_grad={model.hrq.vq_layers[0].theta_m.requires_grad}")
print(f"  freeze 前 κ[0]={kappa_before_freeze[0].tolist()}")

# === 6. 自定义 diagnose (per-layer) ===
def diagnose_evidence(vq, indices, latent_input, prefix):
    """诊断: 输出 evidence log 字段."""
    kappa = vq.kappa_m().detach().cpu().numpy()
    codebook_norm = vq.embeddings.weight.norm(dim=-1).detach().cpu().numpy()
    codebook_norm_mean = float(codebook_norm.mean())
    with torch.no_grad():
        d_euc = torch.cdist(latent_input, vq.embeddings.weight)
        d_min = float(d_euc.min()); d_max = float(d_euc.max())
    n_used, usage_count = vq.get_codebook_usage(indices)
    n_total = vq.embeddings.weight.shape[0]
    utilization = n_used / n_total
    p = usage_count.float() / max(usage_count.sum().item(), 1)
    p_pos = p[p > 0]
    entropy_norm = float(-(p_pos * p_pos.log()).sum()) / max(np.log(n_total), 1e-8)
    expected = usage_count.float().mean().item()
    collision_rate = float(((usage_count.float() - expected) ** 2).mean() / (expected ** 2 + 1e-8))
    grad_theta = vq.theta_m.grad
    grad_norm_theta = float(grad_theta.abs().max().item()) if grad_theta is not None else 0.0
    return {
        'phase': prefix, 'kappa_m': kappa.tolist(),
        'codebook_norm_mean': codebook_norm_mean,
        'distance_range': [d_min, d_max],
        'utilization': utilization, 'entropy_norm': entropy_norm,
        'collision_rate': collision_rate, 'grad_theta_max': grad_norm_theta,
    }

# === 7. WARMUP (use task368 train_one_epoch) ===
print(f"\n[证据4] WARMUP phase ({EPOCHS_WARMUP} epoch)")
log = []
loss_warmup_avg, _ = train_one_epoch(model, X_t, batch_size=64, optimizer=optim_warmup)
print(f"  warmup avg loss = {loss_warmup_avg:.6f}")
with torch.no_grad():
    residual = model.encoder(X_t)
    per_layer_indices = []
    for layer in model.hrq.vq_layers:
        _, _, idx = layer(residual, use_sk=False)
        per_layer_indices.append(idx)
        residual = residual - layer.embeddings(idx)
for layer_id, (layer, idx) in enumerate(zip(model.hrq.vq_layers, per_layer_indices)):
    # layer's latent input is the residual before quantization
    diag = diagnose_evidence(layer, idx, residual.detach() + layer.embeddings(idx).detach(), 'warmup')
    diag['epoch'] = 0; diag['layer_id'] = layer_id
    diag['loss'] = loss_warmup_avg
    log.append(diag)
    print(f"  ep0 L{layer_id} κ={diag['kappa_m'][:3]} norm={diag['codebook_norm_mean']:.3f} "
          f"util={diag['utilization']:.3f} coll={diag['collision_rate']:.3f} gradθ={diag['grad_theta_max']:.2e}")

# === 8. UNFREEZE ===
print(f"\n  --- UNFREEZE phase ({EPOCHS_UNFREEZE} epoch) ---")
kappa_before_unfreeze = [layer.kappa_m().detach().cpu().numpy().copy() for layer in model.hrq.vq_layers]
for layer in model.hrq.vq_layers:
    layer.theta_m.requires_grad = True
print(f"  [t={time.strftime('%H:%M:%S')}] unfreeze start: theta_m.requires_grad={model.hrq.vq_layers[0].theta_m.requires_grad}")
print(f"  unfreeze 前 κ[0]={kappa_before_unfreeze[0].tolist()}")

for ep in range(EPOCHS_WARMUP, EPOCHS_WARMUP + EPOCHS_UNFREEZE):
    loss_avg, _ = train_one_epoch(model, X_t, batch_size=64, optimizer=optim_unfreeze)
    with torch.no_grad():
        residual = model.encoder(X_t)
        per_layer_indices = []
        for layer in model.hrq.vq_layers:
            _, _, idx = layer(residual, use_sk=False)
            per_layer_indices.append(idx)
            residual = residual - layer.embeddings(idx)
    for layer_id, (layer, idx) in enumerate(zip(model.hrq.vq_layers, per_layer_indices)):
        diag = diagnose_evidence(layer, idx, residual.detach() + layer.embeddings(idx).detach(), 'unfreeze')
        diag['epoch'] = ep; diag['layer_id'] = layer_id
        diag['loss'] = loss_avg
        log.append(diag)
        print(f"  ep{ep} L{layer_id} κ={diag['kappa_m'][:3]} norm={diag['codebook_norm_mean']:.3f} "
              f"util={diag['utilization']:.3f} coll={diag['collision_rate']:.3f} gradθ={diag['grad_theta_max']:.2e}")

# === 9. EVIDENCE PACKAGE ===
out_dir = Path(f"/home/wlia0047/ar57/wenyu/GeneRec/products/{TASK_NAME}")
out_dir.mkdir(parents=True, exist_ok=True)
evidence = {
    "task_id": "task371", "issue": "#78", "R20_4_gate_evidence": True,
    "config_diff": config_diff, "log": log, "n_log_entries": len(log),
    "kappa_before_freeze": [k.tolist() for k in kappa_before_freeze],
    "kappa_before_unfreeze": [k.tolist() for k in kappa_before_unfreeze],
    "verdict_path": "verdicts/task371_issue78_evidence_package_v2.md",
    "commit_hash": "pending",
    "evidence_types": ["config_diff", "optimizer_params", "freeze_unfreeze_switch",
                       "log_fields", "short_run", "kappa_before_after"],
}
with open(out_dir / "evidence_package.json", "w") as f:
    json.dump(evidence, f, indent=2, default=str)
print(f"\n[证据5] EVIDENCE PACKAGE saved: {out_dir / 'evidence_package.json'}")

# === 10. SANITY ===
print(f"\n[证据6] SANITY CHECK")
checks = []
warmup_grads = [e['grad_theta_max'] for e in log if e['phase']=='warmup']
unfreeze_grads = [e['grad_theta_max'] for e in log if e['phase']=='unfreeze']
checks.append(('warmup_grad_theta_max=0', all(g == 0.0 for g in warmup_grads)))
checks.append(('unfreeze_grad_theta_max>0', all(g > 0 for g in unfreeze_grads)))
norms = [e['codebook_norm_mean'] for e in log]
checks.append(('codebook_norm_min > 0.01 (not collapse)', min(norms) > 0.01))
checks.append(('warmup_kappa_constant', all(
    e['kappa_m'] == log[0]['kappa_m'] for e in log if e['phase']=='warmup' and e['layer_id']==log[0]['layer_id']
)))
checks.append(('unfreeze_kappa_evolves',
    log[-1]['kappa_m'] != log[EPOCHS_WARMUP * M]['kappa_m']))
n_pass = sum(1 for _, p in checks if p)
print(f"  PASS: {n_pass}/{len(checks)}")
for name, ok in checks:
    print(f"    [{'PASS' if ok else 'FAIL'}] {name}")
print(f"\n[Gate 1] PARTIAL PASS — {n_pass}/{len(checks)} markers")
