#!/usr/bin/env python3
"""
Task #384 / Issue #91 [方向A Gate1] kmeans_init + β=0 first epoch 修复 step1 坍缩 (R22 + R19, GPU 0)

FreeCurvHRQVAE 50 epoch + kmeans_init=True + β schedule (β=0 first epoch, 0.25 after):
- 修复: 用真实数据点 kmeans 初始化 codebook (替代 random)
- 修复: first epoch β=0 (跳过 commitment loss, 只走 recon loss)
- 验证: step 1 不再立即 50%+ 坍缩; 最终三层 util ≥90%
"""
import os, sys, time, json, torch, numpy as np, pandas as pd
from pathlib import Path

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = "cuda:0"

from model.hrqvae_free_curv import FreeCurvHRQVAE

# === 0. config ===
TASK_NAME = "task384_issue91_kmeans_init_beta_warmup_fix"
SEED = 42
NUM_EMB_LIST = [64, 128, 256]
E_DIM = 32
M = 3
KAPPA_MAX = 2.0
EPOCHS = 50  # R11.5: 50 epoch 完整训练验证修复
LOG_EVERY_STEPS = 50  # per-step trace
BATCH = 64
LR = 1e-3
DATASET_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet"
OUT_DIR = Path(f"/home/wlia0047/ar57/wenyu/GeneRec/products/{TASK_NAME}")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TRACE_FILE = OUT_DIR / "trace_per_step.jsonl"
CKPT_PATH = OUT_DIR / "kmeans_init_beta_warmup_ckpt.pt"

torch.manual_seed(SEED); np.random.seed(SEED)
print("=" * 70)
print(f"Task #384 / Issue #91 [方向A Gate1] kmeans_init + β=0 first epoch 修复")
print(f"Device: {DEVICE}, Epochs: {EPOCHS}, K: {NUM_EMB_LIST}, kmeans_init=True, β_schedule: 0→0.25")
print("=" * 70)

# === 1. 数据 ===
df = pd.read_parquet(DATASET_PATH)
emb_col = 'embedding' if 'embedding' in df.columns else 'emb'
X = np.stack([np.asarray(e, dtype=np.float32) for e in df[emb_col]])
X = X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8) * 0.85
X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
print(f"[数据] shape={X_t.shape}")

# === 2. 模型 (修复: kmeans_init=True 替代 baseline recipe 的 kmeans_init=False) ===
model = FreeCurvHRQVAE(
    in_dim=X.shape[1], num_emb_list=NUM_EMB_LIST, e_dim=E_DIM, M=M,
    kappa_max=KAPPA_MAX, layers=[512, 256, 128],
    loss_type='poincare', quant_loss_weight=1.0, beta=0.25,
    kmeans_init=True, kmeans_iters=10,  # ← 修复: kmeans_init=True
    sk_eps=[0.0, 0.0, 0.0], sk_iters=100,
).to(DEVICE)
print(f"[模型] {sum(p.numel() for p in model.parameters()):,} params")

# R-FIX: kmeans_init 在 first forward 时调用 init_emb, 但 batch=64 < K=128/256 → sklearn 拒绝
# init_emb 标志: `if not self.initted and self.training:` — eval 模式不触发 init_emb
# 必须用 model.train() 模式跑 full dataset (n=9922 >= max K=256) 触发 init_emb
print("[R-FIX] 触发 kmeans_init on full dataset (train mode, n=9922 ≥ max K=256)")
with torch.no_grad():
    model.train()  # 必须 train 模式, 否则 init_emb 跳过
    _ = model(X_t, use_sk=False)  # 第一次 forward 触发每层 init_emb on full data
print("[R-FIX] kmeans_init 触发完毕, 每层 initted=True")

# === 3. 训练 + β schedule (epoch 0: β=0, epoch 1+: β=0.25) + per-step trace ===
optim = torch.optim.Adam(model.parameters(), lr=LR)
N = X_t.shape[0]
steps_per_epoch = (N + BATCH - 1) // BATCH
print(f"[steps_per_epoch] {steps_per_epoch}, total_steps={steps_per_epoch * EPOCHS}")

TRACE_FILE.unlink(missing_ok=True)
collapse_step_50pct = [None, None, None]
collapse_step_90pct = [None, None, None]
collapse_step_99pct = [None, None, None]

def compute_trace(model, X_t, step, ep, beta):
    model.eval()
    with torch.no_grad():
        z_e = model.encoder(X_t)
        traces = []
        for l_id, layer in enumerate(model.hrq.vq_layers):
            latent = z_e.view(-1, layer.e_dim)
            cb = layer.embeddings.weight
            d = layer._per_component_dist_sq(latent, cb)
            soft_w = torch.softmax(-d, dim=-1)
            entropy = -(soft_w * torch.log(soft_w + 1e-8)).sum(dim=-1).mean().item()
            indices = torch.argmin(d, dim=-1)
            n_unique = len(set(indices.cpu().numpy().tolist()))
            util = n_unique / cb.shape[0]
            counts = torch.bincount(indices, minlength=cb.shape[0])
            max_load = counts.max().item() / N
            cb_norm = cb.norm(dim=-1).mean().item()
            traces.append({
                "layer_id": l_id,
                "cb_norm": cb_norm,
                "soft_entropy": entropy,
                "util": util,
                "max_load": max_load,
                "n_unique": n_unique,
                "current_beta": beta,
            })
            if max_load > 0.5 and collapse_step_50pct[l_id] is None:
                collapse_step_50pct[l_id] = step
            if max_load > 0.9 and collapse_step_90pct[l_id] is None:
                collapse_step_90pct[l_id] = step
            if max_load > 0.99 and collapse_step_99pct[l_id] is None:
                collapse_step_99pct[l_id] = step
    model.train()
    return traces

start_time = time.time()
for ep in range(EPOCHS):
    # β schedule: epoch 0 = 0 (only recon), epoch 1+ = 0.25 (commitment + codebook)
    current_beta = 0.0 if ep == 0 else 0.25
    for layer in model.hrq.vq_layers:
        layer.beta = current_beta
    print(f"\n[ep {ep}/{EPOCHS}] β={current_beta}")

    model.train()
    perm = torch.randperm(N)
    for i in range(0, N, BATCH):
        idx = perm[i:i+BATCH]
        x_b = X_t[idx]
        optim.zero_grad()
        out, qloss, _ = model(x_b)
        recon_loss = torch.nn.functional.mse_loss(out, x_b)
        loss = recon_loss + current_beta * qloss  # 用 outer β 控制 commitment loss
        loss.backward()
        optim.step()
        step = ep * steps_per_epoch + (i // BATCH) + 1
        if step % LOG_EVERY_STEPS == 0 or step == 1:
            traces = compute_trace(model, X_t, step, ep, current_beta)
            for l_tr in traces:
                l_tr['step'] = step
                l_tr['epoch'] = ep
                l_tr['loss'] = loss.item()
                with open(TRACE_FILE, 'a') as f:
                    f.write(json.dumps(l_tr, default=str) + '\n')
            elapsed = time.time() - start_time
            utils_str = ' '.join(f"L{l_tr['layer_id']}=util{l_tr['util']*100:.1f}%maxload{l_tr['max_load']*100:.1f}%" for l_tr in traces)
            print(f"  step {step}/{steps_per_epoch * EPOCHS} ep{ep} loss={loss.item():.4f} β={current_beta} {utils_str} elapsed={elapsed:.0f}s")

# R12: ckpt save
if CKPT_PATH.exists():
    CKPT_PATH.unlink()
torch.save(model.state_dict(), CKPT_PATH)
print(f"\n[R12 ckpt] saved to {CKPT_PATH}")

# === 4. 总结 + final eval ===
print(f"\n[坍缩点定位]")
for l_id in range(M):
    print(f"  L{l_id}: 50% step = {collapse_step_50pct[l_id]}, 90% step = {collapse_step_90pct[l_id]}, 99% step = {collapse_step_99pct[l_id]}")

model.eval()
with torch.no_grad():
    final_utils = []
    final_collisions = []
    z_e = model.encoder(X_t)
    residual = z_e
    for layer_id, layer in enumerate(model.hrq.vq_layers):
        z_q, _, _ = layer(residual, use_sk=False)
        residual = residual - z_q
    for layer_id, layer in enumerate(model.hrq.vq_layers):
        z_q_full = model.encoder(X_t)
        d = layer._per_component_dist_sq(z_q_full.view(-1, layer.e_dim), layer.embeddings.weight)
        idx = torch.argmin(d, dim=-1)
        n_unique = len(set(idx.cpu().numpy().tolist()))
        final_utils.append(n_unique / layer.embeddings.weight.shape[0])

# === 5. evidence ===
all_traces = []
with open(TRACE_FILE, 'r') as f:
    for line in f:
        all_traces.append(json.loads(line))

evidence = {
    "task_id": "task384",
    "issue": "#91",
    "epochs": EPOCHS,
    "fix_applied": "kmeans_init=True + β=0 first epoch",
    "collapse_step_50pct": collapse_step_50pct,
    "collapse_step_90pct": collapse_step_90pct,
    "collapse_step_99pct": collapse_step_99pct,
    "final_util_per_layer": final_utils,
    "trace_records": len(all_traces),
    "verdict_path": "verdicts/task384_issue91_kmeans_init_beta_warmup_fix_v2.md",
}
with open(OUT_DIR / "evidence_package.json", "w") as f:
    json.dump(evidence, f, indent=2, default=str)
print(f"\n[证据] saved to {OUT_DIR / 'evidence_package.json'}")

# === 6. sanity + Issue #91 PASS 判定 ===
# Issue #91 PASS: step1 不出现单码字 >50% load; 最终三层 util ≥90%
step1_traces = [t for t in all_traces if t['step'] == 1]
max_load_step1 = max(t['max_load'] for t in step1_traces) if step1_traces else 0.0
util_all_pass = all(u >= 0.9 for u in final_utils)
collision_ok = True  # 占位
checks = [
    ("step1 max_load < 50%", max_load_step1 < 0.5),
    ("L0 final util >= 90%", final_utils[0] >= 0.9),
    ("L1 final util >= 90%", final_utils[1] >= 0.9),
    ("L2 final util >= 90%", final_utils[2] >= 0.9),
    ("ckpt_saved", CKPT_PATH.exists()),
    ("no_nan_inf", all(np.isfinite(t['loss']) and np.isfinite(t['util']) for t in all_traces)),
    ("trace_records >= 100", len(all_traces) >= 100),
]
n_pass = sum(1 for _, p in checks if p)
print(f"\n[sanity] PASS: {n_pass}/{len(checks)}")
for name, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

print(f"\n[Issue #91 Gate 1 决策]")
print(f"  Gate 1: {'✅ PASS' if n_pass == len(checks) else '❌ FAIL'}")
print(f"  step1 max_load = {max_load_step1*100:.2f}% (目标 <50%)")
print(f"  final_util L0/L1/L2 = {final_utils[0]*100:.2f}% / {final_utils[1]*100:.2f}% / {final_utils[2]*100:.2f}%")
print(f"  collapse 50% step: L0={collapse_step_50pct[0]}, L1={collapse_step_50pct[1]}, L2={collapse_step_50pct[2]}")