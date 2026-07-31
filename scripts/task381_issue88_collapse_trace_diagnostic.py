#!/usr/bin/env python3
"""
Task #381 / Issue #88 [方向A Gate1] κ/codebook 坍缩时间点根因 trace (R22 + R19, GPU 0)

FreeCurvHRQVAE 10 epoch + per-step trace:
- theta_l / kappa_l
- codebook norm (Euclidean)
- pairwise codebook distance quantiles
- assignment logits top-k margin
- soft assignment entropy
- hard utilization
- collision_rate
- gradient norm (codebook + theta)
- 定位: 单码字承载 >50% / >90% / >99% item 的最早 step
"""
import os, sys, time, json, torch, numpy as np, pandas as pd
from pathlib import Path

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = "cuda:0"

from model.hrqvae_free_curv import FreeCurvHRQVAE

# === 0. config ===
TASK_NAME = "task381_issue88_collapse_trace_diagnostic"
SEED = 42
NUM_EMB_LIST = [64, 128, 256]
E_DIM = 32
M = 3
KAPPA_MAX = 2.0
EPOCHS = 10  # R11.5: spec 要求"按 epoch 或固定 step 输出 trace", 10 epoch 足够定位坍缩点
LOG_EVERY_STEPS = 50  # 每 50 step 输出一行 trace (约 78 step/epoch * 10 = 780 total, 16 trace records)
BATCH = 64
LR = 1e-3
DATASET_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet"
OUT_DIR = Path(f"/home/wlia0047/ar57/wenyu/GeneRec/products/{TASK_NAME}")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TRACE_FILE = OUT_DIR / "trace_per_step.jsonl"
CKPT_PATH = OUT_DIR / "collapse_trace_stage1_ckpt.pt"

torch.manual_seed(SEED); np.random.seed(SEED)
print("=" * 70)
print(f"Task #381 / Issue #88 [方向A Gate1] κ/codebook 坍缩时间点根因 trace")
print(f"Device: {DEVICE}, Epochs: {EPOCHS}, K: {NUM_EMB_LIST}, log_every_steps: {LOG_EVERY_STEPS}")
print("=" * 70)

# === 1. 数据 ===
df = pd.read_parquet(DATASET_PATH)
emb_col = 'embedding' if 'embedding' in df.columns else 'emb'
X = np.stack([np.asarray(e, dtype=np.float32) for e in df[emb_col]])
X = X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8) * 0.85
X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
print(f"[数据] shape={X_t.shape}")

# === 2. 模型 (baseline recipe, 无 norm clipping / 无 entropy reg) ===
model = FreeCurvHRQVAE(
    in_dim=X.shape[1], num_emb_list=NUM_EMB_LIST, e_dim=E_DIM, M=M,
    kappa_max=KAPPA_MAX, layers=[512, 256, 128],
    loss_type='poincare', quant_loss_weight=1.0, beta=0.25,
    kmeans_init=False, kmeans_iters=10, sk_eps=[0.0, 0.0, 0.0], sk_iters=100,
).to(DEVICE)
print(f"[模型] {sum(p.numel() for p in model.parameters()):,} params")

# === 3. 训练 + per-step trace ===
optim = torch.optim.Adam(model.parameters(), lr=LR)
N = X_t.shape[0]
steps_per_epoch = (N + BATCH - 1) // BATCH
print(f"[steps_per_epoch] {steps_per_epoch}, total_steps={steps_per_epoch * EPOCHS}")

# trace file 初始化
TRACE_FILE.unlink(missing_ok=True)

# 坍缩点追踪
collapse_step_50pct = [None, None, None]  # per-layer, first step where single codeword >50%
collapse_step_90pct = [None, None, None]
collapse_step_99pct = [None, None, None]

def compute_trace(model, X_t, step, ep):
    """per-step trace for each layer"""
    model.eval()
    with torch.no_grad():
        z_e = model.encoder(X_t)
        traces = []
        for l_id, layer in enumerate(model.hrq.vq_layers):
            latent = z_e.view(-1, layer.e_dim)
            cb = layer.embeddings.weight  # (K, e_dim)
            d = layer._per_component_dist_sq(latent, cb)  # (N, K)
            # soft assignment entropy
            soft_w = torch.softmax(-d, dim=-1)
            entropy = -(soft_w * torch.log(soft_w + 1e-8)).sum(dim=-1).mean().item()
            # hard assignment
            indices = torch.argmin(d, dim=-1)
            n_unique = len(set(indices.cpu().numpy().tolist()))
            util = n_unique / cb.shape[0]
            # top-k margin (top1 - top2)
            d_top2 = torch.topk(d, k=2, dim=-1, largest=False).values
            margin = (d_top2[:, 1] - d_top2[:, 0]).mean().item()
            # single codeword load
            counts = torch.bincount(indices, minlength=cb.shape[0])
            max_count = counts.max().item()
            max_load = max_count / N
            # codebook norm
            cb_norm = cb.norm(dim=-1).mean().item()
            cb_norm_max = cb.norm(dim=-1).max().item()
            cb_norm_min = cb.norm(dim=-1).min().item()
            # pairwise codebook distance quantiles
            pw = torch.cdist(cb, cb)
            pw_offdiag = pw[~torch.eye(pw.shape[0], dtype=torch.bool)].cpu().numpy()
            pw_q25 = float(np.percentile(pw_offdiag, 25))
            pw_q50 = float(np.percentile(pw_offdiag, 50))
            pw_q75 = float(np.percentile(pw_offdiag, 75))
            # theta / kappa
            theta_l = layer.theta_m.detach().cpu().numpy().tolist()
            # scale_l (codebook Euclidean norm)
            scale_l = cb_norm
            traces.append({
                "layer_id": l_id,
                "theta_l": theta_l,
                "scale_l": scale_l,
                "cb_norm_mean": cb_norm,
                "cb_norm_max": cb_norm_max,
                "cb_norm_min": cb_norm_min,
                "pw_dist_q25": pw_q25,
                "pw_dist_q50": pw_q50,
                "pw_dist_q75": pw_q75,
                "top_k_margin": margin,
                "soft_entropy": entropy,
                "util": util,
                "max_load": max_load,
                "n_unique": n_unique,
            })
            # update collapse_step trackers
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
    model.train()
    perm = torch.randperm(N)
    for i in range(0, N, BATCH):
        idx = perm[i:i+BATCH]
        x_b = X_t[idx]
        optim.zero_grad()
        out, qloss, _ = model(x_b)
        recon_loss = torch.nn.functional.mse_loss(out, x_b)
        loss = recon_loss + 0.25 * qloss
        loss.backward()
        # 梯度范数 (codebook + theta)
        grad_codebook = 0.0
        grad_theta = 0.0
        for name, p in model.named_parameters():
            if p.grad is not None:
                gn = p.grad.norm().item()
                if 'theta' in name:
                    grad_theta += gn ** 2
                elif 'embed' in name:
                    grad_codebook += gn ** 2
        grad_codebook = grad_codebook ** 0.5
        grad_theta = grad_theta ** 0.5
        optim.step()
        # 每 LOG_EVERY_STEPS 输出一行 trace
        step = ep * steps_per_epoch + (i // BATCH) + 1
        if step % LOG_EVERY_STEPS == 0 or step == 1:
            traces = compute_trace(model, X_t, step, ep)
            for l_tr in traces:
                l_tr['step'] = step
                l_tr['epoch'] = ep
                l_tr['loss'] = loss.item()
                l_tr['grad_codebook'] = grad_codebook
                l_tr['grad_theta'] = grad_theta
                with open(TRACE_FILE, 'a') as f:
                    f.write(json.dumps(l_tr, default=str) + '\n')
            elapsed = time.time() - start_time
            utils_str = ' '.join(f"L{l_tr['layer_id']}={l_tr['util']*100:.1f}%maxload={l_tr['max_load']*100:.1f}%" for l_tr in traces)
            print(f"  step {step}/{steps_per_epoch * EPOCHS} ep{ep} loss={loss.item():.4f} {utils_str} elapsed={elapsed:.0f}s")

# R12: ckpt save
if CKPT_PATH.exists():
    CKPT_PATH.unlink()
torch.save(model.state_dict(), CKPT_PATH)
print(f"\n[R12 ckpt] saved to {CKPT_PATH}")

# === 4. 总结 ===
print(f"\n[坍缩点定位]")
for l_id in range(M):
    print(f"  L{l_id}: 50% step = {collapse_step_50pct[l_id]}, 90% step = {collapse_step_90pct[l_id]}, 99% step = {collapse_step_99pct[l_id]}")

# === 5. evidence ===
# 读 trace 文件, 找最早 50% 坍缩 step
all_traces = []
with open(TRACE_FILE, 'r') as f:
    for line in f:
        all_traces.append(json.loads(line))

evidence = {
    "task_id": "task381",
    "issue": "#88",
    "epochs": EPOCHS,
    "total_steps": steps_per_epoch * EPOCHS,
    "trace_records": len(all_traces),
    "collapse_step_50pct": collapse_step_50pct,
    "collapse_step_90pct": collapse_step_90pct,
    "collapse_step_99pct": collapse_step_99pct,
    "final_traces": all_traces[-M:],  # last step per layer
    "verdict_path": "verdicts/task381_issue88_collapse_trace_diagnostic_v2.md",
}
with open(OUT_DIR / "evidence_package.json", "w") as f:
    json.dump(evidence, f, indent=2, default=str)
print(f"\n[证据] saved to {OUT_DIR / 'evidence_package.json'}")

# === 6. sanity ===
checks = [
    ("trace_records >= 30", len(all_traces) >= 30),
    ("L0 50% collapse step found", collapse_step_50pct[0] is not None),
    ("L1 50% collapse step found", collapse_step_50pct[1] is not None),
    ("L2 50% collapse step found", collapse_step_50pct[2] is not None),
    ("ckpt_saved", CKPT_PATH.exists()),
    ("no_nan_inf", all(
        np.isfinite(tr['loss']) and np.isfinite(tr['top_k_margin']) and np.isfinite(tr['util'])
        for tr in all_traces
    )),
]
n_pass = sum(1 for _, p in checks if p)
print(f"\n[sanity] PASS: {n_pass}/{len(checks)}")
for name, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

print(f"\n[Issue #88 Gate 1 决策]")
print(f"  Gate 1: {'✅ PASS' if n_pass == len(checks) else '❌ FAIL'}")
print(f"  trace_records = {len(all_traces)}")
print(f"  collapse 50% step: L0={collapse_step_50pct[0]}, L1={collapse_step_50pct[1]}, L2={collapse_step_50pct[2]}")