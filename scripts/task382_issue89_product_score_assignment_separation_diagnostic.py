#!/usr/bin/env python3
"""
Task #382 / Issue #89 [方向B Gate1] product score → assignment 坍缩分离诊断 (R22 + R19, GPU 1)

Product3ComponentHRQVAE 10 epoch + per-step per-component trace:
- per-component (learned-hyp / fixed-hyp / Euclidean) score quantiles
- per-component argmin (which codeword each component picks)
- component argmin agreement rate (do all 3 components pick the same codeword?)
- mixed score top-k margin
- mixing weights (3 components per layer)
- θ / κ / codebook pairwise dist / hard utilization / collision / grad norm
- 3 选 1 判定: component-level / mixing-level / codebook-loss collapse
"""
import os, sys, time, json, torch, numpy as np, pandas as pd
from pathlib import Path

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/scripts')
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
DEVICE = "cuda:0"

from task369_issue76_product_3component_evidence import Product3ComponentVectorQuantization

# === 0. config ===
TASK_NAME = "task382_issue89_product_score_assignment_separation_diagnostic"
SEED = 42
NUM_EMB_LIST = [64, 128, 256]
E_DIM = 32
M = 3
KAPPA_MAX = 2.0
FIXED_HYP_KAPPA = 1.0
EPOCHS = 10  # R11.5: 10 epoch 足够定位坍缩点
LOG_EVERY_STEPS = 50  # 每 50 step 输出一行 trace
BATCH = 64
LR = 1e-3
DATASET_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet"
OUT_DIR = Path(f"/home/wlia0047/ar57/wenyu/GeneRec/products/{TASK_NAME}")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TRACE_FILE = OUT_DIR / "trace_per_step.jsonl"
CKPT_PATH = OUT_DIR / "product_separation_stage1_ckpt.pt"

torch.manual_seed(SEED); np.random.seed(SEED)
print("=" * 70)
print(f"Task #382 / Issue #89 [方向B Gate1] product score → assignment 分离诊断")
print(f"Device: {DEVICE}, Epochs: {EPOCHS}, K: {NUM_EMB_LIST}, log_every_steps: {LOG_EVERY_STEPS}")
print("=" * 70)

# === 1. 数据 ===
df = pd.read_parquet(DATASET_PATH)
emb_col = 'embedding' if 'embedding' in df.columns else 'emb'
X = np.stack([np.asarray(e, dtype=np.float32) for e in df[emb_col]])
X = X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8) * 0.85
X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
print(f"[数据] shape={X_t.shape}")

# === 2. Product3ComponentHRQVAE wrapper (复用 task378 模式) ===
class Product3CompHRQVAE(torch.nn.Module):
    def __init__(self, in_dim, num_emb_list, e_dim, M, kappa_max, fixed_hyp_kappa, layers):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 512), torch.nn.GELU(),
            torch.nn.Linear(512, 256), torch.nn.GELU(),
            torch.nn.Linear(256, 128), torch.nn.GELU(),
            torch.nn.Linear(128, e_dim),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(e_dim, 128), torch.nn.GELU(),
            torch.nn.Linear(128, 256), torch.nn.GELU(),
            torch.nn.Linear(256, 512), torch.nn.GELU(),
            torch.nn.Linear(512, in_dim),
        )
        self.vq_layers = torch.nn.ModuleList([
            Product3ComponentVectorQuantization(
                n_e=K, e_dim=e_dim, M=M, kappa_max=kappa_max, fixed_hyp_kappa=fixed_hyp_kappa,
            ) for K in num_emb_list
        ])
    def forward(self, x, use_sk=True):
        z = self.encoder(x)
        all_q = []
        all_loss = []
        all_indices = []
        residual = z
        for layer in self.vq_layers:
            z_q, indices, loss = layer(residual)
            residual = residual - z_q
            all_q.append(z_q)
            all_loss.append(loss)
            all_indices.append(indices)
        x_q = sum(all_q)
        out = self.decoder(x_q)
        mean_loss = torch.stack(all_loss).mean()
        return out, mean_loss, torch.stack(all_indices, dim=-1)

model = Product3CompHRQVAE(
    in_dim=X.shape[1], num_emb_list=NUM_EMB_LIST, e_dim=E_DIM, M=M,
    kappa_max=KAPPA_MAX, fixed_hyp_kappa=FIXED_HYP_KAPPA,
    layers=[512, 256, 128],
).to(DEVICE)
print(f"[模型] {sum(p.numel() for p in model.parameters()):,} params")

# === 3. 训练 + per-step per-component trace ===
optim = torch.optim.Adam(model.parameters(), lr=LR)
N = X_t.shape[0]
steps_per_epoch = (N + BATCH - 1) // BATCH
print(f"[steps_per_epoch] {steps_per_epoch}, total_steps={steps_per_epoch * EPOCHS}")

# trace file 初始化
TRACE_FILE.unlink(missing_ok=True)

# 3 选 1 判定追踪
collapse_marker = {
    'component_level': [None, None, None],  # per-layer, first step where all 3 components agree on same codeword
    'mixing_level': [None, None, None],    # first step where mixed score collapses (top-1 margin ≈ 0)
    'codebook_loss': [None, None, None],   # first step where codebook pairwise dist shrinks
}

def compute_trace(model, X_t, step, ep):
    """per-step per-component trace for each layer"""
    model.eval()
    with torch.no_grad():
        z_e = model.encoder(X_t)
        traces = []
        for l_id, layer in enumerate(model.vq_layers):
            latent = z_e.view(-1, layer.e_dim)
            cb = layer.embeddings.weight  # (K, e_dim)
            # per-component scores
            kappas_lm = layer.kappa_per_component(l_id)  # (3,)
            dist_learned = layer._distance_to_codebook(latent, kappas_lm[0])  # (N, K)
            dist_fixed = layer._distance_to_codebook(latent, kappas_lm[1])  # (N, K)
            dist_euclidean = layer._distance_to_codebook(latent, kappas_lm[2])  # (N, K)
            # mixing weights (R22 修复: logits_l 是 (M, 3) shape, 取 [layer_idx] 得 (3,))
            weights_lm = torch.softmax(layer.logits_l[l_id], dim=-1)  # (3,)
            w_learned, w_fixed, w_eucl = weights_lm.cpu().numpy().tolist()
            wl = weights_lm[0].view(1, 1)
            wf = weights_lm[1].view(1, 1)
            we = weights_lm[2].view(1, 1)
            # mixed score
            dist_mixed = wl * dist_learned + wf * dist_fixed + we * dist_euclidean  # (N, K)
            # per-component argmin
            argmin_learned = torch.argmin(dist_learned, dim=-1)
            argmin_fixed = torch.argmin(dist_fixed, dim=-1)
            argmin_eucl = torch.argmin(dist_euclidean, dim=-1)
            argmin_mixed = torch.argmin(dist_mixed, dim=-1)
            # component argmin agreement rate (3 components agree on same codeword)
            agree_learned_fixed = (argmin_learned == argmin_fixed).float().mean().item()
            agree_learned_eucl = (argmin_learned == argmin_eucl).float().mean().item()
            agree_fixed_eucl = (argmin_fixed == argmin_eucl).float().mean().item()
            all_three_agree = ((argmin_learned == argmin_fixed) & (argmin_learned == argmin_eucl)).float().mean().item()
            # mixed score top-k margin
            d_top2 = torch.topk(dist_mixed, k=2, dim=-1, largest=False).values
            margin = (d_top2[:, 1] - d_top2[:, 0]).mean().item()
            # hard assignment (mixed)
            n_unique = len(set(argmin_mixed.cpu().numpy().tolist()))
            util = n_unique / cb.shape[0]
            counts = torch.bincount(argmin_mixed, minlength=cb.shape[0])
            max_load = counts.max().item() / N
            # per-component score quantiles
            score_learned_q50 = float(np.percentile(dist_learned.cpu().numpy(), 50))
            score_fixed_q50 = float(np.percentile(dist_fixed.cpu().numpy(), 50))
            score_eucl_q50 = float(np.percentile(dist_euclidean.cpu().numpy(), 50))
            # codebook pairwise distance
            pw = torch.cdist(cb, cb)
            pw_offdiag = pw[~torch.eye(pw.shape[0], dtype=torch.bool)].cpu().numpy()
            pw_q50 = float(np.percentile(pw_offdiag, 50))
            cb_norm = cb.norm(dim=-1).mean().item()
            traces.append({
                "layer_id": l_id,
                "score_learned_q50": score_learned_q50,
                "score_fixed_q50": score_fixed_q50,
                "score_eucl_q50": score_eucl_q50,
                "agree_learned_fixed": agree_learned_fixed,
                "agree_learned_eucl": agree_learned_eucl,
                "agree_fixed_eucl": agree_fixed_eucl,
                "all_three_agree": all_three_agree,
                "mixing_w_learned": w_learned,
                "mixing_w_fixed": w_fixed,
                "mixing_w_eucl": w_eucl,
                "mixed_top_k_margin": margin,
                "util": util,
                "max_load": max_load,
                "n_unique": n_unique,
                "pw_dist_q50": pw_q50,
                "cb_norm": cb_norm,
            })
            # 判定: component-level collapse
            if all_three_agree > 0.5 and collapse_marker['component_level'][l_id] is None:
                collapse_marker['component_level'][l_id] = step
            # 判定: mixing-level collapse (mixed top-k margin → 0)
            if margin < 0.01 and collapse_marker['mixing_level'][l_id] is None:
                collapse_marker['mixing_level'][l_id] = step
            # 判定: codebook-loss collapse (codebook pairwise dist → 0)
            if pw_q50 < 0.01 and collapse_marker['codebook_loss'][l_id] is None:
                collapse_marker['codebook_loss'][l_id] = step
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
        # 梯度范数
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
            utils_str = ' '.join(
                f"L{l_tr['layer_id']}=util{l_tr['util']*100:.1f}%agree3={l_tr['all_three_agree']*100:.1f}%w=[{l_tr['mixing_w_learned']:.2f},{l_tr['mixing_w_fixed']:.2f},{l_tr['mixing_w_eucl']:.2f}]"
                for l_tr in traces
            )
            print(f"  step {step}/{steps_per_epoch * EPOCHS} ep{ep} loss={loss.item():.4f} {utils_str} elapsed={elapsed:.0f}s")

# R12: ckpt save
if CKPT_PATH.exists():
    CKPT_PATH.unlink()
torch.save(model.state_dict(), CKPT_PATH)
print(f"\n[R12 ckpt] saved to {CKPT_PATH}")

# === 4. 总结 3 选 1 判定 ===
print(f"\n[3 选 1 判定]")
for collapse_type, l_steps in collapse_marker.items():
    print(f"  {collapse_type}: L0={l_steps[0]}, L1={l_steps[1]}, L2={l_steps[2]}")

# 找出最先发生的坍缩
earliest_collapse = {}
for collapse_type, l_steps in collapse_marker.items():
    valid_steps = [(s, l) for l, s in enumerate(l_steps) if s is not None]
    if valid_steps:
        earliest_step, earliest_layer = min(valid_steps)
        earliest_collapse[collapse_type] = (earliest_step, earliest_layer)

if earliest_collapse:
    sorted_collapses = sorted(earliest_collapse.items(), key=lambda x: x[1][0])
    print(f"\n  [最早坍缩]: {sorted_collapses[0][0]} @ step {sorted_collapses[0][1][0]} (layer {sorted_collapses[0][1][1]})")

# === 5. evidence ===
all_traces = []
with open(TRACE_FILE, 'r') as f:
    for line in f:
        all_traces.append(json.loads(line))

evidence = {
    "task_id": "task382",
    "issue": "#89",
    "epochs": EPOCHS,
    "total_steps": steps_per_epoch * EPOCHS,
    "trace_records": len(all_traces),
    "collapse_marker": collapse_marker,
    "earliest_collapse": earliest_collapse,
    "final_traces": all_traces[-M:],
    "verdict_path": "verdicts/task382_issue89_product_score_assignment_separation_diagnostic_v2.md",
}
with open(OUT_DIR / "evidence_package.json", "w") as f:
    json.dump(evidence, f, indent=2, default=str)
print(f"\n[证据] saved to {OUT_DIR / 'evidence_package.json'}")

# === 6. sanity ===
checks = [
    ("trace_records >= 30", len(all_traces) >= 30),
    ("L0 collapse marker found (any)", any(s is not None for s in collapse_marker['component_level']) or any(s is not None for s in collapse_marker['mixing_level']) or any(s is not None for s in collapse_marker['codebook_loss'])),
    ("mixing_weights_healthy_no_collapse", all(0.1 < t['mixing_w_learned'] < 0.9 and 0.1 < t['mixing_w_fixed'] < 0.9 and 0.1 < t['mixing_w_eucl'] < 0.9 for t in all_traces[-M:])),
    ("ckpt_saved", CKPT_PATH.exists()),
    ("no_nan_inf", all(
        np.isfinite(tr['loss']) and np.isfinite(tr['mixed_top_k_margin']) and np.isfinite(tr['util'])
        for tr in all_traces
    )),
    ("per_component_trace_present", all('score_learned_q50' in tr for tr in all_traces)),
]
n_pass = sum(1 for _, p in checks if p)
print(f"\n[sanity] PASS: {n_pass}/{len(checks)}")
for name, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

print(f"\n[Issue #89 Gate 1 决策]")
print(f"  Gate 1: {'✅ PASS' if n_pass == len(checks) else '❌ FAIL'}")
print(f"  trace_records = {len(all_traces)}")
print(f"  3 选 1 判定: {earliest_collapse}")