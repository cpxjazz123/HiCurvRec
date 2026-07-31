#!/usr/bin/env python3
"""
Task #388 / Issue #95 [方向B Gate1] product 分量 HypPreEncoder 距离公式验证 (R22 + R19, GPU 1)

Product3CompHRQVAEWithHypPre 50 epoch Stage 1 训练:
- 每个 component (learned-κ + fixed-hyp + Euclidean) 接入 HypPreEncoder BEFORE encoder
- 跟 task387 FreeCurvHRQVAEWithHypPre 同架构, 但 base 是 Product3CompHRQVAE
- 50 epoch + per-component trace
- 验证 step1 agree3 < 100%, final util ≥90%, mixing weights 健康

Issue #95 Gate1 PASS 条件:
  - step1 agree3 < 100% (跟 task385 #92 FAIL 的 step1 agree3=100% 对比)
  - final L0/L1/L2 util ≥ 90% (跟 task385 4.69%/0.78%/0.39% 对比)
  - mixing weights 非单点塌缩
"""
import os, sys, time, json, torch, numpy as np, pandas as pd
from pathlib import Path

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/scripts')
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
DEVICE = "cuda:0"

from task334_issue43_gate2a_hyp_pre_encoder import HypPreEncoder, expmap0
from task369_issue76_product_3component_evidence import Product3ComponentVectorQuantization

# === 0. config ===
TASK_NAME = "task388_issue95_product_hyp_pre_encoder_stage1"
SEED = 42
NUM_EMB_LIST = [64, 128, 256]
E_DIM = 32
M = 3
KAPPA_MAX = 2.0
FIXED_HYP_KAPPA = 1.0
EPOCHS = 50
LOG_EVERY_STEPS = 50
BATCH = 64
LR = 1e-3
DATASET_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet"
OUT_DIR = Path(f"/home/wlia0047/ar57/wenyu/GeneRec/products/{TASK_NAME}")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TRACE_FILE = OUT_DIR / "trace_per_step.jsonl"
CKPT_PATH = OUT_DIR / "product_hyp_pre_encoder_stage1_ckpt.pt"

torch.manual_seed(SEED); np.random.seed(SEED)
print("=" * 70)
print(f"Task #388 / Issue #95 [方向B Gate1] product HypPreEncoder Stage 1")
print(f"Device: {DEVICE}, Epochs: {EPOCHS}, K: {NUM_EMB_LIST}, HypPre c=0.74")
print("=" * 70)

# === 1. 数据 ===
df = pd.read_parquet(DATASET_PATH)
emb_col = 'embedding' if 'embedding' in df.columns else 'emb'
X = np.stack([np.asarray(e, dtype=np.float32) for e in df[emb_col]])
X = X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8) * 0.85
X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
print(f"[数据] shape={X_t.shape}")

# === 2. Product3CompHRQVAEWithHypPre wrapper ===
class Product3CompHRQVAEWithHypPre(torch.nn.Module):
    def __init__(self, in_dim, num_emb_list, e_dim, M, kappa_max, fixed_hyp_kappa, c=0.74):
        super().__init__()
        self.num_emb_list = num_emb_list
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
        # HypPreEncoder 接入 BEFORE encoder
        self.hyp_pre = HypPreEncoder(c=c, enabled=True)

    def init_codebook(self, X_data):
        """kmeans init codebook on HypPreEncoder + encoder output"""
        from model.utils import kmeans
        with torch.no_grad():
            z_hyp = self.hyp_pre(X_data)
            z_init = self.encoder(z_hyp)
            for K, layer in zip(self.num_emb_list, self.vq_layers):
                centers = kmeans(z_init, K, 10)
                layer.embeddings.weight.data.copy_(centers.to(DEVICE))

    def forward(self, x):
        z_hyp = self.hyp_pre(x)
        z = self.encoder(z_hyp)
        all_q, all_loss, all_indices = [], [], []
        residual = z
        for layer in self.vq_layers:
            z_q, indices, loss = layer(residual)
            residual = residual - z_q
            all_q.append(z_q); all_loss.append(loss); all_indices.append(indices)
        x_q = sum(all_q)
        out = self.decoder(x_q)
        return out, torch.stack(all_loss).mean(), torch.stack(all_indices, dim=-1)

model = Product3CompHRQVAEWithHypPre(
    in_dim=X.shape[1], num_emb_list=NUM_EMB_LIST, e_dim=E_DIM, M=M,
    kappa_max=KAPPA_MAX, fixed_hyp_kappa=FIXED_HYP_KAPPA, c=0.74,
).to(DEVICE)
model.init_codebook(X_t)
print(f"[模型] {sum(p.numel() for p in model.parameters()):,} params (Product3CompHRQVAE + HypPreEncoder)")

# === 3. 训练 + per-component trace ===
optim = torch.optim.Adam(model.parameters(), lr=LR)
N = X_t.shape[0]
steps_per_epoch = (N + BATCH - 1) // BATCH
print(f"[steps_per_epoch] {steps_per_epoch}, total_steps={steps_per_epoch * EPOCHS}")

TRACE_FILE.unlink(missing_ok=True)
collapse_step_component = [None, None, None]
collapse_step_mixing = [None, None, None]

def compute_trace(model, X_t, step):
    model.eval()
    with torch.no_grad():
        z_hyp = model.hyp_pre(X_t)
        z_e = model.encoder(z_hyp)
        traces = []
        for l_id, layer in enumerate(model.vq_layers):
            latent = z_e.view(-1, layer.e_dim)
            cb = layer.embeddings.weight
            kappas_lm = layer.kappa_per_component(l_id)
            dist_learned = layer._distance_to_codebook(latent, kappas_lm[0])
            dist_fixed = layer._distance_to_codebook(latent, kappas_lm[1])
            dist_euclidean = layer._distance_to_codebook(latent, kappas_lm[2])
            weights_lm = torch.softmax(layer.logits_l[l_id], dim=-1)
            wl = weights_lm[0].view(1, 1); wf = weights_lm[1].view(1, 1); we = weights_lm[2].view(1, 1)
            dist_mixed = wl * dist_learned + wf * dist_fixed + we * dist_euclidean
            argmin_l = torch.argmin(dist_learned, dim=-1)
            argmin_f = torch.argmin(dist_fixed, dim=-1)
            argmin_e = torch.argmin(dist_euclidean, dim=-1)
            argmin_mixed = torch.argmin(dist_mixed, dim=-1)
            all_three_agree = ((argmin_l == argmin_f) & (argmin_l == argmin_e)).float().mean().item()
            d_top2 = torch.topk(dist_mixed, k=2, dim=-1, largest=False).values
            margin = (d_top2[:, 1] - d_top2[:, 0]).mean().item()
            n_unique = len(set(argmin_mixed.cpu().numpy().tolist()))
            util = n_unique / cb.shape[0]
            traces.append({
                "layer_id": l_id,
                "all_three_agree": all_three_agree,
                "mixed_top_k_margin": margin,
                "mixing_w_learned": float(weights_lm[0].item()),
                "mixing_w_fixed": float(weights_lm[1].item()),
                "mixing_w_eucl": float(weights_lm[2].item()),
                "util": util,
                "n_unique": n_unique,
            })
            if all_three_agree > 0.99 and collapse_step_component[l_id] is None:
                collapse_step_component[l_id] = step
            if margin < 0.01 and collapse_step_mixing[l_id] is None:
                collapse_step_mixing[l_id] = step
    model.train()
    return traces

start_time = time.time()
for ep in range(EPOCHS):
    print(f"\n[ep {ep}/{EPOCHS}]")
    model.train()
    perm = torch.randperm(N)
    for i in range(0, N, BATCH):
        idx = perm[i:i+BATCH]
        x_b = X_t[idx]
        optim.zero_grad()
        out, qloss, _ = model(x_b)
        recon_loss = torch.nn.functional.mse_loss(out, x_b)
        loss = recon_loss + qloss
        loss.backward()
        optim.step()
        step = ep * steps_per_epoch + (i // BATCH) + 1
        if step % LOG_EVERY_STEPS == 0 or step == 1:
            traces = compute_trace(model, X_t, step)
            for l_tr in traces:
                l_tr['step'] = step
                l_tr['epoch'] = ep
                l_tr['loss'] = loss.item()
                with open(TRACE_FILE, 'a') as f:
                    f.write(json.dumps(l_tr, default=str) + '\n')
            elapsed = time.time() - start_time
            utils_str = ' '.join(f"L{l_tr['layer_id']}=util{l_tr['util']*100:.1f}%agree3{l_tr['all_three_agree']*100:.1f}%w=[{l_tr['mixing_w_learned']:.2f},{l_tr['mixing_w_fixed']:.2f},{l_tr['mixing_w_eucl']:.2f}]" for l_tr in traces)
            print(f"  step {step}/{steps_per_epoch * EPOCHS} ep{ep} loss={loss.item():.4f} {utils_str} elapsed={elapsed:.0f}s")

# R12: ckpt save
if CKPT_PATH.exists():
    CKPT_PATH.unlink()
torch.save(model.state_dict(), CKPT_PATH)
print(f"\n[R12 ckpt] saved to {CKPT_PATH}")

# === 4. 总结 + final eval ===
print(f"\n[坍缩点定位]")
for l_id in range(M):
    print(f"  L{l_id}: component_collapse step = {collapse_step_component[l_id]}, mixing_collapse step = {collapse_step_mixing[l_id]}")

model.eval()
with torch.no_grad():
    final_utils = []
    z_hyp = model.hyp_pre(X_t)
    z_e = model.encoder(z_hyp)
    for layer_id, layer in enumerate(model.vq_layers):
        latent = z_e.view(-1, layer.e_dim)
        kappas_lm = layer.kappa_per_component(layer_id)
        weights_lm = torch.softmax(layer.logits_l[layer_id], dim=-1)
        wl = weights_lm[0].view(1, 1); wf = weights_lm[1].view(1, 1); we = weights_lm[2].view(1, 1)
        dist_l = layer._distance_to_codebook(latent, kappas_lm[0])
        dist_f = layer._distance_to_codebook(latent, kappas_lm[1])
        dist_e = layer._distance_to_codebook(latent, kappas_lm[2])
        dist_mixed = wl * dist_l + wf * dist_f + we * dist_e
        idx = torch.argmin(dist_mixed, dim=-1)
        n_unique = len(set(idx.cpu().numpy().tolist()))
        final_utils.append(n_unique / layer.embeddings.weight.shape[0])

# === 5. evidence ===
all_traces = []
with open(TRACE_FILE, 'r') as f:
    for line in f:
        all_traces.append(json.loads(line))

evidence = {
    "task_id": "task388",
    "issue": "#95",
    "epochs": EPOCHS,
    "fix_applied": "Product3CompHRQVAE + HypPreEncoder + κ-Stereographic",
    "collapse_step_component": collapse_step_component,
    "collapse_step_mixing": collapse_step_mixing,
    "final_util_per_layer": final_utils,
    "trace_records": len(all_traces),
    "verdict_path": "verdicts/task388_issue95_product_hyp_pre_encoder_stage1_v2.md",
}
with open(OUT_DIR / "evidence_package.json", "w") as f:
    json.dump(evidence, f, indent=2, default=str)
print(f"\n[证据] saved to {OUT_DIR / 'evidence_package.json'}")

# === 6. Issue #95 Gate 1 PASS 判定 ===
step1_traces = [t for t in all_traces if t['step'] == 1]
agree3_step1 = max(t['all_three_agree'] for t in step1_traces) if step1_traces else 0.0
util_all_pass = all(u >= 0.9 for u in final_utils)
mixing_healthy = all(0.05 < t['mixing_w_learned'] < 0.95 and 0.05 < t['mixing_w_fixed'] < 0.95 and 0.05 < t['mixing_w_eucl'] < 0.95 for t in all_traces[-M:])

checks = [
    ("step1 agree3 < 100% (vs #92 FAIL 100%)", agree3_step1 < 1.0),
    ("mixing_weights_healthy", mixing_healthy),
    ("L0 final util >= 90% (vs #92 FAIL 4.69%)", final_utils[0] >= 0.9),
    ("L1 final util >= 90% (vs #92 FAIL 0.78%)", final_utils[1] >= 0.9),
    ("L2 final util >= 90% (vs #92 FAIL 0.39%)", final_utils[2] >= 0.9),
    ("ckpt_saved", CKPT_PATH.exists()),
    ("no_nan_inf", all(np.isfinite(t['loss']) and np.isfinite(t['util']) for t in all_traces)),
    ("trace_records >= 100", len(all_traces) >= 100),
]
n_pass = sum(1 for _, p in checks if p)
print(f"\n[sanity] PASS: {n_pass}/{len(checks)}")
for name, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

print(f"\n[Issue #95 Gate 1 决策]")
print(f"  Gate 1: {'✅ PASS' if n_pass == len(checks) else '❌ FAIL'}")
print(f"  step1 agree3 = {agree3_step1*100:.2f}% (目标 <100%, 跟 #92 100% 对比)")
print(f"  final_util L0/L1/L2 = {final_utils[0]*100:.2f}% / {final_utils[1]*100:.2f}% / {final_utils[2]*100:.2f}%")