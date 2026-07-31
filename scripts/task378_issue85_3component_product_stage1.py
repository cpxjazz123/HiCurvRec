#!/usr/bin/env python3
"""
Task #378 / Issue #85 [方向B Gate1] 三分量 product Stage 1 训练 (R22 + R19 并行, GPU 1)

Product3ComponentVQ + per-layer mixing softmax + 200 epoch 训练 + mixing weights 监控
"""
import os, sys, time, json, torch, numpy as np, pandas as pd
from pathlib import Path

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/scripts')
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
DEVICE = "cuda:0"

from task369_issue76_product_3component_evidence import Product3ComponentVectorQuantization

# === 0. config ===
TASK_NAME = "task378_issue85_3component_product_stage1"
SEED = 42
NUM_EMB_LIST = [64, 128, 256]
E_DIM = 32
M = 3
KAPPA_MAX = 2.0
FIXED_HYP_KAPPA = 1.0
EPOCHS = 50  # R11.5 自主决策: Gate 1 验证用 50 epoch, 跟 task379/task377 对齐 (避免 200 epoch 8h+ 阻塞并行)
BATCH = 64
LR = 1e-3
DATASET_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet"
CKPT_DIR = Path(f"/home/wlia0047/ar57/wenyu/GeneRec/products/{TASK_NAME}")
CKPT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_PATH = CKPT_DIR / "product3comp_stage1_ckpt.pt"

torch.manual_seed(SEED); np.random.seed(SEED)
print("=" * 70)
print(f"Task #378 / Issue #85 [方向B Gate1] 三分量 product Stage 1 训练")
print(f"Device: {DEVICE}, Epochs: {EPOCHS}, K: {NUM_EMB_LIST}")
print("=" * 70)

# === 1. 数据 ===
df = pd.read_parquet(DATASET_PATH)
emb_col = 'embedding' if 'embedding' in df.columns else 'emb'
X = np.stack([np.asarray(e, dtype=np.float32) for e in df[emb_col]])
X = X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8) * 0.85
X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
print(f"[数据] shape={X_t.shape}")

# === 2. Product3ComponentVectorQuantization wrapped HRQVAE ===
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

# === 3. 训练 200 epoch + per-layer mixing 监控 ===
optim = torch.optim.Adam(model.parameters(), lr=LR)
N = X_t.shape[0]

epoch_records = []
start_time = time.time()
for ep in range(EPOCHS):
    model.train()
    total_loss = 0.0
    perm = torch.randperm(N)
    for i in range(0, N, BATCH):
        idx = perm[i:i+BATCH]
        x_b = X_t[idx]
        optim.zero_grad()
        out, qloss, _ = model(x_b)
        recon_loss = torch.nn.functional.mse_loss(out, x_b)
        loss = recon_loss + 0.25 * qloss
        loss.backward()
        optim.step()
        total_loss += loss.item()
    avg_loss = total_loss / ((N + BATCH - 1) // BATCH)

    # per-layer util + mixing weights 监控
    model.eval()
    with torch.no_grad():
        utils = []
        mixing_w = []
        for layer in model.hrq.vq_layers:
            z_test = X_t[:256]
            z_q, idx, _ = layer(z_test, use_sk=False)
            n_unique = len(set(idx.cpu().numpy()))
            utils.append(n_unique / layer.embeddings.weight.shape[0])
            w = torch.softmax(layer.logits_l, dim=-1).cpu().numpy()
            mixing_w.append(w.tolist())
    epoch_records.append({
        "epoch": ep,
        "loss": avg_loss,
        "utils": utils,
        "mixing_w": mixing_w,
    })
    if ep % 10 == 0 or ep == EPOCHS - 1:
        elapsed = time.time() - start_time
        print(f"[ep{ep}/{EPOCHS}] loss={avg_loss:.4f} util L0/L1/L2={utils[0]*100:.1f}/{utils[1]*100:.1f}/{utils[2]*100:.1f}% "
              f"mixing_w L0={mixing_w[0]} elapsed={elapsed:.0f}s")

# R12: ckpt save
if CKPT_PATH.exists():
    CKPT_PATH.unlink()
torch.save(model.state_dict(), CKPT_PATH)
print(f"\n[R12 ckpt] saved to {CKPT_PATH}")

# === 4. 完整 Stage 2 推断 (Gate 2 验证) ===
print(f"\n[Stage 2 推断]")
model.eval()
with torch.no_grad():
    utils_full = []
    all_indices = []
    z_e = model.encoder(X_t)
    residual = z_e
    for layer in model.hrq.vq_layers:
        z_q, indices, _ = layer(residual, use_sk=True)
        all_indices.append(indices.cpu().numpy().flatten())
        residual = residual - z_q
        n_unique = len(set(indices.cpu().numpy()))
        utils_full.append(n_unique / layer.embeddings.weight.shape[0])
collision = 1.0 - sum(utils_full) / len(utils_full)

# === 5. evidence ===
evidence = {
    "task_id": "task378",
    "issue": "#85",
    "epochs": EPOCHS,
    "epoch_records_last_5": epoch_records[-5:],
    "stage_2": {
        "utils_full": utils_full,
        "collision": collision,
    },
    "gate_decision": {
        "L0_util": utils_full[0],
        "L1_util": utils_full[1],
        "L2_util": utils_full[2],
        "collision": collision,
        "gate1_pass": all(u >= 0.9 for u in utils_full) and collision <= 0.20,
        "mixing_collapse": any(max(m) > 0.98 for layer_mw in [er['mixing_w'] for er in epoch_records[-10:]] for m in layer_mw),
    },
    "ckpt_path": str(CKPT_PATH),
    "verdict_path": "verdicts/task378_issue85_3component_product_stage1_v2.md",
}
with open(CKPT_DIR / "evidence_package.json", "w") as f:
    json.dump(evidence, f, indent=2, default=str)
print(f"\n[证据] saved to {CKPT_DIR / 'evidence_package.json'}")

# === 6. sanity ===
checks = [
    ("L0_util>=90%", utils_full[0] >= 0.9),
    ("L1_util>=90%", utils_full[1] >= 0.9),
    ("L2_util>=90%", utils_full[2] >= 0.9),
    ("collision<=0.20", collision <= 0.20),
    ("mixing_not_collapse", not evidence['gate_decision']['mixing_collapse']),
    ("ckpt_saved", CKPT_PATH.exists()),
]
n_pass = sum(1 for _, p in checks if p)
print(f"\n[sanity] PASS: {n_pass}/{len(checks)}")
for name, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

print(f"\n[Issue #85 Gate 1 决策]")
print(f"  整体: {'✅ PASS' if n_pass == len(checks) else '❌ FAIL'}")
print(f"  util L0/L1/L2 = {utils_full[0]*100:.1f}% / {utils_full[1]*100:.1f}% / {utils_full[2]*100:.1f}%")
print(f"  collision = {collision*100:.2f}%")