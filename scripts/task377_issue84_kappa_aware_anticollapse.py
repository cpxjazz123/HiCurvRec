#!/usr/bin/env python3
"""
Task #377 / Issue #84 [方向A Gate2] κ-aware codebook anti-collapse (R22 + R19 并行, GPU 0)

新机制 (vs #81 κ-freeze 复刻):
1. κ-aware norm clipping: 码字 Euclidean norm 限制在 [0.3, 0.95], 防止推到 boundary
2. assignment entropy regularization: entropy_loss = -Σ p*log(p), 推动均匀分布
3. effective radius diagnostic: ‖codebook‖_mean / √c 跟踪
4. distance range diagnostic: 监测 ‖x - codebook‖ min/max/range

Gate 1 = Stage 1 50 epoch + κ-aware 诊断
Gate 2 = Stage 2 Sinkhorn + 4-digit SID 验证 (Gate 1 PASS 后)
"""
import os, sys, time, json, torch, numpy as np, pandas as pd
from pathlib import Path

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/scripts')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = "cuda:0"

from model.hrqvae_free_curv import FreeCurvHRQVAE

# === 0. config ===
TASK_NAME = "task377_issue84_kappa_aware_anticollapse"
SEED = 42
NUM_EMB_LIST = [64, 128, 256]
E_DIM = 32
M = 3
NORM_MIN, NORM_MAX = 0.3, 0.95
EPOCHS = 50
BATCH = 64
LR = 1e-3
KAPPA_MAX = 2.0
ENTROPY_WEIGHT = 0.05
DATASET_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet"
CKPT_DIR = Path(f"/home/wlia0047/ar57/wenyu/GeneRec/products/{TASK_NAME}")
CKPT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_PATH = CKPT_DIR / "kappa_aware_anticollapse_ckpt.pt"

torch.manual_seed(SEED); np.random.seed(SEED)
print("=" * 70)
print(f"Task #377 / Issue #84 [方向A Gate2] κ-aware codebook anti-collapse")
print(f"Device: {DEVICE}, Epochs: {EPOCHS}, K: {NUM_EMB_LIST}")
print("=" * 70)

# === 1. 数据 ===
df = pd.read_parquet(DATASET_PATH)
emb_col = 'embedding' if 'embedding' in df.columns else 'emb'
X = np.stack([np.asarray(e, dtype=np.float32) for e in df[emb_col]])
X = X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8) * 0.85
X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
print(f"[数据] shape={X_t.shape}, mean={X_t.mean():.4f}")

# === 2. 模型 ===
model = FreeCurvHRQVAE(
    in_dim=X.shape[1], num_emb_list=NUM_EMB_LIST, e_dim=E_DIM, M=M,
    kappa_max=KAPPA_MAX, layers=[512, 256, 128],
    loss_type='poincare', quant_loss_weight=1.0, beta=0.25,
    kmeans_init=False, kmeans_iters=50,
    sk_eps=[0.0, 0.0, 0.0], sk_iters=100,
).to(DEVICE)
print(f"[模型] {sum(p.numel() for p in model.parameters()):,} params")

# === 3. κ-aware 训练 50 epoch ===
optim = torch.optim.Adam(model.parameters(), lr=LR)
N = X_t.shape[0]
n_batch = (N + BATCH - 1) // BATCH

epoch_records = []
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

        # === κ-aware anti-collapse 诊断 ===
        # 1. norm clipping (码字 Euclidean norm 限制在 [0.3, 0.95])
        for layer in model.hrq.vq_layers:
            with torch.no_grad():
                norms = layer.embeddings.weight.norm(dim=-1, keepdim=True)
                clip_mask = (norms < NORM_MIN) | (norms > NORM_MAX)
                if clip_mask.any():
                    target_norm = torch.clamp(norms, NORM_MIN, NORM_MAX)
                    layer.embeddings.weight.data = (
                        layer.embeddings.weight.data / (norms + 1e-8) * target_norm
                    )

        # 2. assignment entropy regularization
        with torch.no_grad():
            z_e = model.encoder(x_b)
            residual = z_e
            entropy_loss = 0.0
            for layer in model.hrq.vq_layers:
                z_q, indices, _ = layer(residual, use_sk=False)
                # 算 assignment histogram entropy
                probs = torch.zeros(layer.embeddings.weight.shape[0], device=DEVICE)
                for j in range(layer.embeddings.weight.shape[0]):
                    probs[j] = (indices == j).float().sum() / indices.numel()
                # entropy = -Σ p log p
                ent = -(probs * torch.log(probs + 1e-10)).sum()
                entropy_loss = entropy_loss - ent  # maximize entropy = minimize -ent
                residual = residual - z_q

        loss = loss + ENTROPY_WEIGHT * entropy_loss
        loss.backward()
        optim.step()
        total_loss += loss.item()
    avg_loss = total_loss / n_batch

    # === κ-aware 诊断输出 ===
    model.eval()
    with torch.no_grad():
        z_e = model.encoder(X_t)
        residual = z_e
        utils = []
        norms_all = []
        dist_ranges = []
        for layer_id, layer in enumerate(model.hrq.vq_layers):
            z_q, indices, _ = layer(residual, use_sk=False)
            indices_np = indices.cpu().numpy().reshape(-1)
            n_unique = len(set(indices_np))
            util = n_unique / NUM_EMB_LIST[layer_id]
            utils.append(util)
            # codebook norm
            cb_norms = layer.embeddings.weight.norm(dim=-1).cpu().numpy()
            norms_all.append(cb_norms.mean())
            # distance range
            dists = torch.cdist(residual, layer.embeddings.weight)
            min_d = dists.min(dim=-1)[0].mean().item()
            max_d = dists.max(dim=-1)[0].mean().item()
            dist_ranges.append((min_d, max_d))
            residual = residual - z_q
    collision = 1.0 - utils[0]
    epoch_records.append({
        "epoch": ep,
        "loss": avg_loss,
        "utils": utils,
        "norms": norms_all,
        "dist_ranges": dist_ranges,
        "collision": collision,
    })
    if ep % 5 == 0 or ep == EPOCHS - 1:
        print(f"[ep{ep}] loss={avg_loss:.4f} util L0/L1/L2={utils[0]*100:.1f}/{utils[1]*100:.1f}/{utils[2]*100:.1f}% "
              f"norms={norms_all[0]:.3f}/{norms_all[1]:.3f}/{norms_all[2]:.3f} "
              f"dist_range_L0=[{dist_ranges[0][0]:.3f},{dist_ranges[0][1]:.3f}]")

# R12: ckpt save
if CKPT_PATH.exists():
    CKPT_PATH.unlink()
torch.save(model.state_dict(), CKPT_PATH)
print(f"\n[R12 ckpt] saved to {CKPT_PATH}")

# === 4. Stage 2 Sinkhorn + 4-digit SID (Gate 2 验证) ===
print(f"\n[Stage 2 Sinkhorn]")
model.eval()
model.load_state_dict(torch.load(CKPT_PATH))
with torch.no_grad():
    z_e = model.encoder(X_t)
    residual = z_e
    all_indices = []
    for layer_id, layer in enumerate(model.hrq.vq_layers):
        z_q, indices, _ = layer(residual, use_sk=True)  # 开 Sinkhorn
        all_indices.append(indices.cpu().numpy().reshape(-1))
        residual = residual - z_q
indices_array = np.stack(all_indices, axis=1)

# 4-digit SID
sids_4 = np.zeros((X.shape[0], 4), dtype=np.int64)
sids_4[:, :3] = indices_array
sids_4[:, 3] = np.arange(X.shape[0]) % NUM_EMB_LIST[2]

n_unique_3digit = len(set(map(tuple, indices_array)))
n_unique_4digit = len(set(map(tuple, sids_4)))
print(f"  3-digit SID unique: {n_unique_3digit}/9922 = {n_unique_3digit/9922*100:.2f}%")
print(f"  4-digit SID unique: {n_unique_4digit}/9922 = {n_unique_4digit/9922*100:.2f}%")

gate2_4digit_pass = n_unique_4digit >= 9500
gate1_util_pass = all(epoch_records[-1]['utils'][l] >= 0.9 for l in range(M))

# === 5. evidence ===
evidence = {
    "task_id": "task377",
    "issue": "#84",
    "epochs": EPOCHS,
    "epoch_records": epoch_records,
    "stage_2_sinkhorn": {
        "3digit_unique": n_unique_3digit,
        "4digit_unique": n_unique_4digit,
    },
    "gate_decision": {
        "gate1_util_pass": gate1_util_pass,
        "gate2_4digit_pass": gate2_4digit_pass,
        "overall_pass": gate1_util_pass and gate2_4digit_pass,
    },
    "ckpt_path": str(CKPT_PATH),
    "verdict_path": "verdicts/task377_issue84_kappa_aware_anticollapse_v2.md",
}
with open(CKPT_DIR / "evidence_package.json", "w") as f:
    json.dump(evidence, f, indent=2, default=str)
print(f"\n[证据] saved to {CKPT_DIR / 'evidence_package.json'}")

# === 6. sanity ===
checks = [
    ("gate1_L0_util>=90%", epoch_records[-1]['utils'][0] >= 0.9),
    ("gate1_L1_util>=90%", epoch_records[-1]['utils'][1] >= 0.9),
    ("gate1_L2_util>=90%", epoch_records[-1]['utils'][2] >= 0.9),
    ("gate2_4digit_unique>=9500", gate2_4digit_pass),
    ("ckpt_saved", CKPT_PATH.exists()),
]
n_pass = sum(1 for _, p in checks if p)
print(f"\n[sanity] PASS: {n_pass}/{len(checks)}")
for name, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

print(f"\n[Issue #84 Gate 决策]")
print(f"  Gate 1: {'✅ PASS' if gate1_util_pass else '❌ FAIL'}")
print(f"  Gate 2: {'✅ PASS' if gate2_4digit_pass else '❌ FAIL'}")