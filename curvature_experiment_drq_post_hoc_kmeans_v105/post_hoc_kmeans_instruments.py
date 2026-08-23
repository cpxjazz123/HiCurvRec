"""v105 DRQ Stage 1b: post-hoc K-Means cascade on frozen Stage 1a continuous VAE latent.

论文支撑: arxiv 2606.01844 (Shopee 2026) — Decoupled Residual Quantization
DRQ Eq.(3)-(4): encoder 训练后冻结, post-hoc 在 latent (32-d) 上跑 K-Means cascade,
decoupled 训练阶段与量化阶段.

Stage 1b 流程:
1. 加载 Stage 1a frozen RqVae ckpt (含 encoder/decoder), DDP 单卡足够 (B=9922 latent)
2. encoder(x_all 9922,768) → z (9922,32) (latent space)
3. L0: KMeans(z, 256, random_state=42) → L0 codes (9922,)
4. res_1 = z - centroids_0[L0 codes] (9922,32)  (latent residual)
5. L1: KMeans(res_1, 256, random_state=42) → L1 codes (9922,)
6. res_2 = res_1 - centroids_1[L1 codes] (9922,32)
7. L2: KMeans(res_2, 256, random_state=42) → L2 codes (9922,)
8. stack → sids (9922, 3) → save raw_sids.npy
9. R41c: KMeans random_state=42 (固定初始中心, Stage 2 A/B 对照初始一致性硬约束)
10. R36i fake breakthrough 检测: SID unique count + per_layer_balance 验证
11. Stage 2 build_sids_for_hgrec.py 接 (9922, 3) 加 pad → (9922, 4) shape

合规:
- R36n: post-hoc K-Means 是 manifold geometry change
- R41c: KMeans random_state=42 固定初始中心 (Stage 2 A/B 对照一致)
- R36i: 必须验证 SID unique count > 9000 (非 fake breakthrough)
"""
import os
import sys
import numpy as np
import torch
from sklearn.cluster import KMeans

# R47 imports
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/Euclidean_Base_M2M3")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/curvature_experiment_drq_post_hoc_kmeans_v105")

from modules.rqvae import RqVae
from modules.quantize import QuantizeForwardMode
from curvature_config import ITEM_EMB_NPY as EMB_NPY, RQVAE_OUT_DIR as OUT_DIR, RAW_SIDS_NPY

# === 超参 (硬编码 R30/R43) ===
SEED = 42
INPUT_DIM = 768
HIDDEN_DIMS = [512, 256, 128]
EMBED_DIM = 32
CODEBOOK_SIZE = 256
N_LAYERS = 3
COMMITMENT_WEIGHT = 1.0  # Stage 1a 用, 这里 dummy
CKPT_PATH = os.path.join(OUT_DIR, "rqvae_final.pt")  # Stage 1a final ckpt

# K-Means 配置
N_CLUSTERS = 256
KMEANS_RANDOM_STATE = 42  # R41c 硬约束
KMEANS_MAX_ITER = 100
KMEANS_N_INIT = 10  # sklearn default, 平衡稳定性

# R36i fake breakthrough 检测阈值
SID_UNIQUE_THRESHOLD = 9000  # baseline SID unique ≈ 9208, 低于此视为 fake breakthrough
PER_LAYER_BALANCE_MIN = 0.05  # 每层最大 cluster 占比 ≤ 1/N_CLUSTERS * PER_LAYER_BALANCE_MIN

print(f"=== v105 DRQ Stage 1b: post-hoc K-Means cascade ===", flush=True)
print(f"Stage 1a ckpt: {CKPT_PATH}", flush=True)
print(f"Stage 0 embeddings: {EMB_NPY}", flush=True)
print(f"Output: {RAW_SIDS_NPY}", flush=True)
print(f"KMeans: n_clusters={N_CLUSTERS} random_state={KMEANS_RANDOM_STATE} n_init={KMEANS_N_INIT}", flush=True)

# === Stage 1: 加载 Stage 0 embeddings (9922, 768) ===
emb_all = np.load(EMB_NPY).astype(np.float32)  # (9922, 768)
n_items = emb_all.shape[0]
assert emb_all.shape[1] == INPUT_DIM, f"Expected dim {INPUT_DIM}, got {emb_all.shape[1]}"
print(f"[data] loaded {EMB_NPY}: shape={emb_all.shape}", flush=True)

# === Stage 2: 加载 Stage 1a frozen RqVae ckpt ===
device = torch.device("cuda:0")
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

model = RqVae(
    input_dim=INPUT_DIM,
    embed_dim=EMBED_DIM,
    hidden_dims=HIDDEN_DIMS,
    codebook_size=CODEBOOK_SIZE,
    codebook_kmeans_init=False,  # post-hoc 阶段不需要 codebook init
    codebook_normalize=False,
    codebook_sim_vq=False,
    codebook_mode=QuantizeForwardMode.STE,
    n_layers=N_LAYERS,
    n_cat_features=0,
    commitment_weight=COMMITMENT_WEIGHT,
    gate_M2_intrinsic=False,    # Stage 1a 不用 M2/M3
    gate_M3_transport=False,
    hyperbolic_distance=True,    # 与 Stage 1a 一致
    sk_eps=0.0,
    prefix_router_layers=None,
    margin_reg_weight=0.0,      # Stage 1a 不用 C5
    spread_loss_weight=0.0,     # Stage 1a 不用 F3
    use_tcu=False,
    use_mcdq=False,
    use_scs=False,
    use_fixed_curvature=True,
    c_fixed=1.0,
    use_curriculum_curvature=False,
    skip_quantization=True,     # v105: 跳过 Quantize layer
).to(device)

# 加载 Stage 1a final ckpt
print(f"[ckpt] loading {CKPT_PATH}", flush=True)
state = torch.load(CKPT_PATH, map_location=device, weights_only=False)
model.load_state_dict(state["model"], strict=False)
model.eval()
print(f"[ckpt] loaded (step={state.get('global_step', state.get('iter', 'unknown'))})", flush=True)

# === Stage 3: encoder(x_all) → z (9922, 32) ===
with torch.no_grad():
    x_t = torch.from_numpy(emb_all).to(device)  # (9922, 768)
    z = model.encode(x_t)  # (9922, 32) — encoder forward only (no Quantize)
    z_np = z.cpu().numpy().astype(np.float32)  # (9922, 32)
print(f"[encode] z shape={z_np.shape}, mean norm={np.linalg.norm(z_np, axis=1).mean():.4f}", flush=True)

# === Stage 4: K-Means cascade L0/L1/L2 on latent (32-d) ===
# R41c: 固定 random_state, Stage 2 A/B 对照初始一致
sids = np.zeros((n_items, N_LAYERS), dtype=np.int64)

# L0: KMeans on z
print(f"[kmeans L0] fitting KMeans(z, n_clusters={N_CLUSTERS})...", flush=True)
km0 = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=KMEANS_RANDOM_STATE,
    n_init=KMEANS_N_INIT,
    max_iter=KMEANS_MAX_ITER,
).fit(z_np)
sids[:, 0] = km0.labels_
l0_centroids = km0.cluster_centers_  # (256, 32)
print(f"[kmeans L0] inertia={km0.inertia_:.4f}, n_iter={km0.n_iter_}", flush=True)

# L1: KMeans on z - L0_centroids[L0_codes] (latent residual)
print(f"[kmeans L1] fitting KMeans(res_1, n_clusters={N_CLUSTERS})...", flush=True)
residual_1 = z_np - l0_centroids[sids[:, 0]]  # (9922, 32)
km1 = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=KMEANS_RANDOM_STATE,
    n_init=KMEANS_N_INIT,
    max_iter=KMEANS_MAX_ITER,
).fit(residual_1)
sids[:, 1] = km1.labels_
l1_centroids = km1.cluster_centers_  # (256, 32)
print(f"[kmeans L1] inertia={km1.inertia_:.4f}, n_iter={km1.n_iter_}", flush=True)

# L2: KMeans on res_1 - L1_centroids[L1_codes]
print(f"[kmeans L2] fitting KMeans(res_2, n_clusters={N_CLUSTERS})...", flush=True)
residual_2 = residual_1 - l1_centroids[sids[:, 1]]  # (9922, 32)
km2 = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=KMEANS_RANDOM_STATE,
    n_init=KMEANS_N_INIT,
    max_iter=KMEANS_MAX_ITER,
).fit(residual_2)
sids[:, 2] = km2.labels_
l2_centroids = km2.cluster_centers_  # (256, 32)
print(f"[kmeans L2] inertia={km2.inertia_:.4f}, n_iter={km2.n_iter_}", flush=True)

# === Stage 5: save raw_sids.npy (9922, 3) ===
os.makedirs(OUT_DIR, exist_ok=True)
np.save(RAW_SIDS_NPY, sids)
print(f"[save] {RAW_SIDS_NPY} shape={sids.shape} dtype={sids.dtype}", flush=True)

# === Stage 6: R36i fake breakthrough 检测 ===
unique_per_layer = [int(np.unique(sids[:, li]).size) for li in range(N_LAYERS)]
total_unique = len(set(map(tuple, sids.tolist())))
print(f"\n=== Stage 1b R36i validation ===", flush=True)
print(f"  total unique SIDs: {total_unique}/{n_items} ({100*total_unique/n_items:.2f}%)", flush=True)
print(f"  L0 unique: {unique_per_layer[0]}/{N_CLUSTERS}", flush=True)
print(f"  L1 unique: {unique_per_layer[1]}/{N_CLUSTERS}", flush=True)
print(f"  L2 unique: {unique_per_layer[2]}/{N_CLUSTERS}", flush=True)
for li in range(N_LAYERS):
    counts = np.bincount(sids[:, li], minlength=N_CLUSTERS)
    max_freq = counts.max() / n_items
    print(f"  L{li} max cluster freq: {max_freq:.4f} ({100*max_freq:.2f}%)", flush=True)

if total_unique < SID_UNIQUE_THRESHOLD:
    print(f"  [R36i WARNING] total_unique={total_unique} < threshold={SID_UNIQUE_THRESHOLD} → fake breakthrough!")
    print(f"  DRQ Stage 1b 失败: K-Means cascade 导致 SID vocab 退化")
    raise RuntimeError(f"R36i fake breakthrough detected: SID unique={total_unique}")
else:
    print(f"  [R36i PASS] total_unique={total_unique} ≥ {SID_UNIQUE_THRESHOLD} → 真非 fake breakthrough", flush=True)

print(f"\n=== v105 DRQ Stage 1b 完成 ===", flush=True)
print(f"Output: {RAW_SIDS_NPY}", flush=True)
print(f"Next: 调 build_sids_for_hgrec.py 加 pad → (9922, 4) shape for Stage 3", flush=True)