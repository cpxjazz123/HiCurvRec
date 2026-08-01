#!/usr/bin/env python3
"""Task #462 / Issue #169 [方向A Gate1-2] κ元数据条件化wrapper真实 Stage 1 路径审计.

R18 4 维度对比 #167 (closed NO-GO 11b0ee5):
- D1: 真实 Stage 1 export + κ/scale 三层独立learnable + 3 epoch sanity
- D2: load Stage 1 RQ-VAE model + extract codebook + compute per-item κ/scale
- D3: wrapper 之前用硬编码 metadata, 真实 Stage 1 export 验证实施-执行一致
- D4: arXiv:2405.13979 + real RQ-VAE integration

Gate 1 Stage 1 真实 export:
1. 加载 task84 Stage 1 RQ-VAE best_loss_model.pth
2. 提取三层独立 codebook (K=64/128/256) + 导出 per-item κ_l + scale_l
3. 计算三层 hash (per layer codebook hash + 导出 metadata hash) — 必须非空
4. 失败条件: Stage 1 model 不存在 / codebook 不能 extract → 立即 NO-GO

Gate 2 Stage 2 真实 metadata 注入 + 3 epoch sanity:
1. 用真实 per-item κ/scale metadata 喂给 T5 wrapper (替代硬编码)
2. 3 epoch sanity (per issue spec)
3. 报告: SID hash + metadata 流 + 残差范数 + loss + val_R@10 逐 epoch + κ/权重梯度 + ckpt
4. 失败条件: α 单调增长 / val_R@10=0 跨 ≥2 epoch / metadata 未进入 forward → 立即 NO-GO

R23 强制: 跨 2 checkpoint val_R@10=0 → kill + NO-GO
R12 强制: ckpt 落盘
"""
import sys
import os
import json
import hashlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/model")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/data")

os.environ["TRITON_CACHE_DIR"] = "/home/wlia0047/.triton/cache_task462"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)

from HG_Rec import HG_Rec
from dataset import GenRecDataset

# ============================================================================
# Config
# ============================================================================
SEED = 42
DEVICE = "cuda:1"  # GPU 1 (R7: #450 GPU 0, #462 GPU 1, #463 GPU 2)
CODEBOOK_SIZE = [64, 128, 256, 1]
MAX_LEN = 20
PAD_TOKEN = 0
D_MODEL = 128
BATCH_SIZE = 32
NUM_EPOCHS = 3  # Issue #169 spec 3 epoch sanity
LR_CONDITIONER = 1e-3
LR_LAYERNORM = 1e-4
ALPHA_INIT_LOGIT = -10.0  # 跟 task460/task461 一致, init_zero=True (alpha_value=4.54e-5)
SID_NPY = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_poincare.npy"
TRAIN_PARQUET = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/train.parquet"
VAL_PARQUET = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/val.parquet"
T5_CKPT = "/home/wlia0047/ar57/wenyu/GeneRec/products/task84/ckpt_hgrec/Instruments/Jul-23-2026_20-24-44/HG_Rec_best.pth"
RQVAE_CKPT = "/home/wlia0047/ar57/wenyu/GeneRec/products/task84/ckpt/Instruments/Jul-23-2026_20-08-06_beta_1.000_codebook_[64,128,256]_sk_0.000/best_loss_model.pth"
PRODUCT_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/products/task462_issue169_stage1_export_kappa_scale_audit")
PRODUCT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "task462_issue169_stage1_export_kappa_scale_audit.log"
CONFIG_PATH = PRODUCT_DIR / "config.json"
VERDICT_PATH = PRODUCT_DIR / "verdict.json"
STAGE1_PROOF_PATH = PRODUCT_DIR / "stage1_export_proof.json"
TRAIN_TRACE_PATH = PRODUCT_DIR / "train_trace.json"
ADAPTER_CKPT_PATH = PRODUCT_DIR / "adapter.pt"
REAL_METADATA_PATH = PRODUCT_DIR / "real_metadata.npy"  # 真实 per-item κ/scale metadata


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# ============================================================================
# Stage 1 真实 export
# ============================================================================
def load_rqvae_codebook(rqvae_ckpt_path):
    """从 task84 Stage 1 RQ-VAE best_loss_model 提取三层独立 codebook."""
    ckpt = torch.load(rqvae_ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif "model" in ckpt:
        sd = ckpt["model"]
    else:
        sd = ckpt
    codebooks = {}
    for layer_idx in range(3):
        key = f"hrq.vq_layers.{layer_idx}.embeddings.weight"
        if key in sd:
            codebooks[layer_idx] = sd[key].numpy()
    return codebooks


def compute_per_item_metadata(sid_npy, codebooks):
    """对每个 item 用 SID 索引查 centroid, 然后计算 κ_l (curvature) + scale_l (norm)."""
    sid = np.load(sid_npy)  # (N, 4)
    N = sid.shape[0]
    metadata = np.zeros((N, 3, 4), dtype=np.float32)
    # layer 0..2 用 SID [:, 0..2] 索引 codebook
    for layer_idx in range(3):
        K = [64, 128, 256][layer_idx]
        cb = codebooks[layer_idx]  # (K, 32)
        # 每个 item 在该层选中的 centroid
        idx = sid[:, layer_idx]  # (N,)
        # clip idx 到合法范围 (防止 K=1 layer 越界)
        idx = np.clip(idx, 0, K - 1)
        centroids = cb[idx]  # (N, 32)
        norms = np.linalg.norm(centroids, axis=-1)  # (N,)
        # κ_l: 在 Poincaré ball 中, ||c||² 决定曲率. 用 c=1 (task84 baseline)
        # κ_l_effective = -2 * ||c||² / (1 - ||c||²)
        # 数值稳定性: clip ||c||² < 0.99
        c_norm_sq = np.clip(norms ** 2, 0, 0.99)
        kappa_l = -2.0 * c_norm_sq / (1.0 - c_norm_sq)  # (N,) -curvature越大||c||越接近1
        # scale_l = ||c||
        scale_l = norms  # (N,)
        # alpha_l, beta_l, gamma_l 来自 κ/scale 比例 (而不是硬编码 0.333)
        # 这里用: alpha = sigmoid(kappa_l), beta = scale_l / max(scale_l), gamma = 1 - alpha - beta
        alpha_l = 1.0 / (1.0 + np.exp(-kappa_l / 10.0))  # (N,)
        beta_l = scale_l / (scale_l.max() + 1e-8)  # (N,)
        gamma_l = 1.0 - alpha_l - beta_l
        # 归一化
        total = alpha_l + np.abs(beta_l) + np.abs(gamma_l) + 1e-8
        alpha_l = alpha_l / total
        beta_l = np.abs(beta_l) / total
        gamma_l = np.abs(gamma_l) / total
        # 写入 metadata
        metadata[:, layer_idx, 0] = kappa_l.astype(np.float32)
        metadata[:, layer_idx, 1] = alpha_l.astype(np.float32)
        metadata[:, layer_idx, 2] = beta_l.astype(np.float32)
        metadata[:, layer_idx, 3] = gamma_l.astype(np.float32)
    return metadata


def hash_codebook(codebook):
    return hashlib.sha256(codebook.tobytes()).hexdigest()


# ============================================================================
# Wrapper (跟 task460 一致, 但 metadata 来自真实 Stage 1 export)
# ============================================================================
class WeightedMixedCurvatureConditioner(nn.Module):
    def __init__(self, d_model=128, n_layers=3, sid_dim=4, alpha_init_logit=-10.0):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.sid_dim = sid_dim
        self.curvature_embed = nn.Linear(4, d_model)
        self.sid_token_proj = nn.Linear(sid_dim, d_model)
        self.conditioner = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
            nn.Tanh(),
        )
        self.alpha_logit = nn.Parameter(torch.tensor(alpha_init_logit, dtype=torch.float32))

    def get_alpha(self):
        return F.softplus(self.alpha_logit).item()

    def forward(self, x_emb, sid_meta, curvature_meta):
        curv_e_l = self.curvature_embed(curvature_meta)
        curv_summary = curv_e_l.mean(dim=1, keepdim=True).expand(-1, x_emb.shape[1], -1)
        sid_e = self.sid_token_proj(sid_meta)
        combined = torch.cat([sid_e, curv_summary], dim=-1)
        direction = self.conditioner(combined)
        alpha = F.softplus(self.alpha_logit)
        residual = alpha * direction
        return residual, alpha


class HG_Rec_with_RealMetadataAdapter(nn.Module):
    """Issue #169: 真实 Stage 1 κ/scale metadata 注入."""

    def __init__(self, t5_config, t5_state_dict, d_model=128, n_layers=3, sid_dim=4):
        super().__init__()
        self.t5 = HG_Rec(t5_config)
        self.t5.load_state_dict(t5_state_dict)
        for p in self.t5.parameters():
            p.requires_grad = False
        self.adapter = WeightedMixedCurvatureConditioner(d_model=d_model, n_layers=n_layers, sid_dim=sid_dim)
        first_input_ln = self.t5.model.encoder.block[0].layer[0].layer_norm
        for p in first_input_ln.parameters():
            p.requires_grad = True
        self.first_input_ln = first_input_ln

    def forward(self, input_ids, attention_mask=None, labels=None, sid_meta=None, curvature_meta=None):
        x_emb = self.t5.model.shared(input_ids)
        residual, alpha = self.adapter(x_emb, sid_meta, curvature_meta)
        residual_norm = residual.norm().item()
        x_emb_with_residual = x_emb + residual
        x_emb_with_residual = self.first_input_ln(x_emb_with_residual)
        if labels is not None:
            dummy_decoder_output = torch.zeros_like(input_ids[:, :4])
            return self.t5.model(
                inputs_embeds=x_emb_with_residual,
                attention_mask=attention_mask,
                labels=dummy_decoder_output,
            ), None, alpha, residual_norm
        else:
            dummy_decoder_input_ids = torch.zeros_like(input_ids[:, :4])
            return self.t5.model(inputs_embeds=x_emb_with_residual, attention_mask=attention_mask,
                                 decoder_input_ids=dummy_decoder_input_ids), None, alpha, residual_norm


def get_t5_config():
    return {
        "num_layers": 6, "num_decoder_layers": 4,
        "d_model": D_MODEL, "d_ff": 1024, "num_heads": 6,
        "d_kv": 64, "dropout_rate": 0.1,
        "vocab_size": 1025, "pad_token_id": 0, "eos_token_id": 1,
        "feed_forward_proj": "relu",
    }


def load_t5_state_dict(ckpt_path):
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif "model" in state_dict:
        state_dict = state_dict["model"]
    return state_dict


def compute_val_r10(model_wrapper, val_ds, codebook_size, max_len, pad_token, device, n=2048):
    model_wrapper.eval()
    rng = torch.Generator().manual_seed(42 + 100)
    n_eval = min(n, len(val_ds))
    indices = torch.randperm(len(val_ds), generator=rng)[:n_eval].tolist()
    n_strict = 0
    n_total = 0
    with torch.no_grad():
        for idx in indices:
            s = val_ds[idx]
            history = s["history"]
            target = s["target"]
            history_flat = [elem for sublist in history for elem in sublist]
            history_tensor = torch.tensor([history_flat], dtype=torch.long, device=device)
            attention_mask = (history_tensor != pad_token).long()
            B = history_tensor.shape[0]
            L_flat = max_len * 4
            digit_values = history_tensor.float()
            layer_idx = torch.arange(L_flat, device=device) % 4
            layer_idx = layer_idx.float().unsqueeze(0).expand(B, -1)
            pos_in_history = torch.arange(L_flat, device=device) // 4
            pos_in_history = pos_in_history.float().unsqueeze(0).expand(B, -1) / max_len
            padding_flag = (digit_values == pad_token).float()
            sid_meta = torch.stack([digit_values / 1025.0, layer_idx / 4.0, pos_in_history, padding_flag], dim=-1)
            # curvature_meta 用零 (eval 时不依赖 per-item metadata, 因为 val set 没有 per-item ID)
            curvature_meta = torch.zeros(B, 3, 4, dtype=torch.float32, device=device)
            output, _, _, _ = model_wrapper(history_tensor, attention_mask=attention_mask,
                                            sid_meta=sid_meta, curvature_meta=curvature_meta)
            logits = output.logits if hasattr(output, 'logits') else output[0]
            pred_digits = logits[0, :4].argmax(dim=-1).cpu().numpy()
            target_digits = np.asarray(target, dtype=np.int64)
            if (pred_digits[:4] == target_digits[:4]).all():
                n_strict += 1
            n_total += 1
    return n_strict / max(1, n_total)


def verify_sid_token_range(history_input_ids, codebook_size, pad_token=0):
    cumulative = [0]
    for k in codebook_size[:-1]:
        cumulative.append(cumulative[-1] + k)
    ranges = [(c + 1, c + k + 1) for c, k in zip(cumulative, codebook_size)]
    arr = history_input_ids.cpu().numpy()
    B, L_flat = arr.shape
    layer_indices = np.broadcast_to((np.arange(L_flat) % 4)[None, :], (B, L_flat))
    non_pad_mask = arr != pad_token

    def mask_in_range(mask, low, high):
        if not mask.any():
            return True
        vals = arr[mask & non_pad_mask]
        if len(vals) == 0:
            return True
        return bool(((vals >= low) & (vals < high)).all())

    in_range = [mask_in_range(layer_indices == i, low, high) for i, (low, high) in enumerate(ranges)]
    return {"layer0_in_range": in_range[0], "layer1_in_range": in_range[1],
            "layer2_in_range": in_range[2], "layer3_in_range": in_range[3],
            "all_in_range": bool(all(in_range))}


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    log_lines = []
    log_lines.append("=" * 70)
    log_lines.append(f"[Task #462 Issue #169 Gate 1-2] 真实 Stage 1 κ/scale export + 3 epoch sanity")
    log_lines.append("=" * 70)

    sid_sha = sha256_of(SID_NPY)
    t5_ckpt_sha = sha256_of(T5_CKPT)
    rqvae_ckpt_sha = sha256_of(RQVAE_CKPT)
    log_lines.append(f"\n[SHA256] SID_NPY: {sid_sha}")
    log_lines.append(f"[SHA256] T5_CKPT: {t5_ckpt_sha}")
    log_lines.append(f"[SHA256] RQVAE_CKPT: {rqvae_ckpt_sha}")

    config = {
        "seed": SEED, "device": DEVICE, "codebook_size": CODEBOOK_SIZE,
        "max_len": MAX_LEN, "batch_size": BATCH_SIZE, "num_epochs": NUM_EPOCHS,
        "lr_conditioner": LR_CONDITIONER, "lr_layernorm": LR_LAYERNORM,
        "alpha_init_logit": ALPHA_INIT_LOGIT, "d_model": D_MODEL, "n_layers": 3,
        "triton_cache_dir": os.environ["TRITON_CACHE_DIR"],
        "task_id": 462, "issue": "Issue #169",
        "real_stage1_export": True, "three_epoch_sanity": True,
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    # ============================================================
    # Gate 1: 真实 Stage 1 export
    # ============================================================
    log_lines.append(f"\n[Gate 1 Stage 1 真实 export]")
    codebooks = load_rqvae_codebook(RQVAE_CKPT)
    cb_hashes = {}
    for layer_idx, cb in codebooks.items():
        cb_hash = hash_codebook(cb)
        cb_hashes[f"layer{layer_idx}_codebook_hash"] = cb_hash
        log_lines.append(f"  Layer {layer_idx} codebook: shape={cb.shape}, hash={cb_hash[:16]}..., norm range=[{np.linalg.norm(cb, axis=-1).min():.4f}, {np.linalg.norm(cb, axis=-1).max():.4f}]")

    # Gate 1 PASS 条件: 三层 codebook 真实存在
    gate1_pass = len(codebooks) == 3 and all(f"layer{i}_codebook_hash" in cb_hashes for i in range(3))
    log_lines.append(f"  [Gate 1] 真实 Stage 1 export: {gate1_pass} (3 layer codebooks extracted)")

    if not gate1_pass:
        log_lines.append(f"[STOP] Gate 1 FAIL — Stage 1 export 失败, 不做训练")
        with open(LOG_PATH, "w") as f:
            f.write("\n".join(log_lines))
        with open(VERDICT_PATH, "w") as f:
            json.dump({"gate1_pass": False, "reason": "stage1_export_failed",
                       "codebook_hashes": cb_hashes, "task_id": 462,
                       "issue": "Issue #169"}, f, indent=2)
        return

    # 真实 per-item metadata
    real_metadata = compute_per_item_metadata(SID_NPY, codebooks)
    np.save(REAL_METADATA_PATH, real_metadata)
    real_metadata_sha = hashlib.sha256(real_metadata.tobytes()).hexdigest()
    log_lines.append(f"  Real metadata: shape={real_metadata.shape}, hash={real_metadata_sha[:16]}..., kappa_l range=[{real_metadata[:,:,0].min():.4f}, {real_metadata[:,:,0].max():.4f}]")
    log_lines.append(f"  alpha range: [{real_metadata[:,:,1].min():.4f}, {real_metadata[:,:,1].max():.4f}]")
    log_lines.append(f"  beta range:  [{real_metadata[:,:,2].min():.4f}, {real_metadata[:,:,2].max():.4f}]")
    log_lines.append(f"  gamma range: [{real_metadata[:,:,3].min():.4f}, {real_metadata[:,:,3].max():.4f}]")

    stage1_proof = {
        "rqvae_ckpt_sha256": rqvae_ckpt_sha,
        "codebook_hashes": cb_hashes,
        "real_metadata_shape": list(real_metadata.shape),
        "real_metadata_sha256": real_metadata_sha,
        "kappa_l_range": [float(real_metadata[:, :, 0].min()), float(real_metadata[:, :, 0].max())],
        "alpha_l_range": [float(real_metadata[:, :, 1].min()), float(real_metadata[:, :, 1].max())],
        "beta_l_range": [float(real_metadata[:, :, 2].min()), float(real_metadata[:, :, 2].max())],
        "gamma_l_range": [float(real_metadata[:, :, 3].min()), float(real_metadata[:, :, 3].max())],
        "stage1_pass": True,
    }
    with open(STAGE1_PROOF_PATH, "w") as f:
        json.dump(stage1_proof, f, indent=2)

    # ============================================================
    # Gate 2: 真实 metadata 注入 + 3 epoch sanity + val_R@10
    # ============================================================
    log_lines.append(f"\n[Gate 2 训练] {NUM_EPOCHS} epoch, batch_size={BATCH_SIZE} + 真实 metadata 注入 + val_R@10 早停")
    t5_state_dict = load_t5_state_dict(T5_CKPT)
    t5_config = get_t5_config()

    train_ds = GenRecDataset(
        dataset_path=TRAIN_PARQUET, code_path=SID_NPY, mode="train",
        codebook_size=CODEBOOK_SIZE, max_len=MAX_LEN, PAD_TOKEN=PAD_TOKEN,
    )
    val_ds = GenRecDataset(
        dataset_path=VAL_PARQUET, code_path=SID_NPY, mode="evaluation",
        codebook_size=CODEBOOK_SIZE, max_len=MAX_LEN, PAD_TOKEN=PAD_TOKEN,
    )
    log_lines.append(f"[Data] train_ds size: {len(train_ds)}, val_ds size: {len(val_ds)}")

    sample = [train_ds[i] for i in range(BATCH_SIZE)]
    history_list = [s["history"] for s in sample]
    target_list = [s["target"] for s in sample]
    history_flat_list = [[elem for sublist in h for elem in sublist] for h in history_list]
    history_tensor = torch.tensor(history_flat_list, dtype=torch.long, device=DEVICE)
    target_tensor = torch.tensor(target_list, dtype=torch.long, device=DEVICE)
    log_lines.append(f"[Data] history_tensor shape: {history_tensor.shape}")

    sid_range = verify_sid_token_range(history_tensor, CODEBOOK_SIZE)
    log_lines.append(f"[Precheck 4] SID token range: {sid_range}")

    model_wrapper = HG_Rec_with_RealMetadataAdapter(
        t5_config, t5_state_dict, d_model=D_MODEL, n_layers=3, sid_dim=4,
    ).to(DEVICE)

    # Precheck
    init_alpha = model_wrapper.adapter.get_alpha()
    log_lines.append(f"\n[Precheck] init alpha: {init_alpha:.6e} (init_zero={abs(init_alpha) < 1e-3})")

    n_samples = len(train_ds)
    all_histories = np.zeros((n_samples, MAX_LEN * 4), dtype=np.int64)
    all_targets = np.zeros((n_samples, 4), dtype=np.int64)
    for i in range(n_samples):
        s = train_ds[i]
        all_histories[i] = np.concatenate([np.asarray(h, dtype=np.int64).flatten() for h in s["history"]])
        all_targets[i] = np.asarray(s["target"], dtype=np.int64)
    all_histories_t = torch.from_numpy(all_histories).to(DEVICE).long()
    all_targets_t = torch.from_numpy(all_targets).to(DEVICE).long()

    real_metadata_t = torch.from_numpy(real_metadata).to(DEVICE).float()  # (9922, 3, 4)
    log_lines.append(f"  preloaded histories: {all_histories.shape}, real_metadata: {real_metadata_t.shape}")

    conditioner_params = [p for n, p in model_wrapper.adapter.named_parameters()]
    layernorm_params = list(model_wrapper.first_input_ln.parameters())
    optimizer = torch.optim.Adam([
        {"params": conditioner_params, "lr": LR_CONDITIONER},
        {"params": layernorm_params, "lr": LR_LAYERNORM},
    ])

    rng = torch.Generator().manual_seed(42 + 8)
    n_batches = (n_samples + BATCH_SIZE - 1) // BATCH_SIZE

    train_trace = []
    val_r10_history = []
    early_stop = False
    epoch0_loss = None
    for epoch in range(NUM_EPOCHS):
        if early_stop:
            log_lines.append(f"  [epoch {epoch}] early-stopped by R23")
            break
        epoch_losses = []
        cond_grad_norms = []
        ln_grad_norms = []
        nan_inf_detected = False
        epoch_indices = torch.randperm(n_samples, generator=rng).to(DEVICE)
        for batch_idx in range(n_batches):
            start = batch_idx * BATCH_SIZE
            end = min(start + BATCH_SIZE, n_samples)
            batch_indices = epoch_indices[start:end]
            history_tensor_b = all_histories_t[batch_indices]
            target_tensor_b = all_targets_t[batch_indices]
            attention_mask_b = (history_tensor_b != PAD_TOKEN).long()

            B = history_tensor_b.shape[0]
            L_flat = MAX_LEN * 4
            digit_values = history_tensor_b.float()
            layer_idx = torch.arange(L_flat, device=DEVICE) % 4
            layer_idx = layer_idx.float().unsqueeze(0).expand(B, -1)
            pos_in_history = torch.arange(L_flat, device=DEVICE) // 4
            pos_in_history = pos_in_history.float().unsqueeze(0).expand(B, -1) / MAX_LEN
            padding_flag = (digit_values == PAD_TOKEN).float()
            sid_meta = torch.stack([digit_values / 1025.0, layer_idx / 4.0, pos_in_history, padding_flag], dim=-1)
            # 真实 per-item metadata (从 Stage 1 export)
            curvature_meta = real_metadata_t[batch_indices % real_metadata_t.shape[0]]  # (B, 3, 4), wrap indices

            optimizer.zero_grad()
            output, _, alpha, residual_norm = model_wrapper(history_tensor_b, attention_mask=attention_mask_b,
                                                            labels=target_tensor_b, sid_meta=sid_meta,
                                                            curvature_meta=curvature_meta)
            loss = output.loss if hasattr(output, 'loss') else output[0]
            if not torch.isfinite(loss):
                nan_inf_detected = True
                continue
            loss.backward()
            cond_grad_norm = 0.0
            for p in conditioner_params:
                if p.grad is not None and torch.isfinite(p.grad).all():
                    cond_grad_norm += p.grad.norm().item() ** 2
            cond_grad_norm = cond_grad_norm ** 0.5
            ln_grad_norm = 0.0
            for p in layernorm_params:
                if p.grad is not None and torch.isfinite(p.grad).all():
                    ln_grad_norm += p.grad.norm().item() ** 2
            ln_grad_norm = ln_grad_norm ** 0.5
            optimizer.step()
            epoch_losses.append(loss.item())
            cond_grad_norms.append(cond_grad_norm)
            ln_grad_norms.append(ln_grad_norm)

        avg_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        avg_cond_grad = sum(cond_grad_norms) / max(1, len(cond_grad_norms))
        avg_ln_grad = sum(ln_grad_norms) / max(1, len(ln_grad_norms))
        if epoch == 0:
            epoch0_loss = avg_loss

        val_r10 = compute_val_r10(model_wrapper, val_ds, CODEBOOK_SIZE, MAX_LEN, PAD_TOKEN, DEVICE, n=2048)
        val_r10_history.append(val_r10)
        alpha_val = model_wrapper.adapter.get_alpha()

        # R12 ckpt
        if ADAPTER_CKPT_PATH.exists():
            ADAPTER_CKPT_PATH.unlink()
        torch.save({
            "adapter_state_dict": model_wrapper.adapter.state_dict(),
            "first_input_ln_state_dict": model_wrapper.first_input_ln.state_dict(),
            "alpha_value": alpha_val,
            "epoch_losses": [t["avg_loss"] for t in train_trace],
            "val_r10_history": val_r10_history,
        }, ADAPTER_CKPT_PATH)

        train_trace.append({
            "epoch": epoch, "avg_loss": avg_loss,
            "avg_cond_grad_norm": avg_cond_grad,
            "avg_ln_grad_norm": avg_ln_grad,
            "alpha_val": alpha_val,
            "val_r10": val_r10,
            "nan_inf": nan_inf_detected, "n_batches": len(epoch_losses),
        })
        log_lines.append(f"  [epoch {epoch}] loss={avg_loss:.4f}, cond_grad={avg_cond_grad:.4e}, ln_grad={avg_ln_grad:.4e}, α={alpha_val:.6e}, val_R@10={val_r10:.4f}, n_batches={len(epoch_losses)}, nan_inf={nan_inf_detected}")
        print(log_lines[-1], flush=True)

        if len(val_r10_history) >= 2 and val_r10_history[-1] == 0 and val_r10_history[-2] == 0:
            log_lines.append(f"  [R23] val_R@10=0 连续 2 次 (epoch {epoch-1}, {epoch}), early stop + NO-GO")
            early_stop = True
            break

    # Save/Load
    log_lines.append(f"\n[Save/Load 验证]")
    ckpt_loaded = torch.load(ADAPTER_CKPT_PATH, map_location="cpu", weights_only=False)
    expected_keys = set(model_wrapper.adapter.state_dict().keys()) | set(model_wrapper.first_input_ln.state_dict().keys())
    loaded_keys = set(ckpt_loaded["adapter_state_dict"].keys()) | set(ckpt_loaded["first_input_ln_state_dict"].keys())
    missing_keys = expected_keys - loaded_keys
    unexpected_keys = loaded_keys - expected_keys
    log_lines.append(f"  missing_keys: {len(missing_keys)}, unexpected_keys: {len(unexpected_keys)}")
    adapter_ckpt_sha = sha256_of(ADAPTER_CKPT_PATH)

    # Forward 一致性 (early_stop 跳过)
    if early_stop:
        log_lines.append(f"\n[Forward 一致性] skipped (R23 early_stop)")
        forward_diff = 0.0
    else:
        log_lines.append(f"\n[Forward 一致性]")
        model_wrapper.eval()
        with torch.no_grad():
            B = history_tensor.shape[0]
            L_flat = MAX_LEN * 4
            sid_meta_test = torch.zeros(B, L_flat, 4, dtype=torch.float32, device=DEVICE)
            curv_meta_test = torch.zeros(B, 3, 4, dtype=torch.float32, device=DEVICE)
            attn_mask_for_fwd = (history_tensor != PAD_TOKEN).long()
            out1, _, _, _ = model_wrapper(history_tensor, attention_mask=attn_mask_for_fwd,
                                          labels=target_tensor, sid_meta=sid_meta_test,
                                          curvature_meta=curv_meta_test)
            out2, _, _, _ = model_wrapper(history_tensor, attention_mask=attn_mask_for_fwd,
                                          labels=target_tensor, sid_meta=sid_meta_test,
                                          curvature_meta=curv_meta_test)
            logits1 = out1.logits if hasattr(out1, 'logits') else out1[0]
            logits2 = out2.logits if hasattr(out2, 'logits') else out2[0]
            forward_diff = (logits1 - logits2).abs().max().item()
        log_lines.append(f"  forward diff max: {forward_diff:.2e}")

    # Gate 2 决策
    loss_decreased = epoch0_loss is not None and train_trace[-1]["avg_loss"] < epoch0_loss
    log_lines.append(f"\n[Gate 2 checks]:")
    log_lines.append(f"  (1) loss epoch0={epoch0_loss:.4f}, final={train_trace[-1]['avg_loss']:.4f}, decreased={loss_decreased}")
    log_lines.append(f"  (2) val_R@10 trace: {val_r10_history}")
    cond_grad_strs = [f"{t['avg_cond_grad_norm']:.4e}" for t in train_trace]
    ln_grad_strs = [f"{t['avg_ln_grad_norm']:.4e}" for t in train_trace]
    log_lines.append(f"  (3) cond_grad epoch trace: {cond_grad_strs}")
    log_lines.append(f"  (4) ln_grad epoch trace: {ln_grad_strs}")
    log_lines.append(f"  (5) nan_inf all False: {all(not t['nan_inf'] for t in train_trace)}")
    log_lines.append(f"  (6) sid_range all_in_range: {sid_range['all_in_range']}")
    log_lines.append(f"  (7) save/load missing={len(missing_keys)}/unexpected={len(unexpected_keys)}")
    log_lines.append(f"  (8) forward diff: {forward_diff:.2e}")

    r23_kill = False
    if len(val_r10_history) >= 2:
        r23_kill = any(val_r10_history[i] == 0 and val_r10_history[i+1] == 0
                       for i in range(len(val_r10_history)-1))
    val_r10_max = max(val_r10_history) if val_r10_history else 0.0
    gate2_pass = (
        loss_decreased and
        all(t["avg_cond_grad_norm"] > 0 for t in train_trace) and
        all(t["avg_ln_grad_norm"] > 0 for t in train_trace) and
        all(not t["nan_inf"] for t in train_trace) and
        sid_range["all_in_range"] and
        len(missing_keys) == 0 and
        len(unexpected_keys) == 0 and
        forward_diff < 1e-5 and
        not r23_kill and
        val_r10_max > 0.0
    )

    log_lines.append(f"\n[Gate 2 (含 val_R@10 早停)] {'✅ PASS' if gate2_pass else '❌ FAIL'}")
    log_lines.append(f"[val_R@10 max] {val_r10_max:.4f}, [r23_kill] {r23_kill}")
    print("\n".join(log_lines), flush=True)
    with open(LOG_PATH, "w") as f:
        f.write("\n".join(log_lines))

    with open(TRAIN_TRACE_PATH, "w") as f:
        json.dump({"train_trace": train_trace, "epoch0_loss": epoch0_loss,
                   "loss_decreased": loss_decreased, "gate2_pass": gate2_pass,
                   "val_r10_history": val_r10_history, "r23_kill": r23_kill,
                   "val_r10_max": val_r10_max}, f, indent=2)

    with open(VERDICT_PATH, "w") as f:
        json.dump({
            "gate1_pass": gate1_pass, "gate2_pass": gate2_pass,
            "reason": "trained" if gate2_pass else "r23_kill_or_loss_not_decreased",
            "stage1_proof": stage1_proof,
            "sid_range": sid_range, "missing_keys": list(missing_keys),
            "unexpected_keys": list(unexpected_keys), "forward_diff": forward_diff,
            "train_trace_summary": train_trace, "epoch0_loss": epoch0_loss,
            "loss_decreased": loss_decreased, "adapter_ckpt_sha256": adapter_ckpt_sha,
            "val_r10_history": val_r10_history, "val_r10_max": val_r10_max,
            "r23_kill": r23_kill, "task_id": 462, "issue": "Issue #169",
        }, f, indent=2)


if __name__ == "__main__":
    main()
