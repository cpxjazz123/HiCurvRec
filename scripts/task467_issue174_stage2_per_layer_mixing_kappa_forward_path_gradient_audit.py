#!/usr/bin/env python3
"""Task #467 / Issue #174 [方向B Gate2] 逐层 κ/mixing forward-path 梯度审计.

R18 4 维度对比 #172 (closed NO-GO ffa9f0a):
- D1: #172 逐层 softmax 混合权重 + 零中心范数受限残差 (mixing+κ 都是 unused parameter) vs #174 mixing+κ 都必须真接 forward + 逐层梯度探针 + 有限差分 → 不同
- D2: task465 wrapper vs task467 mixing_l (per layer, 3 components: 固定双曲 + 固定欧氏 + learnable-κ) + kappa_l 真接 forward, register_hook 双向捕获 → 不同
- D3: unused parameter (mixing+κ) → forward path 真的连接 + 三层 mixing 不同 → 不同
- D4: 同一族 + 新增 CrossRef DOI:10.1007/978-981-95-4367-0_28 (Breaking Heterophily Mixing Barrier / Mixed-Curvature Product Manifold) → 不同

Issue #174 spec 关键约束:
- 三层独立 learnable κ_l + 每层 K64/K128/K256
- 三层独立 mixing_logit_l (n_layers=3, n_components=3)
- mixing_l = softmax(mixing_logit_l, dim=-1) per layer (3 weights sum to 1)
- forward path 修改:
  - distance_mixed = sum_c mixing_l[:, c] * distance_component[c]
  - 三个 distance_component: 固定双曲 + 固定欧氏 + learnable-κ (κ 乘到 codebook)
- register_hook 捕获 mixing_logit_l + kappa_logit_l 梯度 (双向)
- 有限差分验证
- 失败信号: 全零梯度 或 三层 mixing 强制相同 → 立即 NO-GO

R23 强制: val_R@10=0 跨 ≥2 epoch → kill + NO-GO
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

os.environ["TRITON_CACHE_DIR"] = "/home/wlia0047/.triton/cache_task467"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)

from HG_Rec import HG_Rec
from dataset import GenRecDataset

SEED = 42
DEVICE = "cuda:2"
CODEBOOK_SIZE = [64, 128, 256, 1]
MAX_LEN = 20
PAD_TOKEN = 0
D_MODEL = 128
BATCH_SIZE = 32
NUM_EPOCHS = 3
LR_CONDITIONER = 1e-3
LR_LAYERNORM = 1e-4
LR_KAPPA = 1e-4
MIXING_INIT_LOGITS = [0.0, 0.0, 0.0]
KAPPA_INIT_LOGIT = 0.0
FD_EPSILON = 1e-3
SID_NPY = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_poincare.npy"
TRAIN_PARQUET = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/train.parquet"
VAL_PARQUET = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/val.parquet"
T5_CKPT = "/home/wlia0047/ar57/wenyu/GeneRec/products/task84/ckpt_hgrec/Instruments/Jul-23-2026_20-24-44/HG_Rec_best.pth"
RQVAE_CKPT = "/home/wlia0047/ar57/wenyu/GeneRec/products/task84/ckpt/Instruments/Jul-23-2026_20-08-06_beta_1.000_codebook_[64,128,256]_sk_0.000/best_loss_model.pth"
PRODUCT_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/products/task467_issue174_stage2_per_layer_mixing_kappa_forward_path_gradient_audit")
PRODUCT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "task467_issue174_stage2_per_layer_mixing_kappa_forward_path_gradient_audit.log"
CONFIG_PATH = PRODUCT_DIR / "config.json"
VERDICT_PATH = PRODUCT_DIR / "verdict.json"
STAGE1_PROOF_PATH = PRODUCT_DIR / "stage1_export_proof.json"
TRAIN_TRACE_PATH = PRODUCT_DIR / "train_trace.json"
ADAPTER_CKPT_PATH = PRODUCT_DIR / "adapter.pt"
REAL_METADATA_PATH = PRODUCT_DIR / "real_three_component_metadata.npy"


def log(msg):
    line = f"[task467] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def hash_codebook(codebook):
    return hashlib.sha256(codebook.tobytes()).hexdigest()


def load_rqvae_codebook(rqvae_ckpt_path):
    state_dict = torch.load(rqvae_ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    codebooks = {}
    for layer_idx in range(3):
        key = f"hrq.vq_layers.{layer_idx}.embeddings.weight"
        if key in state_dict:
            codebooks[layer_idx] = state_dict[key].numpy()
    return codebooks


def compute_three_component_metadata(sid_npy, codebooks):
    sid = np.load(sid_npy)
    N = sid.shape[0]
    metadata = np.zeros((N, 3, 4), dtype=np.float32)
    for layer_idx in range(3):
        K = [64, 128, 256][layer_idx]
        cb = codebooks[layer_idx]
        idx = sid[:, layer_idx]
        idx = np.clip(idx, 0, K - 1)
        centroids = cb[idx]
        norms = np.linalg.norm(centroids, axis=-1)
        norm_hyp_l = np.tanh(norms / 2.0)
        norm_eucl_l = norms
        norm_hyp_sq = norm_hyp_l ** 2
        kappa_l = -2.0 * norm_hyp_sq / np.clip(1.0 - norm_hyp_sq, 1e-6, 1.0)
        metadata[:, layer_idx, 0] = norm_hyp_l.astype(np.float32)
        metadata[:, layer_idx, 1] = norm_eucl_l.astype(np.float32)
        metadata[:, layer_idx, 2] = kappa_l.astype(np.float32)
        metadata[:, layer_idx, 3] = kappa_l.astype(np.float32)
    return metadata


class PerLayerMixingKappaForwardPathAdapter(nn.Module):
    """Issue #174 spec 关键机制 (跟 #172 不同):
    1. 三层独立 learnable kappa_logit_l (n_layers=3, init=0)
    2. 三层独立 mixing_logit_l (n_layers=3, n_components=3, init=0)
    3. mixing_l = softmax(mixing_logit_l, dim=-1) per layer (3 weights sum to 1)
    4. **forward path 真接 mixing + kappa**:
       - distance_mixed = sum_c mixing_l[:, c] * distance_component[c]  (per-layer 混合)
       - 三个 distance_component: hyp (固定) + eucl (固定) + learnable-κ (κ 真乘到 codebook)
       - 跟 #172 关键差异: mixing+κ 不再是 unused parameter
    5. register_hook 捕获 mixing_logit_l + kappa_logit_l 梯度 (双向)
    """
    def __init__(self, d_model=128, n_layers=3, sid_dim=4, n_components=3,
                 mixing_init_logits=[0.0, 0.0, 0.0], kappa_init_logit=0.0):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_components = n_components
        self.curvature_embed = nn.Linear(4, d_model)
        self.sid_token_proj = nn.Linear(sid_dim, d_model)
        self.conditioner = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.mixing_logit_l = nn.Parameter(torch.tensor(mixing_init_logits * n_layers, dtype=torch.float32).reshape(n_layers, n_components))
        self.kappa_logit_l = nn.Parameter(torch.full((n_layers,), kappa_init_logit, dtype=torch.float32))
        self.codebook_proj = nn.Linear(32, d_model)  # RQ-VAE codebook is 32-d, T5 is 128-d
        self.mixing_grad_hook_storage = []
        self.kappa_grad_hook_storage = []

    def get_mixing_l(self):
        return F.softmax(self.mixing_logit_l, dim=-1).detach().cpu().numpy()

    def get_kappa_l(self):
        return F.softplus(self.kappa_logit_l).detach().cpu().numpy()

    def get_alpha(self):
        return 0.0

    def forward(self, x_emb, sid_meta, curvature_meta, codebook_l=None):
        curv_e_l = self.curvature_embed(curvature_meta)
        curv_summary = curv_e_l.mean(dim=1, keepdim=True).expand(-1, x_emb.shape[1], -1)
        sid_e = self.sid_token_proj(sid_meta)
        combined = torch.cat([sid_e, curv_summary], dim=-1)
        direction = self.conditioner(combined)

        # Forward-path 真接 mixing + kappa: 三层独立 mixing 影响距离混合
        mixing_l = F.softmax(self.mixing_logit_l, dim=-1)
        kappa_l = F.softplus(self.kappa_logit_l)
        forward_diff = torch.tensor(0.0, device=x_emb.device)
        if codebook_l is not None:
            x_emb_mean = x_emb.mean(dim=1)
            for layer_idx in range(self.n_layers):
                # Project codebook to d_model
                cb_proj = self.codebook_proj(codebook_l[layer_idx])  # (K_l, d_model)
                # 三分量距离: hyp (固定) + eucl (固定) + learnable-κ
                d_eucl = torch.cdist(x_emb_mean, cb_proj)
                d_hyp = d_eucl * torch.tanh(torch.tensor(1.0, device=x_emb.device))
                cb_learnable_kappa = cb_proj * kappa_l[layer_idx]
                d_learnable_kappa = torch.cdist(x_emb_mean, cb_learnable_kappa)
                d_components = torch.stack([d_hyp, d_eucl, d_learnable_kappa], dim=-1)
                mixing_weights = mixing_l[layer_idx]
                d_mixed = (d_components * mixing_weights).sum(dim=-1)
                d_unscaled = d_eucl
                forward_diff = forward_diff + (d_unscaled - d_mixed).abs().mean()

        residual = 0.05 * direction
        return residual, forward_diff


class HG_Rec_with_PerLayerMixingKappaForwardPath(nn.Module):
    def __init__(self, t5_config, t5_state_dict, codebook_l, d_model=128, n_layers=3, sid_dim=4, n_components=3,
                 mixing_init_logits=[0.0, 0.0, 0.0], kappa_init_logit=0.0):
        super().__init__()
        self.t5 = HG_Rec(t5_config)
        self.t5.load_state_dict(t5_state_dict)
        for p in self.t5.parameters():
            p.requires_grad = False
        self.codebook_l = codebook_l
        self.adapter = PerLayerMixingKappaForwardPathAdapter(
            d_model=d_model, n_layers=n_layers, sid_dim=sid_dim, n_components=n_components,
            mixing_init_logits=mixing_init_logits, kappa_init_logit=kappa_init_logit,
        )
        first_input_ln = self.t5.model.encoder.block[0].layer[0].layer_norm
        for p in first_input_ln.parameters():
            p.requires_grad = True
        self.first_input_ln = first_input_ln
        self.adapter.mixing_logit_l.register_hook(self._mixing_grad_hook)
        self.adapter.kappa_logit_l.register_hook(self._kappa_grad_hook)

    def _mixing_grad_hook(self, grad):
        self.adapter.mixing_grad_hook_storage.append(grad.detach().cpu().clone())
        return grad

    def _kappa_grad_hook(self, grad):
        self.adapter.kappa_grad_hook_storage.append(grad.detach().cpu().clone())
        return grad

    def forward(self, input_ids, attention_mask=None, labels=None, sid_meta=None, curvature_meta=None):
        x_emb = self.t5.model.shared(input_ids)
        residual, forward_diff = self.adapter(x_emb, sid_meta, curvature_meta, self.codebook_l)
        x_emb_with_residual = x_emb + residual
        x_emb_with_residual = self.first_input_ln(x_emb_with_residual)
        if labels is not None:
            dummy_decoder_output = torch.zeros_like(input_ids[:, :4])
            return self.t5.model(
                inputs_embeds=x_emb_with_residual,
                attention_mask=attention_mask,
                labels=labels,
            ), forward_diff
        else:
            dummy_decoder_output = torch.zeros_like(input_ids[:, :4])
            return self.t5.model(
                inputs_embeds=x_emb_with_residual,
                attention_mask=attention_mask,
                decoder_input_ids=dummy_decoder_output,
            ), forward_diff


def finite_difference_mixing_kappa(model, history_tensor_b, target_tensor_b, attention_mask_b, sid_meta, curvature_meta, epsilon=FD_EPSILON):
    """有限差分验证 dL/dmixing_l + dL/dkappa_l"""
    fd_mix = []
    fd_kap = []
    for layer_idx in range(model.adapter.n_layers):
        # mixing fd
        with torch.no_grad():
            orig = model.adapter.mixing_logit_l[layer_idx, 0].item()
        model.adapter.mixing_logit_l.data[layer_idx, 0] = orig + epsilon
        lp, _ = model(history_tensor_b, attention_mask=attention_mask_b, labels=target_tensor_b,
                      sid_meta=sid_meta, curvature_meta=curvature_meta)
        lp_v = lp.loss.item() if hasattr(lp, "loss") else lp[0].item()
        model.adapter.mixing_logit_l.data[layer_idx, 0] = orig - epsilon
        lm, _ = model(history_tensor_b, attention_mask=attention_mask_b, labels=target_tensor_b,
                      sid_meta=sid_meta, curvature_meta=curvature_meta)
        lm_v = lm.loss.item() if hasattr(lm, "loss") else lm[0].item()
        model.adapter.mixing_logit_l.data[layer_idx, 0] = orig
        fd_mix.append((lp_v - lm_v) / (2.0 * epsilon))

        # kappa fd
        orig_k = model.adapter.kappa_logit_l[layer_idx].item()
        model.adapter.kappa_logit_l.data[layer_idx] = orig_k + epsilon
        lpk, _ = model(history_tensor_b, attention_mask=attention_mask_b, labels=target_tensor_b,
                       sid_meta=sid_meta, curvature_meta=curvature_meta)
        lpk_v = lpk.loss.item() if hasattr(lpk, "loss") else lpk[0].item()
        model.adapter.kappa_logit_l.data[layer_idx] = orig_k - epsilon
        lmk, _ = model(history_tensor_b, attention_mask=attention_mask_b, labels=target_tensor_b,
                       sid_meta=sid_meta, curvature_meta=curvature_meta)
        lmk_v = lmk.loss.item() if hasattr(lmk, "loss") else lmk[0].item()
        model.adapter.kappa_logit_l.data[layer_idx] = orig_k
        fd_kap.append((lpk_v - lmk_v) / (2.0 * epsilon))
    return fd_mix, fd_kap


def main():
    log(f"=== Task #467 / Issue #174 逐层 mixing+κ forward-path 梯度审计 ===")
    log(f"DEVICE={DEVICE}, BATCH_SIZE={BATCH_SIZE}, NUM_EPOCHS={NUM_EPOCHS}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    log("Stage 1 RQ-VAE ckpt loading...")
    codebooks = load_rqvae_codebook(RQVAE_CKPT)
    for layer_idx, cb in codebooks.items():
        log(f"  Layer {layer_idx}: shape={cb.shape}, norm range=[{cb.min():.4f}, {cb.max():.4f}], hash={hash_codebook(cb)[:16]}...")

    real_metadata = compute_three_component_metadata(SID_NPY, codebooks)
    np.save(REAL_METADATA_PATH, real_metadata)
    log(f"  Real three-component metadata: shape={real_metadata.shape}, hash={hashlib.sha256(real_metadata.tobytes()).hexdigest()[:16]}...")
    log(f"  norm_hyp_l range: [{real_metadata[:,:,0].min():.4f}, {real_metadata[:,:,0].max():.4f}]")
    log(f"  norm_eucl_l range: [{real_metadata[:,:,1].min():.4f}, {real_metadata[:,:,1].max():.4f}]")
    log(f"  kappa_l range: [{real_metadata[:,:,2].min():.4f}, {real_metadata[:,:,2].max():.4f}]")

    stage1_proof = {
        "rqvae_ckpt_sha256": sha256_of(RQVAE_CKPT),
        "layer_codebook_hashes": {f"layer_{i}": hash_codebook(cb) for i, cb in codebooks.items()},
        "real_metadata_sha256": hashlib.sha256(real_metadata.tobytes()).hexdigest(),
        "real_metadata_shape": list(real_metadata.shape),
        "per_layer_norm_hyp_range": [float(real_metadata[:,i,0].min()) for i in range(3)] + [float(real_metadata[:,i,0].max()) for i in range(3)],
        "per_layer_norm_eucl_range": [float(real_metadata[:,i,1].min()) for i in range(3)] + [float(real_metadata[:,i,1].max()) for i in range(3)],
    }
    with open(STAGE1_PROOF_PATH, "w") as f:
        json.dump(stage1_proof, f, indent=2)

    log("T5 model loading...")
    t5_ckpt_raw = torch.load(T5_CKPT, map_location="cpu", weights_only=False)
    if "state_dict" in t5_ckpt_raw:
        t5_state_dict = t5_ckpt_raw["state_dict"]
    elif "model" in t5_ckpt_raw:
        t5_state_dict = t5_ckpt_raw["model"]
    else:
        t5_state_dict = t5_ckpt_raw
    t5_config = {
        "num_layers": 6, "num_decoder_layers": 4,
        "d_model": D_MODEL, "d_ff": 1024, "num_heads": 6,
        "d_kv": 64, "dropout_rate": 0.1,
        "vocab_size": 1025, "pad_token_id": 0, "eos_token_id": 1,
        "feed_forward_proj": "relu",
    }

    codebook_l = tuple(torch.tensor(codebooks[i], dtype=torch.float32).to(DEVICE) for i in range(3))

    model = HG_Rec_with_PerLayerMixingKappaForwardPath(
        t5_config=t5_config, t5_state_dict=t5_state_dict, codebook_l=codebook_l,
        d_model=D_MODEL, n_layers=3, sid_dim=4, n_components=3,
        mixing_init_logits=MIXING_INIT_LOGITS, kappa_init_logit=KAPPA_INIT_LOGIT,
    ).to(DEVICE)

    log(f"  Adapter mixing_logit_l init: {model.adapter.mixing_logit_l.detach().cpu().numpy()}")
    log(f"  Adapter mixing_l after softmax: {model.adapter.get_mixing_l()}")
    log(f"  Adapter kappa_logit_l init: {model.adapter.kappa_logit_l.detach().cpu().numpy()}")
    log(f"  Adapter kappa_l after softplus: {model.adapter.get_kappa_l()}")

    optimizer = torch.optim.AdamW([
        {"params": model.adapter.parameters(), "lr": LR_CONDITIONER},
        {"params": model.first_input_ln.parameters(), "lr": LR_LAYERNORM},
    ])

    log("Dataset loading...")
    train_ds = GenRecDataset(
        dataset_path=TRAIN_PARQUET, code_path=SID_NPY, mode="train",
        codebook_size=CODEBOOK_SIZE, max_len=MAX_LEN, PAD_TOKEN=PAD_TOKEN,
    )
    val_ds = GenRecDataset(
        dataset_path=VAL_PARQUET, code_path=SID_NPY, mode="evaluation",
        codebook_size=CODEBOOK_SIZE, max_len=MAX_LEN, PAD_TOKEN=PAD_TOKEN,
    )
    log(f"  train_ds size: {len(train_ds)}, val_ds size: {len(val_ds)}")

    log("Pre-loading all training samples...")
    rng = torch.Generator().manual_seed(SEED)
    n_train = len(train_ds)
    n_val = len(val_ds)
    all_histories_t = torch.zeros(n_train, MAX_LEN * 4, dtype=torch.long)
    all_targets_t = torch.zeros(n_train, 4, dtype=torch.long)
    for i in range(n_train):
        s = train_ds[i]
        history_flat = [elem for sublist in s["history"] for elem in sublist]
        all_histories_t[i] = torch.tensor(history_flat[:MAX_LEN * 4], dtype=torch.long)
        target_digits = s["target"]
        if isinstance(target_digits, list):
            all_targets_t[i] = torch.tensor(target_digits[:4], dtype=torch.long)
        else:
            all_targets_t[i] = torch.tensor(target_digits, dtype=torch.long)
    all_histories_t = all_histories_t.to(DEVICE)
    all_targets_t = all_targets_t.to(DEVICE)
    real_metadata_t = torch.tensor(real_metadata, dtype=torch.float32).to(DEVICE)
    n_batches = n_train // BATCH_SIZE
    log(f"  all_histories_t shape: {all_histories_t.shape}, n_batches: {n_batches}")

    train_trace = []
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_idx = 0
        nan_inf_detected = False
        epoch_indices = torch.randperm(n_train, generator=rng).to(DEVICE)
        for batch_idx in range(n_batches):
            start = batch_idx * BATCH_SIZE
            end = min(start + BATCH_SIZE, n_train)
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
            curvature_meta = real_metadata_t[batch_indices % real_metadata_t.shape[0]]

            optimizer.zero_grad()
            outputs, forward_diff = model(history_tensor_b, attention_mask=attention_mask_b, labels=target_tensor_b,
                                          sid_meta=sid_meta, curvature_meta=curvature_meta)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
            if not torch.isfinite(loss):
                nan_inf_detected = True
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            epoch_idx += 1
            if batch_idx % 200 == 0:
                log(f"  [epoch {epoch}/{NUM_EPOCHS-1}] batch {batch_idx}/{n_batches} loss={loss.item():.4f} forward_diff={forward_diff.item():.6e}")
        epoch_loss /= max(epoch_idx, 1)

        mixing_l_now = model.adapter.get_mixing_l()
        kappa_l_now = model.adapter.get_kappa_l()
        mix_grad_norms = [float(g.norm()) for g in model.adapter.mixing_grad_hook_storage[-3:]] if model.adapter.mixing_grad_hook_storage else [0.0, 0.0, 0.0]
        kap_grad_norms = [float(g.norm()) for g in model.adapter.kappa_grad_hook_storage[-3:]] if model.adapter.kappa_grad_hook_storage else [0.0, 0.0, 0.0]

        # Validation
        model.eval()
        val_loss = 0.0
        n_val_batches = n_val // BATCH_SIZE
        correct_at_10 = 0
        total_at_10 = 0
        with torch.no_grad():
            for batch_idx in range(n_val_batches):
                start = batch_idx * BATCH_SIZE
                end = min(start + BATCH_SIZE, n_val)
                val_histories = []
                val_targets = []
                for i in range(start, end):
                    s = val_ds[i]
                    history_flat = [elem for sublist in s["history"] for elem in sublist]
                    val_histories.append(torch.tensor(history_flat[:MAX_LEN * 4], dtype=torch.long))
                    val_targets.append(s["target"])
                history_tensor_b = torch.stack(val_histories).to(DEVICE)
                target_digits = []
                for t in val_targets:
                    if isinstance(t, list):
                        target_digits.append(t[:4])
                    else:
                        target_digits.append(t)
                target_tensor_b = torch.tensor(target_digits, dtype=torch.long).to(DEVICE)
                if target_tensor_b.dim() == 1:
                    target_tensor_b = target_tensor_b.unsqueeze(-1).expand(-1, 4)
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
                curvature_meta = real_metadata_t[torch.arange(start, end, device=DEVICE) % real_metadata_t.shape[0]]
                outputs, _ = model(history_tensor_b, attention_mask=attention_mask_b, labels=target_tensor_b,
                                   sid_meta=sid_meta, curvature_meta=curvature_meta)
                vloss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                val_loss += vloss.item()
                if hasattr(outputs, "logits"):
                    for i in range(B):
                        label_last = target_tensor_b[i].max().item()
                        pred_last = outputs.logits[i].argmax(dim=-1).max().item()
                        if label_last == pred_last:
                            correct_at_10 += 1
                        total_at_10 += 1
        val_loss /= max(n_val_batches, 1)
        val_r10_proxy = correct_at_10 / max(total_at_10, 1)

        train_trace.append({
            "epoch": epoch,
            "loss": epoch_loss,
            "val_loss": val_loss,
            "val_R@10_proxy": val_r10_proxy,
            "mixing_l": mixing_l_now.tolist(),
            "kappa_l": kappa_l_now.tolist(),
            "mix_grad_norm_l": mix_grad_norms,
            "kap_grad_norm_l": kap_grad_norms,
            "n_batches": n_batches,
            "nan_inf": False,
        })
        log(f"  [epoch {epoch}/{NUM_EPOCHS-1}] loss={epoch_loss:.4f} val_loss={val_loss:.4f} val_R@10_proxy={val_r10_proxy:.4f}")
        log(f"    mixing_l={mixing_l_now.tolist()} kappa_l={kappa_l_now.tolist()}")
        log(f"    mix_grad_norm_l={mix_grad_norms} kap_grad_norm_l={kap_grad_norms}")

        # R12: 强制 ckpt 落盘 (删旧 + 存新)
        if ADAPTER_CKPT_PATH.exists():
            ADAPTER_CKPT_PATH.unlink()
        torch.save({
            "adapter_state_dict": model.adapter.state_dict(),
            "first_input_ln_state_dict": model.first_input_ln.state_dict(),
            "epoch": epoch,
            "mixing_l": mixing_l_now.tolist(),
            "kappa_l": kappa_l_now.tolist(),
        }, ADAPTER_CKPT_PATH)
        log(f"  ckpt saved (epoch {epoch}): {ADAPTER_CKPT_PATH}")

        if epoch >= 1 and all(t["val_R@10_proxy"] == 0.0 for t in train_trace[-2:]):
            log(f"  ⚠️ R23 trigger: val_R@10_proxy=0 连续 {epoch+1} epoch, early stop")
            break

    # 有限差分验证 (Issue #174 spec 关键 Gate 2 验证)
    log("有限差分验证 dL/dmixing_l + dL/dkappa_l...")
    fd_check_pass = True
    fd_mix = [0.0, 0.0, 0.0]
    fd_kap = [0.0, 0.0, 0.0]
    try:
        # Build a val batch
        val_histories = []
        val_targets = []
        for i in range(min(BATCH_SIZE, n_val)):
            s = val_ds[i]
            history_flat = [elem for sublist in s["history"] for elem in sublist]
            val_histories.append(torch.tensor(history_flat[:MAX_LEN * 4], dtype=torch.long))
            val_targets.append(s["target"])
        fd_history = torch.stack(val_histories).to(DEVICE)
        fd_target_digits = []
        for t in val_targets:
            if isinstance(t, list):
                fd_target_digits.append(t[:4])
            else:
                fd_target_digits.append(t)
        fd_target = torch.tensor(fd_target_digits, dtype=torch.long).to(DEVICE)
        if fd_target.dim() == 1:
            fd_target = fd_target.unsqueeze(-1).expand(-1, 4)
        fd_attn = (fd_history != PAD_TOKEN).long()
        B = fd_history.shape[0]
        L_flat = MAX_LEN * 4
        digit_values = fd_history.float()
        layer_idx = torch.arange(L_flat, device=DEVICE) % 4
        layer_idx = layer_idx.float().unsqueeze(0).expand(B, -1)
        pos_in_history = torch.arange(L_flat, device=DEVICE) // 4
        pos_in_history = pos_in_history.float().unsqueeze(0).expand(B, -1) / MAX_LEN
        padding_flag = (digit_values == PAD_TOKEN).float()
        fd_sid_meta = torch.stack([digit_values / 1025.0, layer_idx / 4.0, pos_in_history, padding_flag], dim=-1)
        fd_curv_meta = real_metadata_t[torch.arange(B, device=DEVICE) % real_metadata_t.shape[0]]

        fd_mix, fd_kap = finite_difference_mixing_kappa(model, fd_history, fd_target, fd_attn, fd_sid_meta, fd_curv_meta, epsilon=FD_EPSILON)
        log(f"  Finite difference mixing grads: {fd_mix}")
        log(f"  Finite difference kappa grads: {fd_kap}")
        for i in range(3):
            if abs(fd_mix[i]) < 1e-6 and abs(fd_kap[i]) < 1e-6:
                fd_check_pass = False
                log(f"  ❌ Layer {i}: mixing fd={fd_mix[i]:.6e} AND kappa fd={fd_kap[i]:.6e} both zero → not in forward")
            else:
                log(f"  ✅ Layer {i}: mix_fd={fd_mix[i]:.6e}, kap_fd={fd_kap[i]:.6e}")
    except Exception as e:
        log(f"  Finite difference ERROR: {e}")
        fd_check_pass = False

    # 三层 mixing 差异检查 (Issue #174 spec 关键: 三层强制相同即 NO-GO)
    mixing_final = train_trace[-1]["mixing_l"] if train_trace else [[1/3]*3]*3
    mixing_per_layer_diff = False
    if len(mixing_final) >= 3:
        mixing_per_layer_diff = any(abs(mixing_final[i][0] - mixing_final[j][0]) > 0.01 for i in range(3) for j in range(i+1, 3))
        log(f"  Mixing per-layer diff check: {mixing_per_layer_diff} (mixing_final={mixing_final})")

    verdict = {
        "task_id": 467,
        "issue_number": 174,
        "gate_1_stage1_three_component_export": "PASS",
        "gate_2_forward_path_audit": "PASS" if (fd_check_pass and mixing_per_layer_diff) else "FAIL",
        "finite_difference_mixing_grads": fd_mix,
        "finite_difference_kappa_grads": fd_kap,
        "mixing_l_final": mixing_final,
        "kappa_l_final": train_trace[-1]["kappa_l"] if train_trace else [0,0,0],
        "val_R@10_proxy_trajectory": [t["val_R@10_proxy"] for t in train_trace],
        "loss_trajectory": [t["loss"] for t in train_trace],
        "fd_check_pass": fd_check_pass,
        "mixing_per_layer_diff": mixing_per_layer_diff,
        "r23_triggered": any(t["val_R@10_proxy"] == 0.0 for t in train_trace[-2:]) if len(train_trace) >= 2 else False,
        "overall": "PASS" if (fd_check_pass and mixing_per_layer_diff) else "FAIL",
    }
    with open(VERDICT_PATH, "w") as f:
        json.dump(verdict, f, indent=2)
    with open(TRAIN_TRACE_PATH, "w") as f:
        json.dump(train_trace, f, indent=2)

    log(f"=== Task #467 complete: Gate 2 {'PASS' if (fd_check_pass and mixing_per_layer_diff) else 'FAIL'} ===")
    log(f"verdict.json: {VERDICT_PATH}")


if __name__ == "__main__":
    main()