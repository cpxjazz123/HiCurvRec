#!/usr/bin/env python3
"""Issue #28 [方向A Gate2] κ更新后重校准与训练稳定性收口 — 短程单 seed 诊断.

Per #28 spec:
- 验证 L0 K64 / L1 K128 / L2 K256 的 κ 更新是否同步影响代码本尺度、距离与 #47 统一公式
- 必须落盘: 每层 κ/梯度/有效步长, 代码本前后 hash, 距离差异, SID hash, checkpoint 与 decoder dtype
- 记录 backbone 漂移 (cosine) + 四组梯度 (adapter/conditioner/kappa_logits/alpha_logit)
- 短程单 seed <=5 epoch, 从 #26 T2 best_adapter (epoch 49, val=0.05) 续训, 禁止 from-0
- 证据完整且真实 canary val_R@10 >= 0.05 后才允许 Gate3->Gate4 单 seed 评估

核心新增 (vs #26 T2): 真正实现 codebook 重校准取证 (Issue #157 逻辑):
- 每 step κ 更新 (Δκ>=1e-4) -> 用新 κ 重投影 Stage2 codebook (proj_to_ball(expmap0(cb,c),c))
- 记录 codebook 投影 hash before/after (每层) + codebook norm before/after
- 每 epoch 记录 Poincaré 距离矩阵统计 (mean/max) + 与 epoch 起始的距离差异
- 证明 κ 更新确实改变了 codebook 尺度与距离度量 (Task-Geometry coupling)

禁止: 改 gradient clip / LR warmup / optimizer / 解冻策略 / 架构; 多 seed.
"""
import os
import sys
import json
import hashlib
import time
import signal
import atexit
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/model")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/data")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec")

os.environ["TRITON_CACHE_DIR"] = "/home/wlia0047/.triton/cache_issue28_gate2"
os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)

from common.stage4_decode import (
    autoregressive_predict_constrained, get_layer_ranges, compute_r_at_k,
)
from common.stage4_eval_beam20 import run_val_beam20
from dataset import GenRecDataset
from utils import proj_to_ball, expmap0, poincare_distance  # HG-Rec Poincaré utils (Issue #157)
import torch.nn as nn
import torch.nn.functional as F
from HG_Rec import HG_Rec


# ============================================================================
# Config
# ============================================================================
SEED = 42
DEVICE = "cuda:0"
CODEBOOK_SIZE = [64, 128, 256, 1]
MAX_LEN = 20
PAD_TOKEN = 0
D_MODEL = 128
BATCH_SIZE = 32
NUM_EPOCHS = 5
START_EPOCH = 0
VAL_EVAL_N = 200
LR_CONDITIONER = 1e-3
LR_LAYERNORM = 1e-4
LR_KAPPA_LOGITS = 1e-2   # #26 T2 已验证生效
DELTA_KAPPA_ASSERT = 1e-4

# 从 #26 T2 best_adapter (epoch 49, val_R@10=0.05) 续训 — 最新已确认 κ 更新状态
PRIOR_CKPT = "/home/wlia0047/ar57/wenyu/GeneRec/taskA/stage3/taskA_stage3_issue26_t2_lr_scan/best_adapter.pt"
T5_CKPT = "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_ckpt/HG_Rec_best.pth"
SID_NPY = "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_data/Instruments/Instruments_t5_hrqvae_poincare.npy"
TRAIN_PARQUET = "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_data/Instruments/train.parquet"
# Issue #30: val 改用 baseline 同 split — valid.parquet (HG-Rec/dataset/Instruments), 替代 train 尾切
VALID_PARQUET = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/valid.parquet"
STAGE2_VQ_CKPT = "/home/wlia0047/ar57/wenyu/GeneRec/taskA/stage2/taskA_stage2_kappa_sync/hrqvae_kappa_sync.ckpt"
VAL_SPLIT_RATIO = 0.2
EXPECTED_SID_SHA = "2dab29229c36a9695a11d70b61d1b80fae95a08d00e3c3675e9bf899b709508a"

PRODUCT_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/taskA/stage3/taskA_stage3_issue28_gate2_diag")
PRODUCT_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_PID_FILE = PRODUCT_DIR / "_TRAINING_PID"
LOG_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/taskA/_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "task_issue28_gate2_diag.log"
RECALIBRATE_LOG_PATH = PRODUCT_DIR / "recalibrate_log.json"
VAL_TRACE_PATH = PRODUCT_DIR / "val_trace.json"
DRIFT_PATH = PRODUCT_DIR / "backbone_drift.json"
GRAD_PATH = PRODUCT_DIR / "grad_norm_history.json"
KAPPA_TRACE_PATH = PRODUCT_DIR / "kappa_trace.json"
ADAPTER_CKPT_PATH = PRODUCT_DIR / "adapter.pt"
BEST_ADAPTER_CKPT_PATH = PRODUCT_DIR / "best_adapter.pt"
VERDICT_PATH = PRODUCT_DIR / "verdict.json"

state = {
    "killed": False, "epoch_losses": [], "val_trace": [], "kappa_history": [],
    "recalibrate_events": [], "drift_history": [], "grad_history": [],
    "dist_stats_history": [], "killed_reason": None,
}


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def flush_evidence():
    try:
        with open(VAL_TRACE_PATH, "w") as f:
            json.dump({"val_trace": state["val_trace"], "kappa_history": state["kappa_history"]}, f, indent=2, default=str)
        with open(RECALIBRATE_LOG_PATH, "w") as f:
            json.dump({"recalibrate_count": len(state["recalibrate_events"]),
                       "events": state["recalibrate_events"][-50:]}, f, indent=2, default=str)
        with open(DRIFT_PATH, "w") as f:
            json.dump({"drift_history": state["drift_history"]}, f, indent=2, default=str)
        with open(GRAD_PATH, "w") as f:
            json.dump({"grad_history": state["grad_history"]}, f, indent=2, default=str)
        with open(KAPPA_TRACE_PATH, "w") as f:
            json.dump({"kappa_history": state["kappa_history"]}, f, indent=2, default=str)
    except Exception as e:
        with open(LOG_PATH, "a") as f:
            f.write(f"[atexit flush error] {e}\n")


def signal_handler(signum, frame):
    state["killed"] = True
    state["killed_reason"] = f"signal {signum}"
    flush_evidence()
    with open(LOG_PATH, "a") as f:
        f.write(f"\n[SIGNAL {signum}] emergency flush done\n")
    sys.exit(0)


signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)
atexit.register(flush_evidence)


# ============================================================================
# Stage2 codebook 加载 + 重校准 (Issue #157 逻辑)
# ============================================================================
def load_stage2_codebooks(stage2_ckpt):
    """加载 Stage2 VQ codebook embedding (每层) + Stage2 κ."""
    d = torch.load(stage2_ckpt, map_location="cpu", weights_only=False)
    sd = d["model_state_dict"]
    cbs = []
    for l in range(3):
        w = sd[f"vq_layers.{l}.embeddings.weight"].clone()  # (K_l, e_dim)
        cbs.append(w)
    stage2_kappas = d.get("final_kappas")
    return cbs, stage2_kappas


def recalibrate_codebook_hashes(cbs, kappa_per_layer):
    """用当前 κ 重投影每层 codebook -> 返回每层投影 hash + norm.
    κ 变化 -> 投影变化 -> hash 变化, 证明 κ 影响 codebook 尺度."""
    out = []
    for l, (cb, kappa) in enumerate(zip(cbs, kappa_per_layer)):
        c = float(kappa)          # 曲率 = adapter κ (softplus>0)
        cb_h = proj_to_ball(expmap0(cb, c), c)  # (K, e_dim)
        arr = cb_h.detach().cpu().numpy().astype(np.float32)
        h = hashlib.sha256(arr.tobytes()).hexdigest()
        norm = float(cb_h.norm(dim=-1).mean())
        out.append({"layer": l, "kappa": c, "hash": h[:16], "norm": norm})
    return out


def codebook_dist_stats(cbs, kappa_per_layer):
    """每层 codebook 中心间 Poincaré 距离矩阵统计 (KxK 全对, 每 epoch 取证).
    返回每层 mean_dist / max_dist / sample of dist."""
    out = []
    for l, (cb, kappa) in enumerate(zip(cbs, kappa_per_layer)):
        c = float(kappa)
        cb_h = proj_to_ball(expmap0(cb, c), c).detach()  # (K, e)
        K = cb_h.shape[0]
        # KxK 全对 Poincaré 距离
        x = cb_h.unsqueeze(0).expand(K, K, -1)   # (K, K, e)
        y = cb_h.unsqueeze(1).expand(K, K, -1)   # (K, K, e)
        d = poincare_distance(x, y, c).squeeze(-1)  # (K, K)
        # 排除对角 (自身距离 0), 取上三角
        tri = torch.triu(d, diagonal=1)
        tri = tri[tri > 0]
        out.append({
            "layer": l, "kappa": c,
            "mean_dist": float(tri.mean()),
            "max_dist": float(tri.max()),
            "median_dist": float(tri.median()),
            "n_pairs": int(tri.numel()),
        })
    return out


# ============================================================================
# val — 已迁移到 common/stage4_eval_beam20.py::run_val_beam20 (baseline 同协议 beam20)
# ============================================================================
# ============================================================================
# backbone 漂移 (frozen HG_Rec vs wrapper T5, 复用 #27 逻辑)
# ============================================================================
def backbone_drift_check(model_wrapper, frozen_t5_state_dict, histories_batch, t5_config, device):
    import torch.nn.functional as F
    from HG_Rec import HG_Rec
    frozen_t5 = HG_Rec(t5_config)
    frozen_t5.load_state_dict(frozen_t5_state_dict)
    frozen_t5.to(device)
    frozen_t5.eval()
    model_wrapper.t5.eval()
    attention_mask = (histories_batch != PAD_TOKEN).long()
    drift_per_layer = []
    with torch.no_grad():
        x_emb_curr = model_wrapper.t5.model.shared(histories_batch)
        x_emb_frozen = frozen_t5.model.shared(histories_batch)
        hidden_curr = x_emb_curr
        hidden_frozen = x_emb_frozen
        L = histories_batch.shape[1]
        extended_mask = attention_mask[:, None, None, :].float()
        extended_mask = (1.0 - extended_mask) * torch.finfo(torch.float32).min
        for layer_i, (block_curr, block_frozen) in enumerate(zip(model_wrapper.t5.model.encoder.block, frozen_t5.model.encoder.block)):
            out_curr = block_curr(hidden_curr, attention_mask=extended_mask)[0]
            out_frozen = block_frozen(hidden_frozen, attention_mask=extended_mask)[0]
            cos = F.cosine_similarity(out_curr, out_frozen, dim=-1).mean().item()
            drift_per_layer.append({"layer": layer_i, "cosine_sim": cos})
            hidden_curr = out_curr
            hidden_frozen = out_frozen
    return drift_per_layer


def grad_norm_per_group(model_wrapper):
    """4 组 grad norm: adapter_total / conditioner / kappa_logits / alpha_logit."""
    gn = {"adapter_total": 0.0, "conditioner_total": 0.0, "kappa_logits": 0.0, "alpha_logit": 0.0}
    for n, p in model_wrapper.named_parameters():
        if p.grad is not None and torch.isfinite(p.grad).all():
            n_sq = p.grad.norm().item() ** 2
            if "kappa_logits" in n:
                gn["kappa_logits"] += n_sq
            elif "alpha_logit" in n:
                gn["alpha_logit"] += n_sq
            elif any(sub in n for sub in ("adapter.conditioner", "adapter.kappa_embed", "adapter.scale_embed", "adapter.sid_token_proj")):
                gn["conditioner_total"] += n_sq
            elif "adapter." in n:
                gn["adapter_total"] += n_sq
    for k in gn:
        gn[k] = gn[k] ** 0.5
    return gn


def main():
    with open(TRAINING_PID_FILE, "w") as f:
        f.write(str(os.getpid()) + "\n")

    log_lines = []
    log_lines.append("=" * 70)
    log_lines.append("[Issue #28 Gate2 diag] codebook 重校准取证 + 训练稳定性")
    log_lines.append("=" * 70)

    sid_sha = sha256_of(SID_NPY)
    t5_sha = sha256_of(T5_CKPT)
    prior_sha = sha256_of(PRIOR_CKPT)
    vq_sha = sha256_of(STAGE2_VQ_CKPT)
    log_lines.append(f"[SHA256] SID_NPY: {sid_sha}")
    log_lines.append(f"[SHA256] T5_CKPT: {t5_sha}")
    log_lines.append(f"[SHA256] PRIOR_CKPT: {prior_sha}")
    log_lines.append(f"[SHA256] STAGE2_VQ_CKPT: {vq_sha}")
    log_lines.append(f"[SID hash match] {sid_sha == EXPECTED_SID_SHA}")

    if sid_sha != EXPECTED_SID_SHA:
        log_lines.append("[FAIL] SID hash mismatch, abort")
        with open(LOG_PATH, "w") as f:
            f.write("\n".join(log_lines) + "\n")
        return

    # Stage2 codebook 加载
    cbs, stage2_kappas = load_stage2_codebooks(STAGE2_VQ_CKPT)
    log_lines.append(f"[Stage2 codebook] layer shapes: {[tuple(c.shape) for c in cbs]}, stage2_kappas: {stage2_kappas}")

    # Load wrapper (类定义内联自 taskA_stage3_issue24_precheck_v3, 见文件底部, 逐字节一致保 ckpt 兼容)
    WrapperCls = HG_Rec_with_BoundedAdapter
    t5_state_dict = load_t5_state_dict(T5_CKPT)
    t5_config = get_t5_config()
    model_wrapper = WrapperCls(t5_config, t5_state_dict, d_model=D_MODEL, n_layers=3, sid_dim=4)
    model_wrapper.to(DEVICE)
    frozen_t5_state_dict = t5_state_dict

    # Load prior ckpt (#26 T2 best_adapter epoch 49)
    prior_ckpt = torch.load(PRIOR_CKPT, map_location=DEVICE, weights_only=False)
    log_lines.append(f"[Load prior ckpt] epoch={prior_ckpt.get('epoch')}, val_r10={prior_ckpt.get('val_r10')}")
    filtered_state = {k: v for k, v in prior_ckpt["adapter_state_dict"].items()
                      if k in model_wrapper.adapter.state_dict()
                      and list(v.shape) == list(model_wrapper.adapter.state_dict()[k].shape)}
    model_wrapper.adapter.load_state_dict(filtered_state, strict=False)
    if "ln_state_dict" in prior_ckpt:
        model_wrapper.first_input_ln.load_state_dict(prior_ckpt["ln_state_dict"])

    kappa_logits_start = model_wrapper.adapter.kappa_logits.detach().clone()
    kappa_per_layer_start = model_wrapper.adapter.get_kappa_per_layer().detach().cpu().tolist()
    log_lines.append(f"\n[kappa_per_layer start] {kappa_per_layer_start}")
    state["kappa_history"].append({"step": 0, "epoch": START_EPOCH, "kappa_logits": kappa_logits_start.cpu().tolist(),
                                   "kappa_per_layer": kappa_per_layer_start,
                                   "codebook_hash_start": recalibrate_codebook_hashes(cbs, kappa_per_layer_start)})

    # Data
    train_ds = GenRecDataset(
        dataset_path=TRAIN_PARQUET, code_path=SID_NPY, mode="train",
        codebook_size=CODEBOOK_SIZE, max_len=MAX_LEN, PAD_TOKEN=PAD_TOKEN,
    )
    n_train = len(train_ds)
    all_histories = np.zeros((n_train, MAX_LEN * 4), dtype=np.int64)
    all_targets = np.zeros((n_train, 4), dtype=np.int64)
    for i in range(n_train):
        s = train_ds[i]
        all_histories[i] = np.concatenate([np.asarray(h, dtype=np.int64).flatten() for h in s["history"]])
        all_targets[i] = np.asarray(s["target"], dtype=np.int64).flatten()
    all_histories_t = torch.from_numpy(all_histories).to(DEVICE).long()
    all_targets_t = torch.from_numpy(all_targets).to(DEVICE).long()
    train_indices = torch.arange(0, n_train, device=DEVICE)

    # Issue #30: val 用 baseline 同 split (valid.parquet), 构建真实 val 数据 + sid_meta
    valid_ds = GenRecDataset(
        dataset_path=VALID_PARQUET, code_path=SID_NPY, mode="evaluation",
        codebook_size=CODEBOOK_SIZE, max_len=MAX_LEN, PAD_TOKEN=PAD_TOKEN,
    )
    n_val_ds = len(valid_ds)
    val_histories = np.zeros((n_val_ds, MAX_LEN * 4), dtype=np.int64)
    val_targets = np.zeros((n_val_ds, 4), dtype=np.int64)
    for i in range(n_val_ds):
        s = valid_ds[i]
        val_histories[i] = np.concatenate([np.asarray(h, dtype=np.int64).flatten() for h in s["history"]])
        val_targets[i] = np.asarray(s["target"], dtype=np.int64).flatten()
    val_histories_t = torch.from_numpy(val_histories).to(DEVICE).long()
    val_targets_t = torch.from_numpy(val_targets).to(DEVICE).long()
    _VB, _VL = val_histories_t.shape
    _vdig = val_histories_t.float()
    _vli = (torch.arange(_VL, device=DEVICE) % 4).float().unsqueeze(0).expand(_VB, -1) / 4.0
    _vpi = (torch.arange(_VL, device=DEVICE) // 4).float().unsqueeze(0).expand(_VB, -1) / MAX_LEN
    _vpf = (_vdig == PAD_TOKEN).float()
    val_sid_meta = torch.stack([_vdig / 1025.0, _vli, _vpi, _vpf], dim=-1)
    val_kappa_meta = torch.zeros(_VB, 3, dtype=torch.float32, device=DEVICE)
    val_scale_meta = torch.ones(_VB, 3, dtype=torch.float32, device=DEVICE)
    log_lines.append(f"[Issue #30 val] valid.parquet n={n_val_ds} (替代 train 尾切), val_eval_n={VAL_EVAL_N}")

    layer_ranges = get_layer_ranges(CODEBOOK_SIZE)

    # Optimizer: kappa_logits 独立 param group (保留 #26 T2 生效配置)
    kappa_logits_param = model_wrapper.adapter.kappa_logits
    alpha_logit_param = model_wrapper.adapter.alpha_logit
    other_cond_params = [p for n, p in model_wrapper.adapter.named_parameters() if n not in ("kappa_logits", "alpha_logit")]
    layernorm_params = list(model_wrapper.first_input_ln.parameters())
    optimizer = torch.optim.Adam([
        {"params": other_cond_params, "lr": LR_CONDITIONER},
        {"params": [kappa_logits_param, alpha_logit_param], "lr": LR_KAPPA_LOGITS},
        {"params": layernorm_params, "lr": LR_LAYERNORM},
    ])
    log_lines.append(f"\n[Optimizer] 3 param groups:")
    log_lines.append(f"  other_cond_params (LR={LR_CONDITIONER})")
    log_lines.append(f"  kappa_logits + alpha_logit (LR={LR_KAPPA_LOGITS})")
    log_lines.append(f"  layernorm (LR={LR_LAYERNORM})")

    rng = torch.Generator().manual_seed(42 + 8)
    n_batches = (len(train_indices) + BATCH_SIZE - 1) // BATCH_SIZE
    best_val_r10 = -1.0
    best_epoch = -1
    start_time = time.time()

    for epoch in range(START_EPOCH, START_EPOCH + NUM_EPOCHS):
        epoch_losses = []
        cond_grad_norms = []
        ln_grad_norms = []
        kappa_logits_grad_norms = []
        kappa_updates = 0
        nan_inf = False
        # epoch 起始距离统计 (用 epoch 前 κ)
        kappa_pre_epoch = model_wrapper.adapter.get_kappa_per_layer().detach().cpu().tolist()
        dist_pre = codebook_dist_stats(cbs, kappa_pre_epoch)
        epoch_indices = torch.randperm(len(train_indices), generator=rng).to(DEVICE)
        for batch_idx in range(n_batches):
            start = batch_idx * BATCH_SIZE
            end = min(start + BATCH_SIZE, len(train_indices))
            bi = epoch_indices[start:end]
            ht = all_histories_t[train_indices[bi]]
            tt = all_targets_t[train_indices[bi]]
            am = (ht != PAD_TOKEN).long()
            B = ht.shape[0]
            L_flat = MAX_LEN * 4
            digit_values = ht.float()
            layer_idx = torch.arange(L_flat, device=DEVICE).float() % 4
            layer_idx = layer_idx.unsqueeze(0).expand(B, -1)
            pos_in_history = torch.arange(L_flat, device=DEVICE).float() // 4 / MAX_LEN
            pos_in_history = pos_in_history.unsqueeze(0).expand(B, -1)
            padding_flag = (digit_values == PAD_TOKEN).float()
            sid_meta = torch.stack([digit_values / 1025.0, layer_idx / 4.0, pos_in_history, padding_flag], dim=-1)
            kappa_meta = torch.zeros(B, 3, dtype=torch.float32, device=DEVICE)
            scale_meta = torch.ones(B, 3, dtype=torch.float32, device=DEVICE)
            optimizer.zero_grad()
            output, _, alpha = model_wrapper(ht, attention_mask=am, labels=tt, sid_meta=sid_meta,
                                             kappa_meta=kappa_meta, scale_meta=scale_meta)
            loss = output.loss if hasattr(output, "loss") else output[0]
            if not torch.isfinite(loss):
                nan_inf = True
                continue
            loss.backward()
            kl_before = kappa_logits_param.detach().clone()
            kappa_before_val = model_wrapper.adapter.get_kappa_per_layer().detach().cpu().tolist()
            cond_grad_norm = sum((p.grad.norm().item() ** 2) for p in other_cond_params if p.grad is not None and torch.isfinite(p.grad).all()) ** 0.5
            ln_grad_norm = sum((p.grad.norm().item() ** 2) for p in layernorm_params if p.grad is not None and torch.isfinite(p.grad).all()) ** 0.5
            kl_grad = kappa_logits_param.grad
            kl_grad_norm = kl_grad.norm().item() if kl_grad is not None and torch.isfinite(kl_grad).all() else 0.0
            optimizer.step()
            kl_after = kappa_logits_param.detach().clone()
            delta = (kl_after - kl_before).abs().max().item()
            if delta < DELTA_KAPPA_ASSERT and kl_grad_norm > 0:
                log_lines.append(f"  [WARN] epoch={epoch} batch={batch_idx} kappa_delta={delta:.6e} < {DELTA_KAPPA_ASSERT} (grad={kl_grad_norm:.6e})")
            epoch_losses.append(loss.item())
            cond_grad_norms.append(cond_grad_norm)
            ln_grad_norms.append(ln_grad_norm)
            kappa_logits_grad_norms.append(kl_grad_norm)

            # === #28 核心: κ 更新后同步重校准 codebook (Issue #157) ===
            if delta >= DELTA_KAPPA_ASSERT:
                kappa_updates += 1
                kappa_after_val = model_wrapper.adapter.get_kappa_per_layer().detach().cpu().tolist()
                cb_before = recalibrate_codebook_hashes(cbs, kappa_before_val)   # 更新前 κ 投影
                cb_after = recalibrate_codebook_hashes(cbs, kappa_after_val)     # 更新后 κ 投影
                hash_changed = [1 if cb_before[l]["hash"] != cb_after[l]["hash"] else 0 for l in range(3)]
                state["recalibrate_events"].append({
                    "epoch": epoch, "batch_idx": batch_idx,
                    "kappa_per_layer_before": kappa_before_val,
                    "kappa_per_layer_after": kappa_after_val,
                    "delta_kappa_max": delta,
                    "codebook_hash_before": [e["hash"] for e in cb_before],
                    "codebook_hash_after": [e["hash"] for e in cb_after],
                    "codebook_norm_before": [e["norm"] for e in cb_before],
                    "codebook_norm_after": [e["norm"] for e in cb_after],
                    "hash_changed_per_layer": hash_changed,
                })
            if state["killed"]:
                flush_evidence()
                return
        avg_loss = np.mean(epoch_losses) if epoch_losses else float("nan")
        avg_cond = np.mean(cond_grad_norms) if cond_grad_norms else 0.0
        avg_ln = np.mean(ln_grad_norms) if ln_grad_norms else 0.0
        avg_kl_grad = np.mean(kappa_logits_grad_norms) if kappa_logits_grad_norms else 0.0
        kappa_per_layer_now = model_wrapper.adapter.get_kappa_per_layer().detach().cpu().tolist()
        delta_from_start = [abs(a - b) for a, b in zip(kappa_per_layer_now, kappa_per_layer_start)]
        elapsed = time.time() - start_time
        msg = f"  [epoch {epoch}] loss={avg_loss:.4f}, cond_grad={avg_cond:.4e}, ln_grad={avg_ln:.4e}, kl_grad={avg_kl_grad:.4e}, kappa_updates={kappa_updates}, kappa={kappa_per_layer_now}, delta_from_start={delta_from_start}, nan_inf={nan_inf}, elapsed={elapsed/60:.1f}m"
        log_lines.append(msg)
        state["kappa_history"].append({"step": batch_idx + 1, "epoch": epoch,
                                       "kappa_per_layer": kappa_per_layer_now,
                                       "kappa_logits": kappa_logits_param.detach().cpu().tolist(),
                                       "delta_from_start": delta_from_start,
                                       "kl_grad_mean": avg_kl_grad, "kappa_updates": kappa_updates})

        # epoch 末距离统计 + 距离差异 (vs epoch 起始)
        dist_post = codebook_dist_stats(cbs, kappa_per_layer_now)
        dist_delta = [abs(d_post["mean_dist"] - d_pre["mean_dist"]) for d_post, d_pre in zip(dist_post, dist_pre)]
        state["dist_stats_history"].append({
            "epoch": epoch,
            "dist_pre": dist_pre, "dist_post": dist_post, "dist_delta_mean_per_layer": dist_delta,
        })
        log_lines.append(f"  [dist] kappa_pre={kappa_pre_epoch}, kappa_post={kappa_per_layer_now}, dist_delta_mean={[f'{x:.6f}' for x in dist_delta]}")

        # backbone 漂移 (每 epoch, 固定 batch 0)
        probe_ht = all_histories_t[train_indices[:BATCH_SIZE]]
        drift = backbone_drift_check(model_wrapper, frozen_t5_state_dict, probe_ht, t5_config, DEVICE)
        state["drift_history"].append({"epoch": epoch, "drift": drift})
        min_cos = min(d["cosine_sim"] for d in drift)
        log_lines.append(f"  [drift] min_cosine_sim={min_cos:.4f} ({len(drift)} layers)")

        # 4 组 grad norm
        gn = grad_norm_per_group(model_wrapper)
        state["grad_history"].append({"epoch": epoch, "grad_norm": gn})
        log_lines.append(f"  [grad] adapter={gn['adapter_total']:.4e}, conditioner={gn['conditioner_total']:.4e}, kappa_logits={gn['kappa_logits']:.4e}, alpha_logit={gn['alpha_logit']:.4e}")

        # val (Issue #30: baseline 同协议 beam20, 替代 greedy 单候选 — 真实 R@10)
        val_r10, val_in_range_pct = run_val_beam20(
            model_wrapper, val_histories_t, val_targets_t, VAL_EVAL_N, BATCH_SIZE, DEVICE,
            val_sid_meta=val_sid_meta, val_kappa_meta=val_kappa_meta, val_scale_meta=val_scale_meta,
        )
        state["val_trace"].append({"epoch": epoch, "val_r10": val_r10, "val_in_range_pct": val_in_range_pct,
                                   "kappa_per_layer": kappa_per_layer_now})
        if val_r10 > best_val_r10:
            best_val_r10 = val_r10
            best_epoch = epoch
            if BEST_ADAPTER_CKPT_PATH.exists():
                BEST_ADAPTER_CKPT_PATH.unlink()
            torch.save({
                "adapter_state_dict": model_wrapper.adapter.state_dict(),
                "ln_state_dict": model_wrapper.first_input_ln.state_dict(),
                "epoch": epoch, "val_r10": val_r10, "alpha": model_wrapper.adapter.get_alpha(),
                "lr_kappa_logits": LR_KAPPA_LOGITS,
            }, BEST_ADAPTER_CKPT_PATH)
        log_lines.append(f"  [val @ epoch {epoch}] val_R@10={val_r10:.4f}, in-range={val_in_range_pct*100:.1f}%, best={best_val_r10:.4f} @ epoch {best_epoch}")

        if ADAPTER_CKPT_PATH.exists():
            ADAPTER_CKPT_PATH.unlink()
        torch.save({
            "adapter_state_dict": model_wrapper.adapter.state_dict(),
            "ln_state_dict": model_wrapper.first_input_ln.state_dict(),
            "epoch": epoch,
        }, ADAPTER_CKPT_PATH)
        flush_evidence()

    # ================= verdict 判定 =================
    final_kappa = model_wrapper.adapter.get_kappa_per_layer().detach().cpu().tolist()
    final_delta = [abs(a - b) for a, b in zip(final_kappa, kappa_per_layer_start)]

    # c3: κ 实际数值更新
    c3_pass = max(final_delta) >= 1e-3 and all(x >= 1e-5 for x in final_delta)

    # c4: codebook 重校准有效
    n_triggers = len(state["recalibrate_events"])
    # step 0 的 kappa_history 首条无 kappa_updates key (仅记录起始状态), 用 .get 安全取
    n_updates = sum(e.get("kappa_updates", 0) for e in state["kappa_history"])
    hash_changed_any = any(
        any(e["hash_changed_per_layer"]) for e in state["recalibrate_events"]
    ) if state["recalibrate_events"] else False
    dist_deltas = [max(e["dist_delta_mean_per_layer"]) for e in state["dist_stats_history"]] if state["dist_stats_history"] else []
    dist_changed_any = any(d > 1e-6 for d in dist_deltas)
    c4_pass = (n_triggers >= 1) and hash_changed_any and dist_changed_any

    # c5: val_R@10 >= 0.05 (真实 canary 水平)
    c5_pass = best_val_r10 >= 0.05 and BEST_ADAPTER_CKPT_PATH.exists()

    # backbone 无漂移
    min_cos_all = min(e["cosine_sim"] for e in [x for h in state["drift_history"] for x in h["drift"]]) if state["drift_history"] else 0.0
    backbone_ok = min_cos_all >= 0.99

    final_verdict = "PASS" if (c3_pass and c4_pass and c5_pass and backbone_ok) else "FAIL"
    verdict = {
        "issue": 28,
        "step": "[方向A Gate2] κ更新后重校准与训练稳定性收口 — codebook 重校准取证诊断",
        "gate": "Gate2 诊断",
        "verdict": final_verdict,
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "c3_kappa_actual_update": {"pass": c3_pass, "kappa_start": kappa_per_layer_start, "kappa_end": final_kappa,
                                   "delta_per_layer": final_delta, "max_delta": max(final_delta)},
        "c4_codebook_recalibrate": {
            "pass": c4_pass,
            "n_recalibrate_triggers": n_triggers,
            "n_kappa_updates": n_updates,
            "count_match": n_triggers == n_updates,
            "hash_changed_any": hash_changed_any,
            "dist_delta_max_per_epoch": dist_deltas,
            "dist_changed_any": dist_changed_any,
            "recalibrate_count": n_triggers,
        },
        "c5_val_canary": {"pass": c5_pass, "best_val_r10": best_val_r10, "best_epoch": best_epoch,
                          "best_adapter_exists": BEST_ADAPTER_CKPT_PATH.exists()},
        "backbone_drift": {"pass": backbone_ok, "min_cosine_sim": min_cos_all, "n_epochs_drifted": len(state["drift_history"])},
        "config": {
            "prior_ckpt": PRIOR_CKPT, "prior_epoch": prior_ckpt.get("epoch"), "prior_val_r10": prior_ckpt.get("val_r10"),
            "lr_kappa_logits": LR_KAPPA_LOGITS, "num_epochs": NUM_EPOCHS,
            "codebook_size": CODEBOOK_SIZE, "stage2_vq_ckpt": STAGE2_VQ_CKPT,
            "stage2_vq_sha256": vq_sha, "stage2_final_kappas": stage2_kappas,
        },
        "dtype_check": {
            "model_dtype": str(next(model_wrapper.parameters()).dtype),
            "lm_head_dtype": str(model_wrapper.t5.model.lm_head.weight.dtype),
            "adapter_dtype": str(next(model_wrapper.adapter.parameters()).dtype),
        },
        "sha256s": {"sid_sha256": sid_sha, "sid_sha_match": sid_sha == EXPECTED_SID_SHA,
                    "t5_ckpt_sha256": t5_sha, "prior_ckpt_sha256": prior_sha,
                    "best_adapter_sha256": sha256_of(BEST_ADAPTER_CKPT_PATH) if BEST_ADAPTER_CKPT_PATH.exists() else None},
        "best_val_r10": best_val_r10,
        "best_epoch": best_epoch,
        "best_adapter_path": str(BEST_ADAPTER_CKPT_PATH),
        "best_adapter_exists": BEST_ADAPTER_CKPT_PATH.exists(),
        "evidence_persisted": {
            "recalibrate_log": str(RECALIBRATE_LOG_PATH),
            "val_trace": str(VAL_TRACE_PATH),
            "backbone_drift": str(DRIFT_PATH),
            "grad_norm_history": str(GRAD_PATH),
            "kappa_trace": str(KAPPA_TRACE_PATH),
            "dist_stats_history": state["dist_stats_history"],
        },
        "verdict_path": str(VERDICT_PATH),
    }
    with open(VERDICT_PATH, "w") as f:
        json.dump(verdict, f, indent=2, default=str)
    log_lines.append(f"\n[Verdict written] {VERDICT_PATH}")
    log_lines.append(f"[c3] {c3_pass} | [c4] {c4_pass} (triggers={n_triggers}, updates={n_updates}, hash_changed={hash_changed_any}, dist_changed={dist_changed_any}) | [c5] {c5_pass} (val={best_val_r10:.4f}) | backbone_ok={backbone_ok} (min_cos={min_cos_all:.4f})")
    log_lines.append(f"[FINAL] {final_verdict}")

    print("\n".join(log_lines), flush=True)




# ============================================================================
# 内联自 archive/taskA_stage3/taskA_stage3_issue24_precheck_v3.py (逐字节, ckpt 兼容)
# ============================================================================
class BoundedKappaScaleConditioner(nn.Module):
    """Issue #177: kappa + sync codebook scale  -> .

     vs #159 (#451):
    - alpha = softplus(alpha_logit).clamp(max=ALPHA_MAX)  alpha <= 0.5 ()
    - direction  tanh bounded [-1, 1]
    - residual = alpha * direction,  |residual| <= ALPHA_MAX

     #159 (Task #451) alpha  18.75 .
    """

    def __init__(self, d_model=128, n_layers=3, sid_dim=4, alpha_init_logit=-10.0, alpha_max=0.5,
                 kappa_init_logit_per_layer=None):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.sid_dim = sid_dim
        self.alpha_max = alpha_max

        # #22 spec:  learnable kappa (per-layer  init )
        if kappa_init_logit_per_layer is None:
            kappa_init_logit_per_layer = [0.5413] * n_layers  # default -> softplus  1.0 
        self.kappa_logits = nn.Parameter(
            torch.tensor(kappa_init_logit_per_layer, dtype=torch.float32)
        )

        self.kappa_embed = nn.Linear(1, d_model)
        self.scale_embed = nn.Linear(1, d_model)
        self.sid_token_proj = nn.Linear(sid_dim, d_model)
        self.conditioner = nn.Sequential(
            nn.Linear(d_model * 5, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
            nn.Tanh(),
        )
        self.alpha_logit = nn.Parameter(torch.tensor(alpha_init_logit, dtype=torch.float32))

    def get_alpha(self):
        return F.softplus(self.alpha_logit).clamp(max=self.alpha_max).item()

    def get_kappa_per_layer(self):
        """ learnable kappa (per-layer softplus of kappa_logits)."""
        return F.softplus(self.kappa_logits)  # (n_layers,)

    def forward(self, x_emb, sid_meta, kappa_meta, scale_meta):
        # #24 spec:  layer  mean(dim=1) ,  kappa 
        # kappa  Linear(1, d_model) -> mean(dim=1) -> Linear,  broadcast  token 
        # dL/dkappa  3*d_model Linear ,  3-scalar Linear 
        B, L = x_emb.shape[0], x_emb.shape[1]
        kappa_per_layer = self.get_kappa_per_layer()  # (n_layers,)
        kappa_e = self.kappa_embed(kappa_per_layer.unsqueeze(-1).unsqueeze(0).expand(B, -1, -1))  # (B, 3, d_model)
        #  collapse:  kappa  (B, L, 3*d_model) ->  conditioner 
        kappa_e_token = kappa_e.unsqueeze(1).expand(-1, L, -1, -1).reshape(B, L, 3 * self.d_model)
        # scale  mean (scale , #24 spec )
        scale_e = self.scale_embed(scale_meta.unsqueeze(-1))  # (B, 3, d_model)
        scale_summary = scale_e.mean(dim=1, keepdim=True).expand(-1, L, -1)  # (B, L, d_model)
        sid_e = self.sid_token_proj(sid_meta)  # (B, L, d_model)
        combined = torch.cat([sid_e, kappa_e_token, scale_summary], dim=-1)  # (B, L, 5*d_model)
        direction = self.conditioner(combined)  # conditioner  Linear(5*d_model, 2*d_model)
        alpha = F.softplus(self.alpha_logit).clamp(max=self.alpha_max)
        residual = alpha * direction
        return residual, alpha


class HG_Rec_with_BoundedAdapter(nn.Module):
    def __init__(self, t5_config, t5_state_dict, d_model=128, n_layers=3, sid_dim=4,
                 kappa_init_logit_per_layer=None):
        super().__init__()
        self.t5 = HG_Rec(t5_config)
        self.t5.load_state_dict(t5_state_dict)
        for p in self.t5.parameters():
            p.requires_grad = False
        self.adapter = BoundedKappaScaleConditioner(
            d_model=d_model, n_layers=n_layers, sid_dim=sid_dim,
            kappa_init_logit_per_layer=kappa_init_logit_per_layer,
        )
        first_input_ln = self.t5.model.encoder.block[0].layer[0].layer_norm
        for p in first_input_ln.parameters():
            p.requires_grad = True
        self.first_input_ln = first_input_ln

    def forward(self, input_ids, attention_mask=None, labels=None, sid_meta=None, kappa_meta=None, scale_meta=None):
        x_emb = self.t5.model.shared(input_ids)
        if kappa_meta is None:
            kappa_meta = torch.zeros(input_ids.shape[0], 3, dtype=torch.float32, device=input_ids.device)
        if scale_meta is None:
            scale_meta = torch.ones(input_ids.shape[0], 3, dtype=torch.float32, device=input_ids.device)
        if sid_meta is None:
            sid_meta = torch.zeros(*input_ids.shape, 4, dtype=torch.float32, device=input_ids.device)
        residual, alpha = self.adapter(x_emb, sid_meta, kappa_meta, scale_meta)
        x_emb_with_residual = x_emb + residual
        x_emb_with_residual = self.first_input_ln(x_emb_with_residual)
        if labels is not None:
            return self.t5.model(
                inputs_embeds=x_emb_with_residual,
                attention_mask=attention_mask,
                labels=labels,
            ), None, alpha
        else:
            return self.t5.model(inputs_embeds=x_emb_with_residual, attention_mask=attention_mask), None, alpha


def get_t5_config():
    return {
        "num_layers": 6, "num_decoder_layers": 4, "d_model": D_MODEL,
        "d_ff": 1024, "num_heads": 6, "d_kv": 64,
        "dropout_rate": 0.1, "vocab_size": 1025,
        "pad_token_id": 0, "eos_token_id": 1,
        "feed_forward_proj": "relu",
    }


def load_t5_state_dict(ckpt_path):
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif "model" in state_dict:
        state_dict = state_dict["model"]
    return state_dict


def load_wrapper_cls():
    return HG_Rec_with_BoundedAdapter, get_t5_config

if __name__ == "__main__":
    main()
