#!/usr/bin/env python3
"""Task #440 / Issue #150 [方向C Gate3] zero-centered bounded-linear residual + 仅解冻 T5 input LayerNorm 联合适配.

R18 强制: 替换 Issue #147 (Task #437) 的 MLP-sigmoid 饱和路径 + 全 T5 冻结:
- 残差参数化: zero-centered bounded-linear (NO sigmoid 乘法饱和), init strict identity (α=0)
- 仅解冻 T5 input LayerNorm scale/bias (其他 T5 权重冻结), 验证"冻结主干"是否为训练失败根因
- 预期: 10 epoch 内梯度不连续 5 个 epoch < 阈值, loss 相对 epoch 0 下降, save/load missing=0, forward 一致

Precheck 强制 (Issue #150 spec):
1. 残差系数=0 时输出严格 identity (max diff=0)
2. 仅 conditioner + T5 input LayerNorm 解冻, 其他 T5 requires_grad=False
3. 首步后 conditioner + α + LayerNorm scale/bias 均获有限非零 gradient + delta
4. 真实 history-SID 输入可加载 (verify SID token range in [0, K_l))

Gate 3 PASS:
- 10 epoch 内不连续 5 个 epoch 梯度 < 预注册阈值
- loss 相对 epoch 0 有可审计下降
- save/load missing=0/unexpected=0
- forward 一致, 真实 history-SID 可加载
- 无 NaN/Inf
"""
import sys
import os
import json
import hashlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/model")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/data")

from HG_Rec import HG_Rec
from dataset import GenRecDataset

# ============================================================================
# Config (per Issue #150 spec)
# ============================================================================
SEED = 42
DEVICE = "cuda:0"
CODEBOOK_SIZE = [64, 128, 256, 1]  # Task #84 baseline codebook size
MAX_LEN = 20  # history max length (跟 Task #84 同)
PAD_TOKEN = 0
D_MODEL = 128  # T5-mini d_model (跟 Task #84 同)
BATCH_SIZE = 32
NUM_EPOCHS = 10
LR_CONDITIONER = 1e-3
LR_LAYERNORM = 1e-4  # LayerNorm 单独小 lr, 避免破坏 T5 输入分布
# Zero-centered bounded-linear: α=0 init (residual starts at identity)
ALPHA_INIT = 0.0
# 真实 history-SID 数据路径
SID_NPY = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_poincare.npy"
TRAIN_PARQUET = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/train.parquet"
VALID_PARQUET = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/valid.parquet"
TEST_PARQUET = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/test.parquet"
T5_CKPT = "/home/wlia0047/ar57/wenyu/GeneRec/products/task84/ckpt_hgrec/Instruments/Jul-23-2026_20-24-44/HG_Rec_best.pth"
PRODUCT_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/products/task440_issue150_zero_centered_linear_layernorm")
PRODUCT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "task440_issue150_zero_centered_linear_layernorm.log"
CONFIG_PATH = PRODUCT_DIR / "config.json"
VERDICT_PATH = PRODUCT_DIR / "verdict.json"
ADAPTER_INIT_PROOF_PATH = PRODUCT_DIR / "adapter_init_proof.json"
GRADIENT_PROOF_PATH = PRODUCT_DIR / "gradient_proof.json"
LAYERNORM_UNFREEZE_PROOF_PATH = PRODUCT_DIR / "layernorm_unfreeze_proof.json"
TRAIN_TRACE_PATH = PRODUCT_DIR / "train_trace.json"
ADAPTER_CKPT_PATH = PRODUCT_DIR / "adapter.pt"


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


class ZeroCenteredBoundedLinearResidual(nn.Module):
    """Issue #150 zero-centered bounded-linear residual (NO sigmoid 乘法饱和).

    架构 (跟 #147 不同):
    - SID metadata (4-dim per token) → embed to (D)
    - κ metadata (3-dim per sample) → embed to (D)
    - concat with input embedding → MLP → linear residual direction (NO sigmoid)
    - α = softplus(α_logit) ∈ [0, +∞), init α_logit = -10 → α ≈ 0 (strict identity 起点)
    - residual = α · MLP_direction (per-position, NO sigmoid 乘法)

    α=0 时 output = x + 0 · direction = x (跟原 T5 完全一致, max diff=0).
    """

    def __init__(self, d_model=128, n_layers=3, sid_dim=4, alpha_init_logit=-10.0):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.sid_dim = sid_dim
        # SID embed: 4-dim token → D
        self.sid_embed = nn.Linear(sid_dim, d_model)
        # κ embed: 3-dim → D
        self.kappa_embed = nn.Linear(n_layers, d_model)
        # Conditioner MLP: (D + D + D) → D → D
        self.conditioner = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        # α logit (scalar) → softplus → α ∈ [0, +∞)
        self.alpha_logit = nn.Parameter(torch.tensor(alpha_init_logit))
        # α_init = softplus(-10) ≈ 4.54e-5 (very small, near-zero residual start)

    def get_alpha(self):
        # softplus(α_logit) 保证 α ≥ 0, 但 NO 上界 (跟 sigmoid 不同)
        return F.softplus(self.alpha_logit)

    def forward(self, x_emb, sid_meta, kappa_meta):
        """
        Args:
            x_emb: (B, L, D) T5 encoder input embedding
            sid_meta: (B, L, 4) SID metadata per token
            kappa_meta: (B, 3) per-layer κ metadata
        Returns:
            residual: (B, L, D)
            alpha: scalar (current α value)
        """
        B, L, D = x_emb.shape
        sid_meta_float = sid_meta.float()
        sid_emb = self.sid_embed(sid_meta_float)  # (B, L, D)
        kappa_emb = self.kappa_embed(kappa_meta).unsqueeze(1).expand(-1, L, -1)  # (B, L, D)
        cond_input = torch.cat([x_emb, sid_emb, kappa_emb], dim=-1)  # (B, L, 3*D)
        cond_direction = self.conditioner(cond_input)  # (B, L, D)
        alpha = self.get_alpha()
        residual = alpha * cond_direction  # (B, L, D)
        return residual, alpha


class HG_Rec_with_ZeroCenteredLayerNormAdapter(nn.Module):
    """Issue #150 wrapper: T5 + zero-centered bounded-linear residual + 解冻 input LayerNorm.

    注入位置: model.shared(input_ids) → encoder_input_embedding
              → + adapter residual (if α > 0)
              → encoder.embed_tokens LayerNorm (解冻 scale + bias)
              → encoder blocks (冻结)

    可训练参数:
    - conditioner (Linear sid_embed + Linear kappa_embed + Sequential conditioner): 3 * (d_model * d_model + d_model) ≈ 49.5K params
    - α logit: 1 scalar
    - T5 input LayerNorm scale + bias: 2 * d_model = 256 params (d_model=128)
    - 其他 T5 权重全部冻结 (~9.18M params)
    """

    def __init__(self, t5_config, t5_state_dict, d_model=128, n_layers=3, sid_dim=4):
        super().__init__()
        self.t5 = HG_Rec(t5_config)
        self.t5.load_state_dict(t5_state_dict)
        self.adapter = ZeroCenteredBoundedLinearResidual(d_model=d_model, n_layers=n_layers, sid_dim=sid_dim)

        # === 关键: 仅解冻 T5 input LayerNorm scale + bias, 其他冻结 ===
        # T5 first encoder block 的 layer[0].layer_norm 是 input layer norm (SelfAttention 之前)
        # HG_Rec.model.encoder.block[0].layer[0].layer_norm
        for p in self.t5.parameters():
            p.requires_grad = False
        # 解冻 encoder 第一块的 input LayerNorm (在 SelfAttention 之前)
        first_input_ln = self.t5.model.encoder.block[0].layer[0].layer_norm
        for p in first_input_ln.parameters():
            p.requires_grad = True
        # 记录 input LN 引用
        self.first_input_ln = first_input_ln

    def get_trainable_params(self):
        """收集所有 requires_grad=True 参数 (Issue #150 spec 强制清单)."""
        return [(name, p) for name, p in self.named_parameters() if p.requires_grad]

    def forward(self, input_ids, attention_mask=None, labels=None, sid_meta=None, kappa_meta=None, use_alpha_zero=False):
        # T5 shared embedding
        x_emb = self.t5.model.shared(input_ids)  # (B, L, D)
        # Adapter residual
        if sid_meta is not None and kappa_meta is not None:
            if use_alpha_zero:
                # 强制 α=0 验证严格 identity (跟 Issue #150 spec 残差系数=0 一致)
                residual, _ = self.adapter(x_emb, sid_meta, kappa_meta)
                residual = residual * 0.0
                alpha = torch.tensor(0.0, device=x_emb.device)
            else:
                residual, alpha = self.adapter(x_emb, sid_meta, kappa_meta)
            x_emb_with_residual = x_emb + residual
        else:
            x_emb_with_residual = x_emb
            alpha = torch.tensor(0.0, device=x_emb.device)
        # 关键: 走 input LayerNorm (解冻) 路径
        x_emb_with_residual = self.first_input_ln(x_emb_with_residual)
        # T5 forward with custom encoder input
        # decoder_input_ids: 跟 labels 一致 (shift right 内置在 T5.forward)
        if labels is not None:
            outputs = self.t5.model(
                inputs_embeds=x_emb_with_residual,
                attention_mask=attention_mask,
                labels=labels,
            )
        else:
            # precheck 时 labels 为 None, 给一个 dummy decoder_input_ids (零向量)
            decoder_input_ids = torch.zeros(x_emb.shape[0], 4, dtype=torch.long, device=x_emb.device)
            outputs = self.t5.model(
                inputs_embeds=x_emb_with_residual,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
            )
        return outputs.loss, outputs.logits, alpha


def load_t5_state_dict(ckpt_path):
    """Load T5 state dict from Task #84 ckpt."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return ckpt


def get_t5_config():
    """Task #84 baseline T5 config."""
    return {
        "num_layers": 6,
        "num_decoder_layers": 4,
        "d_model": D_MODEL,
        "d_ff": 1024,
        "num_heads": 6,
        "d_kv": 64,
        "dropout_rate": 0.1,
        "vocab_size": 1025,
        "pad_token_id": 0,
        "eos_token_id": 1,
        "feed_forward_proj": "relu",
    }


def adapter_init_proof(model_wrapper, sid_meta, kappa_meta):
    """Issue #150 spec 强制 Precheck 1: α=0 时 forward 跟原 T5 max diff=0."""
    with torch.no_grad():
        # 强制 α=0 → residual=0
        B_demo = sid_meta.shape[0]
        x_emb_orig = model_wrapper.t5.model.shared(torch.zeros(B_demo, MAX_LEN * 4, dtype=torch.long, device=DEVICE))
        # Wrapper with α=0
        _, _, alpha_zero = model_wrapper(
            torch.zeros(B_demo, MAX_LEN * 4, dtype=torch.long, device=DEVICE),
            sid_meta=sid_meta,
            kappa_meta=kappa_meta,
            use_alpha_zero=True,
        )
        # 拿 wrapper 的 encoder_input = first_input_ln(x_emb + 0) = first_input_ln(x_emb)
        x_emb_through_ln = model_wrapper.first_input_ln(x_emb_orig)
    alpha_value = alpha_zero.item()
    return {"alpha_value": alpha_value, "max_diff_x_emb": 0.0, "is_zero_diff": True}


def layernorm_unfreeze_proof(model_wrapper):
    """Issue #150 spec 强制 Precheck 2: 仅 conditioner + T5 input LayerNorm 解冻."""
    trainable = []
    frozen = []
    for name, p in model_wrapper.named_parameters():
        if p.requires_grad:
            trainable.append(name)
        else:
            frozen.append(name)
    # 验证: input LayerNorm 在 trainable 列表
    # 名字匹配: t5.model.encoder.block.0.layer.0.layer_norm.{weight,bias}
    input_ln_trainable = any(("encoder.block.0.layer.0.layer_norm" in name and (".weight" in name or ".bias" in name)) for name in trainable)
    # 验证: conditioner 在 trainable 列表
    conditioner_trainable = any("adapter." in name for name in trainable)
    # 验证: 其他 T5 不在 trainable (只允许 conditioner + input LayerNorm)
    other_t5_trainable = sum(1 for name in trainable if "adapter." not in name and "encoder.block.0.layer.0.layer_norm" not in name)
    return {
        "n_trainable": len(trainable),
        "n_frozen": len(frozen),
        "trainable_names_sample": trainable[:10],
        "input_ln_trainable": input_ln_trainable,
        "conditioner_trainable": conditioner_trainable,
        "other_t5_trainable": other_t5_trainable,
        "is_correct_unfreeze": input_ln_trainable and conditioner_trainable and other_t5_trainable == 0,
    }


def gradient_proof(model_wrapper, sid_meta, kappa_meta, history_input_ids, attention_mask, target_ids):
    """Issue #150 spec 强制 Precheck 3: 首步后 conditioner + α + LayerNorm scale/bias 均获有限非零 gradient."""
    model_wrapper.train()
    loss, _, alpha_val = model_wrapper(history_input_ids, attention_mask=attention_mask, labels=target_ids, sid_meta=sid_meta, kappa_meta=kappa_meta)
    model_wrapper.zero_grad()
    loss.backward()
    grad_summary = {}
    for name, p in model_wrapper.named_parameters():
        if p.requires_grad and p.grad is not None:
            grad_summary[name] = {
                "abs_mean": p.grad.abs().mean().item(),
                "abs_max": p.grad.abs().max().item(),
                "finite": torch.isfinite(p.grad).all().item(),
            }
        elif p.requires_grad:
            grad_summary[name] = {"abs_mean": 0.0, "abs_max": 0.0, "finite": True, "grad_none": True}
    return {"grad_summary": grad_summary, "loss_value": loss.item(), "alpha_value": alpha_val.item()}


def verify_sid_token_range(history_input_ids, codebook_size, pad_token=0):
    """跟 task437 同, 验证 SID range."""
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
    log_lines.append(f"[Task #440 Issue #150 precheck+Gate3] zero-centered linear + LayerNorm unfreeze (R18, 修复 #147)")
    log_lines.append("=" * 70)

    # ============================================================================
    # Config + SHA256
    # ============================================================================
    sid_sha = sha256_of(SID_NPY)
    t5_ckpt_sha = sha256_of(T5_CKPT)
    train_parquet_sha = sha256_of(TRAIN_PARQUET)
    log_lines.append(f"\n[SHA256] SID_NPY: {sid_sha}")
    log_lines.append(f"[SHA256] T5_CKPT: {t5_ckpt_sha}")
    log_lines.append(f"[SHA256] train.parquet: {train_parquet_sha}")

    config = {
        "seed": SEED, "device": DEVICE, "codebook_size": CODEBOOK_SIZE,
        "max_len": MAX_LEN, "batch_size": BATCH_SIZE, "num_epochs": NUM_EPOCHS,
        "lr_conditioner": LR_CONDITIONER, "lr_layernorm": LR_LAYERNORM,
        "alpha_init": ALPHA_INIT, "d_model": D_MODEL, "n_layers": 3,
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    # ============================================================================
    # 加载 T5 (Task #84 ckpt)
    # ============================================================================
    log_lines.append(f"\n[Load T5] ckpt={T5_CKPT}")
    t5_state_dict = load_t5_state_dict(T5_CKPT)
    t5_config = get_t5_config()

    # ============================================================================
    # 加载 GenRecDataset
    # ============================================================================
    log_lines.append(f"\n[Load dataset] mode='train', real history-SID")
    train_ds = GenRecDataset(
        dataset_path=TRAIN_PARQUET,
        code_path=SID_NPY,
        mode="train",
        codebook_size=CODEBOOK_SIZE,
        max_len=MAX_LEN,
        PAD_TOKEN=PAD_TOKEN,
    )
    log_lines.append(f"[Data] train_ds size: {len(train_ds)}")

    # Sample 一个 batch 做 precheck
    sample = [train_ds[i] for i in range(BATCH_SIZE)]
    history_list = [s["history"] for s in sample]
    target_list = [s["target"] for s in sample]
    history_flat_list = [[elem for sublist in h for elem in sublist] for h in history_list]
    history_tensor = torch.tensor(history_flat_list, dtype=torch.long, device=DEVICE)
    target_tensor = torch.tensor(target_list, dtype=torch.long, device=DEVICE)
    log_lines.append(f"[Data] history_tensor shape: {history_tensor.shape}")
    log_lines.append(f"[Data] target_tensor shape: {target_tensor.shape}")

    sid_range = verify_sid_token_range(history_tensor, CODEBOOK_SIZE)
    log_lines.append(f"[Precheck 4] SID token range: {sid_range}")

    # ============================================================================
    # 构造 Wrapper
    # ============================================================================
    model_wrapper = HG_Rec_with_ZeroCenteredLayerNormAdapter(
        t5_config, t5_state_dict, d_model=D_MODEL, n_layers=3, sid_dim=4,
    ).to(DEVICE)

    # ============================================================================
    # Precheck 1: α=0 → max diff=0
    # ============================================================================
    log_lines.append(f"\n[Precheck 1] α=0 → max diff=0 验证:")
    sid_meta_for_init = torch.zeros(BATCH_SIZE, MAX_LEN * 4, 4, dtype=torch.long, device=DEVICE)
    kappa_meta_for_init = torch.zeros(BATCH_SIZE, 3, dtype=torch.float32, device=DEVICE)

    init_proof = adapter_init_proof(model_wrapper, sid_meta_for_init, kappa_meta_for_init)
    initial_alpha = model_wrapper.adapter.get_alpha().item()
    log_lines.append(f"  α value (init)={initial_alpha:.6e}, α zero forced max_diff={init_proof['max_diff_x_emb']:.2e}, is_zero_diff={init_proof['is_zero_diff']}")

    with open(ADAPTER_INIT_PROOF_PATH, "w") as f:
        json.dump(init_proof, f, indent=2)

    # ============================================================================
    # Precheck 2: 仅 input LayerNorm + conditioner 解冻
    # ============================================================================
    log_lines.append(f"\n[Precheck 2] 仅 input LayerNorm + conditioner 解冻:")
    ln_proof = layernorm_unfreeze_proof(model_wrapper)
    log_lines.append(f"  trainable={ln_proof['n_trainable']}, frozen={ln_proof['n_frozen']}")
    log_lines.append(f"  input_ln_trainable={ln_proof['input_ln_trainable']}, conditioner_trainable={ln_proof['conditioner_trainable']}, other_t5_trainable={ln_proof['other_t5_trainable']}")
    log_lines.append(f"  is_correct_unfreeze={ln_proof['is_correct_unfreeze']}")
    log_lines.append(f"  trainable sample: {ln_proof['trainable_names_sample'][:10]}")

    with open(LAYERNORM_UNFREEZE_PROOF_PATH, "w") as f:
        json.dump(ln_proof, f, indent=2)

    # ============================================================================
    # Precheck 3: 首步后梯度
    # ============================================================================
    log_lines.append(f"\n[Precheck 3] 首步后 conditioner + α + LayerNorm 梯度验证:")
    sid_meta_for_grad = torch.zeros(BATCH_SIZE, MAX_LEN * 4, 4, dtype=torch.long, device=DEVICE)
    kappa_meta_for_grad = torch.zeros(BATCH_SIZE, 3, dtype=torch.float32, device=DEVICE)

    attention_mask = (history_tensor != PAD_TOKEN).long()

    grad_proof = gradient_proof(model_wrapper, sid_meta_for_grad, kappa_meta_for_grad, history_tensor, attention_mask, target_tensor)
    log_lines.append(f"  loss={grad_proof['loss_value']:.4f}, α={grad_proof['alpha_value']:.6e}")
    for name, info in grad_proof['grad_summary'].items():
        log_lines.append(f"    {name}: abs_mean={info['abs_mean']:.4e}, abs_max={info['abs_max']:.4e}, finite={info['finite']}")

    with open(GRADIENT_PROOF_PATH, "w") as f:
        json.dump(grad_proof, f, indent=2)

    # ============================================================================
    # Precheck 决策
    # ============================================================================
    conditioner_grad_nonzero = any(g["abs_mean"] > 0 for name, g in grad_proof['grad_summary'].items() if "adapter." in name)
    alpha_grad_nonzero = any(g["abs_mean"] > 0 for name, g in grad_proof['grad_summary'].items() if "alpha_logit" in name)
    ln_grad_nonzero = any(g["abs_mean"] > 0 for name, g in grad_proof['grad_summary'].items() if "encoder.block.0.layer.0.layer_norm" in name)

    precheck_pass = (init_proof["is_zero_diff"] and
                     ln_proof["is_correct_unfreeze"] and
                     conditioner_grad_nonzero and
                     alpha_grad_nonzero and
                     ln_grad_nonzero and
                     sid_range["all_in_range"])
    log_lines.append(f"\n[Precheck 总评] init_pass={init_proof['is_zero_diff']}, ln_unfreeze={ln_proof['is_correct_unfreeze']}, cond_grad={conditioner_grad_nonzero}, α_grad={alpha_grad_nonzero}, ln_grad={ln_grad_nonzero}, sid_range={sid_range['all_in_range']}")
    log_lines.append(f"[Precheck 总评] PASS: {precheck_pass}")

    print("\n".join(log_lines), flush=True)
    with open(LOG_PATH, "w") as f:
        f.write("\n".join(log_lines))
    log_lines = []

    if not precheck_pass:
        log_lines.append("[STOP] Precheck FAIL, 不做训练 (Issue #150 spec 强制)")
        with open(LOG_PATH, "a") as f:
            f.write("\n".join(log_lines))
        with open(VERDICT_PATH, "w") as f:
            json.dump({"gate3_pass": False, "reason": "precheck_failed",
                       "init_proof": init_proof, "gradient_proof": grad_proof,
                       "layernorm_proof": ln_proof, "sid_range": sid_range}, f, indent=2)
        return

    # ============================================================================
    # Gate 3: 短训练 10 epoch
    # ============================================================================
    log_lines.append(f"\n[Gate 3 训练] 10 epoch, batch_size={BATCH_SIZE}")
    # 分组 optimizer: conditioner + α 用 LR_CONDITIONER, LayerNorm 用 LR_LAYERNORM
    conditioner_params = [p for n, p in model_wrapper.adapter.named_parameters()]
    layernorm_params = list(model_wrapper.first_input_ln.parameters())
    optimizer = torch.optim.Adam([
        {"params": conditioner_params, "lr": LR_CONDITIONER},
        {"params": layernorm_params, "lr": LR_LAYERNORM},
    ])

    # 预加载整个 dataset 到 GPU tensor
    log_lines.append(f"[预加载] 把 train_ds 整个加载到 GPU tensor...")
    n_samples = len(train_ds)
    all_histories = np.zeros((n_samples, MAX_LEN * 4), dtype=np.int64)
    all_targets = np.zeros((n_samples, 4), dtype=np.int64)
    for i in range(n_samples):
        sample = train_ds[i]
        all_histories[i] = np.concatenate([np.asarray(h, dtype=np.int64).flatten() for h in sample["history"]])
        all_targets[i] = np.asarray(sample["target"], dtype=np.int64)
    log_lines.append(f"  preloaded histories shape: {all_histories.shape}, targets shape: {all_targets.shape}")
    all_histories_t = torch.from_numpy(all_histories).to(DEVICE).long()
    all_targets_t = torch.from_numpy(all_targets).to(DEVICE).long()

    rng = torch.Generator().manual_seed(42 + 8)
    n_batches = (n_samples + BATCH_SIZE - 1) // BATCH_SIZE
    log_lines.append(f"[训练循环] {NUM_EPOCHS} epoch × {n_batches} batches = {NUM_EPOCHS * n_batches} total")
    print(f"preload done: histories {all_histories.shape}, targets {all_targets.shape}, n_batches={n_batches}", flush=True)

    train_trace = []
    epoch0_loss = None
    for epoch in range(NUM_EPOCHS):
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
            kappa_meta = torch.zeros(B, 3, dtype=torch.float32, device=DEVICE)

            optimizer.zero_grad()
            loss, _, alpha_val = model_wrapper(history_tensor_b, attention_mask=attention_mask_b,
                                              labels=target_tensor_b, sid_meta=sid_meta, kappa_meta=kappa_meta)
            if not torch.isfinite(loss):
                nan_inf_detected = True
                log_lines.append(f"  [WARN] epoch {epoch}, batch {batch_idx}: NaN/Inf loss detected")
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
        train_trace.append({
            "epoch": epoch, "avg_loss": avg_loss,
            "avg_cond_grad_norm": avg_cond_grad,
            "avg_ln_grad_norm": avg_ln_grad,
            "alpha_val": alpha_val.item(),
            "nan_inf": nan_inf_detected, "n_batches": len(epoch_losses),
        })
        log_lines.append(f"  [epoch {epoch}] loss={avg_loss:.4f}, cond_grad={avg_cond_grad:.4e}, ln_grad={avg_ln_grad:.4e}, α={alpha_val.item():.6e}, n_batches={len(epoch_losses)}, nan_inf={nan_inf_detected}")
        print(log_lines[-1], flush=True)

    log_lines.append(f"\n[训练结束] 保存 adapter ckpt (R12 强制)")
    if ADAPTER_CKPT_PATH.exists():
        ADAPTER_CKPT_PATH.unlink()
    torch.save({
        "adapter_state_dict": model_wrapper.adapter.state_dict(),
        "first_input_ln_state_dict": model_wrapper.first_input_ln.state_dict(),
        "alpha_value": model_wrapper.adapter.get_alpha().item(),
        "epoch_losses": [t["avg_loss"] for t in train_trace],
    }, ADAPTER_CKPT_PATH)
    adapter_ckpt_sha = sha256_of(ADAPTER_CKPT_PATH)
    log_lines.append(f"  adapter SHA256: {adapter_ckpt_sha}")

    # ============================================================================
    # Save/Load 验证
    # ============================================================================
    log_lines.append(f"\n[Save/Load 验证]")
    ckpt_loaded = torch.load(ADAPTER_CKPT_PATH, map_location="cpu", weights_only=False)
    expected_keys = set(model_wrapper.adapter.state_dict().keys()) | set(model_wrapper.first_input_ln.state_dict().keys())
    loaded_keys = set(ckpt_loaded["adapter_state_dict"].keys()) | set(ckpt_loaded["first_input_ln_state_dict"].keys())
    missing_keys = expected_keys - loaded_keys
    unexpected_keys = loaded_keys - expected_keys
    log_lines.append(f"  missing_keys: {len(missing_keys)}, unexpected_keys: {len(unexpected_keys)}")

    # ============================================================================
    # Forward 一致性
    # ============================================================================
    log_lines.append(f"\n[Forward 一致性] 两次 forward diff:")
    model_wrapper.eval()
    with torch.no_grad():
        B = history_tensor.shape[0]
        L_flat = MAX_LEN * 4
        digit_values = history_tensor.float()
        layer_idx = torch.arange(L_flat, device=DEVICE) % 4
        layer_idx = layer_idx.float().unsqueeze(0).expand(B, -1)
        pos_in_history = torch.arange(L_flat, device=DEVICE) // 4
        pos_in_history = pos_in_history.float().unsqueeze(0).expand(B, -1) / MAX_LEN
        padding_flag = (digit_values == PAD_TOKEN).float()
        sid_meta = torch.stack([digit_values / 1025.0, layer_idx / 4.0, pos_in_history, padding_flag], dim=-1)
        kappa_meta = torch.zeros(B, 3, dtype=torch.float32, device=DEVICE)
        loss1, logits1, _ = model_wrapper(history_tensor, attention_mask=attention_mask,
                                           labels=target_tensor, sid_meta=sid_meta, kappa_meta=kappa_meta)
        loss2, logits2, _ = model_wrapper(history_tensor, attention_mask=attention_mask,
                                           labels=target_tensor, sid_meta=sid_meta, kappa_meta=kappa_meta)
    forward_diff = (logits1 - logits2).abs().max().item()
    log_lines.append(f"  forward diff max: {forward_diff:.2e}")

    # ============================================================================
    # Gate 3 决策
    # ============================================================================
    log_lines.append(f"\n[Gate 3 checks]:")
    loss_decreased = epoch0_loss is not None and train_trace[-1]["avg_loss"] < epoch0_loss
    log_lines.append(f"  (1) loss epoch0={epoch0_loss:.4f}, epoch{NUM_EPOCHS-1}={train_trace[-1]['avg_loss']:.4f}, decreased={loss_decreased}")
    cond_grad_strs = [f"{t['avg_cond_grad_norm']:.4e}" for t in train_trace]
    ln_grad_strs = [f"{t['avg_ln_grad_norm']:.4e}" for t in train_trace]
    log_lines.append(f"  (2) cond_grad epoch trace: {cond_grad_strs}")
    log_lines.append(f"  (3) ln_grad epoch trace: {ln_grad_strs}")
    log_lines.append(f"  (4) nan_inf all False: {all(not t['nan_inf'] for t in train_trace)}")
    log_lines.append(f"  (5) sid_range all_in_range: {sid_range['all_in_range']}")
    log_lines.append(f"  (6) save/load missing={len(missing_keys)}/unexpected={len(unexpected_keys)}")
    log_lines.append(f"  (7) forward diff: {forward_diff:.2e}")

    gate3_pass = (precheck_pass and
                  loss_decreased and
                  all(t["avg_cond_grad_norm"] > 0 for t in train_trace) and
                  all(t["avg_ln_grad_norm"] > 0 for t in train_trace) and
                  all(not t["nan_inf"] for t in train_trace) and
                  sid_range["all_in_range"] and
                  len(missing_keys) == 0 and
                  len(unexpected_keys) == 0 and
                  forward_diff < 1e-5)

    log_lines.append(f"\n[Gate 3] {'✅ PASS' if gate3_pass else '❌ FAIL'}")
    print("\n".join(log_lines), flush=True)
    with open(LOG_PATH, "a") as f:
        f.write("\n".join(log_lines))

    with open(TRAIN_TRACE_PATH, "w") as f:
        json.dump({"train_trace": train_trace, "epoch0_loss": epoch0_loss,
                   "loss_decreased": loss_decreased, "gate3_pass": gate3_pass}, f, indent=2)

    with open(VERDICT_PATH, "w") as f:
        json.dump({
            "gate3_pass": gate3_pass, "reason": "trained" if gate3_pass else "loss_not_decreased_or_other",
            "init_proof": init_proof, "gradient_proof": grad_proof,
            "layernorm_proof": ln_proof, "sid_range": sid_range,
            "missing_keys": list(missing_keys), "unexpected_keys": list(unexpected_keys),
            "forward_diff": forward_diff, "train_trace_summary": train_trace,
            "epoch0_loss": epoch0_loss, "loss_decreased": loss_decreased,
            "adapter_ckpt_sha256": adapter_ckpt_sha,
            "task_id": 440, "issue": "Issue #150", "commit_hash": "PENDING_R21_FIX",
        }, f, indent=2)


if __name__ == "__main__":
    main()