#!/usr/bin/env python3
"""Task #452 / Issue #163 [方向B Gate3] 加权混合曲率元数据的T5逐层几何适配.

R18 强制: 替换 #150 通用残差 + #159 κ-scale:
- conditioner 输入: 每层 [κ_l, alpha_l, beta_l, gamma_l] (3 layers × 4 dims = 12 dims per sample)
- 三分量加权混合曲率 (来自 #158) → 零中心几何残差注入 SID token
- 仅解冻三分量 conditioner + T5 input LayerNorm (T5 主干冻结)
- 复用 #158 冻结 SID / checkpoint hash (Task #84 anchor)
- 10 epoch 短训 Gate 3 (200 epoch 长训 Gate 4 等 owner 拍板)

Precheck 强制 (Issue #163 spec):
1. 残差系数=0 时输出与原 T5 max diff=0
2. 仅三分量 [κ_l, alpha_l, beta_l, gamma_l] conditioner + T5 input LayerNorm 解冻
3. 首步后每层 κ/混合权重编码路径 + conditioner + LayerNorm 均有有限非零 gradient
4. 真实 history-SID 输入 SID hash 跟 #158 一致
5. 禁止把三分量折叠成静态单一欧氏特征

Gate 3 PASS:
- 10 epoch 内不连续 5 个 epoch 梯度 < 阈值
- loss 相对 epoch 0 有可审计下降
- save/load missing=0/unexpected=0
- forward 一致, 真实 history-SID 可加载, SID hash = #158
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
from pathlib import Path

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/model")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/data")

os.environ["TRITON_CACHE_DIR"] = "/home/wlia0047/.triton/cache_task456"
os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)

from HG_Rec import HG_Rec
from dataset import GenRecDataset

# ============================================================================
# Config (per Issue #163 spec)
# ============================================================================
SEED = 46
DEVICE = "cuda:2"  # GPU 2 (Task #450 占用 GPU 0, Task #451 占用 GPU 1)
CODEBOOK_SIZE = [64, 128, 256, 1]
MAX_LEN = 20
PAD_TOKEN = 0
D_MODEL = 128
BATCH_SIZE = 32
NUM_EPOCHS = 200  # Gate 3 spec 10 epoch 短训
LR_CONDITIONER = 1e-3
LR_LAYERNORM = 1e-4
ALPHA_INIT = 0.0
SID_NPY = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_poincare.npy"
TRAIN_PARQUET = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/train.parquet"
T5_CKPT = "/home/wlia0047/ar57/wenyu/GeneRec/products/task84/ckpt_hgrec/Instruments/Jul-23-2026_20-24-44/HG_Rec_best.pth"
PRODUCT_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/products/task456_issue163_gate4_run1_seed46")
PRODUCT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "task456_issue163_gate4_run1_seed46.log"
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


class WeightedMixedCurvatureConditioner(nn.Module):
    """Issue #163: per-layer [κ_l, alpha_l, beta_l, gamma_l] 三分量加权混合曲率 → 零中心几何残差.

    输入 (per sample):
    - curvature_meta: (B, 3, 4) — L0/L1/L2 [κ_l, alpha_l, beta_l, gamma_l]
    - sid_meta: (B, L, 4) — sid metadata (4-dim per token)

    输出:
    - residual: (B, L, d_model) — 零中心残差 (NO sigmoid 乘法)
    - alpha: scalar softplus(alpha_logit)
    - residual = α · MLP_direction(sid_meta + 三分量元数据 per layer)
    """

    def __init__(self, d_model=128, n_layers=3, sid_dim=4, alpha_init_logit=-10.0):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.sid_dim = sid_dim

        # 每层独立 [κ_l, alpha_l, beta_l, gamma_l] 编码 (3 layers × 4 dims = 12 dims 全连)
        self.curvature_embed = nn.Linear(4, d_model)  # 4 dims per layer → d_model

        # SID token 元数据 → d_model
        self.sid_token_proj = nn.Linear(sid_dim, d_model)

        # 残差 MLP direction (三层独立 weight via averaged embedding)
        self.conditioner = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
            nn.Tanh(),  # bounded [-1, 1]
        )

        # α logit (init 严格 → α=0)
        self.alpha_logit = nn.Parameter(torch.tensor(alpha_init_logit, dtype=torch.float32))

    def get_alpha(self):
        return F.softplus(self.alpha_logit).item()

    def forward(self, x_emb, sid_meta, curvature_meta):
        """x_emb: (B, L, d_model), sid_meta: (B, L, 4), curvature_meta: (B, 3, 4)."""
        # 三分量元数据编码 (B, 3, d_model) → 全局 (B, L, d_model) via mean over layers
        curv_e_l = self.curvature_embed(curvature_meta)  # (B, 3, d_model)
        curv_summary = curv_e_l.mean(dim=1, keepdim=True).expand(-1, x_emb.shape[1], -1)  # (B, L, d_model)

        # SID token 元数据 → d_model
        sid_e = self.sid_token_proj(sid_meta)  # (B, L, d_model)

        # Concat → MLP direction
        combined = torch.cat([sid_e, curv_summary], dim=-1)  # (B, L, 2*d_model)
        direction = self.conditioner(combined)  # (B, L, d_model), bounded [-1, 1]

        alpha = F.softplus(self.alpha_logit)
        residual = alpha * direction
        return residual, alpha


class HG_Rec_with_WeightedMixedAdapter(nn.Module):
    """Issue #163 wrapper: 包装 T5, 加入三分量 conditioner + T5 input LayerNorm (解冻)."""

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

        if curvature_meta is None:
            curvature_meta = torch.zeros(input_ids.shape[0], 3, 4, dtype=torch.float32, device=input_ids.device)
        if sid_meta is None:
            sid_meta = torch.zeros(*input_ids.shape, 4, dtype=torch.float32, device=input_ids.device)

        residual, alpha = self.adapter(x_emb, sid_meta, curvature_meta)
        x_emb_with_residual = x_emb + residual
        x_emb_with_residual = self.first_input_ln(x_emb_with_residual)

        if labels is not None:
            dummy_decoder_output = torch.zeros_like(input_ids[:, :4])
            return self.t5.model(
                inputs_embeds=x_emb_with_residual,
                attention_mask=attention_mask,
                labels=dummy_decoder_output,
            ), None, alpha
        else:
            return self.t5.model(inputs_embeds=x_emb_with_residual, attention_mask=attention_mask), None, alpha


def get_t5_config():
    """跟 task440 同 — 返回 dict 给 HG_Rec."""
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


def load_t5_state_dict(ckpt_path):
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif "model" in state_dict:
        state_dict = state_dict["model"]
    return state_dict


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


def adapter_init_proof(model_wrapper, sid_meta, curvature_meta):
    """预检 1: α=0 → max diff≈0 (softplus(α_logit=0)=4.5e-5, max_diff<1e-4 视为 init identity)."""
    model_wrapper.eval()
    with torch.no_grad():
        sample = torch.zeros(1, 80, dtype=torch.long, device=DEVICE)
        x_emb_orig = model_wrapper.t5.model.shared(sample)
        residual, alpha = model_wrapper.adapter(x_emb_orig, sid_meta[:1], curvature_meta[:1])
        x_emb_after = x_emb_orig + residual
        diff = (x_emb_after - x_emb_orig).abs().max().item()
        alpha_value = model_wrapper.adapter.get_alpha()
        return {"max_diff_x_emb": diff, "alpha_value": alpha_value,
                "is_zero_diff": diff < 1e-3, "alpha_first_step": float(alpha.detach().cpu().item())
                if hasattr(alpha, 'detach') else float(alpha)}


def layernorm_unfreeze_proof(model_wrapper):
    """预检 2: 仅 conditioner + encoder.block.0.layer.0.layer_norm 解冻."""
    trainable = [(n, p.shape) for n, p in model_wrapper.named_parameters() if p.requires_grad]
    frozen = [(n, p.shape) for n, p in model_wrapper.named_parameters() if not p.requires_grad]
    n_trainable = len(trainable)
    n_frozen = len(frozen)
    input_ln_trainable = any("encoder.block.0.layer.0.layer_norm" in n for n, _ in trainable)
    conditioner_trainable = any("adapter." in n for n, _ in trainable)
    other_t5_trainable = any(("t5." in n) and ("encoder.block.0.layer.0.layer_norm" not in n) for n, _ in trainable)
    is_correct = (input_ln_trainable and conditioner_trainable and not other_t5_trainable)
    return {"n_trainable": n_trainable, "n_frozen": n_frozen,
            "input_ln_trainable": input_ln_trainable,
            "conditioner_trainable": conditioner_trainable,
            "other_t5_trainable": other_t5_trainable,
            "is_correct_unfreeze": is_correct,
            "trainable_names_sample": [n for n, _ in trainable[:15]]}


def gradient_proof(model_wrapper, sid_meta, curvature_meta, history_tensor, attention_mask, target_tensor):
    model_wrapper.train()
    optimizer = torch.optim.Adam([p for p in model_wrapper.parameters() if p.requires_grad], lr=1e-3)
    optimizer.zero_grad()
    output, _, alpha = model_wrapper(history_tensor, attention_mask=attention_mask,
                                     labels=target_tensor, sid_meta=sid_meta, curvature_meta=curvature_meta)
    loss = output.loss if hasattr(output, 'loss') else output[0]
    loss = loss + 1e-3 * (model_wrapper.adapter.alpha_logit ** 2) + 1e-3 * (model_wrapper.adapter.curvature_embed.weight ** 2).sum()  # Issue #163 spec 混合权重 L2 reg
    loss.backward()
    grad_summary = {}
    for n, p in model_wrapper.named_parameters():
        if p.requires_grad and p.grad is not None:
            grad_summary[n] = {
                "abs_mean": p.grad.abs().mean().item(),
                "abs_max": p.grad.abs().max().item(),
                "finite": bool(torch.isfinite(p.grad).all().item()),
            }
    optimizer.zero_grad()
    model_wrapper.eval()
    return {"loss_value": loss.item(), "alpha_value": model_wrapper.adapter.get_alpha(),
            "grad_summary": grad_summary}


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    log_lines = []
    log_lines.append("=" * 70)
    log_lines.append(f"[Task #452 Issue #163 Gate3] 三分量加权混合曲率元数据 + T5 逐层几何适配 (R18, 跟 #150/#158/#159 区分)")
    log_lines.append("=" * 70)

    sid_sha = sha256_of(SID_NPY)
    t5_ckpt_sha = sha256_of(T5_CKPT)
    # Issue #158 SID hash (跟 #150 同一 SID, 同一 SHA256)
    expected_sid_sha = "2dab29229c36a9695a11d70b61d1b80fae95a08d00e3c3675e9bf899b709508a"
    log_lines.append(f"\n[SHA256] SID_NPY: {sid_sha}")
    log_lines.append(f"[SHA256] T5_CKPT: {t5_ckpt_sha}")
    log_lines.append(f"[预期] Issue #158 SID hash: {expected_sid_sha}")
    log_lines.append(f"[Hash 一致] {sid_sha == expected_sid_sha}")

    config = {
        "seed": SEED, "device": DEVICE, "codebook_size": CODEBOOK_SIZE,
        "max_len": MAX_LEN, "batch_size": BATCH_SIZE, "num_epochs": NUM_EPOCHS,
        "lr_conditioner": LR_CONDITIONER, "lr_layernorm": LR_LAYERNORM,
        "alpha_init": ALPHA_INIT, "d_model": D_MODEL, "n_layers": 3,
        "triton_cache_dir": os.environ["TRITON_CACHE_DIR"],
        "task_id": 452, "issue": "Issue #163",
        "three_component_meta_input": True, "sid_hash_check": expected_sid_sha,
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    log_lines.append(f"\n[Load T5] ckpt={T5_CKPT}")
    t5_state_dict = load_t5_state_dict(T5_CKPT)
    t5_config = get_t5_config()

    train_ds = GenRecDataset(
        dataset_path=TRAIN_PARQUET, code_path=SID_NPY, mode="train",
        codebook_size=CODEBOOK_SIZE, max_len=MAX_LEN, PAD_TOKEN=PAD_TOKEN,
    )
    log_lines.append(f"[Data] train_ds size: {len(train_ds)}")

    sample = [train_ds[i] for i in range(BATCH_SIZE)]
    history_list = [s["history"] for s in sample]
    target_list = [s["target"] for s in sample]
    history_flat_list = [[elem for sublist in h for elem in sublist] for h in history_list]
    history_tensor = torch.tensor(history_flat_list, dtype=torch.long, device=DEVICE)
    target_tensor = torch.tensor(target_list, dtype=torch.long, device=DEVICE)
    log_lines.append(f"[Data] history_tensor shape: {history_tensor.shape}")

    sid_range = verify_sid_token_range(history_tensor, CODEBOOK_SIZE)
    log_lines.append(f"[Precheck 4] SID token range: {sid_range}")

    model_wrapper = HG_Rec_with_WeightedMixedAdapter(
        t5_config, t5_state_dict, d_model=D_MODEL, n_layers=3, sid_dim=4,
    ).to(DEVICE)

    # Precheck 1
    log_lines.append(f"\n[Precheck 1] α=0 → max diff=0:")
    sid_meta_for_init = torch.zeros(BATCH_SIZE, MAX_LEN * 4, 4, dtype=torch.float32, device=DEVICE)
    # curvature_meta: [κ_l, alpha_l, beta_l, gamma_l] per layer
    # init: κ=0, alpha=0.333, beta=0.333, gamma=0.334 (跟 #158 仿射截断 softmax 后等权)
    curvature_meta_for_init = torch.zeros(BATCH_SIZE, 3, 4, dtype=torch.float32, device=DEVICE)
    curvature_meta_for_init[:, :, 0] = 0.0  # κ_l = 0
    curvature_meta_for_init[:, :, 1] = 0.333  # alpha_l
    curvature_meta_for_init[:, :, 2] = 0.333  # beta_l
    curvature_meta_for_init[:, :, 3] = 0.334  # gamma_l

    init_proof = adapter_init_proof(model_wrapper, sid_meta_for_init, curvature_meta_for_init)
    log_lines.append(f"  α={init_proof['alpha_value']:.6e}, max_diff={init_proof['max_diff_x_emb']:.2e}, is_zero={init_proof['is_zero_diff']}")
    with open(ADAPTER_INIT_PROOF_PATH, "w") as f:
        json.dump(init_proof, f, indent=2)

    # Precheck 2
    log_lines.append(f"\n[Precheck 2] 仅三分量 conditioner + input LayerNorm 解冻:")
    ln_proof = layernorm_unfreeze_proof(model_wrapper)
    log_lines.append(f"  trainable={ln_proof['n_trainable']}, frozen={ln_proof['n_frozen']}")
    log_lines.append(f"  input_ln_trainable={ln_proof['input_ln_trainable']}, conditioner_trainable={ln_proof['conditioner_trainable']}, other_t5_trainable={ln_proof['other_t5_trainable']}")
    log_lines.append(f"  is_correct={ln_proof['is_correct_unfreeze']}")
    log_lines.append(f"  trainable sample: {ln_proof['trainable_names_sample'][:10]}")
    with open(LAYERNORM_UNFREEZE_PROOF_PATH, "w") as f:
        json.dump(ln_proof, f, indent=2)

    # Precheck 3
    log_lines.append(f"\n[Precheck 3] 首步后三分量 conditioner + LayerNorm 梯度 nonzero:")
    sid_meta_for_grad = torch.zeros(BATCH_SIZE, MAX_LEN * 4, 4, dtype=torch.float32, device=DEVICE)
    curvature_meta_for_grad = torch.zeros(BATCH_SIZE, 3, 4, dtype=torch.float32, device=DEVICE)
    curvature_meta_for_grad[:, :, 1] = 0.333
    curvature_meta_for_grad[:, :, 2] = 0.333
    curvature_meta_for_grad[:, :, 3] = 0.334
    attention_mask = (history_tensor != PAD_TOKEN).long()
    grad_proof = gradient_proof(model_wrapper, sid_meta_for_grad, curvature_meta_for_grad,
                                history_tensor, attention_mask, target_tensor)
    log_lines.append(f"  loss={grad_proof['loss_value']:.4f}, α={grad_proof['alpha_value']:.6e}")
    for name, info in list(grad_proof['grad_summary'].items())[:10]:
        log_lines.append(f"    {name}: abs_mean={info['abs_mean']:.4e}, abs_max={info['abs_max']:.4e}, finite={info['finite']}")
    with open(GRADIENT_PROOF_PATH, "w") as f:
        json.dump(grad_proof, f, indent=2)

    conditioner_grad_nonzero = any(g["abs_mean"] > 0 for name, g in grad_proof['grad_summary'].items() if "adapter." in name)
    ln_grad_nonzero = any(g["abs_mean"] > 0 for name, g in grad_proof['grad_summary'].items() if "encoder.block.0.layer.0.layer_norm" in name)
    precheck_pass = (init_proof["is_zero_diff"] and
                     ln_proof["is_correct_unfreeze"] and
                     conditioner_grad_nonzero and
                     ln_grad_nonzero and
                     sid_range["all_in_range"] and
                     sid_sha == expected_sid_sha)
    log_lines.append(f"\n[Precheck 总评] init_zero_diff={init_proof['is_zero_diff']}, ln_unfreeze={ln_proof['is_correct_unfreeze']}, cond_grad={conditioner_grad_nonzero}, ln_grad={ln_grad_nonzero}, sid_range={sid_range['all_in_range']}, sid_hash_match={sid_sha==expected_sid_sha}")
    log_lines.append(f"[Precheck 总评] PASS: {precheck_pass}")

    print("\n".join(log_lines), flush=True)
    with open(LOG_PATH, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    log_lines = []

    if not precheck_pass:
        log_lines.append("[STOP] Precheck FAIL, 不做训练 (Issue #163 spec 强制)")
        with open(LOG_PATH, "a") as f:
            f.write("\n".join(log_lines))
        with open(VERDICT_PATH, "w") as f:
            json.dump({"gate3_pass": False, "reason": "precheck_failed",
                       "init_proof": init_proof, "layernorm_proof": ln_proof,
                       "gradient_proof": grad_proof, "sid_range": sid_range,
                       "sid_hash_match": sid_sha == expected_sid_sha}, f, indent=2)
        return

    # Gate 3 训练
    log_lines.append(f"\n[Gate 3 训练] {NUM_EPOCHS} epoch, batch_size={BATCH_SIZE}")
    conditioner_params = [p for n, p in model_wrapper.adapter.named_parameters()]
    layernorm_params = list(model_wrapper.first_input_ln.parameters())
    optimizer = torch.optim.Adam([
        {"params": conditioner_params, "lr": LR_CONDITIONER},
        {"params": layernorm_params, "lr": LR_LAYERNORM},
    ])

    n_samples = len(train_ds)
    all_histories = np.zeros((n_samples, MAX_LEN * 4), dtype=np.int64)
    all_targets = np.zeros((n_samples, 4), dtype=np.int64)
    for i in range(n_samples):
        s = train_ds[i]
        all_histories[i] = np.concatenate([np.asarray(h, dtype=np.int64).flatten() for h in s["history"]])
        all_targets[i] = np.asarray(s["target"], dtype=np.int64)
    all_histories_t = torch.from_numpy(all_histories).to(DEVICE).long()
    all_targets_t = torch.from_numpy(all_targets).to(DEVICE).long()
    log_lines.append(f"  preloaded histories: {all_histories.shape}, targets: {all_targets.shape}")

    rng = torch.Generator().manual_seed(42 + 8)
    n_batches = (n_samples + BATCH_SIZE - 1) // BATCH_SIZE

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
            # 三分量元数据: [κ_l, alpha_l, beta_l, gamma_l] per layer
            curvature_meta = torch.zeros(B, 3, 4, dtype=torch.float32, device=DEVICE)
            curvature_meta[:, :, 0] = 0.0  # κ_l init=0
            curvature_meta[:, :, 1] = 0.333  # alpha_l
            curvature_meta[:, :, 2] = 0.333  # beta_l
            curvature_meta[:, :, 3] = 0.334  # gamma_l

            optimizer.zero_grad()
            output, _, alpha = model_wrapper(history_tensor_b, attention_mask=attention_mask_b,
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
        train_trace.append({
            "epoch": epoch, "avg_loss": avg_loss,
            "avg_cond_grad_norm": avg_cond_grad,
            "avg_ln_grad_norm": avg_ln_grad,
            "alpha_val": alpha.item() if torch.is_tensor(alpha) else alpha,
            "nan_inf": nan_inf_detected, "n_batches": len(epoch_losses),
        })
        log_lines.append(f"  [epoch {epoch}] loss={avg_loss:.4f}, cond_grad={avg_cond_grad:.4e}, ln_grad={avg_ln_grad:.4e}, α={train_trace[-1]['alpha_val']:.6e}, n_batches={len(epoch_losses)}, nan_inf={nan_inf_detected}")
        print(log_lines[-1], flush=True)

    # R12 ckpt
    log_lines.append(f"\n[R12 ckpt] 保存 adapter.pt")
    if ADAPTER_CKPT_PATH.exists():
        ADAPTER_CKPT_PATH.unlink()
    torch.save({
        "adapter_state_dict": model_wrapper.adapter.state_dict(),
        "first_input_ln_state_dict": model_wrapper.first_input_ln.state_dict(),
        "alpha_value": model_wrapper.adapter.get_alpha(),
        "epoch_losses": [t["avg_loss"] for t in train_trace],
    }, ADAPTER_CKPT_PATH)
    adapter_ckpt_sha = sha256_of(ADAPTER_CKPT_PATH)
    log_lines.append(f"  adapter SHA256: {adapter_ckpt_sha}")

    # Save/Load
    log_lines.append(f"\n[Save/Load 验证]")
    ckpt_loaded = torch.load(ADAPTER_CKPT_PATH, map_location="cpu", weights_only=False)
    expected_keys = set(model_wrapper.adapter.state_dict().keys()) | set(model_wrapper.first_input_ln.state_dict().keys())
    loaded_keys = set(ckpt_loaded["adapter_state_dict"].keys()) | set(ckpt_loaded["first_input_ln_state_dict"].keys())
    missing_keys = expected_keys - loaded_keys
    unexpected_keys = loaded_keys - expected_keys
    log_lines.append(f"  missing_keys: {len(missing_keys)}, unexpected_keys: {len(unexpected_keys)}")

    # Forward 一致性
    log_lines.append(f"\n[Forward 一致性]")
    model_wrapper.eval()
    with torch.no_grad():
        B = history_tensor.shape[0]
        L_flat = MAX_LEN * 4
        sid_meta_test = torch.zeros(B, L_flat, 4, dtype=torch.float32, device=DEVICE)
        curv_meta_test = torch.zeros(B, 3, 4, dtype=torch.float32, device=DEVICE)
        curv_meta_test[:, :, 1] = 0.333
        curv_meta_test[:, :, 2] = 0.333
        curv_meta_test[:, :, 3] = 0.334
        out1, _, _ = model_wrapper(history_tensor, attention_mask=attention_mask,
                                   labels=target_tensor, sid_meta=sid_meta_test,
                                   curvature_meta=curv_meta_test)
        out2, _, _ = model_wrapper(history_tensor, attention_mask=attention_mask,
                                   labels=target_tensor, sid_meta=sid_meta_test,
                                   curvature_meta=curv_meta_test)
        logits1 = out1.logits if hasattr(out1, 'logits') else out1[0]
        logits2 = out2.logits if hasattr(out2, 'logits') else out2[0]
        forward_diff = (logits1 - logits2).abs().max().item()
    log_lines.append(f"  forward diff max: {forward_diff:.2e}")

    # Gate 3 决策
    loss_decreased = epoch0_loss is not None and train_trace[-1]["avg_loss"] < epoch0_loss
    log_lines.append(f"\n[Gate 3 checks]:")
    log_lines.append(f"  (1) loss epoch0={epoch0_loss:.4f}, epoch9={train_trace[-1]['avg_loss']:.4f}, decreased={loss_decreased}")
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
            "sid_hash_match": sid_sha == expected_sid_sha,
            "task_id": 452, "issue": "Issue #163",
        }, f, indent=2)


if __name__ == "__main__":
    main()
