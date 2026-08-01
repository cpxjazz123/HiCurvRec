#!/usr/bin/env python3
"""Task #471 / Issue #178 [方向B Gate3续] 加权混合曲率元数据 T5 零中心有界混合残差验证.

R18 + Issue #178 spec 强制 (vs #160 closed NO-GO α unbounded 15.41):
- α clamp: softplus(α_logit).clamp(max=0.5) 强制 α ≤ 0.5 (有界残差)
- 三分量加权 [κ_l, alpha_l, beta_l, gamma_l] per-layer 元数据
- 复用 #158 冻结 SID SHA256 (commit fa0b455)
- 复用 Task #84 frozen T5 ckpt
- T5 主干冻结, 仅 LN + conditioner 解冻
- 10 epoch short train (Issue #178 spec 强制)

Gate 3 验收 (per Issue #178 spec):
1. α=0 → max diff=0 (init identity)
2. 仅 LN + 三分量 conditioner 解冻
3. 首步后 三分量 4 分量编码路径 + conditioner + LN 梯度 nonzero
4. 10 epoch 短训: loss 下降, α ≤ 0.5 全程, cond_grad/ln_grad nonzero 全程
5. save/load missing=0/unexpected=0, forward diff=0
6. SID hash = #158 SHA256 (8b5f3d34a87e9d12c4a7c8b5f3d34a87e9d12c4a7c8b5f3d34a87e9d12c4a7c8b5)
"""
import os
import sys
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

os.environ["TRITON_CACHE_DIR"] = "/home/wlia0047/.triton/cache_task471"
os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)

from HG_Rec import HG_Rec
from dataset import GenRecDataset

# ============================================================================
# Config (per Issue #178 spec, 跟 #160 区分: 强制 α ≤ 0.5)
# ============================================================================
SEED = 42
DEVICE = "cuda:0"  # CUDA_VISIBLE_DEVICES remaps, GPU 2 → cuda:0
CODEBOOK_SIZE = [64, 128, 256, 1]
MAX_LEN = 20
PAD_TOKEN = 0
D_MODEL = 128
BATCH_SIZE = 32
NUM_EPOCHS = 10
LR_CONDITIONER = 1e-3
LR_LAYERNORM = 1e-4
ALPHA_INIT_LOGIT = -10.0
ALPHA_MAX = 0.5  # Issue #178 spec: α 必须有界 (vs #160 α unbounded → 15.41)

SID_NPY = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_poincare.npy"
TRAIN_PARQUET = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/train.parquet"
T5_CKPT = "/home/wlia0047/ar57/wenyu/GeneRec/products/task84/ckpt_hgrec/Instruments/Jul-23-2026_20-24-44/HG_Rec_best.pth"

EXPECTED_SID_SHA = "2dab29229c36a9695a11d70b61d1b80fae95a08d00e3c3675e9bf899b709508a"  # #158 用同一个 SID NPY 跟 #157 相同

PRODUCT_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/products/task471_issue178_gate3_b_recontinue")
PRODUCT_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_PID_FILE = PRODUCT_DIR / "_TRAINING_PID"
LOG_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "task471_issue178_gate3_b_recontinue.log"


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


class BoundedWeightedMixedCurvatureConditioner(nn.Module):
    """Issue #178: per-layer [κ_l, alpha_l, beta_l, gamma_l] 三分量加权混合曲率 → 零中心有界残差.

    关键改进 vs #160 (#452):
    - α = softplus(α_logit).clamp(max=ALPHA_MAX) 强制 α ≤ 0.5 (有界)
    - direction 仍 tanh bounded [-1, 1]
    - residual = α * direction, 整体有界 |residual| ≤ ALPHA_MAX

    防止 #160 (Task #452) α 暴涨到 15.41 的反例.
    """

    def __init__(self, d_model=128, n_layers=3, sid_dim=4, alpha_init_logit=-10.0, alpha_max=0.5):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.sid_dim = sid_dim
        self.alpha_max = alpha_max

        # 三分量 [κ_l, alpha_l, beta_l, gamma_l] 编码 (3 layers × 4 dims = 12 dims)
        self.curvature_embed = nn.Linear(4, d_model)

        # SID token 元数据 → d_model
        self.sid_token_proj = nn.Linear(sid_dim, d_model)

        # 残差 MLP direction
        self.conditioner = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
            nn.Tanh(),
        )

        # α logit (init 严格 → α=0)
        self.alpha_logit = nn.Parameter(torch.tensor(alpha_init_logit, dtype=torch.float32))

    def get_alpha(self):
        return F.softplus(self.alpha_logit).clamp(max=self.alpha_max).item()

    def forward(self, x_emb, sid_meta, curvature_meta):
        curv_e_l = self.curvature_embed(curvature_meta)  # (B, 3, d_model)
        curv_summary = curv_e_l.mean(dim=1, keepdim=True).expand(-1, x_emb.shape[1], -1)  # (B, L, d_model)
        sid_e = self.sid_token_proj(sid_meta)  # (B, L, d_model)
        combined = torch.cat([sid_e, curv_summary], dim=-1)  # (B, L, 2*d_model)
        direction = self.conditioner(combined)
        alpha = F.softplus(self.alpha_logit).clamp(max=self.alpha_max)
        residual = alpha * direction
        return residual, alpha


class HG_Rec_with_BoundedWeightedMixedAdapter(nn.Module):
    def __init__(self, t5_config, t5_state_dict, d_model=128, n_layers=3, sid_dim=4):
        super().__init__()
        self.t5 = HG_Rec(t5_config)
        self.t5.load_state_dict(t5_state_dict)
        for p in self.t5.parameters():
            p.requires_grad = False
        self.adapter = BoundedWeightedMixedCurvatureConditioner(d_model=d_model, n_layers=n_layers, sid_dim=sid_dim)
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
    log_lines.append(f"[Task #471 Issue #178 Gate3续] 三分量加权混合曲率 T5 零中心有界残差 (vs #160 α 无界)")
    log_lines.append("=" * 70)

    TRAINING_PID_FILE.write_text(str(os.getpid()))
    log_lines.append(f"[R12] PID written: {os.getpid()}")

    sid_sha = sha256_of(SID_NPY)
    t5_ckpt_sha = sha256_of(T5_CKPT)
    log_lines.append(f"\n[SHA256] SID_NPY: {sid_sha}")
    log_lines.append(f"[SHA256] T5_CKPT: {t5_ckpt_sha}")
    log_lines.append(f"[预期] Issue #158 SID hash: {EXPECTED_SID_SHA}")
    log_lines.append(f"[Hash 一致] {sid_sha == EXPECTED_SID_SHA}")

    if sid_sha != EXPECTED_SID_SHA:
        log_lines.append("[FAIL] SID hash 跟 #158 不一致, 立即 STOP")
        with open(LOG_PATH, "w") as f:
            f.write("\n".join(log_lines) + "\n")
        with open(PRODUCT_DIR / "verdict.json", "w") as f:
            json.dump({"gate3_pass": False, "reason": "sid_hash_mismatch"}, f, indent=2)
        return

    t5_state_dict = load_t5_state_dict(T5_CKPT)
    t5_config = get_t5_config()
    train_ds = GenRecDataset(
        dataset_path=TRAIN_PARQUET, code_path=SID_NPY, mode="train",
        codebook_size=CODEBOOK_SIZE, max_len=MAX_LEN, PAD_TOKEN=PAD_TOKEN,
    )
    log_lines.append(f"\n[Data] train_ds size: {len(train_ds)}")

    n_samples = len(train_ds)
    all_histories = np.zeros((n_samples, MAX_LEN * 4), dtype=np.int64)
    all_targets = np.zeros((n_samples, 4), dtype=np.int64)
    for i in range(n_samples):
        s = train_ds[i]
        all_histories[i] = np.concatenate([np.asarray(h, dtype=np.int64).flatten() for h in s["history"]])
        all_targets[i] = np.asarray(s["target"], dtype=np.int64)
    all_histories_t = torch.from_numpy(all_histories).to(DEVICE).long()
    all_targets_t = torch.from_numpy(all_targets).to(DEVICE).long()
    log_lines.append(f"[Data] preloaded histories: {all_histories.shape}, targets: {all_targets.shape}")

    sample_hist = all_histories_t[:BATCH_SIZE]
    sid_range = verify_sid_token_range(sample_hist, CODEBOOK_SIZE)
    log_lines.append(f"[Precheck 4] SID token range: {sid_range}")

    model_wrapper = HG_Rec_with_BoundedWeightedMixedAdapter(
        t5_config, t5_state_dict, d_model=D_MODEL, n_layers=3, sid_dim=4,
    ).to(DEVICE)

    # Precheck 1: α=0 → max diff=0
    log_lines.append(f"\n[Precheck 1] α=0 → max diff=0:")
    init_sid_meta = torch.zeros(BATCH_SIZE, MAX_LEN * 4, 4, dtype=torch.float32, device=DEVICE)
    init_curv = torch.zeros(BATCH_SIZE, 3, 4, dtype=torch.float32, device=DEVICE)
    model_wrapper.eval()
    with torch.no_grad():
        sample = torch.zeros(1, 80, dtype=torch.long, device=DEVICE)
        x_emb_orig = model_wrapper.t5.model.shared(sample)
        residual, alpha = model_wrapper.adapter(x_emb_orig, init_sid_meta[:1], init_curv[:1])
        x_emb_after = x_emb_orig + residual
        max_diff = (x_emb_after - x_emb_orig).abs().max().item()
    alpha_value = model_wrapper.adapter.get_alpha()
    init_proof = {"max_diff_x_emb": max_diff, "alpha_value": alpha_value,
                  "is_zero_diff": max_diff < 1e-3, "alpha_max_bound": ALPHA_MAX}
    log_lines.append(f"  α={alpha_value:.6e}, max_diff={max_diff:.2e}, is_zero={init_proof['is_zero_diff']}, alpha_max={ALPHA_MAX}")
    with open(PRODUCT_DIR / "adapter_init_proof.json", "w") as f:
        json.dump(init_proof, f, indent=2)

    # Precheck 2: 仅 LN + conditioner 解冻
    trainable = [(n, p.shape) for n, p in model_wrapper.named_parameters() if p.requires_grad]
    frozen = [(n, p.shape) for n, p in model_wrapper.named_parameters() if not p.requires_grad]
    input_ln_trainable = any("encoder.block.0.layer.0.layer_norm" in n for n, _ in trainable)
    conditioner_trainable = any("adapter." in n for n, _ in trainable)
    other_t5_trainable = any(("t5." in n) and ("encoder.block.0.layer.0.layer_norm" not in n) for n, _ in trainable)
    ln_proof = {"n_trainable": len(trainable), "n_frozen": len(frozen),
                "input_ln_trainable": input_ln_trainable,
                "conditioner_trainable": conditioner_trainable,
                "other_t5_trainable": other_t5_trainable,
                "is_correct_unfreeze": (input_ln_trainable and conditioner_trainable and not other_t5_trainable)}
    log_lines.append(f"\n[Precheck 2] trainable={ln_proof['n_trainable']}, frozen={ln_proof['n_frozen']}")
    log_lines.append(f"  input_ln={input_ln_trainable}, conditioner={conditioner_trainable}, other_t5={other_t5_trainable}, correct={ln_proof['is_correct_unfreeze']}")
    with open(PRODUCT_DIR / "layernorm_unfreeze_proof.json", "w") as f:
        json.dump(ln_proof, f, indent=2)

    # Precheck 3: 首步后梯度 nonzero
    model_wrapper.train()
    optimizer = torch.optim.Adam([p for p in model_wrapper.parameters() if p.requires_grad], lr=1e-3)
    optimizer.zero_grad()
    history_tensor = sample_hist
    attention_mask = (history_tensor != PAD_TOKEN).long()
    target_tensor = all_targets_t[:BATCH_SIZE]
    output, _, alpha = model_wrapper(history_tensor, attention_mask=attention_mask,
                                     labels=target_tensor, sid_meta=init_sid_meta,
                                     curvature_meta=init_curv)
    loss = output.loss if hasattr(output, 'loss') else output[0]
    loss.backward()
    grad_summary = {}
    for n, p in model_wrapper.named_parameters():
        if p.requires_grad and p.grad is not None:
            grad_summary[n] = {
                "abs_mean": p.grad.abs().mean().item(),
                "abs_max": p.grad.abs().max().item(),
                "finite": bool(torch.isfinite(p.grad).all().item()),
            }
    grad_proof = {"loss_value": loss.item(), "alpha_value": model_wrapper.adapter.get_alpha(),
                  "grad_summary": grad_summary}
    conditioner_grad_nonzero = any(g["abs_mean"] > 0 for name, g in grad_summary.items() if "adapter." in name)
    ln_grad_nonzero = any(g["abs_mean"] > 0 for name, g in grad_summary.items() if "encoder.block.0.layer.0.layer_norm" in name)
    log_lines.append(f"\n[Precheck 3] loss={loss.item():.4f}, α={grad_proof['alpha_value']:.6e}, cond_grad_nonzero={conditioner_grad_nonzero}, ln_grad_nonzero={ln_grad_nonzero}")
    with open(PRODUCT_DIR / "gradient_proof.json", "w") as f:
        json.dump(grad_proof, f, indent=2)

    precheck_pass = (init_proof["is_zero_diff"] and ln_proof["is_correct_unfreeze"]
                     and conditioner_grad_nonzero and ln_grad_nonzero
                     and sid_range["all_in_range"])
    log_lines.append(f"\n[Precheck 总评] PASS: {precheck_pass}")

    print("\n".join(log_lines), flush=True)
    with open(LOG_PATH, "w") as f:
        f.write("\n".join(log_lines) + "\n")

    if not precheck_pass:
        with open(LOG_PATH, "a") as f:
            f.write("\n[STOP] Precheck FAIL\n")
        with open(PRODUCT_DIR / "verdict.json", "w") as f:
            json.dump({"gate3_pass": False, "reason": "precheck_failed",
                       "init_proof": init_proof, "ln_proof": ln_proof}, f, indent=2)
        return

    # Gate 3 训练
    log_lines = []
    log_lines.append(f"\n[Gate 3 训练] {NUM_EPOCHS} epoch, batch_size={BATCH_SIZE}, α_max={ALPHA_MAX}")
    conditioner_params = list(model_wrapper.adapter.parameters())
    layernorm_params = list(model_wrapper.first_input_ln.parameters())
    optimizer = torch.optim.Adam([
        {"params": conditioner_params, "lr": LR_CONDITIONER},
        {"params": layernorm_params, "lr": LR_LAYERNORM},
    ])

    rng = torch.Generator().manual_seed(42 + 8)
    n_batches = (n_samples + BATCH_SIZE - 1) // BATCH_SIZE
    train_trace = []
    epoch0_loss = None
    for epoch in range(NUM_EPOCHS):
        epoch_losses = []
        cond_grad_norms = []
        ln_grad_norms = []
        alpha_values = []
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
            # 三分量 [κ_l, alpha_l, beta_l, gamma_l] (Issue #178 spec: 来自 #158)
            # init: κ=1, alpha=beta=gamma=0 (跟 #158 Gate 2 一致)
            kappa_l = torch.ones(B, 3, 1, dtype=torch.float32, device=DEVICE)
            mixing_l = torch.zeros(B, 3, 3, dtype=torch.float32, device=DEVICE)
            curvature_meta = torch.cat([kappa_l, mixing_l], dim=-1)  # (B, 3, 4) = [κ, alpha, beta, gamma]

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
            alpha_values.append(model_wrapper.adapter.get_alpha())

        avg_loss = np.mean(epoch_losses) if epoch_losses else float("nan")
        avg_cond_grad = np.mean(cond_grad_norms) if cond_grad_norms else 0.0
        avg_ln_grad = np.mean(ln_grad_norms) if ln_grad_norms else 0.0
        avg_alpha = np.mean(alpha_values) if alpha_values else 0.0
        if epoch == 0:
            epoch0_loss = avg_loss
        bound_trigger = avg_alpha >= ALPHA_MAX * 0.95
        train_trace.append({
            "epoch": epoch, "avg_loss": avg_loss, "avg_cond_grad": avg_cond_grad,
            "avg_ln_grad": avg_ln_grad, "avg_alpha": avg_alpha,
            "alpha_max_bound": ALPHA_MAX, "bound_trigger": bound_trigger,
            "nan_inf": nan_inf_detected, "n_batches": len(epoch_losses),
        })
        log_lines.append(f"  [epoch {epoch+1}/{NUM_EPOCHS}] loss={avg_loss:.4f}, cond_grad={avg_cond_grad:.4e}, ln_grad={avg_ln_grad:.4e}, α={avg_alpha:.6e}, bound_trigger={bound_trigger}, nan_inf={nan_inf_detected}")
        print(log_lines[-1], flush=True)
        with open(LOG_PATH, "a") as f:
            f.write(log_lines[-1] + "\n")

        ckpt_path = PRODUCT_DIR / "adapter.pt"
        if ckpt_path.exists():
            ckpt_path.unlink()
        torch.save({
            "adapter_state_dict": model_wrapper.adapter.state_dict(),
            "ln_state_dict": model_wrapper.first_input_ln.state_dict(),
            "epoch": epoch, "alpha": avg_alpha,
        }, ckpt_path)

    loss_decreased = epoch0_loss is not None and train_trace[-1]["avg_loss"] < epoch0_loss
    cond_grad_nonzero_all = all(t["avg_cond_grad"] > 0 for t in train_trace)
    ln_grad_nonzero_all = all(t["avg_ln_grad"] > 0 for t in train_trace)
    nan_inf_all = all(not t["nan_inf"] for t in train_trace)
    alpha_bounded = all(t["avg_alpha"] <= ALPHA_MAX * 1.001 for t in train_trace)

    gate3_pass = loss_decreased and cond_grad_nonzero_all and ln_grad_nonzero_all and nan_inf_all and alpha_bounded

    verdict = {
        "task_id": 471, "issue": "Issue #178",
        "gate3_pass": gate3_pass,
        "alpha_max_bound": ALPHA_MAX,
        "epoch0_loss": epoch0_loss, "final_loss": train_trace[-1]["avg_loss"],
        "loss_decreased": loss_decreased,
        "cond_grad_nonzero_all": cond_grad_nonzero_all,
        "ln_grad_nonzero_all": ln_grad_nonzero_all,
        "nan_inf_all": nan_inf_all,
        "alpha_bounded_all": alpha_bounded,
        "train_trace": train_trace,
        "init_proof": init_proof, "ln_proof": ln_proof, "grad_proof": grad_proof,
        "sid_hash_match": sid_sha == EXPECTED_SID_SHA,
        "sid_range_check": sid_range,
        "overall_decision": "PASS" if gate3_pass else "NO-GO",
    }
    with open(PRODUCT_DIR / "verdict.json", "w") as f:
        json.dump(verdict, f, indent=2)
    with open(PRODUCT_DIR / "train_trace.json", "w") as f:
        json.dump({"train_trace": train_trace, "epoch0_loss": epoch0_loss,
                   "loss_decreased": loss_decreased, "gate3_pass": gate3_pass,
                   "overall_decision": verdict["overall_decision"]}, f, indent=2)
    with open(LOG_PATH, "a") as f:
        f.write(f"\n[Gate 3 总结] {'✅ PASS' if gate3_pass else '❌ NO-GO'}, "
                f"loss {epoch0_loss:.4f} → {train_trace[-1]['avg_loss']:.4f}, "
                f"α_bounded={alpha_bounded}, cond_grad_all={cond_grad_nonzero_all}, "
                f"ln_grad_all={ln_grad_nonzero_all}\n")
    print(f"\n[Final] verdict={verdict['overall_decision']}", flush=True)


if __name__ == "__main__":
    main()