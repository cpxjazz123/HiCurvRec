#!/usr/bin/env python3
"""Issue #29 [方向B Gate2] 混合曲率训练稳定性与表示流收口 — 短程单 seed 诊断.

Per #29 spec:
- 诊断固定双曲分量、欧氏分量、L0/L1/L2 独立 κ 与 mixing logits 的实际 forward 贡献
- 记录: 逐层 dL/dκ, dL/dmix, 有限差分敏感度, 分量范数, SID hash, checkpoint/dtype, backbone cosine 漂移
- 先定位训练损失与真实受约束 SID 解码之间的差距; 只有诊断通过后才做一次 Gate3->Gate4 单 seed 正式评估
- 短程单 seed <=5 epoch, 从 #27 best_adapter (epoch 60, val=0.05) 续训, 禁止 from-0
- 禁止多 seed / 重复 199 epoch / 只看 mixing 不退化

验收: precheck 保留 L0 K64/L1 K128/L2 K256 独立 learnable κ, 禁止 global κ / fixed-only / 纯欧氏绕过.
Gate2 逐层混合/κ 影响 forward 且 SID 一致; mixing 非退化; 每层分量贡献可审计.
"""
import os
import sys
import json
import hashlib
import time
import signal
import atexit
import importlib.util
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/model")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/data")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec")

os.environ["TRITON_CACHE_DIR"] = "/home/wlia0047/.triton/cache_issue29_gate2"
os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)

from common.stage4_decode import (
    autoregressive_predict_constrained, get_layer_ranges, compute_r_at_k,
)
from common.stage4_eval_beam20 import run_val_beam20
from dataset import GenRecDataset

# ============================================================================
# Config
# ============================================================================
SEED = 42
DEVICE = "cuda:1"
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
FD_EPS = 1e-3          # 有限差分扰动
MIXING_DEGENERATE_TH = 0.95  # softmax 分量权重 > 该值 = 退化

PRIOR_CKPT = "/home/wlia0047/ar57/wenyu/GeneRec/taskB/stage3/taskB_stage3_issue27_close/best_adapter.pt"
T5_CKPT = "/home/wlia0047/ar57/wenyu/GeneRec/taskB/_ckpt/HG_Rec_best.pth"
SID_NPY = "/home/wlia0047/ar57/wenyu/GeneRec/taskB/_data/Instruments/Instruments_t5_hrqvae_poincare.npy"
TRAIN_PARQUET = "/home/wlia0047/ar57/wenyu/GeneRec/taskB/_data/Instruments/train.parquet"
EXPECTED_SID_SHA = "2dab29229c36a9695a11d70b61d1b80fae95a08d00e3c3675e9bf899b709508a"

PRODUCT_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/taskB/stage3/taskB_stage3_issue29_gate2_diag")
PRODUCT_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_PID_FILE = PRODUCT_DIR / "_TRAINING_PID"
LOG_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/taskB/_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "task_issue29_gate2_diag.log"
VAL_TRACE_PATH = PRODUCT_DIR / "val_trace.json"
BACKBONE_DRIFT_PATH = PRODUCT_DIR / "backbone_drift.json"
GRAD_NORM_PATH = PRODUCT_DIR / "grad_norm_history.json"
FD_SENS_PATH = PRODUCT_DIR / "fd_sensitivity.json"
COMPONENT_PATH = PRODUCT_DIR / "component_norms.json"
MIXING_PATH = PRODUCT_DIR / "mixing_weights.json"
ADAPTER_CKPT_PATH = PRODUCT_DIR / "adapter.pt"
BEST_ADAPTER_CKPT_PATH = PRODUCT_DIR / "best_adapter.pt"
VERDICT_PATH = PRODUCT_DIR / "verdict.json"

state = {
    "killed": False, "val_trace": [], "grad_norm_history": [], "backbone_drift": [],
    "fd_sensitivity": [], "component_norms": [], "mixing_weights": [],
    "kappa_trace": [], "mixing_grad_trace": [], "killed_reason": None,
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
            json.dump({"val_trace": state["val_trace"]}, f, indent=2, default=str)
        with open(GRAD_NORM_PATH, "w") as f:
            json.dump({"grad_norm_history": state["grad_norm_history"]}, f, indent=2, default=str)
        with open(BACKBONE_DRIFT_PATH, "w") as f:
            json.dump({"backbone_drift": state["backbone_drift"]}, f, indent=2, default=str)
        with open(FD_SENS_PATH, "w") as f:
            json.dump({"fd_sensitivity": state["fd_sensitivity"]}, f, indent=2, default=str)
        with open(COMPONENT_PATH, "w") as f:
            json.dump({"component_norms": state["component_norms"]}, f, indent=2, default=str)
        with open(MIXING_PATH, "w") as f:
            json.dump({"mixing_weights": state["mixing_weights"]}, f, indent=2, default=str)
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


# val — 已迁移到 common/stage4_eval_beam20.py::run_val_beam20 (baseline 同协议 beam20)


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
    gn = {"adapter_total": 0.0, "conditioner_total": 0.0, "kappa_logits": 0.0, "mixing_logits": 0.0}
    for n, p in model_wrapper.named_parameters():
        if p.grad is not None and torch.isfinite(p.grad).all():
            n_sq = p.grad.norm().item() ** 2
            if "kappa_logits" in n:
                gn["kappa_logits"] += n_sq
            elif "mixing_logits" in n:
                gn["mixing_logits"] += n_sq
            elif any(sub in n for sub in ("curvature_embed", "conditioner_heads", "sid_token_proj")):
                gn["conditioner_total"] += n_sq
            elif "adapter." in n:
                gn["adapter_total"] += n_sq
    for k in gn:
        gn[k] = gn[k] ** 0.5
    return gn


def finite_diff_sensitivity(model_wrapper, ht, B):
    """有限差分敏感度: 对 kappa_logits / mixing_logits 扰动 FD_EPS, 测 adapter residual 变化 / eps.
    证明 κ 与 mixing 真正影响 forward (Task-Geometry coupling)."""
    model_wrapper.eval()
    with torch.no_grad():
        x_emb = model_wrapper.t5.model.shared(ht)  # (B, L, d_model)
        sid_meta = torch.zeros(B, ht.shape[1], 4, device=DEVICE)
        curv = model_wrapper.adapter.build_curvature_meta(B)
        residual_base, _ = model_wrapper.adapter(x_emb, sid_meta, curv)
        base_norm = residual_base.norm().item()
        # kappa sensitivity
        kappa_sens = []
        for l in range(3):
            orig = model_wrapper.adapter.kappa_logits[l].item()
            model_wrapper.adapter.kappa_logits.data[l] += FD_EPS
            curv_p = model_wrapper.adapter.build_curvature_meta(B)
            res_p, _ = model_wrapper.adapter(x_emb, sid_meta, curv_p)
            model_wrapper.adapter.kappa_logits.data[l] = orig
            d = (res_p - residual_base).norm().item()
            kappa_sens.append(round(d / FD_EPS, 6))
        # mixing sensitivity
        mixing_sens = []
        for l in range(3):
            row = []
            for j in range(3):
                orig = model_wrapper.adapter.mixing_logits[l, j].item()
                model_wrapper.adapter.mixing_logits.data[l, j] += FD_EPS
                curv_p = model_wrapper.adapter.build_curvature_meta(B)
                res_p, _ = model_wrapper.adapter(x_emb, sid_meta, curv_p)
                model_wrapper.adapter.mixing_logits.data[l, j] = orig
                d = (res_p - residual_base).norm().item()
                row.append(round(d / FD_EPS, 6))
            mixing_sens.append(row)
    return {"kappa_sens_per_layer": kappa_sens, "mixing_sens_per_layer": mixing_sens,
            "residual_base_norm": round(base_norm, 6)}


def per_head_component_norms(model_wrapper, ht, B, L):
    """每层 3 个 conditioner_heads 输出方向范数 (固定双曲/欧氏分量贡献)."""
    model_wrapper.eval()
    with torch.no_grad():
        x_emb = model_wrapper.t5.model.shared(ht)
        sid_meta = torch.zeros(B, L, 4, device=DEVICE)
        curv = model_wrapper.adapter.build_curvature_meta(B)
        sid_e = model_wrapper.adapter.sid_token_proj(sid_meta)
        norms = []
        for layer_i in range(3):
            curv_e = model_wrapper.adapter.curvature_embed_per_layer[layer_i](curv[:, layer_i, :])
            curv_e = curv_e.unsqueeze(1).expand(-1, L, -1)
            combined = torch.cat([sid_e, curv_e], dim=-1)
            head_out = model_wrapper.adapter.conditioner_heads[layer_i](combined)
            norms.append(round(head_out.norm().item(), 6))
    return norms


def main():
    with open(TRAINING_PID_FILE, "w") as f:
        f.write(str(os.getpid()) + "\n")

    log_lines = []
    log_lines.append("=" * 70)
    log_lines.append("[Issue #29 Gate2 diag] 混合曲率 forward 贡献诊断")
    log_lines.append("=" * 70)

    sid_sha = sha256_of(SID_NPY)
    t5_sha = sha256_of(T5_CKPT)
    prior_sha = sha256_of(PRIOR_CKPT)
    log_lines.append(f"\n[SHA256] SID_NPY: {sid_sha}")
    log_lines.append(f"[SHA256] T5_CKPT: {t5_sha}")
    log_lines.append(f"[SHA256] PRIOR_CKPT: {prior_sha}")
    log_lines.append(f"[SID hash match] {sid_sha == EXPECTED_SID_SHA}")
    if sid_sha != EXPECTED_SID_SHA:
        log_lines.append("[FAIL] SID hash mismatch, abort")
        with open(LOG_PATH, "w") as f:
            f.write("\n".join(log_lines) + "\n")
        return

    spec = importlib.util.spec_from_file_location(
        "opt_d",
        "/home/wlia0047/ar57/wenyu/GeneRec/taskB/stage3/taskB_stage3_issue23_option_d.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    WrapperCls, get_t5_config = mod.load_wrapper_cls()
    t5_state_dict = torch.load(T5_CKPT, map_location="cpu", weights_only=False)
    if "state_dict" in t5_state_dict:
        t5_state_dict = t5_state_dict["state_dict"]
    elif "model" in t5_state_dict:
        t5_state_dict = t5_state_dict["model"]
    t5_config = get_t5_config()
    model_wrapper = WrapperCls(t5_config, t5_state_dict, d_model=D_MODEL, n_layers=3, sid_dim=4)
    model_wrapper.to(DEVICE)
    frozen_t5_state_dict = t5_state_dict

    # prior = #27 best_adapter (epoch 60)
    prior_ckpt = torch.load(PRIOR_CKPT, map_location=DEVICE, weights_only=False)
    log_lines.append(f"[Load prior ckpt] epoch={prior_ckpt.get('epoch')}, alpha={prior_ckpt.get('alpha')}")
    model_wrapper.adapter.load_state_dict(prior_ckpt["adapter_state_dict"], strict=False)
    if "ln_state_dict" in prior_ckpt:
        model_wrapper.first_input_ln.load_state_dict(prior_ckpt["ln_state_dict"])
    elif "first_input_ln_state_dict" in prior_ckpt:
        model_wrapper.first_input_ln.load_state_dict(prior_ckpt["first_input_ln_state_dict"])

    kappa_start = model_wrapper.adapter.get_kappa_per_layer().detach().cpu().tolist()
    mixing_start = model_wrapper.adapter.get_mixing_per_layer().detach().cpu().tolist()
    log_lines.append(f"\n[kappa_per_layer start] {kappa_start}")
    log_lines.append(f"[mixing_per_layer start] {mixing_start}")

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
    layer_ranges = get_layer_ranges(CODEBOOK_SIZE)

    # Optimizer (issue193 结构, 3 param groups)
    conditioner_params = [p for n, p in model_wrapper.adapter.named_parameters()
                          if "kappa_logits" not in n and "mixing_logits" not in n]
    kappa_mixing_params = [model_wrapper.adapter.kappa_logits, model_wrapper.adapter.mixing_logits]
    layernorm_params = list(model_wrapper.first_input_ln.parameters())
    optimizer = torch.optim.Adam([
        {"params": conditioner_params, "lr": LR_CONDITIONER},
        {"params": kappa_mixing_params, "lr": LR_CONDITIONER},
        {"params": layernorm_params, "lr": LR_LAYERNORM},
    ])

    rng = torch.Generator().manual_seed(42 + 8)
    n_batches = (len(train_indices) + BATCH_SIZE - 1) // BATCH_SIZE
    best_val_r10 = -1.0
    best_epoch = -1
    start_time = time.time()
    B_probe = 32

    for epoch in range(START_EPOCH, START_EPOCH + NUM_EPOCHS):
        epoch_losses = []
        cond_grad_norms = []
        ln_grad_norms = []
        kappa_grads = []   # 逐层 dL/dkappa (3,)
        mixing_grads = []  # 逐层 dL/dmix (3,3)
        nan_inf = False
        epoch_indices = torch.randperm(len(train_indices), generator=rng).to(DEVICE)
        for batch_idx in range(n_batches):
            start = batch_idx * BATCH_SIZE
            end = min(start + BATCH_SIZE, len(train_indices))
            bi = epoch_indices[start:end]
            ht = all_histories_t[train_indices[bi]]
            tt = all_targets_t[train_indices[bi]]
            am = (ht != PAD_TOKEN).long()
            B = ht.shape[0]
            sid_meta = torch.zeros(B, ht.shape[1], 4, device=DEVICE)
            curvature_meta = model_wrapper.adapter.build_curvature_meta(B)
            optimizer.zero_grad()
            output, _, alpha = model_wrapper(ht, attention_mask=am, labels=tt, sid_meta=sid_meta, curvature_meta=curvature_meta)
            loss = output.loss if hasattr(output, "loss") else output[0]
            if not torch.isfinite(loss):
                nan_inf = True
                continue
            loss.backward()
            cond_grad_norm = sum((p.grad.norm().item() ** 2) for p in conditioner_params if p.grad is not None and torch.isfinite(p.grad).all()) ** 0.5
            ln_grad_norm = sum((p.grad.norm().item() ** 2) for p in layernorm_params if p.grad is not None and torch.isfinite(p.grad).all()) ** 0.5
            if model_wrapper.adapter.kappa_logits.grad is not None:
                kappa_grads.append(model_wrapper.adapter.kappa_logits.grad.detach().cpu().tolist())
            if model_wrapper.adapter.mixing_logits.grad is not None:
                mixing_grads.append(model_wrapper.adapter.mixing_logits.grad.detach().cpu().tolist())
            gn = grad_norm_per_group(model_wrapper)
            optimizer.step()
            epoch_losses.append(loss.item())
            cond_grad_norms.append(cond_grad_norm)
            ln_grad_norms.append(ln_grad_norm)
            if state["killed"]:
                flush_evidence()
                return
        avg_loss = np.mean(epoch_losses) if epoch_losses else float("nan")
        avg_cond = np.mean(cond_grad_norms) if cond_grad_norms else 0.0
        avg_ln = np.mean(ln_grad_norms) if ln_grad_norms else 0.0
        kappa_per_layer = model_wrapper.adapter.get_kappa_per_layer().detach().cpu().tolist()
        mixing_per_layer = model_wrapper.adapter.get_mixing_per_layer().detach().cpu().tolist()
        avg_dLdkappa = np.mean(kappa_grads, axis=0).tolist() if kappa_grads else None
        avg_dLdmix = np.mean(mixing_grads, axis=0).tolist() if mixing_grads else None
        elapsed = time.time() - start_time
        msg = f"  [epoch {epoch}] loss={avg_loss:.4f}, cond_grad={avg_cond:.4e}, ln_grad={avg_ln:.4e}, kappa={kappa_per_layer}, nan_inf={nan_inf}, elapsed={elapsed/60:.1f}m"
        log_lines.append(msg)
        log_lines.append(f"    avg dL/dkappa = {[f'{x:.2e}' for x in avg_dLdkappa] if avg_dLdkappa else 'n/a'}")
        log_lines.append(f"    avg dL/dmix  = {[[f'{v:.2e}' for v in row] for row in avg_dLdmix] if avg_dLdmix else 'n/a'}")
        log_lines.append(f"    mixing = {mixing_per_layer}")
        state["kappa_trace"].append({"epoch": epoch, "kappa_per_layer": kappa_per_layer, "mixing_per_layer": mixing_per_layer,
                                     "avg_dLdkappa": avg_dLdkappa, "avg_dLdmix": avg_dLdmix})

        # 有限差分敏感度 (每 epoch, probe batch)
        fd = finite_diff_sensitivity(model_wrapper, val_histories_t[:B_probe], B_probe)
        state["fd_sensitivity"].append({"epoch": epoch, **fd})
        log_lines.append(f"    fd kappa_sens = {fd['kappa_sens_per_layer']}")
        log_lines.append(f"    fd mixing_sens = {fd['mixing_sens_per_layer']}")

        # 分量范数 (每 epoch, probe batch)
        comp = per_head_component_norms(model_wrapper, val_histories_t[:B_probe], B_probe, val_histories_t.shape[1])
        state["component_norms"].append({"epoch": epoch, "per_head_norms": comp})
        log_lines.append(f"    per_head_component_norms = {comp}")

        # backbone 漂移
        drift = backbone_drift_check(model_wrapper, frozen_t5_state_dict, val_histories_t[:32], t5_config, DEVICE)
        state["backbone_drift"].append({"epoch": epoch, "drift_per_layer": drift})
        min_cos = min(d["cosine_sim"] for d in drift)
        log_lines.append(f"    [drift] min_cosine_sim={min_cos:.4f}")

        # 4 组 grad norm
        gn = grad_norm_per_group(model_wrapper)
        state["grad_norm_history"].append({"epoch": epoch, **{f"avg_{k}": v for k, v in gn.items()}})

        # val (共享路径)
        # Issue #30: baseline 同协议 beam20, 替代 greedy 单候选 — 真实 R@10
        val_r10, val_in_range_pct = run_val_beam20(
            model_wrapper, val_histories_t, val_targets_t, VAL_EVAL_N, BATCH_SIZE, DEVICE,
            val_sid_meta=val_sid_meta, val_kappa_meta=val_kappa_meta,
        )
        state["val_trace"].append({"epoch": epoch, "val_r10": val_r10, "val_in_range_pct": val_in_range_pct,
                                   "kappa_per_layer": kappa_per_layer, "mixing_per_layer": mixing_per_layer})
        if val_r10 > best_val_r10:
            best_val_r10 = val_r10
            best_epoch = epoch
            if BEST_ADAPTER_CKPT_PATH.exists():
                BEST_ADAPTER_CKPT_PATH.unlink()
            torch.save({
                "adapter_state_dict": model_wrapper.adapter.state_dict(),
                "ln_state_dict": model_wrapper.first_input_ln.state_dict(),
                "epoch": epoch, "val_r10": val_r10, "alpha": model_wrapper.adapter.get_alpha(),
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

    # ================= verdict =================
    kappa_end = model_wrapper.adapter.get_kappa_per_layer().detach().cpu().tolist()
    mixing_end = model_wrapper.adapter.get_mixing_per_layer().detach().cpu().tolist()
    kappa_delta = [abs(a - b) for a, b in zip(kappa_end, kappa_start)]

    # 判定项
    kappa_updated = max(kappa_delta) >= 1e-3                       # κ 有效更新
    mixing_max = max(max(row) for row in mixing_end)
    mixing_nondegenerate = mixing_max < MIXING_DEGENERATE_TH        # mixing 非退化
    all_fd = [e for h in state["fd_sensitivity"] for e in h["kappa_sens_per_layer"]] + \
             [e for h in state["fd_sensitivity"] for row in h["mixing_sens_per_layer"] for e in row]
    fd_sensitive = (len(all_fd) > 0 and max(all_fd) > 1e-3)        # 有限差分敏感度 > 0
    comp_all = [e for h in state["component_norms"] for e in h["per_head_norms"]]
    comp_differentiated = len(comp_all) > 0 and (max(comp_all) - min(comp_all)) > 1e-3  # 分量贡献分化
    min_cos_all = min(e["cosine_sim"] for h in state["backbone_drift"] for e in h["drift_per_layer"]) if state["backbone_drift"] else 0.0
    backbone_ok = min_cos_all >= 0.99
    val_ok = best_val_r10 >= 0.05 and BEST_ADAPTER_CKPT_PATH.exists()

    final_verdict = "PASS" if (kappa_updated and mixing_nondegenerate and fd_sensitive and comp_differentiated and backbone_ok) else "FAIL"
    verdict = {
        "issue": 29,
        "step": "[方向B Gate2] 混合曲率训练稳定性与表示流收口 — forward 贡献诊断",
        "gate": "Gate2 诊断",
        "verdict": final_verdict,
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "diagnostics": {
            "kappa_update": {"pass": kappa_updated, "start": kappa_start, "end": kappa_end, "delta": kappa_delta},
            "mixing_nondegenerate": {"pass": mixing_nondegenerate, "mixing_max_weight": mixing_max, "end": mixing_end},
            "fd_sensitivity": {"pass": fd_sensitive, "max_sens": max(all_fd) if all_fd else 0.0,
                               "kappa_sens_max": max(e for h in state["fd_sensitivity"] for e in h["kappa_sens_per_layer"]) if state["fd_sensitivity"] else 0.0,
                               "mixing_sens_max": max(e for h in state["fd_sensitivity"] for row in h["mixing_sens_per_layer"] for e in row) if state["fd_sensitivity"] else 0.0},
            "component_differentiation": {"pass": comp_differentiated, "per_head_norms_epochs": state["component_norms"]},
            "backbone_drift": {"pass": backbone_ok, "min_cosine_sim": min_cos_all},
            "val_canary": {"pass": val_ok, "best_val_r10": best_val_r10, "best_epoch": best_epoch,
                           "best_adapter_exists": BEST_ADAPTER_CKPT_PATH.exists()},
        },
        "config": {
            "prior_ckpt": PRIOR_CKPT, "prior_epoch": prior_ckpt.get("epoch"),
            "lr_conditioner": LR_CONDITIONER, "lr_layernorm": LR_LAYERNORM, "fd_eps": FD_EPS,
            "codebook_size": CODEBOOK_SIZE,
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
        "best_adapter_exists": BEST_ADAPTER_CKPT_PATH.exists(),
        "evidence_persisted": {
            "fd_sensitivity": str(FD_SENS_PATH), "component_norms": str(COMPONENT_PATH),
            "mixing_weights": str(MIXING_PATH), "val_trace": str(VAL_TRACE_PATH),
            "backbone_drift": str(BACKBONE_DRIFT_PATH), "grad_norm_history": str(GRAD_NORM_PATH),
        },
        "verdict_path": str(VERDICT_PATH),
    }
    with open(VERDICT_PATH, "w") as f:
        json.dump(verdict, f, indent=2, default=str)
    log_lines.append(f"\n[Verdict written] {VERDICT_PATH}")
    log_lines.append(f"[kappa_updated] {kappa_updated} | [mixing_nondeg] {mixing_nondegenerate} | [fd_sens] {fd_sensitive} | [comp_diff] {comp_differentiated} | [backbone] {backbone_ok} | [val>=0.05] {val_ok}")
    log_lines.append(f"[FINAL] {final_verdict}")
    print("\n".join(log_lines), flush=True)


if __name__ == "__main__":
    main()
