#!/usr/bin/env python3
"""Issue #27    + best_adapter  + backbone .

Per #27 spec ( =  epoch, ):
-  issue193_long_run/adapter.pt (epoch 59)  ,  <=5 epoch (epoch 60-64)
-  common/stage4_decode.py::autoregressive_predict_constrained  val ( ,
- best_adapter.pt   (issue193 long_run epoch 50 val_R@10=0.058,  issue27   val_R@10  )
- val_trace.json  epoch  sid_sha256 + ckpt_sha256 + dtype + in_range_pct
- SIGTERM/SIGINT handler + atexit ( kill )
- T2: backbone   cosine similarity per layer + 4  grad norm (adapter / conditioner / kappa_logits / mixing_logits)

:  LR / gradient clip / warmup /  /  /
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

os.environ["TRITON_CACHE_DIR"] = "/home/wlia0047/.triton/cache_issue27"
os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)

from common.stage4_decode import (
    autoregressive_predict_constrained, get_layer_ranges, compute_r_at_k,
)
from dataset import GenRecDataset

# ============================================================================
# Config (per #27 spec  =  epoch, )
# ============================================================================
SEED = 42
DEVICE = "cuda:1"
CODEBOOK_SIZE = [64, 128, 256, 1]
MAX_LEN = 20
PAD_TOKEN = 0
D_MODEL = 128
BATCH_SIZE = 32
NUM_EPOCHS = 5
START_EPOCH = 60
VAL_EVAL_N = 200

#  issue193 long_run  (per spec )
LR_CONDITIONER = 1e-3
LR_LAYERNORM = 1e-4
LR_KAPPA = 1e-3
LR_MIXING = 1e-3
EARLY_STOP_PATIENCE = 10
MIN_DELTA = 1e-4

PRIOR_CKPT = "/home/wlia0047/ar57/wenyu/GeneRec/taskB/stage3/taskB_stage3_issue193_long_run/adapter.pt"
T5_CKPT = "/home/wlia0047/ar57/wenyu/GeneRec/taskB/_ckpt/HG_Rec_best.pth"
SID_NPY = "/home/wlia0047/ar57/wenyu/GeneRec/taskB/_data/Instruments/Instruments_t5_hrqvae_poincare.npy"
TRAIN_PARQUET = "/home/wlia0047/ar57/wenyu/GeneRec/taskB/_data/Instruments/train.parquet"
EXPECTED_SID_SHA = "2dab29229c36a9695a11d70b61d1b80fae95a08d00e3c3675e9bf899b709508a"
T5_FROZEN_CKPT = T5_CKPT  #  T5_CKPT  HG-Rec_best.pth = frozen T5 ckpt

PRODUCT_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/taskB/stage3/taskB_stage3_issue27_close")
PRODUCT_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_PID_FILE = PRODUCT_DIR / "_TRAINING_PID"
LOG_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/taskB/_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "task_issue27_close.log"
VAL_TRACE_PATH = PRODUCT_DIR / "val_trace.json"
BACKBONE_DRIFT_PATH = PRODUCT_DIR / "backbone_drift.json"
GRAD_NORM_PATH = PRODUCT_DIR / "grad_norm_history.json"
ADAPTER_CKPT_PATH = PRODUCT_DIR / "adapter.pt"
BEST_ADAPTER_CKPT_PATH = PRODUCT_DIR / "best_adapter.pt"
VERDICT_PATH = PRODUCT_DIR / "verdict.json"

state = {"killed": False, "val_trace": [], "grad_norm_history": [], "backbone_drift": []}


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
    except Exception as e:
        with open(LOG_PATH, "a") as f:
            f.write(f"[atexit flush error] {e}\n")


def signal_handler(signum, frame):
    state["killed"] = True
    flush_evidence()
    with open(LOG_PATH, "a") as f:
        f.write(f"\n[SIGNAL {signum}] emergency flush done\n")
    sys.exit(0)


signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)
atexit.register(flush_evidence)


def backbone_drift_check(model_wrapper, frozen_t5_state_dict, histories_batch, t5_config, device):
    """T2: backbone drift. frozen ckpt + wrapper T5 encoder vs frozen encoder (use full encoder.forward, not per-block to avoid HF T5 4D mask issue)."""
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
        # Per-layer via encoder.block forward with 4D extended mask
        x_emb_curr = model_wrapper.t5.model.shared(histories_batch)
        x_emb_frozen = frozen_t5.model.shared(histories_batch)
        hidden_curr = x_emb_curr
        hidden_frozen = x_emb_frozen
        # T5 v1 expects extended_attention_mask (B, 1, L, L) for encoder self-attention
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
    """T2: 4  grad norm (adapter / conditioner / kappa_logits / mixing_logits)."""
    gn = {"adapter_total": 0.0, "conditioner_total": 0.0, "kappa_logits": 0.0, "mixing_logits": 0.0}
    for n, p in model_wrapper.named_parameters():
        if p.grad is not None and torch.isfinite(p.grad).all():
            n_sq = p.grad.norm().item() ** 2
            if "adapter.curvature_embed" in n or "adapter.conditioner" in n:
                gn["conditioner_total"] += n_sq
            elif "kappa_logits" in n:
                gn["kappa_logits"] += n_sq
            elif "mixing_logits" in n:
                gn["mixing_logits"] += n_sq
            elif "adapter." in n:
                gn["adapter_total"] += n_sq
    for k in gn:
        gn[k] = gn[k] ** 0.5
    return gn


def run_val_shared(model_wrapper, val_histories_t, val_targets_t, val_n, batch_size, layer_ranges, device):
    """ val."""
    model_wrapper.eval()
    n = min(val_n, val_histories_t.shape[0])
    rng = torch.Generator().manual_seed(42 + 13)
    sample_idx = torch.randperm(val_histories_t.shape[0], generator=rng)[:n]
    histories = val_histories_t[sample_idx]
    targets = val_targets_t[sample_idx]
    preds = []
    targets_all = []
    in_range_count = 0
    with torch.no_grad():
        for batch_start in range(0, n, batch_size):
            batch_end = min(batch_start + batch_size, n)
            B = batch_end - batch_start
            ht = histories[batch_start:batch_end]
            tt = targets[batch_start:batch_end]
            am = (ht != PAD_TOKEN).long()
            curvature_meta = model_wrapper.adapter.build_curvature_meta(B)
            pred = autoregressive_predict_constrained(
                model_wrapper, ht, am, layer_ranges,
                sid_meta=None, curvature_meta=curvature_meta,
                max_len=4, mask_value=-1e9, device=device,
            )
            for i in range(B):
                p = pred[i].cpu().tolist()
                t = tt[i].cpu().tolist()
                preds.append(p[:4])
                targets_all.append(t[:4])
                for li, tok in enumerate(p[:4]):
                    lo, hi = layer_ranges[li]
                    if lo <= tok <= hi:
                        in_range_count += 1
    r10 = float(np.mean([1.0 if p == t else 0.0 for p, t in zip(preds, targets_all)]))
    return r10, in_range_count / (len(preds) * 4), preds, targets_all


def main():
    pid = os.getpid()
    with open(TRAINING_PID_FILE, "w") as f:
        f.write(str(pid) + "\n")

    log_lines = []
    log_lines.append("=" * 70)
    log_lines.append("[Issue #27] taskB Gate3  +  + best_adapter  + backbone ")
    log_lines.append("=" * 70)

    sid_sha = sha256_of(SID_NPY)
    t5_sha = sha256_of(T5_CKPT)
    prior_sha = sha256_of(PRIOR_CKPT)
    t5_frozen_sha = sha256_of(T5_FROZEN_CKPT) if os.path.exists(T5_FROZEN_CKPT) else "n/a"
    log_lines.append(f"\n[SHA256] SID_NPY: {sid_sha}")
    log_lines.append(f"[SHA256] T5_CKPT: {t5_sha}")
    log_lines.append(f"[SHA256] PRIOR_CKPT: {prior_sha}")
    log_lines.append(f"[SHA256] T5_FROZEN_CKPT: {t5_frozen_sha}")
    log_lines.append(f"[SID hash match expected #157] {sid_sha == EXPECTED_SID_SHA}")

    if sid_sha != EXPECTED_SID_SHA:
        log_lines.append("[FAIL] SID hash mismatch, abort")
        with open(LOG_PATH, "w") as f:
            f.write("\n".join(log_lines) + "\n")
        return

    #  wrapper
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

    #  prior ckpt (issue193 epoch 59 adapter.pt)
    prior_ckpt = torch.load(PRIOR_CKPT, map_location=DEVICE, weights_only=False)
    log_lines.append(f"[Load prior ckpt] epoch={prior_ckpt.get('epoch')}, alpha={prior_ckpt.get('alpha')}")
    model_wrapper.adapter.load_state_dict(prior_ckpt["adapter_state_dict"], strict=False)
    if "ln_state_dict" in prior_ckpt:
        model_wrapper.first_input_ln.load_state_dict(prior_ckpt["ln_state_dict"])
    elif "first_input_ln_state_dict" in prior_ckpt:
        model_wrapper.first_input_ln.load_state_dict(prior_ckpt["first_input_ln_state_dict"])

    #  frozen T5  backbone drift  (per #27 T2)
    frozen_t5_state_dict = torch.load(T5_FROZEN_CKPT, map_location="cpu", weights_only=False) if os.path.exists(T5_FROZEN_CKPT) else t5_state_dict
    if "state_dict" in frozen_t5_state_dict:
        frozen_t5_state_dict = frozen_t5_state_dict["state_dict"]
    elif "model" in frozen_t5_state_dict:
        frozen_t5_state_dict = frozen_t5_state_dict["model"]

    # 
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
    n_val = int(n_train * 0.2)
    val_split_idx = n_train - n_val
    train_indices = torch.arange(0, val_split_idx, device=DEVICE)
    val_histories_t = torch.from_numpy(all_histories[val_split_idx:]).to(DEVICE).long()
    val_targets_t = torch.from_numpy(all_targets[val_split_idx:]).to(DEVICE).long()

    layer_ranges = get_layer_ranges(CODEBOOK_SIZE)
    log_lines.append(f"[Layer ranges] {layer_ranges}")
    log_lines.append(f"[Data] n_train={len(train_indices)}, n_val={n_val}")

    # Optimizer (issue193 )
    conditioner_params = list(model_wrapper.adapter.parameters())
    layernorm_params = list(model_wrapper.first_input_ln.parameters())
    optimizer = torch.optim.Adam([
        {"params": conditioner_params, "lr": LR_CONDITIONER},
        {"params": layernorm_params, "lr": LR_LAYERNORM},
    ])

    rng = torch.Generator().manual_seed(42 + 8)
    n_batches = (len(train_indices) + BATCH_SIZE - 1) // BATCH_SIZE
    best_val_r10 = -1.0
    best_epoch = -1
    early_stop_triggered = False
    start_time = time.time()
    kappa_per_layer = model_wrapper.adapter.get_kappa_per_layer().detach().cpu().tolist()

    #  backbone drift  (epoch 60 baseline)
    log_lines.append("\n[T2 backbone drift @ epoch 60 baseline]")
    drift_base = backbone_drift_check(model_wrapper, frozen_t5_state_dict, val_histories_t[:32], t5_config, DEVICE)
    state["backbone_drift"].append({"epoch": START_EPOCH, "drift_per_layer": drift_base})
    for d in drift_base:
        log_lines.append(f"  layer {d['layer']}: cosine_sim = {d['cosine_sim']:.4f}")

    for epoch in range(START_EPOCH, START_EPOCH + NUM_EPOCHS):
        epoch_losses = []
        cond_grad_norms = []
        ln_grad_norms = []
        gn_history_step = []
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
            gn = grad_norm_per_group(model_wrapper)
            gn_history_step.append(gn)
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
        gn_avg = {k: float(np.mean([g[k] for g in gn_history_step])) for k in gn_history_step[0]}
        kappa_per_layer = model_wrapper.adapter.get_kappa_per_layer().detach().cpu().tolist()
        elapsed = time.time() - start_time
        msg = f"  [epoch {epoch}] loss={avg_loss:.4f}, cond_grad={avg_cond:.4e}, ln_grad={avg_ln:.4e}, kappa_per_layer={kappa_per_layer}, nan_inf={nan_inf}, elapsed={elapsed/60:.1f}m"
        log_lines.append(msg)
        state["grad_norm_history"].append({"epoch": epoch, **{f"avg_{k}": v for k, v in gn_avg.items()}})

        # backbone drift
        drift = backbone_drift_check(model_wrapper, frozen_t5_state_dict, val_histories_t[:32], t5_config, DEVICE)
        state["backbone_drift"].append({"epoch": epoch, "drift_per_layer": drift})

        # val ()
        val_r10, val_in_range_pct, preds, tgts = run_val_shared(model_wrapper, val_histories_t, val_targets_t, VAL_EVAL_N, BATCH_SIZE, layer_ranges, DEVICE)
        ckpt_sha_now = sha256_of(ADAPTER_CKPT_PATH) if ADAPTER_CKPT_PATH.exists() else "n/a"
        state["val_trace"].append({
            "epoch": epoch,
            "val_r10": val_r10,
            "val_in_range_pct": val_in_range_pct,
            "sid_sha256": sid_sha,
            "sid_sha_match": sid_sha == EXPECTED_SID_SHA,
            "ckpt_sha256": ckpt_sha_now,
            "model_dtype": str(next(model_wrapper.parameters()).dtype),
            "kappa_per_layer": kappa_per_layer,
            "in_range_pct": val_in_range_pct,
        })
        if val_r10 > best_val_r10:
            best_val_r10 = val_r10
            best_epoch = epoch
            if BEST_ADAPTER_CKPT_PATH.exists():
                BEST_ADAPTER_CKPT_PATH.unlink()
            torch.save({
                "adapter_state_dict": model_wrapper.adapter.state_dict(),
                "ln_state_dict": model_wrapper.first_input_ln.state_dict(),
                "epoch": epoch, "val_r10": val_r10,
                "alpha": model_wrapper.adapter.get_alpha(),
            }, BEST_ADAPTER_CKPT_PATH)
        log_lines.append(f"  [val @ epoch {epoch}] val_R@10={val_r10:.4f}, in-range={val_in_range_pct*100:.1f}%, best={best_val_r10:.4f} @ epoch {best_epoch}, best_adapter={'YES' if BEST_ADAPTER_CKPT_PATH.exists() else 'NO'}")

        if ADAPTER_CKPT_PATH.exists():
            ADAPTER_CKPT_PATH.unlink()
        torch.save({
            "adapter_state_dict": model_wrapper.adapter.state_dict(),
            "ln_state_dict": model_wrapper.first_input_ln.state_dict(),
            "epoch": epoch,
        }, ADAPTER_CKPT_PATH)
        flush_evidence()

    # verdict
    verdict = {
        "issue": 27,
        "step": "T1  +  + best_adapter  + T2 backbone ",
        "gate": "Gate 3",
        "verdict": "PARTIAL PASS" if best_val_r10 >= 0.058 else "FAIL  best_adapter  val  baseline",
        "best_val_r10": best_val_r10,
        "best_epoch": best_epoch,
        "best_adapter_exists": BEST_ADAPTER_CKPT_PATH.exists(),
        "best_adapter_sha256": sha256_of(BEST_ADAPTER_CKPT_PATH) if BEST_ADAPTER_CKPT_PATH.exists() else None,
        "val_trace": state["val_trace"],
        "backbone_drift": state["backbone_drift"],
        "grad_norm_history": state["grad_norm_history"],
        "sid_sha256": sid_sha,
        "sid_sha_match": sid_sha == EXPECTED_SID_SHA,
        "t5_sha256": t5_sha,
        "prior_sha256": prior_sha,
        "kappa_per_layer_final": model_wrapper.adapter.get_kappa_per_layer().detach().cpu().tolist(),
        "mixing_per_layer_final": model_wrapper.adapter.get_mixing_per_layer().detach().cpu().tolist(),
    }
    with open(VERDICT_PATH, "w") as f:
        json.dump(verdict, f, indent=2, default=str)
    log_lines.append(f"\n[Verdict written] {VERDICT_PATH}")
    log_lines.append(f"[best_val_R@10] {best_val_r10}, best_epoch={best_epoch}, best_adapter={BEST_ADAPTER_CKPT_PATH.exists()}")

    print("\n".join(log_lines), flush=True)
    with open(LOG_PATH, "w") as f:
        f.write("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    main()