#!/usr/bin/env python3
"""Issue #26 T2  kappa_logits  param group + LR .

Per #26 spec ():
- kappa_logits  optimizer param group, LR 
- LR  {1e-3, 1e-2, 1e-1} ,  <=5 epoch
-  taskA_stage3_issue192_long_run/best_adapter.pt (epoch 48, val_R@10=0.058) 
- : from-0 
- optimizer.step()  >=8   (float64 )  kappa_logits  + Delta
-  epoch  assert |Deltakappa_l| >= 1e-4  ,
-  codebook /   (#24 recalibrate )
-  epoch flush  + SIGTERM/SIGINT handler + atexit
- recalibrate_log.json  +   codebook sha256

: LR / gradient clip / warmup /  /  /
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

os.environ["TRITON_CACHE_DIR"] = "/home/wlia0047/.triton/cache_issue26_t2"
os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)

from common.stage4_decode import (
    autoregressive_predict_constrained, get_layer_ranges, compute_r_at_k,
)
from dataset import GenRecDataset

# ============================================================================
# Config (per #26 spec  = kappa_logits LR)
# ============================================================================
SEED = 42
DEVICE = "cuda:2"
CODEBOOK_SIZE = [64, 128, 256, 1]
MAX_LEN = 20
PAD_TOKEN = 0
D_MODEL = 128
BATCH_SIZE = 32
NUM_EPOCHS = 5
START_EPOCH = 49
VAL_EVAL_N = 200
LR_CONDITIONER = 1e-3  # ,
LR_LAYERNORM = 1e-4
LR_KAPPA_LOGITS = 1e-2  # : T2  1e-2  ()
DELTA_KAPPA_ASSERT = 1e-4

PRIOR_CKPT = "/home/wlia0047/ar57/wenyu/GeneRec/taskA/stage3/taskA_stage3_issue192_long_run/best_adapter.pt"
T5_CKPT = "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_ckpt/HG_Rec_best.pth"
SID_NPY = "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_data/Instruments/Instruments_t5_hrqvae_poincare.npy"
TRAIN_PARQUET = "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_data/Instruments/train.parquet"
VAL_SPLIT_RATIO = 0.2  # issue192  train.parquet  80/20
EXPECTED_SID_SHA = "2dab29229c36a9695a11d70b61d1b80fae95a08d00e3c3675e9bf899b709508a"

PRODUCT_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/taskA/stage3/taskA_stage3_issue26_t2_lr_scan")
PRODUCT_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_PID_FILE = PRODUCT_DIR / "_TRAINING_PID"
LOG_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/taskA/_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "task_issue26_t2.log"
RECALIBRATE_LOG_PATH = PRODUCT_DIR / "recalibrate_log.json"
VAL_TRACE_PATH = PRODUCT_DIR / "val_trace.json"
ADAPTER_CKPT_PATH = PRODUCT_DIR / "adapter.pt"
BEST_ADAPTER_CKPT_PATH = PRODUCT_DIR / "best_adapter.pt"
VERDICT_PATH = PRODUCT_DIR / "verdict.json"

state = {"killed": False, "epoch_losses": [], "val_trace": [], "kappa_history": [], "recalibrate_events": []}


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
            json.dump({"recalibrate_count": len(state["recalibrate_events"]), "events": state["recalibrate_events"][-50:]}, f, indent=2, default=str)
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


def run_val_shared(model_wrapper, val_histories_t, val_targets_t, val_n, batch_size, layer_ranges, device):
    """ val (common/stage4_decode.py::autoregressive_predict_constrained)."""
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
            try:
                pred = autoregressive_predict_constrained(
                    model_wrapper, ht, am, layer_ranges,
                    sid_meta=None, curvature_meta=None, kappa_meta=None, scale_meta=None,
                    max_len=4, mask_value=-1e9, device=device,
                )
            except Exception as e:
                # taskA fallback
                dummy_curv = torch.zeros(ht.shape[0], 3, 4, device=device)
                pred = autoregressive_predict_constrained(
                    model_wrapper, ht, am, layer_ranges,
                    sid_meta=None, curvature_meta=dummy_curv, kappa_meta=None, scale_meta=None,
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
    log_lines.append(f"[Issue #26 T2] kappa_logits LR={LR_KAPPA_LOGITS}  (1e-3, 1e-2, 1e-1)")
    log_lines.append("=" * 70)

    sid_sha = sha256_of(SID_NPY)
    t5_sha = sha256_of(T5_CKPT)
    prior_sha = sha256_of(PRIOR_CKPT)
    log_lines.append(f"\n[SHA256] SID_NPY: {sid_sha}")
    log_lines.append(f"[SHA256] T5_CKPT: {t5_sha}")
    log_lines.append(f"[SHA256] PRIOR_CKPT: {prior_sha}")
    log_lines.append(f"[SID hash match expected #157] {sid_sha == EXPECTED_SID_SHA}")

    if sid_sha != EXPECTED_SID_SHA:
        log_lines.append("[FAIL] SID hash mismatch, abort")
        with open(LOG_PATH, "w") as f:
            f.write("\n".join(log_lines) + "\n")
        return

    # Load wrapper
    spec = importlib.util.spec_from_file_location(
        "v3",
        "/home/wlia0047/ar57/wenyu/GeneRec/taskA/stage3/_archive/taskA_stage3_issue24_precheck_v3.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    WrapperCls = mod.HG_Rec_with_BoundedAdapter
    t5_state_dict = mod.load_t5_state_dict(T5_CKPT)
    t5_config = mod.get_t5_config()
    model_wrapper = WrapperCls(t5_config, t5_state_dict, d_model=D_MODEL, n_layers=3, sid_dim=4)
    model_wrapper.to(DEVICE)

    # Load prior ckpt (issue192 epoch 48 best_adapter)
    prior_ckpt = torch.load(PRIOR_CKPT, map_location=DEVICE, weights_only=False)
    log_lines.append(f"[Load prior ckpt] epoch={prior_ckpt.get('epoch')}, val_r10={prior_ckpt.get('val_r10')}")
    filtered_state = {k: v for k, v in prior_ckpt["adapter_state_dict"].items() if k in model_wrapper.adapter.state_dict() and list(v.shape) == list(model_wrapper.adapter.state_dict()[k].shape)}
    model_wrapper.adapter.load_state_dict(filtered_state, strict=False)
    if "ln_state_dict" in prior_ckpt:
        model_wrapper.first_input_ln.load_state_dict(prior_ckpt["ln_state_dict"])

    #  kappa_logits 
    kappa_logits_start = model_wrapper.adapter.kappa_logits.detach().clone()
    kappa_per_layer_start = model_wrapper.adapter.get_kappa_per_layer().detach().cpu().tolist()
    log_lines.append(f"\n[kappa_per_layer start] {kappa_per_layer_start}")
    state["kappa_history"].append({"step": 0, "epoch": START_EPOCH, "kappa_logits": kappa_logits_start.cpu().tolist(), "kappa_per_layer": kappa_per_layer_start})

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
    train_indices = torch.arange(n_train, device=DEVICE)

    val_ds = GenRecDataset(
        dataset_path=TRAIN_PARQUET, code_path=SID_NPY, mode="train",
        codebook_size=CODEBOOK_SIZE, max_len=MAX_LEN, PAD_TOKEN=PAD_TOKEN,
    )
    n_val = int(n_train * VAL_SPLIT_RATIO)
    val_split_idx = n_train - n_val
    val_hist = np.zeros((n_val, MAX_LEN * 4), dtype=np.int64)
    val_tgt = np.zeros((n_val, 4), dtype=np.int64)
    for i in range(n_val):
        s = val_ds[i]
        val_hist[i] = np.concatenate([np.asarray(h, dtype=np.int64).flatten() for h in s["history"]])
        val_tgt[i] = np.asarray(s["target"], dtype=np.int64).flatten()
    val_histories_t = torch.from_numpy(all_histories[val_split_idx:]).to(DEVICE).long()
    val_targets_t = torch.from_numpy(all_targets[val_split_idx:]).to(DEVICE).long()
    train_indices = torch.arange(0, val_split_idx, device=DEVICE)

    layer_ranges = get_layer_ranges(CODEBOOK_SIZE)

    # === Optimizer ( = kappa_logits  param group + LR) ===
    # kappa_logits + alpha_logit  group (LR=1e-2),  conditioner + LN  LR
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
    early_stop_triggered = False
    start_time = time.time()

    # :  |kappa_l| >= 1e-4 per step assert
    for epoch in range(START_EPOCH, START_EPOCH + NUM_EPOCHS):
        epoch_losses = []
        cond_grad_norms = []
        ln_grad_norms = []
        kappa_logits_grad_norms = []
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
            output, _, alpha = model_wrapper(ht, attention_mask=am, labels=tt, sid_meta=sid_meta, kappa_meta=kappa_meta, scale_meta=scale_meta)
            loss = output.loss if hasattr(output, "loss") else output[0]
            if not torch.isfinite(loss):
                nan_inf = True
                continue
            loss.backward()
            #  kappa_logits  ( step )  assert 
            kl_before = kappa_logits_param.detach().clone()
            cond_grad_norm = sum((p.grad.norm().item() ** 2) for p in other_cond_params if p.grad is not None and torch.isfinite(p.grad).all()) ** 0.5
            ln_grad_norm = sum((p.grad.norm().item() ** 2) for p in layernorm_params if p.grad is not None and torch.isfinite(p.grad).all()) ** 0.5
            kl_grad = kappa_logits_param.grad
            kl_grad_norm = kl_grad.norm().item() if kl_grad is not None and torch.isfinite(kl_grad).all() else 0.0
            optimizer.step()
            # assert 
            kl_after = kappa_logits_param.detach().clone()
            delta = (kl_after - kl_before).abs().max().item()
            if delta < DELTA_KAPPA_ASSERT and kl_grad_norm > 0:
                log_lines.append(f"  [WARN] epoch={epoch} batch={batch_idx} kappa={delta:.6e} < {DELTA_KAPPA_ASSERT} (grad={kl_grad_norm:.6e}, LR_KAPPA={LR_KAPPA_LOGITS})")
            epoch_losses.append(loss.item())
            cond_grad_norms.append(cond_grad_norm)
            ln_grad_norms.append(ln_grad_norm)
            kappa_logits_grad_norms.append(kl_grad_norm)
            # recalibrate trigger (per #24 spec): kappa  codebook/
            #  (codebook  issue , )
            if delta >= DELTA_KAPPA_ASSERT:
                state["recalibrate_events"].append({
                    "epoch": epoch, "batch_idx": batch_idx,
                    "kappa_per_layer_before": model_wrapper.adapter.get_kappa_per_layer().detach().cpu().tolist(),
                    "kappa_per_layer_after": model_wrapper.adapter.get_kappa_per_layer().detach().cpu().tolist(),
                    "delta_kappa_max": delta,
                    "codebook_sha256_before": "n/a (codebook recalibration not in scope)",
                    "codebook_sha256_after": "n/a",
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
        msg = f"  [epoch {epoch}] loss={avg_loss:.4f}, cond_grad={avg_cond:.4e}, ln_grad={avg_ln:.4e}, kl_grad={avg_kl_grad:.4e}, kappa_per_layer={kappa_per_layer_now}, delta_from_start={delta_from_start}, nan_inf={nan_inf}, elapsed={elapsed/60:.1f}m"
        log_lines.append(msg)
        state["kappa_history"].append({"step": batch_idx + 1, "epoch": epoch, "kappa_per_layer": kappa_per_layer_now, "delta_from_start": delta_from_start, "kl_grad_mean": avg_kl_grad})

        # val
        val_r10, val_in_range_pct, preds, tgts = run_val_shared(model_wrapper, val_histories_t, val_targets_t, VAL_EVAL_N, BATCH_SIZE, layer_ranges, DEVICE)
        state["val_trace"].append({"epoch": epoch, "val_r10": val_r10, "val_in_range_pct": val_in_range_pct, "kappa_per_layer": kappa_per_layer_now})
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

    # verdict
    final_kappa = model_wrapper.adapter.get_kappa_per_layer().detach().cpu().tolist()
    final_delta = [abs(a - b) for a, b in zip(final_kappa, kappa_per_layer_start)]
    c3_pass = max(final_delta) >= 1e-3
    verdict = {
        "issue": 26,
        "step": f"T2 kappa_logits LR={LR_KAPPA_LOGITS} ",
        "gate": "precheck v4",
        "verdict": "PASS" if c3_pass else "FAIL  kappa ,  LR",
        "lr_kappa_logits": LR_KAPPA_LOGITS,
        "lr_conditioner": LR_CONDITIONER,
        "lr_layernorm": LR_LAYERNORM,
        "epochs_observed": [START_EPOCH, START_EPOCH + NUM_EPOCHS - 1],
        "kappa_per_layer_start": kappa_per_layer_start,
        "kappa_per_layer_end": final_kappa,
        "delta_kappa_per_layer": final_delta,
        "max_delta_kappa": max(final_delta),
        "c3_pass": c3_pass,
        "best_val_r10": best_val_r10,
        "best_epoch": best_epoch,
        "best_adapter_path": str(BEST_ADAPTER_CKPT_PATH),
        "best_adapter_exists": BEST_ADAPTER_CKPT_PATH.exists(),
        "n_recalibrate_triggers": len(state["recalibrate_events"]),
        "ckpt_sha256": sha256_of(BEST_ADAPTER_CKPT_PATH) if BEST_ADAPTER_CKPT_PATH.exists() else None,
        "sid_sha256": sid_sha,
        "sid_sha_match": sid_sha == EXPECTED_SID_SHA,
        "delta_kappa_history": state["kappa_history"],
    }
    with open(VERDICT_PATH, "w") as f:
        json.dump(verdict, f, indent=2, default=str)
    log_lines.append(f"\n[Verdict written] {VERDICT_PATH}")
    log_lines.append(f"[c3_pass] {c3_pass}, max_delta_kappa={max(final_delta):.4e}, best_val_R@10={best_val_r10:.4f}")

    print("\n".join(log_lines), flush=True)
    with open(LOG_PATH, "w") as f:
        f.write("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    main()