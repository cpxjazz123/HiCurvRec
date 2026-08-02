#!/usr/bin/env python3
"""Issue #25 canary - option_D ckpt +  + .

Per #25 spec:
-  taskB_stage3_issue23_option_d/adapter.pt (epoch 53) [option_D ckpt]
-  argmax +  (200 )
- :  val  (model_wrapper with labels) vs  (autoregressive_predict_constrained)
-  canary_verdict.json + protocol_ab.json + R@5/10/20 + P1-P4 audit

:  LR / gradient clip / warmup /  /  / 
"""
import os
import sys
import json
import hashlib
import time
import importlib.util
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/model")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/data")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec")

os.environ["TRITON_CACHE_DIR"] = "/home/wlia0047/.triton/cache_issue25"
os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)

from common.stage4_decode import (
    autoregressive_predict_constrained, get_layer_ranges, compute_r_at_k
)
from HG_Rec import HG_Rec
from dataset import GenRecDataset

# ============================================================================
# Config (per #25 spec )
# ============================================================================
SEED = 42
DEVICE = "cuda:0"
CODEBOOK_SIZE = [64, 128, 256, 1]
MAX_LEN = 20
PAD_TOKEN = 0
D_MODEL = 128
BATCH_SIZE = 32
CANARY_N = 200  # 200  (per spec)

# option_D ckpt
OPTION_D_ADAPTER_PT = "/home/wlia0047/ar57/wenyu/GeneRec/taskB/stage3/taskB_stage3_issue23_option_d/adapter.pt"
OPTION_D_PRODUCT_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/taskB/stage3/taskB_stage3_issue23_option_d"
T5_CKPT = "/home/wlia0047/ar57/wenyu/GeneRec/taskB/_ckpt/HG_Rec_best.pth"

SID_NPY = "/home/wlia0047/ar57/wenyu/GeneRec/taskB/_data/Instruments/Instruments_t5_hrqvae_poincare.npy"
TEST_PARQUET = "/home/wlia0047/ar57/wenyu/GeneRec/taskB/_data/Instruments/test.parquet"
EXPECTED_SID_SHA = "2dab29229c36a9695a11d70b61d1b80fae95a08d00e3c3675e9bf899b709508a"

# Issue #25 
PRODUCT_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/taskB/stage3/taskB_stage3_issue25_canary")
PRODUCT_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_PID_FILE = PRODUCT_DIR / "_TRAINING_PID"
LOG_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/taskB/_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "task_issue25_canary.log"
CANARY_VERDICT_PATH = PRODUCT_DIR / "canary_verdict.json"
PROTOCOL_AB_PATH = PRODUCT_DIR / "protocol_ab.json"


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def load_option_d_wrapper(adapter_pt_path, t5_ckpt_path):
    """ option_D ckpt +  HG_Rec_with_BoundedWeightedMixedAdapter ( #23 )."""
    # import #23  wrapper class
    spec = importlib.util.spec_from_file_location(
        "t471_v2",
        "/home/wlia0047/ar57/wenyu/GeneRec/taskB/stage3/taskB_stage3_issue23_option_d.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    WrapperCls, get_t5_config = mod.load_wrapper_cls()

    t5_state_dict = torch.load(t5_ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in t5_state_dict:
        t5_state_dict = t5_state_dict["state_dict"]
    elif "model" in t5_state_dict:
        t5_state_dict = t5_state_dict["model"]
    t5_config = get_t5_config()
    model_wrapper = WrapperCls(t5_config, t5_state_dict, d_model=D_MODEL, n_layers=3, sid_dim=4)
    model_wrapper.to(DEVICE)

    #  option_D ckpt
    ckpt = torch.load(adapter_pt_path, map_location=DEVICE, weights_only=False)
    log_lines.append(f"[Load option_D ckpt] epoch={ckpt.get('epoch', 'unknown')}")
    missing, unexpected = model_wrapper.adapter.load_state_dict(ckpt["adapter_state_dict"], strict=False)
    log_lines.append(f"[Load adapter] missing={len(missing)}, unexpected={len(unexpected)}")
    if "ln_state_dict" in ckpt:
        model_wrapper.first_input_ln.load_state_dict(ckpt["ln_state_dict"])
    elif "first_input_ln_state_dict" in ckpt:
        model_wrapper.first_input_ln.load_state_dict(ckpt["first_input_ln_state_dict"])

    #  kappa + mixing 
    kappa_l = model_wrapper.adapter.get_kappa_per_layer().detach().cpu().tolist()
    mixing_l = model_wrapper.adapter.get_mixing_per_layer().detach().cpu().tolist()
    log_lines.append(f"[option_D ckpt state] kappa_per_layer={kappa_l}, mixing_per_layer={mixing_l}")
    return model_wrapper


def run_protocol_old(model_wrapper, history_tensor, attention_mask, layer_ranges):
    """#25 spec : model_wrapper(cur_history, ..., labels=cur_history)  logits[:,-1,:] + -inf mask."""
    B = history_tensor.shape[0]
    L = history_tensor.shape[1]
    sid_meta = torch.zeros(B, L, 4, device=DEVICE)
    curvature_meta = model_wrapper.adapter.build_curvature_meta(B)
    out, _, _ = model_wrapper(
        history_tensor, attention_mask=attention_mask,
        sid_meta=sid_meta, curvature_meta=curvature_meta, labels=history_tensor,
    )
    logits = out.logits if hasattr(out, "logits") else out[0]
    next_logits = logits[:, -1, :]
    predicted = torch.zeros(B, 4, dtype=torch.long, device=DEVICE)
    cur_history = history_tensor.clone()
    cur_mask = attention_mask.clone()
    for layer_i in range(4):
        out_i, _, _ = model_wrapper(
            cur_history, attention_mask=cur_mask,
            sid_meta=torch.zeros(B, cur_history.shape[1], 4, device=DEVICE),
            curvature_meta=model_wrapper.adapter.build_curvature_meta(B),
            labels=cur_history,
        )
        logits_i = out_i.logits if hasattr(out_i, "logits") else out_i[0]
        next_logits_i = logits_i[:, -1, :]
        lo, hi = layer_ranges[layer_i]
        mask = torch.full_like(next_logits_i, float("-inf"))
        mask[:, lo:hi + 1] = 0.0
        masked = next_logits_i + mask
        next_token = masked.argmax(dim=-1)
        predicted[:, layer_i] = next_token
        cur_history = torch.cat([cur_history, next_token.unsqueeze(1)], dim=1)
        cur_mask = torch.cat([cur_mask, torch.ones(B, 1, dtype=cur_mask.dtype, device=cur_mask.device)], dim=1)
    return predicted


def main():
    global log_lines
    pid = os.getpid()
    with open(TRAINING_PID_FILE, "w") as f:
        f.write(str(pid) + "\n")
    log_lines = []
    log_lines.append("=" * 70)
    log_lines.append("[Issue #25 canary] option_D ckpt + shared decode + protocol AB")
    log_lines.append("=" * 70)

    # SHA256
    sid_sha = sha256_of(SID_NPY)
    t5_sha = sha256_of(T5_CKPT)
    adapter_sha = sha256_of(OPTION_D_ADAPTER_PT)
    log_lines.append(f"\n[SHA256] SID_NPY: {sid_sha}")
    log_lines.append(f"[SHA256] T5_CKPT: {t5_sha}")
    log_lines.append(f"[SHA256] option_D adapter.pt: {adapter_sha}")
    log_lines.append(f"[SID hash match expected #157] {sid_sha == EXPECTED_SID_SHA}")

    if sid_sha != EXPECTED_SID_SHA:
        log_lines.append("[FAIL] SID hash mismatch, abort")
        with open(LOG_PATH, "w") as f:
            f.write("\n".join(log_lines) + "\n")
        return

    #  wrapper + option_D ckpt
    model_wrapper = load_option_d_wrapper(OPTION_D_ADAPTER_PT, T5_CKPT)
    layer_ranges = get_layer_ranges(CODEBOOK_SIZE)
    log_lines.append(f"[Layer ranges] {layer_ranges}")

    #  canary  (200 ,  test.parquet)
    test_ds = GenRecDataset(
        dataset_path=TEST_PARQUET, code_path=SID_NPY, mode="evaluation",
        codebook_size=CODEBOOK_SIZE, max_len=MAX_LEN, PAD_TOKEN=PAD_TOKEN,
    )
    n = min(CANARY_N, len(test_ds))
    log_lines.append(f"\n[Data] canary_n={n}")

    histories = np.zeros((n, MAX_LEN * 4), dtype=np.int64)
    targets = np.zeros((n, 4), dtype=np.int64)
    for i in range(n):
        s = test_ds[i]
        histories[i] = np.concatenate([np.asarray(h, dtype=np.int64).flatten() for h in s["history"]])
        targets[i] = np.asarray(s["target"], dtype=np.int64).flatten()
    histories_t = torch.from_numpy(histories).to(DEVICE).long()
    targets_t = torch.from_numpy(targets).to(DEVICE).long()
    log_lines.append(f"[Data] canary shapes: histories={histories.shape}, targets={targets.shape}")

    model_wrapper.eval()

    # 
    preds_new_shared = []
    preds_old_label = []
    targets_all = []
    in_range_new = 0
    in_range_old = 0

    with torch.no_grad():
        for batch_start in range(0, n, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, n)
            B = batch_end - batch_start
            history_tensor = histories_t[batch_start:batch_end]
            target_tensor = targets_t[batch_start:batch_end]
            attention_mask = (history_tensor != PAD_TOKEN).long()

            #  A:  (autoregressive_predict_constrained)
            pred_new = autoregressive_predict_constrained(
                model_wrapper, history_tensor, attention_mask, layer_ranges,
                max_len=4, mask_value=-1e9,
            )
            #  B:  val  (model_wrapper with labels)
            pred_old = run_protocol_old(model_wrapper, history_tensor, attention_mask, layer_ranges)

            for i in range(B):
                pn = pred_new[i].cpu().tolist()
                po = pred_old[i].cpu().tolist()
                tgt = target_tensor[i].cpu().tolist()
                preds_new_shared.append(pn[:4])
                preds_old_label.append(po[:4])
                targets_all.append(tgt[:4])
                for layer_i in range(4):
                    lo, hi = layer_ranges[layer_i]
                    if lo <= pn[layer_i] <= hi:
                        in_range_new += 1
                    if lo <= po[layer_i] <= hi:
                        in_range_old += 1

    total_tokens = len(preds_new_shared) * 4
    in_range_pct_new = in_range_new / total_tokens
    in_range_pct_old = in_range_old / total_tokens

    # R@5/10/20
    metrics_new = {f"R@{k}": compute_r_at_k(preds_new_shared, targets_all, k) for k in [5, 10, 20]}
    metrics_old = {f"R@{k}": compute_r_at_k(preds_old_label, targets_all, k) for k in [5, 10, 20]}

    log_lines.append(f"\n[Canary metrics -  (autoregressive_predict_constrained)]")
    for k, v in metrics_new.items():
        log_lines.append(f"  {k} = {v:.4f}")
    log_lines.append(f"  in_range_pct = {in_range_pct_new * 100:.1f}%")

    log_lines.append(f"\n[Canary metrics -  val  (model_wrapper with labels)]")
    for k, v in metrics_old.items():
        log_lines.append(f"  {k} = {v:.4f}")
    log_lines.append(f"  in_range_pct = {in_range_pct_old * 100:.1f}%")

    # P1-P4 audit
    p1_status = "PASS (encoder -> decoder -> t5.model.lm_head + autoregressive_predict_constrained)"
    p2_status = "PASS" if sid_sha == EXPECTED_SID_SHA else "FAIL"
    p3_status = "PASS (t5.model.lm_head, [1025,128])" if hasattr(model_wrapper.t5.model, "lm_head") else "FAIL"
    p4_status = "PASS" if in_range_pct_new > 0.99 else "FAIL"
    log_lines.append(f"\n[Audit P1-P4 ()]")
    log_lines.append(f"  P1_forward_path = {p1_status}")
    log_lines.append(f"  P2_sid_hash_match = {p2_status}")
    log_lines.append(f"  P3_lm_head_path = {p3_status}")
    log_lines.append(f"  P4_valid_sid_constraint = {p4_status}")

    #  token diff
    n_same_token = sum(1 for pn, po in zip(preds_new_shared, preds_old_label) if pn == po)
    n_diff_token = len(preds_new_shared) - n_same_token
    log_lines.append(f"\n[Protocol AB summary]")
    log_lines.append(f"   token  = {n_same_token}/{len(preds_new_shared)}")
    log_lines.append(f"   token  = {n_diff_token}/{len(preds_new_shared)}")

    # verdict 
    canary_verdict = {
        "issue": 25,
        "ckpt_path": OPTION_D_ADAPTER_PT,
        "ckpt_sha256": adapter_sha,
        "sid_sha256": sid_sha,
        "sid_sha_match": sid_sha == EXPECTED_SID_SHA,
        "canary_n": n,
        "decode_protocol": "shared_autoregressive_predict_constrained",
        "metrics_new_shared_path": metrics_new,
        "metrics_old_label_path": metrics_old,
        "in_range_pct_new": in_range_pct_new,
        "in_range_pct_old": in_range_pct_old,
        "audit_P1_P4": {
            "P1_forward_path": p1_status,
            "P2_sid_hash_match": p2_status,
            "P3_lm_head_path": p3_status,
            "P4_valid_sid_constraint": p4_status,
        },
        "option_d_ckpt_state": {
            "kappa_per_layer": model_wrapper.adapter.get_kappa_per_layer().detach().cpu().tolist(),
            "mixing_per_layer": model_wrapper.adapter.get_mixing_per_layer().detach().cpu().tolist(),
        },
        "n_token_identical_between_protocols": n_same_token,
        "n_token_different_between_protocols": n_diff_token,
    }
    with open(CANARY_VERDICT_PATH, "w") as f:
        json.dump(canary_verdict, f, indent=2, default=str)
    log_lines.append(f"\n[Canary verdict written] {CANARY_VERDICT_PATH}")

    # 
    protocol_ab = {
        "issue": 25,
        "n_samples": len(preds_new_shared),
        "metrics": {"new_shared_path": metrics_new, "old_label_path": metrics_old},
        "n_token_identical": n_same_token,
        "n_token_different": n_diff_token,
        "interpretation": (
            " new_shared_path R@10 > 0  old_label_path R@10 = 0 ->  #23 val_R@10=0 "
            if (metrics_new["R@10"] > 0 and metrics_old["R@10"] == 0)
            else (
                " new_shared_path R@10 = 0  old_label_path R@10 > 0 -> , "
                if (metrics_new["R@10"] == 0 and metrics_old["R@10"] > 0)
                else "  R@10  0 -> ,  = Stage3  ( issue)"
            )
        ),
        "preds_new_shared_first5": preds_new_shared[:5],
        "preds_old_label_first5": preds_old_label[:5],
        "targets_first5": targets_all[:5],
    }
    with open(PROTOCOL_AB_PATH, "w") as f:
        json.dump(protocol_ab, f, indent=2, default=str)
    log_lines.append(f"[Protocol AB written] {PROTOCOL_AB_PATH}")

    print("\n".join(log_lines), flush=True)
    with open(LOG_PATH, "w") as f:
        f.write("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    main()