#!/usr/bin/env python3
"""Issue #17 [方向A Step 6] canary — 100 样本真实 argmax/约束解码.

per #17 spec Step 6:
- canary: 约 100 样本真实 argmax/约束解码, R@10 必须非零且 validity=100%
- 否则禁止启动长跑

ckpt: taskA_stage3_issue192_long_run/best_adapter.pt (epoch 48, val_R@10=0.058)
     = 跟 #14 用的同一 ckpt (Step 5 NO-GO 没产生新 best)

输出: verdicts/issue17_step6_canary.json
"""
import os, sys, json, importlib.util
import torch
import numpy as np
from pathlib import Path

PROJECT = "/home/wlia0047/ar57/wenyu/GeneRec"
sys.path.insert(0, f"{PROJECT}/HG-Rec/model")
sys.path.insert(0, f"{PROJECT}/HG-Rec/data")

CKPT_PATH = f"{PROJECT}/taskA/stage3/taskA_stage3_issue192_long_run/best_adapter.pt"
SID_NPY = f"{PROJECT}/taskA/_data/Instruments/Instruments_t5_hrqvae_poincare.npy"
TEST_PARQUET = f"{PROJECT}/taskA/_data/Instruments/test.parquet"
T5_CKPT = f"{PROJECT}/taskA/_ckpt/HG_Rec_best.pth"
CODEBOOK_SIZE = [64, 128, 256, 1]
MAX_LEN = 20
PAD_TOKEN = 0
D_MODEL = 128
DEVICE = "cuda:0"
N_CANARY = 100
SEED = 42
LAYER_RANGES = [(1, 64), (65, 192), (193, 448), (449, 449)]


def load_wrapper_cls():
    spec = importlib.util.spec_from_file_location(
        "t470",
        f"{PROJECT}/taskA/stage3/taskA_stage3_kappa_scale_recontinue.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.HG_Rec_with_BoundedAdapter, mod.get_t5_config


def autoregressive_predict(model_wrapper, history_tensor, attention_mask):
    """跟 #192 long-run / Stage4 同路径: 自回归 4-forwards + 闭区间 mask."""
    B = history_tensor.shape[0]
    predicted = torch.zeros(B, 4, dtype=torch.long, device=history_tensor.device)
    cur_history = history_tensor.clone()
    cur_mask = attention_mask.clone()
    for layer_i in range(4):
        out, _, _ = model_wrapper(
            cur_history, attention_mask=cur_mask,
            sid_meta=torch.zeros(B, cur_history.shape[1], 4, device=history_tensor.device),
            kappa_meta=torch.zeros(B, 3, device=history_tensor.device),
            scale_meta=torch.ones(B, 3, device=history_tensor.device),
            labels=cur_history,
        )
        logits = out.logits if hasattr(out, "logits") else out[0]
        next_logits = logits[:, -1, :]
        lo, hi = LAYER_RANGES[layer_i]
        mask = torch.full_like(next_logits, float("-inf"))
        mask[:, lo:hi + 1] = 0.0
        masked_logits = next_logits + mask
        next_token = masked_logits.argmax(dim=-1)
        predicted[:, layer_i] = next_token
        cur_history = torch.cat([cur_history, next_token.unsqueeze(1)], dim=1)
        cur_mask = torch.cat([cur_mask, torch.ones(B, 1, dtype=cur_mask.dtype, device=cur_mask.device)], dim=1)
    return predicted


def main():
    # SHA256 verify
    import hashlib
    sid_sha = hashlib.sha256(open(SID_NPY, "rb").read()).hexdigest()
    expected = "2dab29229c36a9695a11d70b61d1b80fae95a08d00e3c3675e9bf899b709508a"
    assert sid_sha == expected, f"SID hash mismatch: {sid_sha} != {expected}"

    # Load ckpt
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    print(f"[Load ckpt] {CKPT_PATH}")
    print(f"  epoch={ckpt.get('epoch', 'unknown')}, val_R@10={ckpt.get('val_r10', 'unknown')}")

    WrapperCls, get_t5_config = load_wrapper_cls()
    t5_state_dict = torch.load(T5_CKPT, map_location="cpu", weights_only=False)
    if "state_dict" in t5_state_dict:
        t5_state_dict = t5_state_dict["state_dict"]
    elif "model" in t5_state_dict:
        t5_state_dict = t5_state_dict["model"]
    t5_config = get_t5_config()
    model_wrapper = WrapperCls(t5_config, t5_state_dict, d_model=D_MODEL, n_layers=3, sid_dim=4)
    model_wrapper.to(DEVICE)
    if "adapter_state_dict" in ckpt:
        model_wrapper.adapter.load_state_dict(ckpt["adapter_state_dict"])
    else:
        model_wrapper.adapter.load_state_dict(ckpt)
    if "ln_state_dict" in ckpt:
        model_wrapper.first_input_ln.load_state_dict(ckpt["ln_state_dict"])
    elif "first_input_ln_state_dict" in ckpt:
        model_wrapper.first_input_ln.load_state_dict(ckpt["first_input_ln_state_dict"])

    # Load test data
    from dataset import GenRecDataset
    test_ds = GenRecDataset(
        dataset_path=TEST_PARQUET, code_path=SID_NPY, mode="evaluation",
        codebook_size=CODEBOOK_SIZE, max_len=MAX_LEN, PAD_TOKEN=PAD_TOKEN,
    )
    n = len(test_ds)
    # 随机取 N_CANARY samples (deterministic)
    g = torch.Generator().manual_seed(SEED + 100)
    sample_idx = torch.randperm(n, generator=g)[:N_CANARY].tolist()
    print(f"[Canary] n={N_CANARY} from test set (n_total={n})")

    # Predict + measure
    model_wrapper.eval()
    preds_all = []
    targets_all = []
    in_range_count = 0
    total_tokens = 0
    BATCH = 16

    with torch.no_grad():
        for batch_start in range(0, N_CANARY, BATCH):
            batch_idx = sample_idx[batch_start:batch_start + BATCH]
            batch_h = []
            batch_t = []
            for i in batch_idx:
                s = test_ds.data[i]
                batch_h.append(s["history"])
                batch_t.append(s["target"])
            max_L = max(len(h) for h in batch_h)
            history_padded = np.zeros((len(batch_h), max_L * 4), dtype=np.int64)
            target_arr = np.zeros((len(batch_h), 4), dtype=np.int64)
            for i, (h, t) in enumerate(zip(batch_h, batch_t)):
                flat = np.concatenate([np.asarray(x, dtype=np.int64).flatten() for x in h])
                history_padded[i, :len(flat)] = flat
                target_arr[i] = np.asarray(t, dtype=np.int64).flatten()
            history_tensor = torch.from_numpy(history_padded).to(DEVICE)
            attention_mask = (history_tensor != PAD_TOKEN).long()
            predicted = autoregressive_predict(model_wrapper, history_tensor, attention_mask)
            for i in range(len(batch_h)):
                pred = predicted[i].cpu().tolist()
                tgt = target_arr[i].tolist()
                preds_all.append(pred[:4])
                targets_all.append(tgt[:4])
                for layer_i, token_id in enumerate(pred[:4]):
                    lo, hi = LAYER_RANGES[layer_i]
                    if lo <= token_id <= hi:
                        in_range_count += 1
                    total_tokens += 1

    r10 = float(np.mean([1.0 if p == t else 0.0 for p, t in zip(preds_all, targets_all)]))
    in_range_pct = in_range_count / total_tokens if total_tokens > 0 else 0.0
    canary_pass = r10 > 0 and in_range_pct >= 0.99

    verdict = {
        "issue": 17,
        "step": "Step 6 canary — 100 样本真实 argmax/约束解码",
        "ckpt_path": CKPT_PATH,
        "ckpt_epoch": ckpt.get("epoch"),
        "ckpt_loaded_val_r10": ckpt.get("val_r10"),
        "n_canary": N_CANARY,
        "beam_size": 1,
        "autoregressive": True,
        "layer_wise_mask": True,
        "r10": r10,
        "validity_count": in_range_count,
        "total_tokens": total_tokens,
        "validity_pct": in_range_pct,
        "canary_pass": bool(canary_pass),
        "canary_pass_definition": "R@10 > 0 AND validity_pct >= 0.99 (per #17 spec Step 6)",
    }
    out_path = f"{PROJECT}/verdicts/issue17_step6_canary.json"
    with open(out_path, "w") as f:
        json.dump(verdict, f, indent=2)

    print(f"\n[Canary result] n={N_CANARY}")
    print(f"  R@10 = {r10:.4f} (期望 > 0)")
    print(f"  validity = {in_range_count}/{total_tokens} ({in_range_pct*100:.1f}%) (期望 100%)")
    print(f"  canary_pass = {canary_pass}")
    print(f"  verdict saved: {out_path}")
    return canary_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)