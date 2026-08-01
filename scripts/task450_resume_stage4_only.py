"""
Task #450 Issue #161 [方向C Gate4] - Stage 4 ONLY resume (R23 fix lm_head path)

加载 R12 best ckpt (adapter_200ep_BEST.pt), 双复跑 Stage 4 R@K 六项指标.

修复 Task #450 Stage 4 crash: lm_head 路径错误 -> 用 wrapper.forward() (内置 T5 forward + decoder LM head).

R18 实证 + R22 立即开工 + R17 Gate + R23 monitor.
"""

import os
import sys
import json
import time
import hashlib
import argparse
import importlib.util
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT = "/home/wlia0047/ar57/wenyu/GeneRec"
sys.path.insert(0, f"{PROJECT}/HG-Rec/model")
sys.path.insert(0, f"{PROJECT}/HG-Rec/data")
sys.path.insert(0, PROJECT)

os.environ["TRITON_CACHE_DIR"] = "/home/wlia0047/.triton/cache_task450_resume"
os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)

from dataset import GenRecDataset  # noqa: E402
from HG_Rec import HG_Rec  # noqa: E402

# Import wrapper class via importlib (no scripts/__init__.py)
_spec = importlib.util.spec_from_file_location(
    "t440",
    "/home/wlia0047/ar57/wenyu/GeneRec/scripts/task440_issue150_zero_centered_linear_layernorm.py",
)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
HG_Rec_with_ZeroCenteredLayerNormAdapter = _m.HG_Rec_with_ZeroCenteredLayerNormAdapter

DEVICE = "cuda"
T5_CKPT = "/home/wlia0047/ar57/wenyu/GeneRec/products/task84/ckpt_hgrec/Instruments/Jul-23-2026_20-24-44/HG_Rec_best.pth"
SID_NPY = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_poincare.npy"
ADAPTER_CKPT = "/home/wlia0047/ar57/wenyu/GeneRec/products/task450_issue150_stage3_long_train_200ep/adapter_200ep_BEST.pt"
TEST_PARQUET = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/test.parquet"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_npy(arr):
    return hashlib.sha256(arr.tobytes()).hexdigest()


def collate_fn(batch):
    histories = [b["history"] for b in batch]
    targets = [b["target"] for b in batch]
    max_L = max(len(h) for h in histories)
    history_padded = np.zeros((len(batch), max_L, 4), dtype=np.int64)
    for i, h in enumerate(histories):
        L = len(h)
        for j in range(L):
            history_padded[i, j] = h[j]
    target_arr = np.stack(targets, axis=0)
    return {
        "input_ids": torch.from_numpy(history_padded.reshape(len(batch), -1)),
        "labels": torch.from_numpy(target_arr),
    }


def load_model_wrapper():
    """Load ZeroCenteredLayerNormAdapter (self-contained T5Config + state_dict load)."""
    t5_config = {
        "num_layers": 6, "num_decoder_layers": 4, "d_model": 128,
        "d_ff": 1024, "num_heads": 6, "d_kv": 64,
        "dropout_rate": 0.1, "vocab_size": 1025,
        "pad_token_id": 0, "eos_token_id": 1,
        "feed_forward_proj": "relu",
    }
    raw = torch.load(T5_CKPT, map_location="cpu", weights_only=False)
    wrapper = HG_Rec_with_ZeroCenteredLayerNormAdapter(t5_config, raw).to(DEVICE)
    print(f"[T5+Wrapper] loaded from {T5_CKPT}")
    wrapper.first_input_ln = wrapper.first_input_ln.to(DEVICE)
    wrapper.adapter = wrapper.adapter.to(DEVICE)

    if os.path.exists(ADAPTER_CKPT):
        adapter_state = torch.load(ADAPTER_CKPT, map_location=DEVICE, weights_only=False)
        loaded_keys = []
        # Format from Task #450: keys are adapter_state_dict / first_input_ln_state_dict / alpha_value / epoch / val_r10 / epoch_losses
        if isinstance(adapter_state, dict):
            if "adapter_state_dict" in adapter_state:
                wrapper.adapter.load_state_dict(adapter_state["adapter_state_dict"])
                loaded_keys.append("adapter_state_dict")
            if "first_input_ln_state_dict" in adapter_state:
                wrapper.first_input_ln.load_state_dict(adapter_state["first_input_ln_state_dict"])
                loaded_keys.append("first_input_ln_state_dict")
            if "alpha_value" in adapter_state:
                loaded_keys.append(f"alpha_value={adapter_state['alpha_value']:.4e}")
        print(f"[Adapter load] keys loaded={loaded_keys}")
        if "adapter_state_dict" not in loaded_keys:
            raise ValueError("Failed to load adapter weights!")
    else:
        raise FileNotFoundError(f"adapter ckpt not found: {ADAPTER_CKPT}")

    wrapper.eval()
    return wrapper


def run_eval(wrapper, test_ds, run_id, log_lines):
    loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=0)
    n_total = 0
    n_correct_5 = 0
    n_correct_10 = 0
    n_correct_20 = 0
    ndcg_5 = 0.0
    ndcg_10 = 0.0
    ndcg_20 = 0.0

    t0 = time.time()
    for batch_idx, batch in enumerate(loader):
        history = batch["input_ids"].to(DEVICE)
        target = batch["labels"].to(DEVICE)
        B = history.size(0)
        # wrapper.forward expects input_ids shape (B, L*4), sid_meta shape (B, L*4, 4)
        history_flat = history.long()  # (B, L*4)
        # sid_meta per-token: each token has its 4-digit SID context
        # Construct sid_meta such that sid_meta[b, i, :] = 4-digit SID of position i
        # For simplicity (per task440 init proof uses zeros), use (B, L*4, 4) zero padding with digit at last pos
        Bsz, Lx4 = history_flat.shape
        # Build sid_meta of shape (B, L*4, 4) - per-token SID context
        sid_meta = torch.zeros(Bsz, Lx4, 4, dtype=torch.long, device=DEVICE)
        # Set the 4th position (digit) to the actual token value
        sid_meta[:, :, 3] = history_flat
        kappa_meta = torch.zeros(Bsz, 3, dtype=torch.float32, device=DEVICE)
        # attention mask: (B, L*4) per-token (T5 expects per-token mask matching inputs_embeds)
        # Each token in L*4 has a position mask = 1 if any of the 4 digits at this step is nonzero
        L = Lx4 // 4
        per_step_mask = (history_flat.view(Bsz, L, 4).abs().sum(dim=-1) > 0).long()  # (B, L)
        attention_mask = per_step_mask.unsqueeze(-1).expand(-1, -1, 4).reshape(Bsz, Lx4)  # (B, L*4)

        with torch.no_grad():
            outputs = wrapper.forward(
                input_ids=history_flat,
                attention_mask=attention_mask,
                sid_meta=sid_meta,
                kappa_meta=kappa_meta,
                use_alpha_zero=False,
            )
        # wrapper returns (loss, logits, alpha) tuple per task440 forward signature
        logits = outputs[1]
        preds = logits.argmax(dim=-1)
        for i in range(B):
            pred = preds[i].cpu().tolist()
            tgt = target[i].cpu().tolist()
            n_total += 1
            if pred[:4] == tgt[:4]:
                n_correct_5 += 1
                n_correct_10 += 1
                n_correct_20 += 1
                ndcg_5 += 1.0
                ndcg_10 += 1.0
                ndcg_20 += 1.0

        if batch_idx % 20 == 0:
            elapsed = time.time() - t0
            speed = (batch_idx + 1) * 32 / max(elapsed, 1e-6)
            log_lines.append(f"[run{run_id}] batch {batch_idx}/{len(loader)} ({speed:.1f} samp/s)")

    return {
        "R@5": n_correct_5 / n_total,
        "R@10": n_correct_10 / n_total,
        "R@20": n_correct_20 / n_total,
        "NDCG@5": ndcg_5 / n_total,
        "NDCG@10": ndcg_10 / n_total,
        "NDCG@20": ndcg_20 / n_total,
        "n_total": n_total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--log", default="/home/wlia0047/ar57/wenyu/GeneRec/logs/task450_resume_stage4_v2.log")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    log_lines = []
    log_lines.append(f"[Start] {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_lines.append(f"[TRITON_CACHE] {os.environ['TRITON_CACHE_DIR']}")
    log_lines.append(f"[GPU] CUDA_VISIBLE_DEVICES={args.gpu}")

    sid_npy = np.load(SID_NPY)
    log_lines.append(f"[SHA256] SID_NPY: {sha256_npy(sid_npy)}")
    log_lines.append(f"[SHA256] T5_CKPT: {sha256_file(T5_CKPT)}")
    log_lines.append(f"[SHA256] ADAPTER_CKPT: {sha256_file(ADAPTER_CKPT)}")
    log_lines.append(f"[SID shape] {sid_npy.shape}")

    test_ds = GenRecDataset(
        dataset_path=TEST_PARQUET, code_path=SID_NPY, mode="evaluation",
        codebook_size=[64, 128, 256, 1], max_len=21, PAD_TOKEN=0,
    )
    log_lines.append(f"[Dataset] test size: {len(test_ds)}")

    wrapper = load_model_wrapper()
    try:
        alpha_val = wrapper.adapter.alpha.item() if hasattr(wrapper.adapter, 'alpha') else float('nan')
    except Exception:
        alpha_val = float('nan')
    log_lines.append(f"[Model] ZeroCenteredAdapter loaded, alpha={alpha_val:.6f}")

    results = {}
    for run_id in [1, 2]:
        log_lines.append(f"\n[Stage 4 run {run_id}] start")
        r = run_eval(wrapper, test_ds, run_id, log_lines)
        results[f"run{run_id}"] = r
        log_lines.append(
            f"[Stage 4 run {run_id}] R@5={r['R@5']:.4f} R@10={r['R@10']:.4f} R@20={r['R@20']:.4f} "
            f"NDCG@5={r['NDCG@5']:.4f} NDCG@10={r['NDCG@10']:.4f} NDCG@20={r['NDCG@20']:.4f} "
            f"(n={r['n_total']})"
        )

    avg = {
        k: float(np.mean([results[f"run{i}"][k] for i in [1, 2]]))
        for k in ["R@5", "R@10", "R@20", "NDCG@5", "NDCG@10", "NDCG@20"]
    }
    log_lines.append(
        f"\n[AVG] R@5={avg['R@5']:.4f} R@10={avg['R@10']:.4f} R@20={avg['R@20']:.4f} "
        f"NDCG@5={avg['NDCG@5']:.4f} NDCG@10={avg['NDCG@10']:.4f} NDCG@20={avg['NDCG@20']:.4f}"
    )
    log_lines.append(f"[Baseline (Task #84)] R@10=0.1020")

    os.makedirs("products/task450_issue150_stage3_long_train_200ep", exist_ok=True)
    stage4_verdict = {
        "task": "task450_resume_stage4_v2",
        "issue": 161,
        "stage3_best_val_R@10_sim": 0.1069,
        "adapter_ckpt_sha256": sha256_file(ADAPTER_CKPT),
        "sid_npy_sha256": sha256_npy(sid_npy),
        "t5_ckpt_sha256": sha256_file(T5_CKPT),
        "test_size": len(test_ds),
        "results": results,
        "avg": avg,
        "decision": "PASS" if avg["R@10"] > 0.1020 else "NO-GO",
    }
    with open(
        "products/task450_issue150_stage3_long_train_200ep/stage4_verdict_v2.json", "w"
    ) as f:
        json.dump(stage4_verdict, f, indent=2)
    log_lines.append(f"\n[verdict saved] products/task450_issue150_stage3_long_train_200ep/stage4_verdict_v2.json")
    log_lines.append(f"[Decision] {stage4_verdict['decision']}")

    log_text = "\n".join(log_lines)
    with open(args.log, "w") as f:
        f.write(log_text)
    print(log_text)


if __name__ == "__main__":
    main()
