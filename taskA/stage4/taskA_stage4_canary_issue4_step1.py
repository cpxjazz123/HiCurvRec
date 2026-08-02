#!/usr/bin/env python3
"""Issue #4 [方向A Gate4 Step 1] canary 验证修复后 compute_r_at_k on long-run best_adapter.pt.

对比:
- 修复前: taskA_stage4_resume.py L50 `p[:k]==t[:k]` (k>4 时退化为 exact match) → 6 指标全同值
- 修复后: 新 compute_r_at_k 接受 (K candidates per sample) → argmax 单 candidate 时 R@K ≡ R@1 数学正确
                                          → K candidates (beam search) 时 R@K 单调非降
canary 目标: 用 long-run best_adapter.pt (epoch=49) 跑 100 samples, 验证 R@10 > 0
"""
import os, sys, json, time, importlib.util
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TRITON_CACHE_DIR"] = "/home/wlia0047/.triton/cache_issue4_step1_canary"
os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)
import torch
import numpy as np
from torch.utils.data import DataLoader

PROJECT = "/home/wlia0047/ar57/wenyu/GeneRec"
sys.path.insert(0, f"{PROJECT}/HG-Rec/model")
sys.path.insert(0, f"{PROJECT}/HG-Rec/data")

# Use load_wrapper_cls from long-run script (跟 #190 同架构)
_spec_lr = importlib.util.spec_from_file_location(
    "t_lr", f"{PROJECT}/taskA/stage3/taskA_stage3_issue192_long_run.py"
)
_m_lr = importlib.util.module_from_spec(_spec_lr)
_spec_lr.loader.exec_module(_m_lr)
WrapperCls, get_t5_config = _m_lr.load_wrapper_cls()
_spec_k = importlib.util.spec_from_file_location(
    "t470", f"{PROJECT}/taskA/stage3/taskA_stage3_kappa_scale_recontinue.py"
)
_m_k = importlib.util.module_from_spec(_spec_k)
_spec_k.loader.exec_module(_m_k)
load_t5_state_dict = _m_k.load_t5_state_dict

# === 修复后 compute_r_at_k (与 taskA_stage4_resume.py 同步) ===
def compute_r_at_k(candidates_per_sample, targets_list, k):
    n_correct = 0
    for cands, target in zip(candidates_per_sample, targets_list):
        top_k = cands[:k]
        if any(c == target for c in top_k):
            n_correct += 1
    return n_correct / max(1, len(candidates_per_sample))


def compute_ndcg_at_k(candidates_per_sample, targets_list, k):
    import math
    ndcgs = []
    for cands, target in zip(candidates_per_sample, targets_list):
        rank = None
        for i, c in enumerate(cands[:k]):
            if c == target:
                rank = i + 1
                break
        ndcgs.append(0.0 if rank is None else 1.0 / math.log2(rank + 1))
    return sum(ndcgs) / max(1, len(ndcgs))

T5_CKPT = f"{PROJECT}/taskA/_ckpt/HG_Rec_best.pth"
SID_NPY = f"{PROJECT}/HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_poincare.npy"
TEST_PARQUET = f"{PROJECT}/HG-Rec/dataset/Instruments/test.parquet"
CKPT_PATH = f"{PROJECT}/taskA/stage3/taskA_stage3_issue192_long_run/best_adapter.pt"
DEVICE = "cuda"
D_MODEL = 128
MAX_LEN = 4
PAD_TOKEN = 0
CODEBOOK_SIZE = [64, 128, 256, 1]
CANARY_N = 100
SEED = 42


def main():
    log = []
    log.append(f"[Issue #4 Step 1 canary] taskA long-run best_adapter.pt @ {CKPT_PATH}")
    log.append(f"[Setup] CANARY_N={CANARY_N}, SEED={SEED}, GPU={os.environ['CUDA_VISIBLE_DEVICES']}")

    t5_config = get_t5_config()
    t5_state_dict = load_t5_state_dict(T5_CKPT)
    model_wrapper = WrapperCls(t5_config, t5_state_dict, d_model=D_MODEL, n_layers=3, sid_dim=4)
    model_wrapper = model_wrapper.to(DEVICE)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    model_wrapper.adapter.load_state_dict(ckpt["adapter_state_dict"])
    # 兼容两种 ckpt 格式: kappa_scale_recontinue 用 first_input_ln_state_dict,
    # issue192_long_run 用 ln_state_dict
    if "first_input_ln_state_dict" in ckpt:
        model_wrapper.first_input_ln.load_state_dict(ckpt["first_input_ln_state_dict"])
    elif "ln_state_dict" in ckpt:
        model_wrapper.first_input_ln.load_state_dict(ckpt["ln_state_dict"])
    alpha_val = ckpt.get("alpha_value", ckpt.get("alpha"))
    alpha_str = f"{alpha_val:.6e}" if isinstance(alpha_val, (int, float)) else "N/A"
    log.append(f"[Load ckpt] epoch={ckpt.get('epoch', 'N/A')}, alpha={alpha_str}")
    model_wrapper.eval()

    # 4 层合法 SID token 范围 (from get_layer_ranges in long-run)
    layer_ranges = _m_lr.get_layer_ranges(CODEBOOK_SIZE)
    log.append(f"[SID layer ranges] {layer_ranges}")

    from dataset import GenRecDataset
    test_ds = GenRecDataset(
        dataset_path=TEST_PARQUET, code_path=SID_NPY, mode="evaluation",
        codebook_size=CODEBOOK_SIZE, max_len=MAX_LEN, PAD_TOKEN=PAD_TOKEN,
    )

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

    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate_fn)
    K_list = [5, 10, 20]
    torch.manual_seed(SEED)
    candidates_per_sample = []
    targets_list = []
    n_processed = 0

    with torch.no_grad():
        for batch in test_loader:
            if n_processed >= CANARY_N:
                break
            history_tensor = batch["input_ids"].to(DEVICE)
            target_tensor = batch["labels"].to(DEVICE)
            attention_mask = (history_tensor != PAD_TOKEN).long()
            # 使用 long-run 的 autoregressive_predict (4-step + layer-wise SID mask)
            # 返回 shape (B, 4) 的 4-tuple SIDs
            preds = _m_lr.autoregressive_predict(model_wrapper, history_tensor, attention_mask, layer_ranges)
            preds = preds.cpu().tolist()
            for i in range(len(preds)):
                if n_processed >= CANARY_N:
                    break
                pred = tuple(preds[i])
                target = tuple(target_tensor[i].cpu().tolist()[:4])
                candidates_per_sample.append([pred])  # 修复后接口: 1-candidate list
                targets_list.append(target)
                n_processed += 1

    metrics = {}
    for k in K_list:
        metrics[f"R@{k}"] = compute_r_at_k(candidates_per_sample, targets_list, k)
    for k in K_list:
        metrics[f"NDCG@{k}"] = compute_ndcg_at_k(candidates_per_sample, targets_list, k)

    log.append(f"[Canary n={n_processed}] metrics={metrics}")
    log.append(f"[Verification] 修复后 6 指标:")
    vals = list(metrics.values())
    if len(set(vals)) == 1:
        log.append(f"  ⚠️  6 指标仍全同值 ({vals[0]}) — 数学正确 (argmax 单 candidate 时 R@K≡R@1 恒等)")
        log.append(f"  解释: 修复前 `p[:k]==t[:k]` (k>4 退化为 exact match) → R@5=R@10=R@20 trivially 恒等")
        log.append(f"        修复后函数接受 (K candidates per sample) 接口, 1-candidate 时数学上必同值, 但接口正确")
        log.append(f"        真正让 6 指标差异化需要 beam search 提供 K candidates (Step 1.5, 下个 tick)")
    else:
        log.append(f"  ✅  6 指标已差异化 (含不同 R@K, 证明函数正确)")

    r10 = metrics["R@10"]
    if r10 > 0:
        log.append(f"[Canary Gate 4 Step 4] ✅ PASS: R@10 = {r10:.4f} > 0 (非零, 满足 issue Step 4 要求)")
    else:
        log.append(f"[Canary Gate 4 Step 4] ❌ FAIL: R@10 = {r10:.4f} = 0 (issue Step 4 要求非零)")

    out_dir = "/home/wlia0047/ar57/wenyu/GeneRec/verdicts"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/issue4_step1_canary_result.json"
    with open(out_path, "w") as f:
        json.dump({
            "issue": 4,
            "ckpt": CKPT_PATH,
            "ckpt_epoch": ckpt.get("epoch", "N/A"),
            "canary_n": n_processed,
            "metrics": metrics,
            "verdict": "PASS" if r10 > 0 else "FAIL",
            "function_fix_note": "compute_r_at_k + compute_ndcg_at_k 改为 K-candidate 接口, argmax 时数学正确 (R@K≡R@1 恒等), beam search 时 R@K 单调非降",
        }, f, indent=2)
    log.append(f"[Verdict saved] {out_path}")

    log_str = "\n".join(log)
    print(log_str)
    with open("/home/wlia0047/.claude/jobs/accff6eb/tmp/issue4_step1_canary.log", "w") as f:
        f.write(log_str)


if __name__ == "__main__":
    main()
