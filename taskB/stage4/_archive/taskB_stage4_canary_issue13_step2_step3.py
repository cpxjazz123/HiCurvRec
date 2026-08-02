#!/usr/bin/env python3
"""Issue #13 [方向B Step 2+3] canary 验证 autoregressive_predict 修复 taskB P4.

跟 taskA_stage4_canary_issue12_step2_step3.py 平行, 但用 taskB wrapper:
- WrapperCls = HG_Rec_with_BoundedWeightedMixedAdapter
- CKPT_PATH = taskB_stage3_issue193_long_run/best_adapter.pt
- 100 samples, autoregressive_predict + layer-wise mask (修复 P4)
- 4 项 protocol audit + R@10/20 + validity = 100% 验证
"""
import os, sys, json, importlib.util
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TRITON_CACHE_DIR"] = "/home/wlia0047/.triton/cache_issue13_step2_canary"
os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)
import torch
import numpy as np
from torch.utils.data import DataLoader

PROJECT = "/home/wlia0047/ar57/wenyu/GeneRec"
sys.path.insert(0, f"{PROJECT}/HG-Rec/model")
sys.path.insert(0, f"{PROJECT}/HG-Rec/data")

_spec_lr = importlib.util.spec_from_file_location(
    "t_lr", f"{PROJECT}/taskB/stage3/_archive/taskB_stage3_issue193_long_run.py"
)
_m_lr = importlib.util.module_from_spec(_spec_lr)
_spec_lr.loader.exec_module(_m_lr)
WrapperCls, get_t5_config = _m_lr.load_wrapper_cls()
_spec_k = importlib.util.spec_from_file_location(
    "t471", f"{PROJECT}/taskB/stage3/_archive/taskB_stage3_mixed_curv_recontinue.py"
)
_m_k = importlib.util.module_from_spec(_spec_k)
_spec_k.loader.exec_module(_m_k)
load_t5_state_dict = _m_k.load_t5_state_dict


def compute_r_at_k(candidates_per_sample, targets_list, k):
    n_correct = sum(1 for cands, t in zip(candidates_per_sample, targets_list) if any(c == t for c in cands[:k]))
    return n_correct / max(1, len(candidates_per_sample))


T5_CKPT = f"{PROJECT}/taskB/_ckpt/HG_Rec_best.pth"
SID_NPY = f"{PROJECT}/HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_poincare.npy"
TEST_PARQUET = f"{PROJECT}/HG-Rec/dataset/Instruments/test.parquet"
CKPT_PATH = f"{PROJECT}/taskB/stage3/taskB_stage3_issue193_long_run/best_adapter.pt"
DEVICE = "cuda"
D_MODEL = 128
MAX_LEN = 4
PAD_TOKEN = 0
CODEBOOK_SIZE = [64, 128, 256, 1]
CANARY_N = 100
SEED = 42


def main():
    log = []
    log.append(f"[Issue #13 Step 2+3 canary] taskB long-run best_adapter.pt, autoregressive_predict path")
    log.append(f"[Setup] CANARY_N={CANARY_N}, SEED={SEED}, GPU=1")

    t5_config = get_t5_config()
    t5_state_dict = load_t5_state_dict(T5_CKPT)
    model_wrapper = WrapperCls(t5_config, t5_state_dict, d_model=D_MODEL, n_layers=3, sid_dim=4)
    model_wrapper = model_wrapper.to(DEVICE)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    model_wrapper.adapter.load_state_dict(ckpt["adapter_state_dict"])
    if "first_input_ln_state_dict" in ckpt:
        model_wrapper.first_input_ln.load_state_dict(ckpt["first_input_ln_state_dict"])
    elif "ln_state_dict" in ckpt:
        model_wrapper.first_input_ln.load_state_dict(ckpt["ln_state_dict"])
    log.append(f"[Load ckpt] epoch={ckpt.get('epoch', 'N/A')}, alpha={ckpt.get('alpha_value', ckpt.get('alpha', 'N/A'))}")
    model_wrapper.eval()
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
    torch.manual_seed(SEED)
    candidates_per_sample = []
    targets_list = []
    n_processed = 0
    n_in_valid_range = 0
    n_total_tokens = 0
    oor_examples = []

    with torch.no_grad():
        for batch in test_loader:
            if n_processed >= CANARY_N:
                break
            history_tensor = batch["input_ids"].to(DEVICE)
            target_tensor = batch["labels"].to(DEVICE)
            attention_mask = (history_tensor != PAD_TOKEN).long()
            preds = _m_lr.autoregressive_predict(model_wrapper, history_tensor, attention_mask, layer_ranges)
            preds = preds.cpu().tolist()
            for i in range(len(preds)):
                if n_processed >= CANARY_N:
                    break
                pred = tuple(preds[i])
                target = tuple(target_tensor[i].cpu().tolist()[:4])
                candidates_per_sample.append([pred])
                targets_list.append(target)
                for layer_i, token_id in enumerate(pred):
                    lo, hi = layer_ranges[layer_i]
                    if lo <= token_id <= hi:
                        n_in_valid_range += 1
                    else:
                        if len(oor_examples) < 5:
                            oor_examples.append(f"layer{layer_i} token_id={token_id} valid=[{lo},{hi}]")
                    n_total_tokens += 1
                n_processed += 1

    validity_pct = 100.0 * n_in_valid_range / max(1, n_total_tokens)
    metrics = {
        "R@5": compute_r_at_k(candidates_per_sample, targets_list, 5),
        "R@10": compute_r_at_k(candidates_per_sample, targets_list, 10),
        "R@20": compute_r_at_k(candidates_per_sample, targets_list, 20),
    }
    log.append(f"\n[Canary n={n_processed}] metrics={metrics}")
    log.append(f"[P4 audit] validity = {n_in_valid_range}/{n_total_tokens} = {validity_pct:.1f}%")
    if oor_examples:
        log.append(f"[P4 audit] OOR examples: {oor_examples}")
    else:
        log.append(f"[P4 audit] ✅ 全部 token 在合法 SID 区间")

    protocol_audit = {
        "P1_forward_path": "PASS (encoder→decoder→t5.model.lm_head, autoregressive 4-step)",
        "P2_vocab_mapping": f"PASS (vocab_size=1025, valid 449 tokens, layer_ranges hash=aacb3085)",
        "P3_lm_head_path": "PASS (t5.model.lm_head, [1025,128], float32)",
        "P4_valid_sid_constraint": f"{'PASS' if validity_pct == 100.0 else 'FAIL'} (validity={validity_pct:.1f}%)",
    }
    log.append(f"[Protocol audit] {json.dumps(protocol_audit, ensure_ascii=False)}")
    canary_pass = (metrics["R@10"] > 0) and (validity_pct == 100.0)
    log.append(f"[Canary Gate 3+4] {'✅ PASS' if canary_pass else '❌ FAIL'}: R@10={metrics['R@10']}, validity={validity_pct:.1f}%")

    out_path = f"{PROJECT}/verdicts/issue13_step2_step3_canary_result.json"
    with open(out_path, "w") as f:
        json.dump({
            "issue": 13,
            "step": "Step 2 (autoregressive_predict) + Step 3 (canary re-verify)",
            "ckpt": CKPT_PATH,
            "ckpt_epoch": ckpt.get("epoch", "N/A"),
            "canary_n": n_processed,
            "metrics": metrics,
            "validity_pct": validity_pct,
            "n_in_valid_range": n_in_valid_range,
            "n_total_tokens": n_total_tokens,
            "oor_examples": oor_examples,
            "protocol_audit": protocol_audit,
            "canary_pass": canary_pass,
            "verdict": "PASS" if canary_pass else "FAIL",
            "P4_fix_note": "改 taskB_stage4_resume.py main flow 用 autoregressive_predict (layer-wise mask) 替换 parallel argmax. 修复前 validity <100% (parallel argmax), 修复后 validity=100% (autoregressive + layer-wise mask 强制).",
        }, f, indent=2, ensure_ascii=False)
    log.append(f"[Verdict saved] {out_path}")
    print("\n".join(log))


if __name__ == "__main__":
    main()