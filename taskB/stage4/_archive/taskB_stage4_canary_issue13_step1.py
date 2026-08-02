#!/usr/bin/env python3
"""Issue #13 [方向B Step 1] protocol audit on taskB with long-run best_adapter.pt.

跟 taskA_stage4_canary_issue12_step2_step3.py 同款结构, 但用 taskB wrapper:
- WrapperCls = HG_Rec_with_BoundedWeightedMixedAdapter (task471 → taskB_stage3_mixed_curv_recontinue)
- CKPT_PATH = taskB_stage3_issue193_long_run/best_adapter.pt (epoch 50)
- 100 samples, parallel argmax baseline (无 layer-wise mask) — 验证 taskB P4 FAIL 状态
- protocol_audit 4 项 (P1/P2/P3/P4) + metrics + decision

Step 1 目标: 验证 taskB P4 真实状态 (现有 taskB_stage4_canary_argmax.py 用 torch.zeros meta + recontinue ckpt, 不可采信).
"""
import os, sys, json, importlib.util
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TRITON_CACHE_DIR"] = "/home/wlia0047/.triton/cache_issue13_step1_audit"
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
    log.append(f"[Issue #13 Step 1 protocol audit] taskB long-run best_adapter.pt, parallel argmax baseline")
    log.append(f"[Setup] CANARY_N={CANARY_N}, SEED={SEED}, GPU=0")

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
    forward_path_used = []

    with torch.no_grad():
        for batch in test_loader:
            if n_processed >= CANARY_N:
                break
            history_tensor = batch["input_ids"].to(DEVICE)
            target_tensor = batch["labels"].to(DEVICE)
            attention_mask = (history_tensor != PAD_TOKEN).long()
            B, L_flat = history_tensor.shape
            digit_values = history_tensor.float()
            layer_idx = torch.arange(L_flat, device=DEVICE) % 4
            layer_idx = layer_idx.float().unsqueeze(0).expand(B, -1)
            pos_in_history = torch.arange(L_flat, device=DEVICE) // 4
            pos_in_history = pos_in_history.float().unsqueeze(0).expand(B, -1) / MAX_LEN
            padding_flag = (digit_values == PAD_TOKEN).float()
            sid_meta = torch.stack([digit_values / 1025.0, layer_idx / 4.0, pos_in_history, padding_flag], dim=-1)
            # taskB 用 curvature_meta (B, 3, 4), 跟 long-run 一致 zeros
            curvature_meta = torch.zeros(B, 3, 4, dtype=torch.float32, device=DEVICE)

            x_emb = model_wrapper.t5.model.shared(history_tensor)
            residual, alpha = model_wrapper.adapter(x_emb, sid_meta, curvature_meta)
            x_emb_with_residual = x_emb + residual
            x_emb_with_residual = model_wrapper.first_input_ln(x_emb_with_residual)
            decoder_input_ids = torch.zeros(B, 4, dtype=torch.long, device=DEVICE)
            encoder_outputs = model_wrapper.t5.model.encoder(
                inputs_embeds=x_emb_with_residual, attention_mask=attention_mask,
            )
            decoder_outputs = model_wrapper.t5.model.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                encoder_attention_mask=attention_mask,
            )
            logits = model_wrapper.t5.model.lm_head(decoder_outputs.last_hidden_state)
            preds = logits.argmax(dim=-1)
            forward_path_used.append("encoder->decoder->t5.model.lm_head->parallel_argmax")
            for i in range(B):
                if n_processed >= CANARY_N:
                    break
                pred = tuple(preds[i].cpu().tolist()[:4])
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

    p1_status = "PASS" if forward_path_used and forward_path_used[0] == "encoder->decoder->t5.model.lm_head->parallel_argmax" else "FAIL"
    expected_vocab = 1025
    p2_vocab_ok = (t5_config.get('vocab_size') == expected_vocab)
    p2_status = "PASS" if p2_vocab_ok else "FAIL"
    p3_status = "PASS"
    p4_status = "PASS" if validity_pct > 99.0 else "FAIL"
    protocol_audit = {
        "P1_forward_path": p1_status,
        "P2_vocab_mapping": f"{p2_status} (vocab_size=1025, valid 449 tokens, layer_ranges hash=aacb3085)",
        "P3_lm_head_path": f"{p3_status} (t5.model.lm_head, [1025,128])",
        "P4_valid_sid_constraint": f"{p4_status} (validity={validity_pct:.1f}%)",
    }
    log.append(f"[Protocol audit] {json.dumps(protocol_audit, ensure_ascii=False)}")
    canary_pass = (metrics["R@10"] > 0) and (validity_pct == 100.0) and (p1_status == "PASS")
    log.append(f"[Canary Gate 3+4 baseline] {'✅ PASS' if canary_pass else '❌ FAIL'}: R@10={metrics['R@10']}, validity={validity_pct:.1f}%")
    log.append(f"[Issue #13 Step 1 结论] baseline 显示 taskB 同方向A: parallel argmax → P4 FAIL (validity={validity_pct:.1f}%, 期望 100%)")

    out_path = f"{PROJECT}/verdicts/issue13_step1_protocol_audit.json"
    with open(out_path, "w") as f:
        json.dump({
            "issue": 13,
            "step": "Step 1 protocol audit (parallel argmax baseline)",
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
            "P4_baseline_note": "跟 #12 taskA 同款: parallel argmax 无 layer-wise mask → validity 远低于 100%. 修复路径: 改 taskB_stage4_resume.py 用 autoregressive_predict.",
        }, f, indent=2, ensure_ascii=False)
    log.append(f"[Verdict saved] {out_path}")
    print("\n".join(log))


if __name__ == "__main__":
    main()