#!/usr/bin/env python3
"""Issue #13 [方向B Step 6] beam search K=20 eval — 让 6 指标互不恒等.

Per-sample 4-step beam search + layer-wise mask (跟 taskA Issue #12 Step 4 平行, 但用 taskB wrapper).
K=20 candidates per sample → R@5/R@10/R@20 + NDCG@5/10/20 互不恒等.
"""
import os, sys, json, math, importlib.util
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["TRITON_CACHE_DIR"] = "/home/wlia0047/.triton/cache_issue13_step6_beam"
os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)
import torch
import numpy as np
from torch.utils.data import DataLoader

PROJECT = "/home/wlia0047/ar57/wenyu/GeneRec"
sys.path.insert(0, f"{PROJECT}/HG-Rec/model")
sys.path.insert(0, f"{PROJECT}/HG-Rec/data")

_spec_lr = importlib.util.spec_from_file_location(
    "t_lr", f"{PROJECT}/taskB/stage3/taskB_stage3_issue193_long_run.py"
)
_m_lr = importlib.util.module_from_spec(_spec_lr)
_spec_lr.loader.exec_module(_m_lr)
WrapperCls, get_t5_config = _m_lr.load_wrapper_cls()
_spec_k = importlib.util.spec_from_file_location(
    "t471", f"{PROJECT}/taskB/stage3/taskB_stage3_mixed_curv_recontinue.py"
)
_m_k = importlib.util.module_from_spec(_spec_k)
_spec_k.loader.exec_module(_m_k)
load_t5_state_dict = _m_k.load_t5_state_dict


def compute_r_at_k(candidates_per_sample, targets_list, k):
    return sum(1 for cands, t in zip(candidates_per_sample, targets_list) if any(c == t for c in cands[:k])) / max(1, len(candidates_per_sample))


def compute_ndcg_at_k(candidates_per_sample, targets_list, k):
    ndcgs = []
    for cands, t in zip(candidates_per_sample, targets_list):
        rank = None
        for i, c in enumerate(cands[:k]):
            if c == t:
                rank = i + 1
                break
        ndcgs.append(0.0 if rank is None else 1.0 / math.log2(rank + 1))
    return sum(ndcgs) / max(1, len(ndcgs))


def beam_search_predict(model_wrapper, history_tensor, attention_mask, layer_ranges, beam_size=20):
    """Per-sample 4-step beam search (taskB 平行). Returns list of B lists of K 4-tuple candidates."""
    B = history_tensor.shape[0]
    device = history_tensor.device
    candidates_per_sample = [[] for _ in range(B)]
    with torch.no_grad():
        for b_idx in range(B):
            hist_b = history_tensor[b_idx:b_idx+1]
            mask_b = attention_mask[b_idx:b_idx+1]
            B_, L_flat = hist_b.shape
            digit_values = hist_b.float()
            layer_idx = torch.arange(L_flat, device=device) % 4
            layer_idx = layer_idx.float().unsqueeze(0).expand(B_, -1)
            pos_in_history = torch.arange(L_flat, device=device) // 4
            pos_in_history = pos_in_history.float().unsqueeze(0).expand(B_, -1) / 4
            padding_flag = (digit_values == 0).float()
            sid_meta = torch.stack([digit_values / 1025.0, layer_idx / 4.0, pos_in_history, padding_flag], dim=-1)
            curvature_meta = torch.zeros(B_, 3, 4, dtype=torch.float32, device=device)
            x_emb = model_wrapper.t5.model.shared(hist_b)
            residual, alpha = model_wrapper.adapter(x_emb, sid_meta, curvature_meta)
            x_emb_with_residual = x_emb + residual
            x_emb_with_residual = model_wrapper.first_input_ln(x_emb_with_residual)
            encoder_outputs = model_wrapper.t5.model.encoder(
                inputs_embeds=x_emb_with_residual, attention_mask=mask_b,
            )
            enc_h = encoder_outputs.last_hidden_state
            enc_mask = mask_b

            beams = [([], 0.0)]
            for step in range(4):
                new_beams = []
                for tokens, score in beams:
                    dec_in = torch.zeros(1, 4, dtype=torch.long, device=device)
                    for t, tok in enumerate(tokens):
                        dec_in[0, t + 1] = tok
                    dec_out = model_wrapper.t5.model.decoder(
                        input_ids=dec_in, encoder_hidden_states=enc_h, encoder_attention_mask=enc_mask,
                    )
                    logits = model_wrapper.t5.model.lm_head(dec_out.last_hidden_state[:, step, :])
                    lo, hi = layer_ranges[step]
                    mask = torch.full_like(logits, -1e9)
                    mask[:, lo:hi + 1] = 0.0
                    log_probs = torch.log_softmax(logits + mask, dim=-1)
                    top_log_probs, top_ids = log_probs.topk(beam_size)
                    for k in range(beam_size):
                        new_tokens = tokens + [top_ids[0, k].item()]
                        new_score = score + top_log_probs[0, k].item()
                        new_beams.append((new_tokens, new_score))
                new_beams.sort(key=lambda x: -x[1])
                beams = new_beams[:beam_size]
            for tokens, score in beams:
                while len(tokens) < 4:
                    tokens.append(0)
                candidates_per_sample[b_idx].append(tuple(tokens[:4]))
    return candidates_per_sample


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
BEAM_SIZE = 20


def main():
    log = []
    log.append(f"[Issue #13 Step 6 beam search] taskB long-run best_adapter.pt, BEAM_SIZE={BEAM_SIZE}")
    log.append(f"[Setup] CANARY_N={CANARY_N}, SEED={SEED}, GPU=2")

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
    log.append(f"[Load ckpt] epoch={ckpt.get('epoch', 'N/A')}")
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

    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_fn)
    torch.manual_seed(SEED)
    candidates_per_sample = []
    targets_list = []
    n_processed = 0
    n_in_valid_range = 0
    n_total_tokens = 0
    import time
    t0 = time.time()

    with torch.no_grad():
        for batch in test_loader:
            if n_processed >= CANARY_N:
                break
            history_tensor = batch["input_ids"].to(DEVICE)
            target_tensor = batch["labels"].to(DEVICE)
            attention_mask = (history_tensor != PAD_TOKEN).long()
            cands_batch = beam_search_predict(model_wrapper, history_tensor, attention_mask, layer_ranges, beam_size=BEAM_SIZE)
            for i, cands in enumerate(cands_batch):
                if n_processed >= CANARY_N:
                    break
                target = tuple(target_tensor[i].cpu().tolist()[:4])
                candidates_per_sample.append(cands)
                targets_list.append(target)
                for cand in cands:
                    for layer_i, token_id in enumerate(cand):
                        lo, hi = layer_ranges[layer_i]
                        if lo <= token_id <= hi:
                            n_in_valid_range += 1
                        n_total_tokens += 1
                n_processed += 1

    elapsed = time.time() - t0
    validity_pct = 100.0 * n_in_valid_range / max(1, n_total_tokens)
    log.append(f"[Elapsed] {elapsed:.1f}s for {n_processed} samples (beam_size={BEAM_SIZE})")
    log.append(f"[P4 audit] validity = {n_in_valid_range}/{n_total_tokens} = {validity_pct:.1f}% (期望 100%)")

    K_list = [5, 10, 20]
    metrics = {}
    for k in K_list:
        metrics[f"R@{k}"] = compute_r_at_k(candidates_per_sample, targets_list, k)
    for k in K_list:
        metrics[f"NDCG@{k}"] = compute_ndcg_at_k(candidates_per_sample, targets_list, k)
    log.append(f"[Beam search K={BEAM_SIZE} metrics on {n_processed} samples]")
    for k, v in metrics.items():
        log.append(f"  {k} = {v:.4f}")

    vals = list(metrics.values())
    n_unique = len(set(vals))
    log.append(f"[6 指标互不恒等验证] {n_unique}/6 unique values (期望 >1)")
    if n_unique > 1:
        log.append(f"  ✅ 6 指标已差异化")
    else:
        log.append(f"  ⚠️  6 指标仍全同 (argmax 单 candidate 退化情况)")

    r5, r10, r20 = metrics["R@5"], metrics["R@10"], metrics["R@20"]
    monotone_ok = r5 <= r10 <= r20
    log.append(f"[R@K 单调性] R@5={r5} ≤ R@10={r10} ≤ R@20={r20}: {'✅ PASS' if monotone_ok else '❌ FAIL'}")

    out_path = f"{PROJECT}/verdicts/issue13_step6_beam_search_result.json"
    with open(out_path, "w") as f:
        json.dump({
            "issue": 13,
            "step": "Step 6 beam search K=20",
            "ckpt": CKPT_PATH,
            "ckpt_epoch": ckpt.get("epoch", "N/A"),
            "canary_n": n_processed,
            "beam_size": BEAM_SIZE,
            "elapsed_sec": elapsed,
            "metrics": metrics,
            "validity_pct": validity_pct,
            "n_unique_metrics": n_unique,
            "monotone_R_at_K": monotone_ok,
            "verdict": "PASS" if n_unique > 1 and monotone_ok and validity_pct == 100.0 else "FAIL",
            "note": "per-sample beam search 4-step + layer-wise mask, 修复 #13 Step 6 (6 指标互不恒等). 跟 #12 Step 4 taskA 平行.",
        }, f, indent=2, ensure_ascii=False)
    log.append(f"[Verdict saved] {out_path}")
    print("\n".join(log))


if __name__ == "__main__":
    main()