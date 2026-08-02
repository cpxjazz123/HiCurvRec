#!/usr/bin/env python3
"""Issue #14 [方向A Step 1] Task84 test 全量评估 — beam K=20 六指标产出.

跟 #12 Step 4 taskA_stage4_beam_search 平行, 但跑全部 24772 samples.
支持多 GPU 并行: --gpu 0/1/2/3 --start_idx X --end_idx Y.

输出 partial JSON (单卡结果) 到 verdicts/issue14_full_eval_gpuN_partial.json.
聚合由外部脚本完成.
"""
import os, sys, json, math, time, argparse, importlib.util
import torch
import numpy as np
from torch.utils.data import DataLoader

PROJECT = "/home/wlia0047/ar57/wenyu/GeneRec"
sys.path.insert(0, f"{PROJECT}/HG-Rec/model")
sys.path.insert(0, f"{PROJECT}/HG-Rec/data")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", default="0")
    p.add_argument("--start_idx", type=int, default=0, help="起始 sample index")
    p.add_argument("--end_idx", type=int, default=24772, help="结束 sample index (exclusive)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--beam_size", type=int, default=20)
    p.add_argument("--canary_n", type=int, default=None, help="override: 限制 sample 数 (smoke test)")
    return p.parse_args()


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
    """Per-sample 4-step beam search. Returns list of B lists of K 4-tuple candidates."""
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
            kappa_meta = torch.zeros(B_, 3, device=device)
            scale_meta = torch.ones(B_, 3, device=device)
            x_emb = model_wrapper.t5.model.shared(hist_b)
            residual, alpha = model_wrapper.adapter(x_emb, sid_meta, kappa_meta, scale_meta)
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


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["TRITON_CACHE_DIR"] = f"/home/wlia0047/.triton/cache_issue14_full_eval_gpu{args.gpu}"
    os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)

    T5_CKPT = f"{PROJECT}/taskA/_ckpt/HG_Rec_best.pth"
    SID_NPY = f"{PROJECT}/HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_poincare.npy"
    TEST_PARQUET = f"{PROJECT}/HG-Rec/dataset/Instruments/test.parquet"
    CKPT_PATH = f"{PROJECT}/taskA/stage3/taskA_stage3_issue192_long_run/best_adapter.pt"
    DEVICE = "cuda"
    D_MODEL = 128
    MAX_LEN = 4
    PAD_TOKEN = 0
    CODEBOOK_SIZE = [64, 128, 256, 1]
    SEED = args.seed
    BEAM_SIZE = args.beam_size

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

    print(f"[Issue #14 full eval] gpu={args.gpu}, samples [{args.start_idx}, {args.end_idx}), beam_size={BEAM_SIZE}, seed={SEED}", flush=True)

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
    print(f"[Load ckpt] epoch={ckpt.get('epoch', 'N/A')}", flush=True)
    model_wrapper.eval()
    layer_ranges = _m_lr.get_layer_ranges(CODEBOOK_SIZE)

    from dataset import GenRecDataset
    test_ds = GenRecDataset(
        dataset_path=TEST_PARQUET, code_path=SID_NPY, mode="evaluation",
        codebook_size=CODEBOOK_SIZE, max_len=MAX_LEN, PAD_TOKEN=PAD_TOKEN,
    )

    # 切片 samples
    all_indices = list(range(len(test_ds)))
    if args.canary_n:
        all_indices = all_indices[: args.canary_n]
    slice_indices = all_indices[args.start_idx: args.end_idx]
    print(f"[Slice] {len(slice_indices)} samples out of {len(all_indices)} total", flush=True)

    def collate_fn(batch_with_idx):
        histories = [b["history"] for b in batch_with_idx]
        targets = [b["target"] for b in batch_with_idx]
        max_L = max(len(h) for h in histories)
        history_padded = np.zeros((len(batch_with_idx), max_L, 4), dtype=np.int64)
        for i, h in enumerate(histories):
            L = len(h)
            for j in range(L):
                history_padded[i, j] = h[j]
        target_arr = np.stack(targets, axis=0)
        return {
            "input_ids": torch.from_numpy(history_padded.reshape(len(batch_with_idx), -1)),
            "labels": torch.from_numpy(target_arr),
        }

    # 用 Subset 切片 dataset
    from torch.utils.data import Subset
    subset = Subset(test_ds, slice_indices)
    test_loader = DataLoader(subset, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_fn)
    torch.manual_seed(SEED)

    candidates_per_sample = []
    targets_list = []
    n_processed = 0
    n_in_valid_range = 0
    n_total_tokens = 0
    t0 = time.time()

    with torch.no_grad():
        for batch in test_loader:
            history_tensor = batch["input_ids"].to(DEVICE)
            target_tensor = batch["labels"].to(DEVICE)
            attention_mask = (history_tensor != PAD_TOKEN).long()
            cands_batch = beam_search_predict(model_wrapper, history_tensor, attention_mask, layer_ranges, beam_size=BEAM_SIZE)
            for i, cands in enumerate(cands_batch):
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
            if n_processed % 200 == 0:
                elapsed = time.time() - t0
                rate = n_processed / elapsed
                eta = (len(slice_indices) - n_processed) / rate
                print(f"  [gpu{args.gpu}] progress {n_processed}/{len(slice_indices)} ({100.0*n_processed/len(slice_indices):.1f}%) elapsed={elapsed:.0f}s rate={rate:.2f}/s eta={eta:.0f}s", flush=True)

    elapsed = time.time() - t0
    validity_pct = 100.0 * n_in_valid_range / max(1, n_total_tokens)
    K_list = [5, 10, 20]
    metrics = {}
    for k in K_list:
        metrics[f"R@{k}"] = compute_r_at_k(candidates_per_sample, targets_list, k)
    for k in K_list:
        metrics[f"NDCG@{k}"] = compute_ndcg_at_k(candidates_per_sample, targets_list, k)

    vals = list(metrics.values())
    n_unique = len(set(vals))
    r5, r10, r20 = metrics["R@5"], metrics["R@10"], metrics["R@20"]
    monotone_ok = r5 <= r10 <= r20

    print(f"\n[Issue #14 full eval gpu={args.gpu} done]", flush=True)
    print(f"  n_processed = {n_processed}", flush=True)
    print(f"  elapsed_sec = {elapsed:.1f}", flush=True)
    print(f"  validity_pct = {validity_pct:.1f}%", flush=True)
    print(f"  metrics = {metrics}", flush=True)
    print(f"  n_unique_metrics = {n_unique}/6", flush=True)
    print(f"  monotone_R_at_K = {monotone_ok}", flush=True)

    out_path = f"{PROJECT}/verdicts/issue14_full_eval_gpu{args.gpu}_partial.json"
    with open(out_path, "w") as f:
        json.dump({
            "issue": 14,
            "gpu": args.gpu,
            "slice": [args.start_idx, args.end_idx],
            "ckpt": CKPT_PATH,
            "ckpt_epoch": ckpt.get("epoch", "N/A"),
            "n_samples": n_processed,
            "beam_size": BEAM_SIZE,
            "seed": SEED,
            "elapsed_sec": elapsed,
            "validity_pct": validity_pct,
            "n_in_valid_range": n_in_valid_range,
            "n_total_tokens": n_total_tokens,
            "metrics": metrics,
            "n_unique_metrics": n_unique,
            "monotone_R_at_K": monotone_ok,
            "layer_ranges": layer_ranges,
            "targets_list_sample": targets_list[:5],  # 前 5 个 target 留底
            "verdict": "PASS" if (validity_pct == 100.0 and n_unique > 1 and monotone_ok) else "FAIL",
        }, f, indent=2, ensure_ascii=False)
    print(f"[Verdict saved] {out_path}", flush=True)


if __name__ == "__main__":
    main()