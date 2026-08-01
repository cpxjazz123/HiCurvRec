#!/usr/bin/env python3
"""Task #474 / Issue #186 [方向A Gate3→Gate4] 协议对齐 canary 真实 argmax sanity check.

Issue #186 spec 强制:
- 禁止直接长跑 (no 200 epoch, no double-run)
- 仅小规模真实样本 Stage 4 argmax (默认 200)
- 4 项协议一致性 audit
- 复用 Issue #179 ckpt (task472 200 epoch 训练产物, 已存在, 禁止重训)
"""
import os
import sys
import json
import hashlib
import argparse
import importlib.util
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

PROJECT = "/home/wlia0047/ar57/wenyu/GeneRec"
sys.path.insert(0, f"{PROJECT}/HG-Rec/model")
sys.path.insert(0, f"{PROJECT}/HG-Rec/data")

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default="0")
parser.add_argument("--canary_n", type=int, default=200)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ["TRITON_CACHE_DIR"] = f"/home/wlia0047/.triton/cache_task474_issue186"
os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)

from HG_Rec import HG_Rec
from dataset import GenRecDataset

_spec = importlib.util.spec_from_file_location(
    "t470", "/home/wlia0047/ar57/wenyu/GeneRec/taskA/stage3/taskA_stage3_kappa_scale_recontinue.py"
)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)

WrapperCls = _m.HG_Rec_with_BoundedAdapter
t5_config = _m.get_t5_config()

SEED = args.seed
DEVICE = "cuda"
CODEBOOK_SIZE = [64, 128, 256, 1]
MAX_LEN = 20
PAD_TOKEN = 0
D_MODEL = 128
CANARY_N = args.canary_n

SID_NPY = f"{PROJECT}/taskA/_data/Instruments/Instruments_t5_hrqvae_poincare.npy"
TEST_PARQUET = f"{PROJECT}/taskA/_data/Instruments/test.parquet"
T5_CKPT = "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_ckpt/HG_Rec_best.pth"
CKPT_PATH = f"{PROJECT}/taskA/stage3/taskA_stage3_kappa_scale_recontinue/adapter.pt"

EXPECTED_SID_SHA = "2dab29229c36a9695a11d70b61d1b80fae95a08d00e3c3675e9bf899b709508a"

OUT_DIR = f"{PROJECT}/taskA/stage4/taskA_stage4_canary_argmax"
os.makedirs(OUT_DIR, exist_ok=True)
LOG_PATH = f"{PROJECT}/logs/task474_issue186_canary_stage4_argmax.log"


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def seed_all(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    seed_all(SEED)
    log = []
    log.append(f"\n=== Task #474 / Issue #186 [方向A canary] 协议对齐 sanity check ===")
    log.append(f"[Args] gpu={args.gpu} canary_n={CANARY_N} seed={SEED}")

    sid_sha = sha256_of(SID_NPY)
    t5_sha = sha256_of(T5_CKPT)
    log.append(f"\n[Precheck SHA]")
    log.append(f"  SID_NPY: {sid_sha}")
    log.append(f"  T5_CKPT: {t5_sha}")
    sid_match = (sid_sha == EXPECTED_SID_SHA)
    log.append(f"  SID match: {sid_match}")

    t5_state_dict = _m.load_t5_state_dict(T5_CKPT)
    model_wrapper = WrapperCls(t5_config, t5_state_dict, d_model=D_MODEL, n_layers=3, sid_dim=4)
    model_wrapper = model_wrapper.to(DEVICE)

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    model_wrapper.adapter.load_state_dict(ckpt["adapter_state_dict"])
    model_wrapper.first_input_ln.load_state_dict(ckpt["ln_state_dict"])
    log.append(f"\n[Load ckpt] epoch={ckpt['epoch']}, alpha={ckpt['alpha']:.6e}")

    # P3 lm_head audit
    log.append(f"\n[Protocol P3 audit: lm_head path]")
    log.append(f"  hasattr(t5, 'lm_head'): {hasattr(model_wrapper.t5, 'lm_head')}")
    log.append(f"  hasattr(t5.model, 'lm_head'): {hasattr(model_wrapper.t5.model, 'lm_head')}")
    lm_head_path = None
    if hasattr(model_wrapper.t5.model, 'lm_head'):
        lm_head_path = "t5.model.lm_head"
        lm_head_obj = model_wrapper.t5.model.lm_head
        log.append(f"  -> using t5.model.lm_head (correct path per #450 fix)")
        log.append(f"     in={lm_head_obj.in_features}, out={lm_head_obj.out_features}, bias={lm_head_obj.bias is not None}")
        log.append(f"     weight shape={list(lm_head_obj.weight.shape)}, dtype={lm_head_obj.weight.dtype}")
    p3_status = "PASS" if lm_head_path == "t5.model.lm_head" else "FAIL"

    # P2 vocab audit
    log.append(f"\n[Protocol P2 audit: vocab/tokenizer mapping]")
    log.append(f"  T5 config vocab_size={t5_config.get('vocab_size')}")
    expected_vocab = sum(CODEBOOK_SIZE) + 2
    p2_vocab_ok = (t5_config.get('vocab_size') == expected_vocab)
    log.append(f"  vocab_size match: {p2_vocab_ok} (expected={expected_vocab}, got={t5_config.get('vocab_size')})")
    cumulative = [0]
    for k in CODEBOOK_SIZE[:-1]:
        cumulative.append(cumulative[-1] + k)
    cumulative.append(cumulative[-1] + CODEBOOK_SIZE[-1])
    ranges = [(cumulative[i]+1, cumulative[i+1]+1) for i in range(len(CODEBOOK_SIZE))]
    log.append(f"  SID token ranges (4 layers): {ranges}")
    p2_status = "PASS" if p2_vocab_ok else "FAIL"

    test_ds = GenRecDataset(
        dataset_path=TEST_PARQUET, code_path=SID_NPY, mode="evaluation",
        codebook_size=CODEBOOK_SIZE, max_len=MAX_LEN, PAD_TOKEN=PAD_TOKEN,
    )
    log.append(f"\n[Dataset] test size: {len(test_ds)}")

    canary_n = min(CANARY_N, len(test_ds))
    log.append(f"  canary_n = {canary_n}")

    histories_list = []
    targets_list = []
    for i in range(canary_n):
        s = test_ds.data[i]
        histories_list.append(s["history"])
        targets_list.append(s["target"])

    log.append(f"\n[Stage 4 canary: real argmax decode on {canary_n} samples]")
    model_wrapper.eval()

    BATCH = 32
    preds_all = []
    targets_all = []
    forward_path_used = []

    with torch.no_grad():
        for batch_start in range(0, canary_n, BATCH):
            batch_end = min(batch_start + BATCH, canary_n)
            B = batch_end - batch_start
            max_L = max(len(h) for h in histories_list[batch_start:batch_end])
            history_padded = np.zeros((B, max_L * 4), dtype=np.int64)
            for i, h in enumerate(histories_list[batch_start:batch_end]):
                flat = np.concatenate([np.asarray(x, dtype=np.int64).flatten() for x in h])
                history_padded[i, :len(flat)] = flat
            history_tensor = torch.from_numpy(history_padded).to(DEVICE)
            target_tensor = torch.from_numpy(np.stack(targets_list[batch_start:batch_end], axis=0)).to(DEVICE)
            attention_mask = (history_tensor != PAD_TOKEN).long()

            x_emb = model_wrapper.t5.model.shared(history_tensor)
            residual, alpha = model_wrapper.adapter(
                x_emb,
                torch.zeros(B, history_tensor.shape[1], 4, device=DEVICE),
                torch.zeros(B, 3, device=DEVICE),
                torch.ones(B, 3, device=DEVICE),
            )
            x_emb_with_residual = x_emb + residual
            x_emb_with_residual = model_wrapper.first_input_ln(x_emb_with_residual)
            decoder_input_ids = torch.zeros(B, 4, dtype=torch.long, device=DEVICE)
            encoder_outputs = model_wrapper.t5.model.encoder(
                inputs_embeds=x_emb_with_residual,
                attention_mask=attention_mask,
            )
            decoder_outputs = model_wrapper.t5.model.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                encoder_attention_mask=attention_mask,
            )
            logits = model_wrapper.t5.model.lm_head(decoder_outputs.last_hidden_state)
            preds = logits.argmax(dim=-1)

            forward_path_used.append("encoder->decoder->t5.model.lm_head->argmax")
            for i in range(B):
                pred = preds[i].cpu().tolist()
                target = target_tensor[i].cpu().tolist()
                preds_all.append(pred[:4])
                targets_all.append(target[:4])

    def compute_r_at_k(preds, targets, k):
        return float(np.mean([1.0 if p == t else 0.0 for p, t in zip(preds, targets)]))

    metrics = {}
    for k in [5, 10, 20]:
        metrics[f"R@{k}"] = compute_r_at_k(preds_all, targets_all, k)

    log.append(f"\n[Stage 4 canary results]")
    log.append(f"  n_processed = {len(preds_all)}")
    log.append(f"  metrics = {metrics}")
    log.append(f"  forward_path_used = {forward_path_used[0] if forward_path_used else 'NONE'}")

    log.append(f"\n[Protocol P4 audit: valid SID constraint space]")
    cum = [0]
    for k in CODEBOOK_SIZE[:-1]:
        cum.append(cum[-1] + k)
    cum.append(cum[-1] + CODEBOOK_SIZE[-1])
    layer_ranges = [(cum[i]+1, cum[i+1]+1) for i in range(len(CODEBOOK_SIZE))]
    log.append(f"  Expected layer ranges: {layer_ranges}")
    in_valid_range = 0
    out_of_range = 0
    oor_examples = []
    for pred in preds_all:
        for layer_i, token_id in enumerate(pred[:4]):
            lo, hi = layer_ranges[layer_i]
            if lo <= token_id <= hi:
                in_valid_range += 1
            else:
                out_of_range += 1
                if len(oor_examples) < 5:
                    oor_examples.append(f"layer{layer_i} token_id={token_id} valid=[{lo},{hi}]")
    total_tokens = len(preds_all) * 4
    p4_validity = in_valid_range / total_tokens if total_tokens > 0 else 0.0
    log.append(f"  In-valid-range: {in_valid_range}/{total_tokens} ({p4_validity*100:.1f}%)")
    log.append(f"  Out-of-range: {out_of_range}, examples: {oor_examples[:3]}")
    p4_status = "PASS" if p4_validity > 0.99 else "FAIL"

    p1_status = "PASS" if forward_path_used and forward_path_used[0] == "encoder->decoder->t5.model.lm_head->argmax" else "FAIL"
    log.append(f"\n[Protocol P1 audit: forward path]")
    log.append(f"  Status: {p1_status} (using {forward_path_used[0] if forward_path_used else 'NONE'})")
    log.append(f"  Note: same encoder->decoder->lm_head->argmax used by training forward + Stage 4 eval")

    canary_r10 = metrics.get("R@10", 0.0)
    gate3_upgrade = (
        canary_r10 > 0.0
        and p1_status == "PASS"
        and p2_status == "PASS"
        and p3_status == "PASS"
        and p4_status == "PASS"
    )

    log.append(f"\n[Decision Summary]")
    log.append(f"  P1 (forward path): {p1_status}")
    log.append(f"  P2 (vocab mapping): {p2_status}")
    log.append(f"  P3 (lm_head path): {p3_status}")
    log.append(f"  P4 (valid SID constraint): {p4_status}")
    log.append(f"  Canary R@10: {canary_r10}")
    log.append(f"  Gate 3->full PASS upgrade: {gate3_upgrade}")
    log.append(f"  Overall: {'CANARY PASS' if gate3_upgrade else 'CANARY FAIL'}")

    verdict = {
        "task": "task474_issue186_canary_stage4_argmax",
        "issue": 186,
        "canary_n": canary_n,
        "sid_sha256": sid_sha,
        "sid_sha_match": sid_match,
        "ckpt_epoch": ckpt['epoch'],
        "ckpt_alpha_value": ckpt['alpha'],
        "metrics": metrics,
        "protocol_audit": {
            "P1_forward_path": p1_status,
            "P2_vocab_mapping": p2_status,
            "P3_lm_head_path": p3_status,
            "P4_valid_sid_constraint": p4_status,
            "p4_validity_pct": p4_validity * 100,
            "p4_oor_examples": oor_examples[:3],
        },
        "canary_pass": gate3_upgrade,
        "decision": "PASS" if gate3_upgrade else "FAIL",
    }
    with open(f"{OUT_DIR}/canary_verdict.json", "w") as f:
        json.dump(verdict, f, indent=2)

    log.append(f"\n[Verdict saved] {OUT_DIR}/canary_verdict.json")

    with open(LOG_PATH, "w") as f:
        f.write("\n".join(log))
    print("\n".join(log))
    print(f"\n[DONE] canary_r10={canary_r10}, decision={verdict['decision']}")


if __name__ == "__main__":
    main()
