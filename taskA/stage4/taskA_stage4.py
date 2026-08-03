#!/usr/bin/env python3
"""Issue #30 [方向A]: baseline 同协议 beam20 重测现有 adapter.

根因修复验证 (verdicts/_analysis_why_no_baseline_exceeded.md):
过去 taskA 报告 val_R@10≈0.03-0.05 是 greedy 单候选协议, 与 baseline 0.1020 标尺不同。
此脚本用 baseline 完全相同协议重测:
  - SID: Instruments_t5_hrqvae_poincare.npy (sha256 2dab2922, 与训练一致)
  - split: HG-Rec/dataset/Instruments/valid.parquet (baseline 用 valid)
  - 解码: t5.model.generate(num_beams=20, num_return_sequences=20) 无约束
  - R@K: target 4-token SID 出现在 top-k beam (train_HG-Rec evaluate)

Sanity: 不 load ckpt (adapter init α≈4.5e-5≈0) 时 beam20 R@10 应复现 baseline 0.1020,
验证数据/解码链路正确; 然后 load 各 ckpt 重测真实增益。

每个 ckpt 必须用其训练时的 wrapper 模块加载 (架构随 issue 演化):
  --ckpt "<wrapper_module.py>@<ckpt_path>"  可多次指定

用法:
  python3 taskA/stage4/taskA_stage4.py \
      --device cuda:0 --n 1000 --tag issue28_diag \
      --ckpt "taskA/stage3/taskA_stage3.py@taskA/_history/taskA_stage3_issue28_gate2_diag/best_adapter.pt"
"""
import argparse
import importlib.util
import json
import os
import sys

import torch

sys.path.insert(0, "/fs04/ar57/wenyu/GeneRec/HG-Rec/model")
sys.path.insert(0, "/fs04/ar57/wenyu/GeneRec/HG-Rec/data")
sys.path.insert(0, "/fs04/ar57/wenyu/GeneRec")

from dataset import GenRecDataset
from common.stage4_eval_beam20 import generate_beam20, recall_at_k
from common.stage4_decode import autoregressive_predict_constrained, get_layer_ranges

DEVICE = None
CODEBOOK_SIZE = [64, 128, 256, 1]
MAX_LEN = 20
PAD_TOKEN = 0
T5_CKPT = "/fs04/ar57/wenyu/GeneRec/taskA/_ckpt/HG_Rec_best.pth"
VALID_PARQUET = "/fs04/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/valid.parquet"
TEST_PARQUET = "/fs04/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/test.parquet"
SID_NPY = os.environ.get("TASKA_STAGE4_SID_NPY",
                         "/fs04/ar57/wenyu/GeneRec/taskA/_data/Instruments/Instruments_t5_hrqvae_poincare.npy")
# split -> parq 映射 (--split test 走 test.parquet)
SPLIT_PARQUET = {"valid": VALID_PARQUET, "test": TEST_PARQUET}
EXPECTED_SID_SHA = os.environ.get("TASKA_STAGE4_SID_SHA",
                                  "2dab29229c36a9695a11d70b61d1b80fae95a08d00e3c3675e9bf899b709508a")
LAYER_RANGES = get_layer_ranges(CODEBOOK_SIZE)
# 默认 (sanity) wrapper: 最新架构 (带 learnable kappa_logits), 指向 stage3 主脚本 (内联类定义, 逐字节一致)
DEFAULT_WRAPPER = "/fs04/ar57/wenyu/GeneRec/taskA/stage3/taskA_stage3.py"


def sha256_of(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def load_wrapper_cls_from(module_path, tag):
    """从脚本模块加载 (WrapperCls, get_t5_config). 兼容 load_wrapper_cls() 或直接类."""
    spec = importlib.util.spec_from_file_location(f"wrap_{tag}", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if hasattr(mod, "load_wrapper_cls"):
        return mod.load_wrapper_cls()
    return mod.HG_Rec_with_BoundedAdapter, mod.get_t5_config


def build_wrapper(module_path, tag):
    WrapperCls, get_t5_config = load_wrapper_cls_from(module_path, tag)
    t5_state_dict = torch.load(T5_CKPT, map_location="cpu", weights_only=False)
    if "state_dict" in t5_state_dict:
        t5_state_dict = t5_state_dict["state_dict"]
    elif "model" in t5_state_dict:
        t5_state_dict = t5_state_dict["model"]
    wrapper = WrapperCls(get_t5_config(), t5_state_dict, d_model=128, n_layers=3, sid_dim=4).to(DEVICE)
    wrapper.eval()
    return wrapper


def build_data(split_parquet, sid_npy, n, stage2_kappas=None, stage2_mix_weights=None):
    ds = GenRecDataset(dataset_path=split_parquet, code_path=sid_npy, mode="evaluation",
                       codebook_size=CODEBOOK_SIZE, max_len=MAX_LEN)
    idxs = list(range(min(n, len(ds))))
    hist, tgt = [], []
    for i in idxs:
        s = ds[i]
        hist.append([e for sub in s["history"] for e in sub])
        tgt.append([int(x) for x in s["target"]])
    history_tensor = torch.tensor(hist, device=DEVICE).long()
    targets = torch.tensor(tgt, device=DEVICE).long()
    # 方向A sid_meta (B, L, 4): [digit/1025, layer_idx/4, pos/MAX_LEN, padding_flag]
    B, L_flat = history_tensor.shape
    digit_values = history_tensor.float()
    layer_idx = (torch.arange(L_flat, device=DEVICE) % 4).float().unsqueeze(0).expand(B, -1) / 4.0
    pos_in_history = (torch.arange(L_flat, device=DEVICE) // 4).float().unsqueeze(0).expand(B, -1) / MAX_LEN
    padding_flag = (digit_values == PAD_TOKEN).float()
    sid_meta = torch.stack([digit_values / 1025.0, layer_idx, pos_in_history, padding_flag], dim=-1)
    kappa_meta = torch.zeros(B, 3, dtype=torch.float32, device=DEVICE)
    scale_meta = torch.ones(B, 3, dtype=torch.float32, device=DEVICE)
    # Issue #55/v2: 把 stage2 学到的 κ / mix_weight 注入 eval meta (与 train 一致)
    if stage2_kappas is not None and any(abs(k) > 1e-6 for k in stage2_kappas):
        kappa_meta = torch.tensor(stage2_kappas, dtype=torch.float32, device=DEVICE).unsqueeze(0).expand(B, -1)
    if stage2_mix_weights is not None:
        scale_meta = torch.tensor(stage2_mix_weights, dtype=torch.float32, device=DEVICE).unsqueeze(0).expand(B, -1)
    attention_mask = (history_tensor != PAD_TOKEN).long()
    return ds, history_tensor, targets, attention_mask, sid_meta, kappa_meta, scale_meta


def eval_all(wrapper, hist, tgt, mask, sid_meta, kappa_meta, scale_meta, batch=32):
    """分批 beam20 评估 (全量 24772 防 OOM). 协议与 fullsplit_baseline_probe.py 一致."""
    B = hist.shape[0]
    n_hit5 = n_hit10 = n_hit20 = 0
    greedy_hit = 0
    alpha_last = 0.0
    for bs in range(0, B, batch):
        be = min(bs + batch, B)
        ht = hist[bs:be]
        am = mask[bs:be]
        tt = tgt[bs:be]
        preds_beam, alpha = generate_beam20(
            wrapper, ht, am, sid_meta=sid_meta[bs:be] if sid_meta is not None else None,
            kappa_meta=kappa_meta[bs:be] if kappa_meta is not None else None,
            scale_meta=scale_meta[bs:be] if scale_meta is not None else None,
            beam=20, max_length=5,
        )
        for i in range(ht.shape[0]):
            t = tt[i].tolist()
            plist = preds_beam[i].tolist()
            n_hit5 += 1 if any(p == t for p in plist[:5]) else 0
            n_hit10 += 1 if any(p == t for p in plist[:10]) else 0
            n_hit20 += 1 if any(p == t for p in plist[:20]) else 0
        preds_greedy = autoregressive_predict_constrained(
            wrapper, ht, am, LAYER_RANGES,
            sid_meta=sid_meta[bs:be] if sid_meta is not None else None,
            kappa_meta=kappa_meta[bs:be] if kappa_meta is not None else None,
            scale_meta=scale_meta[bs:be] if scale_meta is not None else None,
        )
        greedy_hit += sum(1 for i in range(ht.shape[0]) if preds_greedy[i].tolist() == tt[i].tolist())
        alpha_last = float(alpha.detach())
    return {"beam20_r5": round(n_hit5 / B, 4), "beam20_r10": round(n_hit10 / B, 4),
            "beam20_r20": round(n_hit20 / B, 4),
            "greedy_layerwise_r10": round(greedy_hit / B, 4), "alpha": round(alpha_last, 6)}


def main():
    global DEVICE
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", default=[],
                    help='"<wrapper_module.py>@<ckpt_path>", 可多次')
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--tag", default="reeval")
    ap.add_argument("--split", choices=["valid", "test"], default="valid", help="评估 split (valid/test)")
    ap.add_argument("--alpha_logit", type=float, default=None,
                    help="强制覆盖 alpha_logit (eval 时设不同值做 sweep); None=ckpt 原值")
    args = ap.parse_args()
    DEVICE = args.device
    if not torch.cuda.is_available():
        DEVICE = "cpu"

    sid_sha = sha256_of(SID_NPY)
    assert sid_sha == EXPECTED_SID_SHA, f"SID hash mismatch: {sid_sha}"
    eval_parquet = SPLIT_PARQUET[args.split]
    # Issue #55/v2: 自动加载 stage2 ckpt 中的 κ/mix_weight 用于 eval meta
    stage2_kappas = None
    stage2_mix_weights = None
    stage2_ckpt = os.environ.get("TASKA_STAGE2_CKPT",
                                  "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_kappa_sync/hrqvae_kappa_sync.ckpt")
    if os.path.exists(stage2_ckpt):
        try:
            _d = torch.load(stage2_ckpt, map_location="cpu", weights_only=False)
            stage2_kappas = _d.get("final_kappas")
            stage2_mix_weights = _d.get("final_mix_weights")
            print(f"[Stage2 meta] κ={stage2_kappas} mix_weight={stage2_mix_weights}", flush=True)
        except Exception as e:
            print(f"[WARN stage2 load] {e}", flush=True)
    ds, hist, tgt, mask, sid_meta, kappa_meta, scale_meta = build_data(
        eval_parquet, SID_NPY, args.n,
        stage2_kappas=stage2_kappas, stage2_mix_weights=stage2_mix_weights)
    print(f"[data] split={args.split} parquet={eval_parquet} n={hist.shape[0]} (ds={len(ds)}), sid_sha={sid_sha[:12]}...", flush=True)

    results = {"n": hist.shape[0], "sid_sha": sid_sha, "split": args.split, "protocol": "beam20",
               "tag": args.tag, "per_ckpt": {}}

    # 0) T5-only sanity: 不 load ckpt, adapter init α≈0 → 应复现 baseline 0.1020
    wrapper = build_wrapper(DEFAULT_WRAPPER, "sanity")
    if args.alpha_logit is not None:
        with torch.no_grad():
            wrapper.adapter.alpha_logit.data.fill_(args.alpha_logit)
    results["t5_only_sanity"] = eval_all(wrapper, hist, tgt, mask, sid_meta, kappa_meta, scale_meta)
    print(f"[sanity T5-only (init α≈0)] {json.dumps(results['t5_only_sanity'])}", flush=True)

    # 1..N) load 各 ckpt (用其训练时的 wrapper 模块)
    for i, ckpt_arg in enumerate(args.ckpt):
        if "@" in ckpt_arg:
            module_path, ckpt = ckpt_arg.split("@", 1)
        else:
            module_path, ckpt = DEFAULT_WRAPPER, ckpt_arg
        wrapper = build_wrapper(module_path, f"c{i}")
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        if "adapter_state_dict" in ck:
            wrapper.adapter.load_state_dict(ck["adapter_state_dict"])
        if "ln_state_dict" in ck:
            wrapper.first_input_ln.load_state_dict(ck["ln_state_dict"])
        elif "first_input_ln_state_dict" in ck:
            wrapper.first_input_ln.load_state_dict(ck["first_input_ln_state_dict"])
        # v5: 解冻 T5 微调后的 encoder 权重 (含在 ckpt 里), 覆盖 wrapper 内 T5
        if "t5_state_dict" in ck:
            missing, unexpected = wrapper.t5.load_state_dict(ck["t5_state_dict"], strict=False)
            if missing or unexpected:
                print(f"[WARN t5_state_dict] missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        if args.alpha_logit is not None:
            with torch.no_grad():
                wrapper.adapter.alpha_logit.data.fill_(args.alpha_logit)
        r = eval_all(wrapper, hist, tgt, mask, sid_meta, kappa_meta, scale_meta)
        results["per_ckpt"][ckpt] = {"wrapper_module": module_path, **r}
        print(f"[{ckpt} (wrap={module_path.split('/')[-1]})] {json.dumps(r)}", flush=True)

    out_path = f"/fs04/ar57/wenyu/GeneRec/verdicts/issue30_{args.tag}_beam20_reeval.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
