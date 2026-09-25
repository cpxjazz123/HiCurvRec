"""Eval per-token prediction accuracy + cross-entropy on valid set.

Loads a stage3 ckpt + its SID file, builds the same GenRecDataset used in training,
runs teacher-forced forward pass to collect logits for each token position
(SOS / T0 / T1 / T2 / EOS, total 5 positions), and reports per-position:
  - mean cross-entropy (token-level CE, ignore_history only for valid target)
  - top-1 accuracy (greedy argmax vs ground truth token at that position)
  - top-5 accuracy
  - top-20 accuracy

Usage (硬编码无 CLI flag):
    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 scripts/eval_token_level.py

Configs (在 main 顶部修改, 不接受外部参数):
    CKPT_PATH, SID_JSON, LABEL (用于 print)
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# === 硬编码配置 (按需手动改 3 个值即可, 无 CLI flag) ===
CKPT_PATH    = "/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/iter10_gumbel_softmax_anneal/ckpt/Amazon_2023_Instruments/Sep-18-2026_14-47-18/HG_Rec_best.pth"
SID_JSON     = "/home/wlia0047/ar57/wenyu/GeneRec/stage3_T5Train/dataset/Amazon_2023_Instruments/item_sids_iter10.json"
LABEL        = "iter10_gumbel_softmax_anneal"
# ============================================

SCRIPT_DIR = Path(__file__).resolve().parent
STAGE3_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(STAGE3_DIR))

from data.dataset import GenRecDataset
from data.dataloader import GenRecDataLoader
from model.hg_rec import HG_Rec
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "train_hg_rec", str(STAGE3_DIR / "train_HG-Rec.py")
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
make_sample_collator = _mod.make_sample_collator


def build_model(sid_json: str, device: torch.device) -> tuple[HG_Rec, GenRecDataset]:
    config = {
        "n_user_tokens": 1,
        "max_len": 20,
        "vocab_size": 0,
        "eos_token_id": 0,
        "pad_token_id": 0,
        "decoder_start_token_id": 0,
        "max_token_seq_len": 82,
        "codebook_size": [256, 256, 256, 1],
        "sid_length": 4,
        "exclude_history": True,
        "dataset_path":  "./dataset/",
        "dataset_name":  "Amazon_2023_Instruments",
        # T5 架构参数 (与 train_HG-Rec.py 硬编码一致)
        "num_layers": 4,
        "num_decoder_layers": 4,
        "d_model": 128,
        "d_ff": 1024,
        "num_heads": 6,
        "d_kv": 64,
        "dropout_rate": 0.1,
        "activation_function": "relu",
        "feed_forward_proj": "relu",
        "exclude_history": True,
        "bf16": False,
    }
    ds = _mod._build_dataset(config, "valid_recbole.parquet", "evaluation", sid_json)
    config["vocab_size"]            = ds.vocab_size
    config["eos_token_id"]          = ds.eos_token
    config["pad_token_id"]          = ds.pad_token_id
    config["decoder_start_token_id"] = ds.pad_token_id
    model = HG_Rec(config).to(device)
    return model, ds


def evaluate_token_level(model: HG_Rec, ds: GenRecDataset, device: torch.device) -> dict:
    collator = make_sample_collator(ds)
    loader = GenRecDataLoader(
        ds, batch_size=512, shuffle=False, num_workers=4,
        sample_collator=collator,
        max_token_seq_len=82,
        pad_token_id=ds.pad_token_id,
        pin_memory=False,
    )
    n_positions = ds.n_digit + 2  # SOS + 3 SID + EOS
    ce_sum = torch.zeros(n_positions)
    acc1_sum = torch.zeros(n_positions)
    acc5_sum = torch.zeros(n_positions)
    acc20_sum = torch.zeros(n_positions)
    count = torch.zeros(n_positions)

    model.eval()
    with torch.inference_mode():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)  # [B, n_positions]
            loss, logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            logits = logits  # [B, n_positions, vocab]
            B, T, V = logits.shape
            for t in range(T):
                valid = labels[:, t] != ds.pad_token_id
                if valid.sum().item() == 0:
                    continue
                lp = logits[valid, t, :]
                tgt = labels[valid, t]
                ce = F.cross_entropy(lp, tgt, reduction="mean")
                ce_sum[t] += ce.item() * valid.sum().item()
                topk = lp.topk(min(20, V), dim=-1).indices
                acc1_sum[t] += (topk[:, 0] == tgt).float().sum().item()
                acc5_sum[t] += (topk[:, :5] == tgt.unsqueeze(-1)).any(dim=-1).float().sum().item()
                acc20_sum[t] += (topk[:, :20] == tgt.unsqueeze(-1)).any(dim=-1).float().sum().item()
                count[t] += valid.sum().item()

    result = {}
    pos_names = ["SOS"] + [f"T{i}" for i in range(ds.n_digit)] + ["EOS"]
    for t in range(n_positions):
        n = count[t].item()
        if n == 0:
            continue
        result[pos_names[t]] = {
            "n": int(n),
            "ce":     ce_sum[t].item() / n,
            "acc@1":  acc1_sum[t].item() / n,
            "acc@5":  acc5_sum[t].item() / n,
            "acc@20": acc20_sum[t].item() / n,
        }
    return result


def main() -> None:
    if not Path(CKPT_PATH).is_file():
        raise FileNotFoundError(f"ckpt 不存在: {CKPT_PATH}")
    if not Path(SID_JSON).is_file():
        raise FileNotFoundError(f"SID json 不存在: {SID_JSON}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== eval_token_level: {LABEL} ===")
    print(f"  ckpt:    {CKPT_PATH}")
    print(f"  sid:     {SID_JSON}")
    print(f"  device:  {device}")

    model, ds = build_model(SID_JSON, device)
    state = torch.load(CKPT_PATH, map_location=device)
    if "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected keys: {unexpected[:3]}")
    print(f"  loaded ckpt, missing_keys={len(missing)}")

    res = evaluate_token_level(model, ds, device)
    print(f"\n{'pos':<6} {'n':>10} {'CE':>10} {'acc@1':>10} {'acc@5':>10} {'acc@20':>10}")
    print("-" * 60)
    for pos, m in res.items():
        print(f"{pos:<6} {m['n']:>10d} {m['ce']:>10.4f} {m['acc@1']:>10.4f} {m['acc@5']:>10.4f} {m['acc@20']:>10.4f}")


if __name__ == "__main__":
    main()
