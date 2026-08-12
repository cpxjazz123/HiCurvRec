"""Issue #136 reference-0 数据画像 — 独立 canary forward (deterministic, no backward)

Issue133 Stage3 协议:
- history (item_id 列表) → item_to_code → 4-token SID per item → flatten to (max_len*4,) tokens
- target (item_id) → item_to_code → 4-token SID
- input_ids (B, 80), target (B, 4), max_len=20
- 不使用 tokenizer, 直接 token id 序列
"""

import os
import sys
import json
import time
from pathlib import Path

TASK_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/tasks/Issue136_reference0_data_profile")
os.chdir(TASK_DIR)
sys.path.insert(0, str(TASK_DIR))
sys.path.insert(0, str(TASK_DIR / "_lib"))

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
REF0 = REPO_ROOT / "tasks/Issue133_stage2_baseline_c_exp"
DATASET_DIR = REPO_ROOT / "dataset"

OUT_DIR = TASK_DIR / "data_profile"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 硬编码 (R30 + R43)
CKPT_PATH = REF0 / "stage2/HG_Rec_best.pth"
SID_PATH = REF0 / "stage2/sid_output.npy"
TEST_PARQUET = DATASET_DIR / "test.parquet"
OUTPUT_JSON = OUT_DIR / "raw_predictions.json"

BEAM_SIZE = 20
MAX_GEN_LEN = 8  # 4 SID tokens + EOS
MIN_GEN_LEN = 4  # 至少 4 个 SID tokens
MAX_SAMPLES = 5000

# Issue133 Stage3 CONFIG (R30 硬编码)
CONFIG = dict(
    num_layers=6, num_decoder_layers=4, d_model=128, d_ff=1024,
    num_heads=6, d_kv=64, dropout_rate=0.1, vocab_size=1025,
    pad_token_id=0, eos_token_id=0, decoder_start_token_id=0,
    feed_forward_proj="relu",
)
MAX_LEN = 20
CODEBOOK_SIZE = [64, 128, 256, 1]
PAD_TOKEN = 0


def item_to_code(item_id, sid_array, codebook_size):
    """Issue133 protocol: item_id → 4 SID tokens via codebook_size offsets.

    offsets = [c0 + 1, c1 + 65, c2 + 193, c3 + 449]
    (cumulative sum of codebook_size + 1 per position)
    """
    if item_id < 0 or item_id >= sid_array.shape[0]:
        return [PAD_TOKEN] * 4
    code = sid_array[item_id]  # (4,)
    offsets = [int(code[i]) + sum(codebook_size[0:i]) + 1 for i in range(4)]
    return offsets


def history_to_tokens(history, sid_array, codebook_size, max_len=20):
    """history (item_id list) → (max_len*4,) SID token sequence (left-padded)."""
    history = list(history[-max_len:])
    history = [0] * (max_len - len(history)) + history  # 0 = PAD item
    tokens = []
    for item_id in history:
        codes = item_to_code(item_id, sid_array, codebook_size)
        tokens.extend(codes)
    return tokens


def main():
    t0 = time.time()
    print(f"[canary] 开始加载 Issue133 ckpt + 跑 forward", flush=True)

    import torch
    import numpy as np
    import pyarrow.parquet as pq

    sid = np.load(SID_PATH)
    n_items = sid.shape[0]
    print(f"  SID shape={sid.shape}, codebook_size={CODEBOOK_SIZE}", flush=True)

    test_df = pq.read_table(TEST_PARQUET).to_pandas()
    print(f"  test.parquet rows={len(test_df)}, columns={list(test_df.columns)}", flush=True)

    from _lib.HG_Rec import HG_Rec
    model = HG_Rec(CONFIG)
    print(f"  HG_Rec created", flush=True)

    if hasattr(model, "install_hab"):
        model.install_hab()
        with torch.no_grad():
            model.hab_module.lambda_raw.data.fill_(0.0)
        print(f"  HAB installed + freeze baseline", flush=True)
    if hasattr(model, "install_per_head_curvature"):
        model.install_per_head_curvature()
        with torch.no_grad():
            model.hab_module.lambda_h_raw.data.fill_(0.0)
        print(f"  per-head installed + freeze", flush=True)

    state = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  ckpt loaded: missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    PAD_ID = 0
    EOS_ID = 0

    raw_preds = []
    n_samples = len(test_df)
    sample_indices = list(range(min(MAX_SAMPLES, n_samples)))

    with torch.no_grad():
        for i in sample_indices:
            row = test_df.iloc[i]
            history_raw = row.get("history")
            if history_raw is None:
                history = []
            elif hasattr(history_raw, "tolist"):
                history = history_raw.tolist()
            else:
                history = list(history_raw)
            target_item = int(row.get("target", 0))

            input_ids = history_to_tokens(history, sid, CODEBOOK_SIZE, MAX_LEN)
            input_ids_t = torch.tensor([input_ids], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids_t)

            target_sid = sid[target_item].tolist() if 0 <= target_item < n_items else [0, 0, 0, 0]

            try:
                gen = model.model.generate(
                    input_ids=input_ids_t,
                    attention_mask=attention_mask,
                    max_length=MAX_GEN_LEN,
                    min_length=MIN_GEN_LEN,
                    num_beams=BEAM_SIZE,
                    num_return_sequences=BEAM_SIZE,
                    early_stopping=True,
                    pad_token_id=PAD_ID,
                    eos_token_id=EOS_ID,
                )
            except Exception as e:
                print(f"  [sample {i}] generate ERROR: {e}", flush=True)
                continue

            preds = []
            for b in range(BEAM_SIZE):
                seq = gen[b].tolist()
                if EOS_ID in seq:
                    seq = seq[:seq.index(EOS_ID)]
                if len(seq) >= 4:
                    sid_pred = tuple(seq[:4])
                else:
                    sid_pred = tuple(seq + [PAD_ID] * (4 - len(seq)))
                preds.append(list(sid_pred))

            raw_preds.append({
                "sample_id": i,
                "history_len": len(history),
                "target_item": target_item,
                "target_sid": target_sid,
                "pred_top20_sids": preds,
            })

            if (i + 1) % 500 == 0:
                print(f"  [{i+1}/{len(sample_indices)}] done, "
                      f"elapsed={time.time()-t0:.1f}s", flush=True)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(raw_preds, f, indent=2)
    print(f"  -> {OUTPUT_JSON} ({len(raw_preds)} samples, "
          f"{OUTPUT_JSON.stat().st_size/1024:.1f}KB)", flush=True)
    print(f"[canary] 完成 ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()