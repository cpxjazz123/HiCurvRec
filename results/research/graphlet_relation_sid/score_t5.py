"""Score the frozen TIGER_RQ-VAE T5 checkpoint on the TEST split, without retraining.

Two readings per test row:

    target_rank_top10   the rank the archived 4-GPU beam=20 evaluation already recorded
                         (read from the run's `_test_shards/rank*.json`, not recomputed)
    target_logprob       teacher-forced log p(SID tokens + EOS | user + history), from one forward pass

The forward pass is only a scorer: no generation, no gradient, no parameter update.  The run's
recorded overall recall@10 is recomputed from the shard ranks and must match `test_final.json`, so a
wrong checkpoint or a wrong row order fails loudly instead of silently joining the wrong model.

    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        results/research/graphlet_relation_sid/score_t5.py

Writes `test_scores.parquet` and `test_scores_meta.json` next to this file.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
STAGE3 = REPO / "stage3_T5Train"
if str(STAGE3) not in sys.path:
    sys.path.insert(0, str(STAGE3))

import torch  # noqa: E402

from data.dataloader import GenRecDataLoader  # noqa: E402
from data.dataset import GenRecDataset  # noqa: E402
from model.hg_rec import HG_Rec  # noqa: E402

TEST = REPO / "results" / "stage0_build_parquet" / "test.parquet"
SID_JSON = REPO / "results" / "stage2_RQ-VAE" / "TIGER_RQ-VAE" / "item_sids.json"
RUN_DIR = (
    REPO
    / "results" / "stage3_T5Train" / "TIGER_RQ-VAE"
    / "logs" / "Amazon_2023_Instruments" / "Nov-12-2026_21-04-04"
)
CHECKPOINT = (
    REPO
    / "results" / "stage3_T5Train" / "TIGER_RQ-VAE"
    / "ckpt" / "Amazon_2023_Instruments" / "Nov-12-2026_21-04-04" / "HG_Rec_best.pth"
)
OUT = HERE / "test_scores.parquet"
META = HERE / "test_scores_meta.json"

MAX_LEN = 20
BATCH = 512
N_USER_TOKENS = 1
CODEBOOK_SIZE = [256, 256, 256, 1]


def make_collator(dataset: GenRecDataset):
    def sample_collator(item):
        history_tokens = [token for sid in item["history"] for token in sid]
        return {
            "input_ids": [dataset.user_token_for(item["user"]), *history_tokens, dataset.eos_token],
            "labels": [*item["target"], dataset.eos_token],
            "target_item": int(item["target_item"]),
        }

    return sample_collator


def load_ranks(world_size: int = 4) -> np.ndarray:
    shard_dir = RUN_DIR / "_test_shards"
    ranks = np.full(57439, np.nan)
    covered = np.zeros(57439, dtype=bool)
    for path in sorted(shard_dir.glob("rank*.json")):
        payload = json.loads(path.read_text())
        rank = int(payload["rank"])
        for entry in payload["per_user"]:
            global_row = rank + int(entry["local_row_index"]) * world_size
            value = entry["target_rank"]
            ranks[global_row] = np.nan if value is None else float(value)
            covered[global_row] = True
    if not covered.all():
        raise SystemExit(f"test shards do not cover every row: {covered.sum()} of {len(covered)}")
    return ranks


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = GenRecDataset(
        dataset_path=TEST,
        code_path=SID_JSON,
        mode="evaluation",
        codebook_size=CODEBOOK_SIZE,
        max_len=MAX_LEN,
        PAD_TOKEN=0,
        n_user_tokens=N_USER_TOKENS,
    )
    config = {
        "num_layers": 4,
        "num_decoder_layers": 4,
        "d_model": 128,
        "d_ff": 1024,
        "num_heads": 6,
        "d_kv": 64,
        "dropout_rate": 0.1,
        "activation_function": "relu",
        "feed_forward_proj": "relu",
        "vocab_size": dataset.vocab_size,
        "pad_token_id": 0,
        "eos_token_id": dataset.eos_token,
        "decoder_start_token_id": 0,
        "max_token_seq_len": dataset.max_token_seq_len,
        "sid_length": dataset.n_digit,
    }
    model = HG_Rec(config).to(device)
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    model.eval()

    loader = GenRecDataLoader(
        dataset,
        batch_size=BATCH,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sample_collator=make_collator(dataset),
        max_token_seq_len=dataset.max_token_seq_len,
        pad_token_id=0,
    )

    rows = {
        "global_row": [],
        "target_item": [],
        "target_sid1": [],
        "target_sid2": [],
        "target_sid3": [],
        "logprob_total": [],
        "logprob_sid1": [],
        "logprob_sid2": [],
        "logprob_sid3": [],
        "logprob_eos": [],
    }
    offset = 0
    with torch.inference_mode():
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            outputs = model.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits.float()
            log_probs = torch.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            width = labels.shape[1]
            if width != dataset.n_digit + 1:
                raise SystemExit(f"label width {width}, expected {dataset.n_digit + 1}")
            rows["global_row"].extend(range(offset, offset + labels.shape[0]))
            rows["target_item"].extend(batch["target_item"].tolist())
            rows["target_sid1"].extend(labels[:, 0].cpu().tolist())
            rows["target_sid2"].extend(labels[:, 1].cpu().tolist())
            rows["target_sid3"].extend(labels[:, 2].cpu().tolist())
            rows["logprob_total"].extend(token_log_probs.sum(dim=1).cpu().tolist())
            rows["logprob_sid1"].extend(token_log_probs[:, 0].cpu().tolist())
            rows["logprob_sid2"].extend(token_log_probs[:, 1].cpu().tolist())
            rows["logprob_sid3"].extend(token_log_probs[:, 2].cpu().tolist())
            rows["logprob_eos"].extend(token_log_probs[:, 3].cpu().tolist())
            offset += labels.shape[0]
            if offset % 5120 == 0:
                print(f"scored {offset}/{len(dataset)}", flush=True)
    if offset != len(dataset):
        raise SystemExit(f"scored {offset} rows, dataset has {len(dataset)}")

    frame = pd.DataFrame(rows)
    ranks = load_ranks()
    frame["target_rank_top10"] = ranks
    test = pd.read_parquet(TEST, columns=["user", "seen_history", "target"])
    frame["last_item"] = [int(history[-1]) if len(history) else -1 for history in test["seen_history"].to_numpy()]
    frame["history_length"] = [len(history) for history in test["seen_history"].to_numpy()]
    frame["target_item_parquet"] = test["target"].to_numpy(np.int64)
    if not (frame["target_item"].to_numpy() == frame["target_item_parquet"].to_numpy()).all():
        raise SystemExit("dataset target order does not match test.parquet")
    frame = frame.drop(columns=["target_item_parquet"])
    frame.to_parquet(OUT, index=False)

    recall10 = float((ranks < 10).mean())
    recall5 = float((ranks < 5).mean())
    recorded = json.loads((RUN_DIR / "test_final.json").read_text())
    meta = {
        "checkpoint": str(CHECKPOINT.relative_to(REPO)),
        "run_dir": str(RUN_DIR.relative_to(REPO)),
        "sid_file": str(SID_JSON.relative_to(REPO)),
        "n_rows": int(len(frame)),
        "recomputed_recall@5": recall5,
        "recomputed_recall@10": recall10,
        "recorded_recall@5": recorded["test_recall@5"],
        "recorded_recall@10": recorded["test_recall@10"],
        "recall_matches": bool(abs(recall10 - recorded["test_recall@10"]) < 1e-12),
        "mean_logprob": float(frame["logprob_total"].mean()),
        "median_logprob": float(frame["logprob_total"].median()),
    }
    META.write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)
    if not meta["recall_matches"]:
        raise SystemExit("shard ranks and test_final.json disagree; refusing to publish")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
