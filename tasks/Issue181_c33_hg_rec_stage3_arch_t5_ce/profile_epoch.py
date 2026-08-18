"""Profile C33 epoch time — train + valid eval 各部分耗时."""
import os
import sys
import time

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "_lib"))

from data.dataset import GenRecDataset  # noqa: E402
from data.dataloader import GenRecDataLoader  # noqa: E402
from model.hg_rec import HG_Rec  # noqa: E402


SEED = 42
MAX_LEN = 20
CODEBOOK_SIZE = [256, 256, 256]
VOCAB_SIZE = 769
PAD_TOKEN_ID = 0

D_MODEL = 128
D_FF = 1024
NUM_HEADS = 6
D_KV = 64
NUM_LAYERS = 6
NUM_DECODER_LAYERS = 4
DROPOUT_RATE = 0.1

BATCH_SIZE = 256
INFER_SIZE = 96
BEAM_SIZE = 20

DATASET_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments"
CODE_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/tasks/Issue181_c33_hg_rec_stage3_arch_t5_ce/dataset/Instruments/Instruments_c28_sids_for_hgrec.npy"


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda")

    config = {
        "num_layers": NUM_LAYERS, "num_decoder_layers": NUM_DECODER_LAYERS,
        "d_model": D_MODEL, "d_ff": D_FF, "num_heads": NUM_HEADS, "d_kv": D_KV,
        "dropout_rate": DROPOUT_RATE, "vocab_size": VOCAB_SIZE,
        "pad_token_id": PAD_TOKEN_ID, "eos_token_id": PAD_TOKEN_ID,
        "decoder_start_token_id": PAD_TOKEN_ID, "feed_forward_proj": "relu",
    }
    model = HG_Rec(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    print(f"[setup] HG_Rec params={sum(p.numel() for p in model.parameters()):,}", flush=True)

    train_ds = GenRecDataset(
        dataset_path=os.path.join(DATASET_DIR, "train.parquet"),
        code_path=CODE_PATH, mode="train",
        codebook_size=CODEBOOK_SIZE, max_len=MAX_LEN, PAD_TOKEN=PAD_TOKEN_ID,
    )
    valid_ds = GenRecDataset(
        dataset_path=os.path.join(DATASET_DIR, "valid.parquet"),
        code_path=CODE_PATH, mode="evaluation",
        codebook_size=CODEBOOK_SIZE, max_len=MAX_LEN, PAD_TOKEN=PAD_TOKEN_ID,
    )
    train_loader = GenRecDataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    valid_loader = GenRecDataLoader(valid_ds, batch_size=INFER_SIZE, shuffle=False, num_workers=2)

    # Profile train: 1 epoch
    print(f"\n=== Train 1 epoch profile (batch_size={BATCH_SIZE}) ===", flush=True)
    model.train()
    t0 = time.time()
    n_batch = 0
    for batch in train_loader:
        input_ids = batch["history"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["target"].to(device)
        optimizer.zero_grad()
        loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss.backward()
        optimizer.step()
        n_batch += 1
    t_train = time.time() - t0
    print(f"  train: {n_batch} batches in {t_train:.1f}s = {t_train/n_batch*1000:.1f}ms/batch", flush=True)

    # Profile valid eval: full + subset
    print(f"\n=== Valid eval profile (beam={BEAM_SIZE}) ===", flush=True)
    model.eval()
    with torch.no_grad():
        # Full valid
        t0 = time.time()
        n_v = 0
        for batch in valid_loader:
            input_ids = batch["history"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            preds = model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=BEAM_SIZE)
            n_v += 1
        t_valid_full = time.time() - t0
        print(f"  full valid: {n_v} batches in {t_valid_full:.1f}s = {t_valid_full/n_v*1000:.1f}ms/batch", flush=True)

        # Subset 1000 samples (~10 batches)
        from torch.utils.data import Subset
        subset = Subset(valid_ds, list(range(1000)))
        sub_loader = GenRecDataLoader(subset, batch_size=INFER_SIZE, shuffle=False, num_workers=0)
        t0 = time.time()
        n_s = 0
        for batch in sub_loader:
            input_ids = batch["history"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            preds = model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=BEAM_SIZE)
            n_s += 1
        t_valid_sub = time.time() - t0
        print(f"  subset 1000 valid: {n_s} batches in {t_valid_sub:.1f}s = {t_valid_sub/n_s*1000:.1f}ms/batch", flush=True)

        # Subset 5000 samples (~52 batches)
        subset2 = Subset(valid_ds, list(range(5000)))
        sub2_loader = GenRecDataLoader(subset2, batch_size=INFER_SIZE, shuffle=False, num_workers=0)
        t0 = time.time()
        n_s2 = 0
        for batch in sub2_loader:
            input_ids = batch["history"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            preds = model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=BEAM_SIZE)
            n_s2 += 1
        t_valid_sub2 = time.time() - t0
        print(f"  subset 5000 valid: {n_s2} batches in {t_valid_sub2:.1f}s = {t_valid_sub2/n_s2*1000:.1f}ms/batch", flush=True)


if __name__ == "__main__":
    main()