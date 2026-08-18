"""C34 standalone test eval — DDP 4 卡, R35b 同口径 (全量 test 24772 samples).

加载 train_stage3_ddp_fast.py 训练产出的 best_ckpt.pt, 4 卡 DistributedSampler 切片
(每样本恰好 1 次), all_reduce SUM hits/NDCG, 输出与单卡完全一致的结果.

差异 vs C33 test_eval.py:
- DDP 4 卡
- bf16 inference
- DistributedSampler 切片
- all_reduce SUM
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "_lib"))

from data.dataset import GenRecDataset  # noqa: E402
from model.hg_rec import HG_Rec  # noqa: E402


# === 与 train_stage3_ddp_fast.py 严格一致 ===
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

INFER_SIZE = 256  # per-rank
BEAM_SIZE = 20
TOPK_LIST = [5, 10, 20]

DATASET_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments"
CODE_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/tasks/Issue181_c33_hg_rec_stage3_arch_t5_ce/dataset/Instruments/Instruments_c28_sids_for_hgrec.npy"
CKPT_PATH = "/home/wlia0047/hj82_scratch2/wenyu/claude_tmp/issue181_c34_ckpt/best_ckpt.pt"


def hgrec_collate(batch, pad_token=PAD_TOKEN_ID):
    flattened_histories = torch.stack(
        [torch.tensor([elem for sublist in item["history"] for elem in sublist], dtype=torch.int64) for item in batch]
    )
    flattened_targets = torch.stack(
        [torch.tensor(item["target"], dtype=torch.int64) for item in batch]
    )
    attention_masks = torch.stack(
        [torch.tensor([1 if elem != pad_token else 0 for elem in h], dtype=torch.int64) for h in flattened_histories]
    )
    return {
        "history": flattened_histories,
        "target": flattened_targets,
        "attention_mask": attention_masks,
    }


def setup_distributed():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def calculate_pos_index(preds, labels, maxk=20):
    preds = preds.detach().cpu()
    labels = labels.detach().cpu()
    pos_index = torch.zeros((preds.shape[0], maxk), dtype=torch.bool)
    for i in range(preds.shape[0]):
        cur_label = labels[i].tolist()
        for j in range(maxk):
            cur_pred = preds[i, j].tolist()
            if cur_pred == cur_label:
                pos_index[i, j] = True
                break
    return pos_index


def recall_at_k(pos_index, k):
    return pos_index[:, :k].sum(dim=1).cpu().float()


def ndcg_at_k(pos_index, k):
    ranks = torch.arange(1, pos_index.shape[-1] + 1).to(pos_index.device)
    dcg = 1.0 / torch.log2(ranks + 1)
    dcg = torch.where(pos_index, dcg, torch.tensor(0.0, dtype=torch.float, device=dcg.device))
    return dcg[:, :k].sum(dim=1).cpu().float()


def main():
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(SEED + rank)
    np.random.seed(SEED + rank)

    config = {
        "num_layers": NUM_LAYERS, "num_decoder_layers": NUM_DECODER_LAYERS,
        "d_model": D_MODEL, "d_ff": D_FF, "num_heads": NUM_HEADS, "d_kv": D_KV,
        "dropout_rate": DROPOUT_RATE, "vocab_size": VOCAB_SIZE,
        "pad_token_id": PAD_TOKEN_ID, "eos_token_id": PAD_TOKEN_ID,
        "decoder_start_token_id": PAD_TOKEN_ID, "feed_forward_proj": "relu",
    }
    model = HG_Rec(config).to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # 加载 best_ckpt (C34 DDP 训练产出, 非 DDP wrapper 的 state_dict)
    state_dict = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model.module.load_state_dict(state_dict)
    model.eval()
    if rank == 0:
        print(f"[load] best ckpt from {CKPT_PATH}", flush=True)
        print(f"[setup] test eval DDP world_size={world_size}, beam={BEAM_SIZE}", flush=True)

    test_ds = GenRecDataset(
        dataset_path=os.path.join(DATASET_DIR, "test.parquet"),
        code_path=CODE_PATH, mode="evaluation",
        codebook_size=CODEBOOK_SIZE, max_len=MAX_LEN, PAD_TOKEN=PAD_TOKEN_ID,
    )
    test_sampler = DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False)
    test_loader = DataLoader(
        test_ds, batch_size=INFER_SIZE, sampler=test_sampler,
        num_workers=2, pin_memory=True, collate_fn=hgrec_collate,
        persistent_workers=True,
    )
    n_test = len(test_ds)
    if rank == 0:
        print(f"[data] test={n_test}, per_rank={n_test//world_size}", flush=True)

    # per-sample metric arrays (R35b: 每样本恰好计数一次)
    local_recall = {k: torch.zeros(n_test // world_size, dtype=torch.float32, device=device) for k in TOPK_LIST}
    local_ndcg = {k: torch.zeros(n_test // world_size, dtype=torch.float32, device=device) for k in TOPK_LIST}
    sample_offset = 0
    t0 = time.time()
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["history"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["target"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                preds = model.module.generate(
                    input_ids=input_ids, attention_mask=attention_mask,
                    num_beams=BEAM_SIZE,
                )
            preds = preds[:, 1:].reshape(input_ids.shape[0], BEAM_SIZE, -1)
            pos_index = calculate_pos_index(preds, labels, maxk=BEAM_SIZE).to(device)
            bs = pos_index.shape[0]
            for k in TOPK_LIST:
                local_recall[k][sample_offset:sample_offset+bs] = recall_at_k(pos_index, k).to(device)
                local_ndcg[k][sample_offset:sample_offset+bs] = ndcg_at_k(pos_index, k).to(device)
            sample_offset += bs

    # R35b: 必须 all_gather 拼接各 rank tensor → 完整 (n_test,) 数组 → 求和 / n_test
    # DistributedSampler 切片连续: rank 0 = [0:w), rank 1 = [w:2w), etc
    gathered_recall = {k: torch.zeros(n_test, dtype=torch.float32, device=device) for k in TOPK_LIST}
    gathered_ndcg = {k: torch.zeros(n_test, dtype=torch.float32, device=device) for k in TOPK_LIST}
    per_rank = n_test // world_size
    for k in TOPK_LIST:
        rbuf = [torch.zeros(per_rank, dtype=torch.float32, device=device) for _ in range(world_size)]
        dist.all_gather(rbuf, local_recall[k])
        gathered_recall[k] = torch.cat(rbuf)[:n_test]
        nbuf = [torch.zeros(per_rank, dtype=torch.float32, device=device) for _ in range(world_size)]
        dist.all_gather(nbuf, local_ndcg[k])
        gathered_ndcg[k] = torch.cat(nbuf)[:n_test]
    total_recall = {k: v.sum().item() / n_test for k, v in gathered_recall.items()}
    total_ndcg = {k: v.sum().item() / n_test for k, v in gathered_ndcg.items()}

    t_eval = time.time() - t0
    if rank == 0:
        print(f"\n=== TEST FINAL (DDP 4 卡, beam={BEAM_SIZE}, R35b 同口径) ===", flush=True)
        print(f"  test eval time: {t_eval:.1f}s", flush=True)
        for k in TOPK_LIST:
            print(f"  Recall@{k}: {total_recall[k]:.4f}", flush=True)
            print(f"  NDCG@{k}: {total_ndcg[k]:.4f}", flush=True)

        result = {
            "ckpt": CKPT_PATH,
            "ddp_world_size": world_size,
            "beam_size": BEAM_SIZE,
            "n_test": n_test,
            "test_eval_time_s": t_eval,
            "test_recall@5": total_recall[5],
            "test_recall@10": total_recall[10],
            "test_recall@20": total_recall[20],
            "test_ndcg@5": total_ndcg[5],
            "test_ndcg@10": total_ndcg[10],
            "test_ndcg@20": total_ndcg[20],
        }
        out_json = "/home/wlia0047/hj82_scratch2/wenyu/claude_tmp/issue181_c34_train/test_final.json"
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n[save] {out_json}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()