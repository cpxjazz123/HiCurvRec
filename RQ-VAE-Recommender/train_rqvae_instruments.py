"""DDP RQ-VAE training on pre-computed Musical_Instruments embeddings.

Loads item_emb.npy (9922, 768) and trains RqVae on 4 GPUs via torchrun.

启动:
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 \
      /home/wlia0047/ar57/wenyu/GeneRec/RQ-VAE-Recommender/train_rqvae_instruments.py

R36/R47: items are precomputed by preprocess_instruments.py using sentence-t5-xxl.
R42: 必须 torchrun --nproc_per_node=4 (DDP 4 卡).
R30/R43: 超参硬编码, 无 CLI 数值超参.

step 计数 = 全球 step (all_reduce SUM 每 step), 与 RQ-VAE-Recommender 论文 400k iter 对齐.
每 50k 全球 step rank 0 保存一次 ckpt; 训完保存 final.
"""
import json
import os
import random as _random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

# === R51+ 6 确定性约束 (硬编码, R47 + R51 联动) ===
os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision("high")

# R47 imports — RQ-VAE-Recommender modules
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/RQ-VAE-Recommender")
from modules.rqvae import RqVae
from modules.quantize import QuantizeForwardMode
from modules.tokenizer.semids import SemanticIdTokenizer
from data.schemas import SeqBatch


# === 超参 (硬编码 R30/R43) ===
SEED = 42
EMB_NPY = "/home/wlia0047/hj82_scratch2/wenyu/rqvae_dataset/instruments/item_emb.npy"
OUT_DIR = "/home/wlia0047/hj82_scratch2/wenyu/rqvae_dataset/instruments/rqvae_out"

INPUT_DIM = 768
HIDDEN_DIMS = [512, 256, 128]
EMBED_DIM = 32
CODEBOOK_SIZE = 256
N_LAYERS = 3
COMMITMENT_WEIGHT = 0.25

# === 训练硬约束 (R30/R43) ===
MAX_GLOBAL_STEPS = 400_000       # RQ-VAE-Recommender 论文 400k iter (全球)
CKPT_EVERY = 50_000              # 每 50k 全球 step 保存一份 ckpt

# === 加速调参 (硬编码, R36 加速 OK) ===
BATCH_SIZE = 640                 # per-GPU batch (4 卡 DDP, 总 batch = BATCH_SIZE * 4)
NUM_WORKERS = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 4
COMPILE = False


class EmbeddingDataset(Dataset):
    def __init__(self, emb_path: str):
        arr = np.load(emb_path).astype(np.float32)
        self.embeddings = torch.from_numpy(arr)
        if dist.get_rank() == 0:
            print(f"[data] loaded {emb_path}: shape={tuple(self.embeddings.shape)}, dtype={self.embeddings.dtype}", flush=True)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx]


def collate_items(batch):
    return torch.stack(batch, dim=0)


def worker_init_fn(worker_id: int):
    """R51+ DataLoader worker RNG 固定 (per-worker seed 由 base_seed + worker_id 派生)."""
    base_seed = SEED + dist.get_rank() * 1000 + worker_id
    _random.seed(base_seed)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)


def setup_distributed():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed():
    dist.destroy_process_group()


def save_ckpt(model, optimizer, global_step, out_dir, tag):
    """rank 0 only: 保存 model.module.state_dict() + optimizer + step."""
    if dist.get_rank() != 0:
        return
    path = os.path.join(out_dir, f"rqvae_{tag}.pt")
    state = {
        "model": model.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": global_step,
    }
    torch.save(state, path)
    print(f"[ckpt] saved {path} (step={global_step})", flush=True)


def main():
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    torch.manual_seed(SEED + rank)
    np.random.seed(SEED + rank)
    _random.seed(SEED + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED + rank)

    if rank == 0:
        os.makedirs(OUT_DIR, exist_ok=True)
        print(f"=== RQ-VAE Musical_Instruments DDP (world_size={world_size}) ===", flush=True)
        print(f"emb: {EMB_NPY}", flush=True)
        print(f"out: {OUT_DIR}", flush=True)
        print(f"per-GPU batch={BATCH_SIZE} | total={BATCH_SIZE*world_size} | workers={NUM_WORKERS} | compile={COMPILE}", flush=True)
        print(f"hidden={HIDDEN_DIMS} embed={EMBED_DIM} codebook={CODEBOOK_SIZE} layers={N_LAYERS}", flush=True)
        print(f"target: MAX_GLOBAL_STEPS={MAX_GLOBAL_STEPS} | ckpt every {CKPT_EVERY}", flush=True)

    ds = EmbeddingDataset(EMB_NPY)
    n_items = len(ds)
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True, seed=SEED, drop_last=True)
    train_loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS and NUM_WORKERS > 0,
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
        drop_last=True,
        collate_fn=collate_items,
        worker_init_fn=worker_init_fn,
    )

    if rank == 0:
        print(f"[data] {n_items} items | {len(train_loader)} batches/rank | {len(train_loader)*world_size} batches/epoch", flush=True)

    model = RqVae(
        input_dim=INPUT_DIM,
        embed_dim=EMBED_DIM,
        hidden_dims=HIDDEN_DIMS,
        codebook_size=CODEBOOK_SIZE,
        codebook_kmeans_init=True,
        codebook_normalize=False,
        codebook_sim_vq=False,
        codebook_mode=QuantizeForwardMode.STE,
        n_layers=N_LAYERS,
        n_cat_features=0,
        commitment_weight=COMMITMENT_WEIGHT,
    ).to(device)

    if COMPILE:
        model = torch.compile(model)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    if rank == 0:
        n_params = sum(p.numel() for p in model.module.parameters())
        print(f"[model] params={n_params} | DDP wrapped", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    global_step = 0   # 全球 step (= 单卡 step × world_size, 用 all_reduce SUM 同步)
    t_start = time.time()
    last_log_t = t_start

    if rank == 0:
        print(f"[train] start, target={MAX_GLOBAL_STEPS} global steps", flush=True)
    model.train()
    done = False
    while not done:
        sampler.set_epoch(global_step)
        for batch in train_loader:
            x = batch.to(device, non_blocking=True)
            bsz = x.shape[0]
            seq_batch = SeqBatch(
                user_ids=torch.zeros(bsz, dtype=torch.long, device=device),
                ids=torch.arange(bsz, dtype=torch.long, device=device),
                ids_fut=torch.zeros(bsz, dtype=torch.long, device=device),
                x=x,
                x_fut=x,
                seq_mask=torch.ones(bsz, dtype=torch.long, device=device),
            )
            optimizer.zero_grad()
            out = model(seq_batch, gumbel_t=0.2)
            loss = out.loss
            loss.backward()
            optimizer.step()
            global_step += 1

            # 同步全球 step 计数 (rank 0 累加其他 rank 的 +1)
            step_tensor = torch.tensor([global_step], dtype=torch.long, device=device)
            dist.all_reduce(step_tensor, op=dist.ReduceOp.SUM)
            global_step_sync = int(step_tensor.item())

            now = time.time()
            if rank == 0 and (now - last_log_t >= 5.0):
                elapsed = now - t_start
                # 全球 it/s = 全球 step / 时间
                ips_global = global_step_sync / elapsed
                ips_per_gpu = (global_step_sync / world_size) / elapsed
                eta_s = (MAX_GLOBAL_STEPS - global_step_sync) / ips_global if ips_global > 0 else float("inf")
                print(
                    f"  step {global_step_sync:6d}/{MAX_GLOBAL_STEPS} "
                    f"| loss={float(loss.item()):.4f} "
                    f"| rl={float(out.reconstruction_loss.item()):.4f} "
                    f"| vl={float(out.rqvae_loss.item()):.4f} "
                    f"| ips_g={ips_global:.1f} ips/rk={ips_per_gpu:.1f} "
                    f"| elapsed={elapsed:.1f}s | eta={eta_s:.1f}s",
                    flush=True,
                )
                last_log_t = now

            # ckpt 保存 (rank 0 only, 每 CKPT_EVERY 步一次)
            if global_step_sync >= MAX_GLOBAL_STEPS or (
                global_step_sync > 0 and global_step_sync % CKPT_EVERY == 0
            ):
                tag = f"step{global_step_sync}"
                if global_step_sync >= MAX_GLOBAL_STEPS:
                    tag = "final"
                save_ckpt(model, optimizer, global_step_sync, OUT_DIR, tag)
                dist.barrier()  # 等所有 rank 到齐再继续

            if global_step_sync >= MAX_GLOBAL_STEPS:
                done = True
                break

    if rank == 0:
        print(f"[train] done at global_step={global_step_sync}, total time={(time.time()-t_start):.1f}s", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_distributed()
