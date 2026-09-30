"""Train TIGER RQ-VAE for a fixed 10,000 synchronized DDP steps and export SIDs."""


from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from tqdm import tqdm

from model import RQVAE


import curvature_config as experiment
# === Fixed experiment paths and global-step budget ===
EMBEDDING_FILE = experiment.EMBEDDING_FILE
TRAIN_FILE = experiment.TRAIN_FILE
OUTPUT_SID = experiment.SIDS_NPY
OUTPUT_JSON = experiment.ITEM_SIDS_JSON
CHECKPOINT = experiment.RQVAE_CKPT_PATH
RAW_SIDS_NPY = experiment.RAW_SIDS_NPY
METRICS_PATH = experiment.STAGE2_LOG_DIR / "training_metrics.jsonl"
LOG_DIR = experiment.STAGE2_LOG_DIR




# === TIGER RQ-VAE hyperparameters (kept from the legacy 0.055 baseline) ===
MAX_GLOBAL_STEPS = experiment.MAX_GLOBAL_STEPS
EVAL_INTERVAL_STEPS = experiment.EVAL_INTERVAL_STEPS
PROGRESS_INTERVAL_STEPS = 1_000
BATCH_SIZE_PER_RANK = 1024
LR = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN_SIZES = (512, 256, 128)
CODEBOOK_NUM = 3
CODEBOOK_SIZE = (256, 256, 256)
CODEBOOK_DIM = 32
BETA = 0.25
VQ_TYPE = "vq"
EMA_DECAY = 0.99
SK_EPSILON = 0.003
SK_ITERS = 50
PCA_DIM = 0
XAVIER_INIT = True
WARMUP_EPOCHS = 50
GRADIENT_CLIP_NORM = 1.0
OPTIMIZER = "AdamW"
SEED = 42
NUM_WORKERS = 0

# === DDP launcher 配置 (与 stage3 不同 master_port 避免冲突) ===
_LAUNCHER = {
    "torchrun":    "/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/torchrun",
    "script":      os.path.abspath(__file__),
    "nproc":       4,
    "master_port": 50202,
    "log":         str(LOG_DIR / "train_migrated.log"),
    "visible_dev": "0,1,2,3",
}


# === 2026-09-22: 全程训练 metrics JSONL 收集器 (rank 0 only) ===
_metrics_lock = threading.Lock()
_T0 = time.time()
# Truncate-on-train_start: every fresh run starts with an empty metrics file
# so a single ``training_metrics.jsonl`` always represents exactly one run
# (no leftover rows from a prior failed attempt).
_metrics_truncated = False
def _record(rank, event, **fields):
    if rank != 0:
        return
    rec = {"event": event, "timestamp": time.time(), "wall_time_s": round(time.time() - _T0, 3)}
    rec.update(fields)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    global _metrics_truncated
    with _metrics_lock:
        # Truncate only the very first time we append — that keeps the
        # file scoped to this run while still letting multiple ``_record``
        # calls in the same epoch run in append mode (cheap).
        if not _metrics_truncated:
            open(METRICS_PATH, "w", encoding="utf-8").close()
            _metrics_truncated = True
        with open(METRICS_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_float32_matmul_precision("high")


def load_embeddings(path: Path) -> np.ndarray:
    """Load an (N, D) item-embedding matrix.

    Accepts either:
    - a Parquet frame with an ``embedding`` column of per-row arrays
      (the historical RecBole3.0 ``item_emb.parquet`` convention), or
    - an ``*.npy`` file containing an ``(N, D)`` ``float32`` array
      (stage1's ``sentence_t5.npy`` output).
    """
    suffix = path.suffix.lower()
    if suffix == ".npy":
        embeddings = np.asarray(np.load(path), dtype="float32")
    elif suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
        if "embedding" not in frame:
            raise ValueError(f"Embedding file must contain an embedding column: {path}")
        embeddings = np.stack(frame["embedding"].to_numpy()).astype("float32", copy=False)
    else:
        raise ValueError(f"Unsupported embedding file suffix: {path}")
    if embeddings.ndim != 2 or not np.isfinite(embeddings).all():
        raise ValueError(f"Invalid embedding matrix: shape={embeddings.shape}")
    return embeddings


def maybe_apply_pca(embeddings: np.ndarray, pca_dim: int, seed: int) -> np.ndarray:
    if pca_dim <= 0:
        return embeddings
    if pca_dim >= embeddings.shape[1]:
        raise ValueError(f"--pca_dim must be smaller than input dimension {embeddings.shape[1]}")
    from sklearn.decomposition import PCA

    reduced = PCA(n_components=pca_dim, whiten=True, random_state=seed).fit_transform(embeddings)
    return reduced.astype("float32", copy=False)


def initialize_tiger_weights(model: RQVAE) -> None:
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def _extend_collisions(tokens: np.ndarray, codebook_sizes: list[int]) -> np.ndarray:
    """Match RecBole3.0's extend path with one fixed extra SID level."""
    if tokens.ndim != 2 or tokens.shape[1] != len(codebook_sizes):
        raise ValueError(f"Token shape {tokens.shape} disagrees with codebook sizes {codebook_sizes}")
    groups: dict[tuple[int, ...], list[int]] = {}
    for item_id, row in enumerate(tokens.tolist()):
        groups.setdefault(tuple(int(value) for value in row), []).append(item_id)
    offsets = np.cumsum([0, *codebook_sizes], dtype=np.int64)
    result = np.empty((len(tokens), tokens.shape[1] + 1), dtype=np.int64)
    for row, item_ids in groups.items():
        for occurrence, item_id in enumerate(item_ids):
            result[item_id, :-1] = np.asarray(row, dtype=np.int64)
            result[item_id, -1] = int(offsets[-1] + occurrence)
    return result


def _tokenizer_config() -> SimpleNamespace:
    sizes = list(CODEBOOK_SIZE)
    if len(sizes) == 1:
        sizes *= CODEBOOK_NUM
    if len(sizes) != CODEBOOK_NUM:
        raise ValueError("CODEBOOK_SIZE must contain one size or one size per codebook level")
    return SimpleNamespace(
        hidden_sizes=HIDDEN_SIZES,
        codebook_num=CODEBOOK_NUM,
        codebook_size=tuple(sizes),
        codebook_dim=CODEBOOK_DIM,
        dropout=0.0,
        beta=BETA,
        vq_type=VQ_TYPE,
        ema_decay=EMA_DECAY,
        fix_code_embs=False,
        sk_epsilon=SK_EPSILON,
        sk_iters=SK_ITERS,
    )


def main() -> None:
    # === DDP 初始化 ===
    if "RANK" in os.environ and int(os.environ.get("RANK", -1)) >= 0:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    set_seed(SEED + rank)
    embeddings = maybe_apply_pca(load_embeddings(EMBEDDING_FILE), PCA_DIM, SEED)
    train_frame = pd.read_parquet(TRAIN_FILE)
    train_ids = np.unique(train_frame["target"].to_numpy(dtype=np.int64))
    if train_ids.size == 0 or train_ids.min() < 0 or train_ids.max() >= len(embeddings):
        raise ValueError("Training target item ids do not match the embedding rows")

    all_embeddings = torch.from_numpy(embeddings)
    train_embeddings = all_embeddings[torch.from_numpy(train_ids)]
    config = _tokenizer_config()
    model = RQVAE(config, in_dim=embeddings.shape[1]).to(device)
    if XAVIER_INIT:
        initialize_tiger_weights(model)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    train_dataset = TensorDataset(train_embeddings)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=SEED, drop_last=False) if world_size > 1 else None
    loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE_PER_RANK,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=False,  # 2026-09-24: 关闭避免每个 epoch 都 fork 一次
    )

    if OPTIMIZER.lower() == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    local_step_target = MAX_GLOBAL_STEPS
    total_steps = max(1, local_step_target)
    warmup_steps = max(0, WARMUP_EPOCHS * len(loader))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if total_steps <= warmup_steps:
            return 1.0
        return max(0.0, float(total_steps - step) / float(total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if rank == 0:
        print(f"[Iter65 TIGER RQ-VAE] device={device} all_items={len(all_embeddings)} train_items={len(train_embeddings)} world_size={world_size}", flush=True)
        print(
            f"[Iter65 TIGER RQ-VAE] max_global_steps={MAX_GLOBAL_STEPS} "
            f"eval_interval_steps={EVAL_INTERVAL_STEPS} "
            f"batch_size_per_rank={BATCH_SIZE_PER_RANK} "
            f"total_batch={BATCH_SIZE_PER_RANK*world_size} lr={LR}",
            flush=True,
        )
        _record(
            rank, "train_start", max_global_steps=MAX_GLOBAL_STEPS,
            eval_interval_steps=EVAL_INTERVAL_STEPS,
            progress_interval_steps=PROGRESS_INTERVAL_STEPS,
            batch_size_per_rank=BATCH_SIZE_PER_RANK,
            total_batch_size=BATCH_SIZE_PER_RANK*world_size,
            lr=LR, weight_decay=WEIGHT_DECAY, hidden_sizes=list(HIDDEN_SIZES),
            codebook_num=CODEBOOK_NUM, codebook_size=list(CODEBOOK_SIZE),
            codebook_dim=CODEBOOK_DIM, beta=BETA, vq_type=VQ_TYPE,
            sk_epsilon=SK_EPSILON, sk_iters=SK_ITERS, pca_dim=PCA_DIM,
            xavier_init=XAVIER_INIT, gradient_clip_norm=GRADIENT_CLIP_NORM,
            optimizer=OPTIMIZER, seed=SEED, num_workers=NUM_WORKERS,
            all_items=len(all_embeddings), train_items=len(train_embeddings),
            embedding_shape=list(embeddings.shape), world_size=world_size,
        )
    raw_module = model.module if isinstance(model, DDP) else model
    if rank == 0:
        print("[Iter65 TIGER RQ-VAE] initializing codebooks with KMeans", flush=True)
    with torch.no_grad():
        raw_module.init_codebook(train_embeddings.to(device))
    if world_size > 1:
        dist.barrier()
    local_optimizer_steps = 0
    global_step = 0
    epoch = 0
    progress_loss_sum = 0.0
    progress_recon_sum = 0.0
    progress_updates = 0
    final_collision = float("nan")
    codebook_sizes = list(config.codebook_size)

    while local_optimizer_steps < local_step_target:
        epoch += 1
        if train_sampler is not None:
            train_sampler.set_epoch(epoch - 1)
        model.train()
        for (batch,) in loader:
            if local_optimizer_steps >= local_step_target:
                break
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            reconstructed, quant_loss, _, _ = model(batch)
            loss, recon_loss = raw_module.compute_loss(batch, reconstructed, quant_loss)
            if not torch.isfinite(loss):
                raise RuntimeError(f"RQ-VAE loss became non-finite at global step {global_step}")
            loss.backward()
            if GRADIENT_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()
            scheduler.step()

            local_optimizer_steps += 1
            global_step = local_optimizer_steps
            progress_loss_sum += float(loss.detach())
            progress_recon_sum += float(recon_loss.detach())
            progress_updates += 1

            should_log = (
                global_step % PROGRESS_INTERVAL_STEPS == 0
                or global_step == MAX_GLOBAL_STEPS
            )
            if should_log:
                progress = torch.tensor(
                    [progress_loss_sum, progress_recon_sum, float(progress_updates)],
                    dtype=torch.float64,
                    device=device,
                )
                if world_size > 1:
                    dist.all_reduce(progress, op=dist.ReduceOp.SUM)
                denominator = max(float(progress[2].item()), 1.0)
                avg_loss = float(progress[0].item()) / denominator
                avg_recon = float(progress[1].item()) / denominator
                if rank == 0:
                    print(
                        f"[Iter65 TIGER RQ-VAE] global_step={global_step}/{MAX_GLOBAL_STEPS} "
                        f"local_updates={local_optimizer_steps} loss={avg_loss:.8f} "
                        f"recon={avg_recon:.8f} lr={optimizer.param_groups[0]['lr']:.3e}",
                        flush=True,
                    )
                    _record(
                        rank, "train", global_step=global_step,
                        local_optimizer_steps=local_optimizer_steps,
                        epoch=epoch, loss=avg_loss, recon=avg_recon,
                        lr=optimizer.param_groups[0]["lr"],
                    )
                progress_loss_sum = 0.0
                progress_recon_sum = 0.0
                progress_updates = 0

            if global_step % EVAL_INTERVAL_STEPS == 0 or global_step == MAX_GLOBAL_STEPS:
                if world_size > 1:
                    dist.barrier()
                raw_module.eval()
                for layer in raw_module.rq.vq_layers:
                    layer._skip_ddp_reduce = True
                try:
                    if rank == 0:
                        with torch.no_grad():
                            raw_tokens = raw_module.get_indices(
                                all_embeddings.to(device), infer_use_sk=False
                            ).cpu().numpy().astype(np.int64)
                        raw_unique = int(len(np.unique(raw_tokens, axis=0)))
                        final_collision = 1.0 - raw_unique / len(raw_tokens)
                        CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(
                            {
                                "state_dict": raw_module.state_dict(),
                                "global_step": global_step,
                                "collision": final_collision,
                            },
                            CHECKPOINT,
                        )
                        print(
                            f"[Iter65 TIGER RQ-VAE] global_step={global_step} "
                            f"raw_unique={raw_unique}/{len(raw_tokens)} "
                            f"collision={final_collision:.6f}",
                            flush=True,
                        )
                        _record(
                            rank, "sid_metrics", global_step=global_step,
                            raw_unique=raw_unique, raw_total=len(raw_tokens),
                            collision=final_collision, checkpoint=str(CHECKPOINT),
                            checkpoint_reason="fixed_step_budget",
                        )
                finally:
                    for layer in raw_module.rq.vq_layers:
                        layer._skip_ddp_reduce = False
                    raw_module.train()
                if world_size > 1:
                    dist.barrier()
            if global_step == MAX_GLOBAL_STEPS:
                break

    if global_step != MAX_GLOBAL_STEPS:
        raise RuntimeError(
            f"Training ended at global_step={global_step}, expected {MAX_GLOBAL_STEPS}"
        )

    if rank == 0:
        raw_module.eval()
        for layer in raw_module.rq.vq_layers:
            layer._skip_ddp_reduce = True
        try:
            with torch.no_grad():
                raw_tokens = raw_module.get_indices(
                    all_embeddings.to(device), infer_use_sk=False
                ).cpu().numpy().astype(np.int64)
        finally:
            for layer in raw_module.rq.vq_layers:
                layer._skip_ddp_reduce = False
        RAW_SIDS_NPY.parent.mkdir(parents=True, exist_ok=True)
        np.save(RAW_SIDS_NPY, raw_tokens)
        sid = _extend_collisions(raw_tokens, codebook_sizes)
        if len(np.unique(sid, axis=0)) != len(sid):
            raise RuntimeError("RecBole SID extension failed to make item codes unique")
        OUTPUT_SID.parent.mkdir(parents=True, exist_ok=True)
        np.save(OUTPUT_SID, sid)
        payload = {str(item_id): [int(value) for value in row] for item_id, row in enumerate(sid)}
        OUTPUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(
            f"[Iter65 TIGER RQ-VAE] exported raw_shape={raw_tokens.shape} "
            f"sid_shape={sid.shape} unique={len(np.unique(sid, axis=0))} -> {OUTPUT_SID}",
            flush=True,
        )
        _record(
            rank, "train_end", global_step=global_step,
            local_optimizer_steps=local_optimizer_steps,
            total_wall_time_s=round(time.time() - _T0, 3),
            final_collision=final_collision, sid_shape=list(sid.shape),
            raw_sid_path=str(RAW_SIDS_NPY), output_sid=str(OUTPUT_SID),
            output_json=str(OUTPUT_JSON),
            unique_after_extend=len(np.unique(sid, axis=0)),
        )

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()



def _launch_via_torchrun() -> None:
    """2026-09-22: 与 stage3 同款, 但不同端口 (50202). 顶层调用时自动 fork 4 卡 DDP."""
    if int(os.environ.get("RANK", -1)) >= 0:
        return  # 已经在 torchrun 启动的子进程里
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = _LAUNCHER["visible_dev"]
    env["NCCL_IB_DISABLE"]     = "1"
    env["NCCL_P2P_DISABLE"]    = "1"
    env["NCCL_SHM_DISABLE"]    = "1"
    env["NCCL_TIMEOUT"]        = "3600"
    env["TORCH_NCCL_BLOCKING_WAIT"] = "1"
    cmd = [
        _LAUNCHER["torchrun"],
        "--standalone",
        f"--nproc_per_node={_LAUNCHER['nproc']}",
        f"--master_port={_LAUNCHER['master_port']}",
        _LAUNCHER["script"],
    ]
    with open(_LAUNCHER["log"], "w", encoding="utf-8") as fout:
        rc = subprocess.call(cmd, stdout=fout, stderr=subprocess.STDOUT, env=env)
    sys.exit(rc)


if __name__ == "__main__":
    _launch_via_torchrun()
    main()
