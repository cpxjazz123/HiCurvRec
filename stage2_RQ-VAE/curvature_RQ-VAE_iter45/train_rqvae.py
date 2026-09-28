"""Train TIGER RQ-VAE with fixed-c=1 Poincare quantization (4-card DDP).

Keep TIGER's architecture, AdamW hyperparameters, initialization, Sinkhorn,
and SID export. Train for the same fixed 100k distributed-step budget as
other iterations; descriptive SID metrics never stop training or select weights.
"""

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


from model import RQVAE
import curvature_config as experiment


EMBEDDING_FILE = Path(experiment.EMBEDDING_FILE)
TRAIN_FILE = Path(experiment.TRAIN_FILE)
OUTPUT_SID = Path(experiment.SIDS_NPY)
OUTPUT_JSON = Path(experiment.ITEM_SIDS_JSON)
CHECKPOINT = Path(experiment.RQVAE_CKPT_PATH)
RAW_SIDS_NPY = Path(experiment.RAW_SIDS_NPY)
METRICS_PATH = Path(experiment.STAGE2_LOG_DIR / "training_metrics.jsonl")
LOG_DIR = Path(experiment.STAGE2_LOG_DIR)


# === TIGER hyperparameters with the shared fixed-step iteration budget ===
MAX_GLOBAL_STEPS = int(experiment.MAX_GLOBAL_STEPS)
EVAL_INTERVAL_STEPS = int(experiment.EVAL_INTERVAL_STEPS)
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

    if rank == 0:
        print(f"[RecBole RQ-VAE] device={device} all_items={len(all_embeddings)} train_items={len(train_embeddings)} world_size={world_size}", flush=True)
        print(
            f"[Strict GFC] max_global_steps={MAX_GLOBAL_STEPS} "
            f"eval_interval_steps={EVAL_INTERVAL_STEPS} "
            f"batch_size_per_rank={BATCH_SIZE_PER_RANK} "
            f"total_batch={BATCH_SIZE_PER_RANK*world_size} lr={LR} c=1.0",
            flush=True,
        )
        _record(
            rank, "train_start", max_global_steps=MAX_GLOBAL_STEPS,
            eval_interval_steps=EVAL_INTERVAL_STEPS,
            batch_size_per_rank=BATCH_SIZE_PER_RANK,
            total_batch_size=BATCH_SIZE_PER_RANK*world_size,
            lr=LR, weight_decay=WEIGHT_DECAY, hidden_sizes=list(HIDDEN_SIZES),
            codebook_num=CODEBOOK_NUM, codebook_size=list(CODEBOOK_SIZE),
            codebook_dim=CODEBOOK_DIM, beta=BETA, vq_type=VQ_TYPE,
            sk_epsilon=SK_EPSILON, sk_iters=SK_ITERS, pca_dim=PCA_DIM,
            xavier_init=XAVIER_INIT,
            gradient_clip_norm=GRADIENT_CLIP_NORM, optimizer=OPTIMIZER,
            seed=SEED, num_workers=NUM_WORKERS, all_items=len(all_embeddings),
            train_items=len(train_embeddings), embedding_shape=list(embeddings.shape),
            world_size=world_size, curvature=1.0,
        )

    if world_size > 1 and MAX_GLOBAL_STEPS % world_size != 0:
        raise ValueError("MAX_GLOBAL_STEPS must be divisible by the DDP world size")

    # === TIGER codebook initialization; unchanged for every rank ===
    if rank == 0:
        print("[RecBole RQ-VAE] initializing codebooks with KMeans", flush=True)
    raw_module = model.module if isinstance(model, DDP) else model
    with torch.no_grad():
        raw_module.init_codebook(train_embeddings.to(device))
    if world_size > 1:
        dist.barrier()

    codebook_sizes = list(config.codebook_size)
    final_raw_tokens = None
    local_optimizer_steps = 0
    global_step_sync = 0
    epoch = 0
    interval_loss_sum = 0.0
    interval_recon_sum = 0.0
    interval_updates = 0
    last_progress_time = time.time()

    while global_step_sync < MAX_GLOBAL_STEPS:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            reconstructed, quant_loss, _, _ = model(batch)
            loss, recon_loss = raw_module.compute_loss(batch, reconstructed, quant_loss)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"RQ-VAE loss became non-finite at distributed step {global_step_sync}"
                )
            loss.backward()
            if GRADIENT_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()

            local_optimizer_steps += 1
            step_tensor = torch.tensor(local_optimizer_steps, dtype=torch.int64, device=device)
            if world_size > 1:
                dist.all_reduce(step_tensor, op=dist.ReduceOp.SUM)
            global_step_sync = int(step_tensor.item())
            interval_loss_sum += float(loss.detach())
            interval_recon_sum += float(recon_loss.detach())
            interval_updates += 1

            now = time.time()
            if rank == 0 and now - last_progress_time >= 5:
                print(
                    f"[Strict GFC] global_step={global_step_sync}/{MAX_GLOBAL_STEPS} "
                    f"local_updates={local_optimizer_steps} "
                    f"loss={float(loss.detach()):.8f} "
                    f"recon={float(recon_loss.detach()):.8f} lr={LR:.3e}",
                    flush=True,
                )
                last_progress_time = now

            should_evaluate = (
                global_step_sync % EVAL_INTERVAL_STEPS == 0
                or global_step_sync >= MAX_GLOBAL_STEPS
            )
            if not should_evaluate:
                if global_step_sync >= MAX_GLOBAL_STEPS:
                    break
                continue

            loss_sum_tensor = torch.tensor(interval_loss_sum, dtype=torch.float64, device=device)
            recon_sum_tensor = torch.tensor(interval_recon_sum, dtype=torch.float64, device=device)
            updates_tensor = torch.tensor(interval_updates, dtype=torch.int64, device=device)
            if world_size > 1:
                dist.all_reduce(loss_sum_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(recon_sum_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(updates_tensor, op=dist.ReduceOp.SUM)
            avg_loss = float(loss_sum_tensor.item()) / max(int(updates_tensor.item()), 1)
            avg_recon = float(recon_sum_tensor.item()) / max(int(updates_tensor.item()), 1)
            interval_loss_sum = 0.0
            interval_recon_sum = 0.0
            interval_updates = 0

            model.eval()
            for layer in raw_module.rq.vq_layers:
                layer._skip_ddp_reduce = True
            raw_tokens = None
            try:
                if rank == 0:
                    with torch.no_grad():
                        raw_tokens = raw_module.get_indices(
                            all_embeddings.to(device), infer_use_sk=False
                        ).cpu().numpy().astype(np.int64)
                    raw_unique = int(len(np.unique(raw_tokens, axis=0)))
                    collision_v = 1.0 - raw_unique / len(raw_tokens)
                    print(
                        f"[Strict GFC] global_step={global_step_sync}/{MAX_GLOBAL_STEPS} "
                        f"loss={avg_loss:.8f} recon={avg_recon:.8f} lr={LR:.3e} "
                        f"raw_unique={raw_unique}/{len(raw_tokens)} "
                        f"collision={collision_v:.6f}",
                        flush=True,
                    )
                    _record(
                        rank, "train", global_step=global_step_sync,
                        local_optimizer_steps=local_optimizer_steps,
                        loss=avg_loss, recon=avg_recon, lr=LR,
                        raw_unique=raw_unique, raw_total=len(raw_tokens),
                        collision=collision_v,
                    )
                    if global_step_sync >= MAX_GLOBAL_STEPS:
                        final_raw_tokens = raw_tokens
            finally:
                for layer in raw_module.rq.vq_layers:
                    layer._skip_ddp_reduce = False
            if world_size > 1:
                dist.barrier()
            model.train()
            last_progress_time = time.time()
            if global_step_sync >= MAX_GLOBAL_STEPS:
                break
        epoch += 1

    if rank == 0:
        if final_raw_tokens is None:
            raise RuntimeError("The fixed Stage2 step budget did not produce final SIDs")
        state = {
            name: value.detach().cpu()
            for name, value in raw_module.state_dict().items()
        }
        CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": state,
                "global_step": global_step_sync,
                "local_optimizer_steps": local_optimizer_steps,
                "curvature": 1.0,
            },
            CHECKPOINT,
        )
        _record(
            rank, "ckpt_saved", global_step=global_step_sync,
            path=str(CHECKPOINT), reason="fixed_stage2_step_budget",
        )
        RAW_SIDS_NPY.parent.mkdir(parents=True, exist_ok=True)
        np.save(RAW_SIDS_NPY, final_raw_tokens)
        sid = _extend_collisions(final_raw_tokens, codebook_sizes)
        if len(np.unique(sid, axis=0)) != len(sid):
            raise RuntimeError("RecBole SID extension failed to make item codes unique")
        OUTPUT_SID.parent.mkdir(parents=True, exist_ok=True)
        np.save(OUTPUT_SID, sid)
        payload = {
            str(item_id): [int(value) for value in row]
            for item_id, row in enumerate(sid)
        }
        OUTPUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(
            f"[Strict GFC] fixed_c=1 global_steps={global_step_sync} "
            f"local_optimizer_steps={local_optimizer_steps} "
            f"raw_shape={final_raw_tokens.shape} sid_shape={sid.shape} "
            f"unique={len(np.unique(sid, axis=0))} -> {OUTPUT_SID}",
            flush=True,
        )
        _record(
            rank, "train_end", total_wall_time_s=round(time.time() - _T0, 3),
            global_step=global_step_sync,
            local_optimizer_steps=local_optimizer_steps,
            sid_shape=list(sid.shape), checkpoint=str(CHECKPOINT),
            output_sid=str(OUTPUT_SID), output_json=str(OUTPUT_JSON),
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
