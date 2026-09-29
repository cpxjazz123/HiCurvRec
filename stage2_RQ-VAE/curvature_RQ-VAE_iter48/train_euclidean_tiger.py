"""Matched 40k-step Euclidean TIGER side arms for Iter48."""
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from tiger_euclidean_model import TIGERBehaviorRQVAE, TIGERRQVAE

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
SOURCE_DIR = REPO_ROOT / "stage2_RQ-VAE/curvature_RQ-VAE_iter48"
EMBEDDING_FILE = REPO_ROOT / "stage1_GeneEmbedding/output/sentence_t5.npy"
TRAIN_FILE = REPO_ROOT / "results/stage0_build_parquet/train.parquet"
RESULTS_ROOT = REPO_ROOT / "results/stage2_RQ-VAE/curvature_RQ-VAE_iter48"
SHARED_INITIALIZATION = (
    RESULTS_ROOT / "versions/TIGERBaseline/out/rqvae/instruments/rqvae_init.pth"
)
LOG_DIR = SOURCE_DIR / "logs"
VERSIONS = {"BASELINE": "TIGERBaseline", "BEHAVIOR": "TIGERBehavior"}
METRICS_PATH = LOG_DIR / "training_metrics_TIGERBaseline.jsonl"
RUN_MODE = "BASELINE"

MAX_GLOBAL_STEPS = 40_000
EVAL_INTERVAL_STEPS = 10_000
BATCH_SIZE_PER_RANK = 1024
PAIR_BATCH_SIZE_PER_RANK = BATCH_SIZE_PER_RANK // 2
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
XAVIER_INIT = True
GRADIENT_CLIP_NORM = 1.0
ADAMW_BETAS = (0.9, 0.999)
ADAMW_EPS = 1e-8
SEED = 42
NUM_WORKERS = 0
BEHAVIOR_WEIGHT_MAX = 0.20
BEHAVIOR_TEMPERATURE = 0.07

_LAUNCHER = {
    "torchrun": "/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/torchrun",
    "script": "",
    "nproc": 4,
    "master_port": 50202,
    "log": "",
    "visible_dev": "0,1,2,3",
}

_metrics_lock = threading.Lock()
_T0 = time.time()
_metrics_truncated = False


def configure_arm(arm: str, launcher_script: str) -> None:
    global RUN_MODE, METRICS_PATH
    if arm not in VERSIONS:
        raise ValueError(f"Unsupported Euclidean TIGER arm: {arm}")
    RUN_MODE = arm
    version = VERSIONS[arm]
    METRICS_PATH = LOG_DIR / f"training_metrics_{version}.jsonl"
    _LAUNCHER["script"] = os.path.abspath(launcher_script)
    _LAUNCHER["log"] = str(LOG_DIR / f"train_{version.lower()}_migrated.log")


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
    embeddings = np.asarray(np.load(path), dtype=np.float32)
    if embeddings.ndim != 2 or not np.isfinite(embeddings).all():
        raise ValueError(f"Invalid embedding matrix at {path}: {embeddings.shape}")
    return embeddings


class TransitionDataset(Dataset):
    """The train-history-to-target pairs used by the matched Iter48 arms."""

    def __init__(self, frame: pd.DataFrame):
        pairs = []
        for history, target in frame[["history", "target"]].itertuples(
            index=False, name=None
        ):
            if history is None or not isinstance(history, (list, tuple, np.ndarray)):
                continue
            if len(history) == 0:
                continue
            pairs.append((int(history[-1]), int(target)))
        self.pairs = torch.tensor(pairs, dtype=torch.long)
        if self.pairs.ndim != 2 or self.pairs.shape[1] != 2 or len(self.pairs) == 0:
            raise ValueError("Training parquet contains no valid history-target pairs")

    def __len__(self) -> int:
        return int(self.pairs.shape[0])

    def __getitem__(self, index: int):
        return self.pairs[index]


def _model_config():
    return SimpleNamespace(
        hidden_sizes=HIDDEN_SIZES,
        codebook_num=CODEBOOK_NUM,
        codebook_size=CODEBOOK_SIZE,
        codebook_dim=CODEBOOK_DIM,
        dropout=0.0,
        beta=BETA,
        vq_type=VQ_TYPE,
        ema_decay=EMA_DECAY,
        fix_code_embs=False,
        sk_epsilon=SK_EPSILON,
        sk_iters=SK_ITERS,
        behavior_temperature=BEHAVIOR_TEMPERATURE,
        behavior_weight_max=BEHAVIOR_WEIGHT_MAX,
    )


def build_model(arm: str, embedding_dim: int):
    config = _model_config()
    model_type = TIGERBehaviorRQVAE if arm == "BEHAVIOR" else TIGERRQVAE
    return model_type(config, in_dim=embedding_dim)


def initialize_tiger_weights(model: nn.Module) -> None:
    if not XAVIER_INIT:
        return
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def _record(rank: int, event: str, **fields) -> None:
    if rank != 0:
        return
    global _metrics_truncated
    rec = {
        "event": event,
        "timestamp": time.time(),
        "wall_time_s": round(time.time() - _T0, 3),
        **fields,
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _metrics_lock:
        if not _metrics_truncated:
            METRICS_PATH.write_text("", encoding="utf-8")
            _metrics_truncated = True
        with METRICS_PATH.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stream.flush()
            try:
                os.fsync(stream.fileno())
            except OSError:
                pass


def _extend_collisions(tokens: np.ndarray) -> np.ndarray:
    groups: dict[tuple[int, ...], list[int]] = {}
    for item_id, row in enumerate(tokens.tolist()):
        groups.setdefault(tuple(int(value) for value in row), []).append(item_id)
    offsets = np.cumsum([0, *CODEBOOK_SIZE], dtype=np.int64)
    extended = np.empty((len(tokens), CODEBOOK_NUM + 1), dtype=np.int64)
    for row, item_ids in groups.items():
        for occurrence, item_id in enumerate(item_ids):
            extended[item_id, :-1] = np.asarray(row, dtype=np.int64)
            extended[item_id, -1] = int(offsets[-1] + occurrence)
    return extended


def _save_snapshot(raw_module, tokens: np.ndarray, global_step: int, local_steps: int) -> None:
    version = VERSIONS[RUN_MODE]
    root = RESULTS_ROOT / "versions" / version
    checkpoint = root / "out/rqvae/instruments/rqvae_best.pth"
    raw_path = root / "out/rqvae/instruments/sids_raw.npy"
    sid_path = root / "dataset/Instruments/sids_for_hgrec.npy"
    json_path = root / "item_sids.json"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {
                name: value.detach().cpu()
                for name, value in raw_module.state_dict().items()
            },
            "global_step": global_step,
            "local_optimizer_steps": local_steps,
            "run_mode": RUN_MODE,
            "version": version,
            "quantization_geometry": "euclidean",
            "sinkhorn_assignment_level": CODEBOOK_NUM - 1,
        },
        checkpoint,
    )
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(raw_path, tokens)
    sids = _extend_collisions(tokens)
    if len(np.unique(sids, axis=0)) != len(sids):
        raise RuntimeError(f"SID extension failed for {version}")
    sid_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(sid_path, sids)
    json_path.write_text(
        json.dumps(
            {str(item_id): [int(value) for value in row] for item_id, row in enumerate(sids)},
            indent=2,
        ),
        encoding="utf-8",
    )
    _record(
        0,
        "snapshot_saved",
        version=version,
        global_step=global_step,
        checkpoint=str(checkpoint),
        raw_sids=str(raw_path),
        sids_for_hgrec=str(sid_path),
        item_sids=str(json_path),
        raw_unique=int(len(np.unique(tokens, axis=0))),
        unique_after_extend=int(len(np.unique(sids, axis=0))),
    )

def _initialization_metadata(embeddings: np.ndarray, target_ids: np.ndarray) -> dict:
    return {
        "source_arm": "TIGERBaseline",
        "seed": SEED,
        "embedding_shape": list(embeddings.shape),
        "train_target_ids_sha256": hashlib.sha256(
            np.asarray(target_ids, dtype=np.int64).tobytes()
        ).hexdigest(),
        "hidden_sizes": list(HIDDEN_SIZES),
        "codebook_size": list(CODEBOOK_SIZE),
        "codebook_dim": CODEBOOK_DIM,
        "beta": BETA,
        "sk_epsilon": SK_EPSILON,
        "sk_iters": SK_ITERS,
    }


def _initialize_or_load_shared(
    raw_module,
    train_embeddings: torch.Tensor,
    embeddings: np.ndarray,
    target_ids: np.ndarray,
    rank: int,
    world_size: int,
) -> None:
    metadata = _initialization_metadata(embeddings, target_ids)
    if RUN_MODE == "BASELINE":
        raw_module.init_codebook(train_embeddings.to(next(raw_module.parameters()).device))
        if rank == 0:
            SHARED_INITIALIZATION.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": {
                        name: value.detach().cpu()
                        for name, value in raw_module.state_dict().items()
                    },
                    "metadata": metadata,
                },
                SHARED_INITIALIZATION,
            )
    else:
        if not SHARED_INITIALIZATION.is_file():
            raise FileNotFoundError(
                f"Run TIGERBaseline first to create {SHARED_INITIALIZATION}"
            )
        payload = torch.load(SHARED_INITIALIZATION, map_location="cpu")
        if payload.get("metadata") != metadata:
            raise RuntimeError("Shared TIGER initialization metadata does not match this run")
        raw_module.load_state_dict(payload["state_dict"], strict=True)
    if world_size > 1:
        dist.barrier()


def _run_torchrun_if_needed() -> None:
    if int(os.environ.get("RANK", -1)) >= 0:
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": _LAUNCHER["visible_dev"],
            "NCCL_IB_DISABLE": "1",
            "NCCL_P2P_DISABLE": "1",
            "NCCL_SHM_DISABLE": "1",
            "NCCL_TIMEOUT": "3600",
            "TORCH_NCCL_BLOCKING_WAIT": "1",
        }
    )
    command = [
        _LAUNCHER["torchrun"],
        "--standalone",
        f"--nproc_per_node={_LAUNCHER['nproc']}",
        f"--master_port={_LAUNCHER['master_port']}",
        _LAUNCHER["script"],
    ]
    print(f"[Euclidean TIGER {RUN_MODE}] launching four-rank DDP", flush=True)
    with open(_LAUNCHER["log"], "w", encoding="utf-8") as stream:
        sys.exit(subprocess.call(command, stdout=stream, stderr=subprocess.STDOUT, env=env))


def main() -> None:
    if int(os.environ.get("RANK", -1)) < 0:
        raise RuntimeError("Stage2 main must run under torchrun")
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    set_seed(SEED + rank)

    embeddings = load_embeddings(EMBEDDING_FILE)
    train_frame = pd.read_parquet(TRAIN_FILE)
    pair_dataset = TransitionDataset(train_frame)
    target_ids = np.unique(train_frame["target"].to_numpy(dtype=np.int64))
    if target_ids.size == 0 or target_ids.min() < 0 or target_ids.max() >= len(embeddings):
        raise ValueError("Training target item ids do not match embedding rows")
    all_embeddings = torch.from_numpy(embeddings)
    train_embeddings = all_embeddings[torch.from_numpy(target_ids)]

    model = build_model(RUN_MODE, embeddings.shape[1]).to(device)
    initialize_tiger_weights(model)
    if RUN_MODE == "BEHAVIOR":
        model.set_global_step(0)
    wrapped = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    raw_module = wrapped.module
    sampler = DistributedSampler(
        pair_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=SEED, drop_last=False
    )
    loader = DataLoader(
        pair_dataset,
        batch_size=PAIR_BATCH_SIZE_PER_RANK,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        persistent_workers=False,
    )
    if len(loader) == 0:
        raise RuntimeError("Distributed transition loader has no complete batch")
    optimizer = torch.optim.AdamW(
        wrapped.parameters(),
        lr=LR,
        betas=ADAMW_BETAS,
        eps=ADAMW_EPS,
        weight_decay=WEIGHT_DECAY,
    )

    if rank == 0:
        _record(
            rank,
            "train_start",
            run_mode=RUN_MODE,
            version=VERSIONS[RUN_MODE],
            max_global_steps=MAX_GLOBAL_STEPS,
            eval_interval_steps=EVAL_INTERVAL_STEPS,
            seed=SEED,
            cold_start=True,
            resume_checkpoint_loaded=False,
            shared_initialization_loaded=RUN_MODE == "BEHAVIOR",
            shared_initialization_path=str(SHARED_INITIALIZATION),
            train_target_ids_sha256=_initialization_metadata(
                embeddings, target_ids
            )["train_target_ids_sha256"],
            embedding_shape=list(embeddings.shape),
            train_target_items=len(train_embeddings),
            transition_pairs=len(pair_dataset),
            pair_batch_size_per_rank=PAIR_BATCH_SIZE_PER_RANK,
            embedding_batch_size_per_rank=BATCH_SIZE_PER_RANK,
            effective_embedding_batch_size=BATCH_SIZE_PER_RANK * world_size,
            world_size=world_size,
            learning_rate=LR,
            weight_decay=WEIGHT_DECAY,
            optimizer="AdamW",
            adamw_betas=list(ADAMW_BETAS),
            adamw_eps=ADAMW_EPS,
            hidden_sizes=list(HIDDEN_SIZES),
            codebook_num=CODEBOOK_NUM,
            codebook_size=list(CODEBOOK_SIZE),
            codebook_dim=CODEBOOK_DIM,
            beta=BETA,
            sk_epsilon=SK_EPSILON,
            sk_iters=SK_ITERS,
            sinkhorn_levels=[CODEBOOK_NUM - 1],
            behavior_enabled=RUN_MODE == "BEHAVIOR",
            behavior_weight_max=BEHAVIOR_WEIGHT_MAX if RUN_MODE == "BEHAVIOR" else 0.0,
            behavior_ramp_steps=[20_000, 40_000] if RUN_MODE == "BEHAVIOR" else None,
            behavior_temperature=BEHAVIOR_TEMPERATURE if RUN_MODE == "BEHAVIOR" else None,
            training_geometry="euclidean",
            dataset_kind="train_history_target_pairs",
        )

    _initialize_or_load_shared(
        raw_module, train_embeddings, embeddings, target_ids, rank, world_size
    )

    local_steps = 0
    global_step = 0
    losses, recons, behavior_losses = [], [], []
    start_time = time.time()
    wrapped.train()
    for epoch in range(1_000_000):
        sampler.set_epoch(epoch)
        for pairs in loader:
            pairs = pairs.to(device, non_blocking=True)
            source_ids, target_ids = pairs[:, 0], pairs[:, 1]
            batch_ids = torch.cat((source_ids, target_ids), dim=0)
            batch = all_embeddings[batch_ids.cpu()].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if RUN_MODE == "BEHAVIOR":
                reconstructed, quant_loss, _, _, behavior_loss = wrapped(
                    batch, behavior_ids=(source_ids, target_ids)
                )
                behavior_weight = raw_module.get_behavior_weight()
            else:
                reconstructed, quant_loss, _, _ = wrapped(batch)
                behavior_loss = quant_loss.new_zeros(())
                behavior_weight = 0.0
            recon_loss = torch.nn.functional.mse_loss(reconstructed, batch)
            loss = recon_loss + quant_loss + behavior_weight * behavior_loss
            if not loss.requires_grad or loss.grad_fn is None or not torch.isfinite(loss):
                raise RuntimeError(f"Invalid total loss at global step {global_step}")
            loss.backward()
            if GRADIENT_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(wrapped.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()
            local_steps += 1
            step_count = torch.tensor(local_steps, device=device, dtype=torch.long)
            dist.all_reduce(step_count, op=dist.ReduceOp.SUM)
            global_step = int(step_count.item())
            if RUN_MODE == "BEHAVIOR":
                raw_module.set_global_step(global_step)
            losses.append(float(loss.detach()))
            recons.append(float(recon_loss.detach()))
            behavior_losses.append(float(behavior_loss.detach()))

            if global_step % EVAL_INTERVAL_STEPS != 0 and global_step != MAX_GLOBAL_STEPS:
                continue
            if global_step > MAX_GLOBAL_STEPS:
                raise RuntimeError(f"Exceeded requested 40k steps: {global_step}")

            train_loss = torch.tensor(float(np.mean(losses)), device=device)
            train_recon = torch.tensor(float(np.mean(recons)), device=device)
            train_behavior = torch.tensor(float(np.mean(behavior_losses)), device=device)
            dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_recon, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_behavior, op=dist.ReduceOp.SUM)
            train_loss /= world_size
            train_recon /= world_size
            train_behavior /= world_size
            losses.clear()
            recons.clear()
            behavior_losses.clear()

            wrapped.eval()
            for layer in raw_module.rq.vq_layers:
                layer._skip_ddp_reduce = True
            dist.barrier()
            if rank == 0:
                with torch.no_grad():
                    tokens = raw_module.get_indices(all_embeddings.to(device)).cpu().numpy()
                raw_unique = int(len(np.unique(tokens, axis=0)))
                raw_total = int(len(tokens))
                usage_counts = [
                    np.bincount(tokens[:, level], minlength=CODEBOOK_SIZE[level]).astype(int).tolist()
                    for level in range(CODEBOOK_NUM)
                ]
                used = [sum(count > 0 for count in counts) for counts in usage_counts]
                behavior_weight = (
                    raw_module.get_behavior_weight() if RUN_MODE == "BEHAVIOR" else 0.0
                )
                _record(
                    rank,
                    "train",
                    run_mode=RUN_MODE,
                    global_step=global_step,
                    local_optimizer_steps=local_steps,
                    loss=float(train_loss.item()),
                    recon=float(train_recon.item()),
                    behavior_loss=float(train_behavior.item()),
                    behavior_weight=behavior_weight,
                    raw_unique=raw_unique,
                    raw_total=raw_total,
                    collision=1.0 - raw_unique / raw_total,
                    codebook_usage_counts=usage_counts,
                    codebook_used=used,
                    elapsed_s=round(time.time() - start_time, 3),
                )
                if global_step == MAX_GLOBAL_STEPS:
                    _save_snapshot(raw_module, tokens, global_step, local_steps)
            dist.barrier()
            for layer in raw_module.rq.vq_layers:
                layer._skip_ddp_reduce = False
            wrapped.train()
            if global_step == MAX_GLOBAL_STEPS:
                break
        if global_step == MAX_GLOBAL_STEPS:
            break

    if global_step != MAX_GLOBAL_STEPS:
        raise RuntimeError(f"Training ended at {global_step}, expected {MAX_GLOBAL_STEPS}")
    if rank == 0:
        _record(
            rank,
            "train_end",
            run_mode=RUN_MODE,
            global_step=global_step,
            local_optimizer_steps=local_steps,
            duration_s=round(time.time() - start_time, 3),
        )
    dist.destroy_process_group()


def launch_and_run() -> None:
    _run_torchrun_if_needed()
    main()
