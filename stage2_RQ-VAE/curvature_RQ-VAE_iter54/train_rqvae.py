"""Matched 40k-step TIGER behavior training with train-graph Ricci curvature."""
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
from scipy.stats import spearmanr
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

import curvature_config as experiment
from tiger_learnable_curvature_model import TIGERLearnableCurvatureRQVAE

REPO_ROOT = experiment.REPO_ROOT
SOURCE_DIR = REPO_ROOT / "stage2_RQ-VAE/curvature_RQ-VAE_iter54"
EMBEDDING_FILE = REPO_ROOT / "stage1_GeneEmbedding/output/sentence_t5.npy"
TRAIN_FILE = REPO_ROOT / "results/stage0_build_parquet/train.parquet"
RESULTS_ROOT = experiment.STAGE2_RESULT_DIR
SHARED_INITIALIZATION = (
    REPO_ROOT
    / "results/stage2_RQ-VAE/curvature_RQ-VAE_iter48/versions/TIGERBaseline/out/rqvae/instruments/rqvae_init.pth"
)
LOG_DIR = experiment.STAGE2_LOG_DIR
METRICS_PATH = LOG_DIR / "training_metrics.jsonl"
RQVAE_OUT_DIR = Path(experiment.RQVAE_OUT_DIR)
ITEM_CURVATURES_PATH = experiment.ITEM_CURVATURES_PATH
ITEM_SIGNALS_PATH = experiment.ITEM_SIGNALS_PATH
ITEM_CONTROLLER_PATH = experiment.ITEM_CONTROLLER_PATH

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
    "script": str(SOURCE_DIR / "curvature_RQ-VAE.py"),
    "nproc": 4,
    "master_port": 50254,
    "log": str(LOG_DIR / "train_migrated.log"),
    "visible_dev": "0,1,2,3",
}
_metrics_lock = threading.Lock()
_T0 = time.time()
_metrics_truncated = False


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
    """Last-history-item to target pairs from the training parquet."""

    def __init__(self, frame: pd.DataFrame):
        pairs = []
        for history, target in frame[["history", "target"]].itertuples(
            index=False, name=None
        ):
            if history is None or not isinstance(history, (list, tuple, np.ndarray)):
                continue
            if len(history):
                pairs.append((int(history[-1]), int(target)))
        self.pairs = torch.tensor(pairs, dtype=torch.long)
        if self.pairs.ndim != 2 or self.pairs.shape[1] != 2 or not len(self.pairs):
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
        controller_hidden=experiment.CONTROLLER_HIDDEN,
        controller_weight_init=experiment.CONTROLLER_WEIGHT_INIT,
        controller_diversity=experiment.CONTROLLER_DIVERSITY,
        controller_bias_init=experiment.CONTROLLER_BIAS_INIT,
        curvature_min=experiment.CURVATURE_MIN,
        curvature_max=experiment.CURVATURE_MAX,
        reference_curvature_mean=experiment.REFERENCE_CURVATURE_MEAN,
        curvature_mean_reg=experiment.CURVATURE_MEAN_REG,
        reference_curvature_std=experiment.REFERENCE_CURVATURE_STD,
        curvature_std_reg=experiment.CURVATURE_STD_REG,
    )


def build_curvature_features(item_signals: np.ndarray) -> np.ndarray:
    """Z-score the three graph signals and append their pairwise interactions."""
    signals = np.asarray(item_signals, dtype=np.float64)
    scale = signals.std(axis=0, dtype=np.float64)
    scale[scale < 1e-8] = 1.0
    z = (signals - signals.mean(axis=0, dtype=np.float64)) / scale
    primary = z.astype(np.float32)
    r, g, h = primary[:, 0], primary[:, 1], primary[:, 2]
    return np.column_stack([r, g, h, r * g, r * h, g * h]).astype(np.float32)


def build_model(embedding_dim: int, features: np.ndarray) -> TIGERLearnableCurvatureRQVAE:
    return TIGERLearnableCurvatureRQVAE(
        _model_config(), in_dim=embedding_dim, curvature_features=torch.from_numpy(features)
    )


def initialize_tiger_weights(model: nn.Module) -> None:
    if not XAVIER_INIT:
        return
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def _record(event: str, **fields) -> None:
    if dist.get_rank() != 0:
        return
    global _metrics_truncated
    rec = {
        "event": event,
        "timestamp": time.time(),
        "wall_time_s": round(time.time() - _T0, 3),
        **fields,
    }
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with _metrics_lock:
        if not _metrics_truncated:
            METRICS_PATH.write_text("", encoding="utf-8")
            _metrics_truncated = True
        with METRICS_PATH.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _curvature_interpretability(
    learned_curvature: np.ndarray, item_signals: np.ndarray
) -> tuple[dict[str, float], list[dict[str, float]]]:
    names = ("negative_orc_need", "two_hop_expansion", "transition_entropy")
    correlations = {
        name: float(spearmanr(learned_curvature, item_signals[:, index]).statistic)
        for index, name in enumerate(names)
    }
    order = np.lexsort((np.arange(len(learned_curvature)), learned_curvature))
    quartiles = []
    for quartile, item_ids in enumerate(np.array_split(order, 4), start=1):
        quartile_curvature = learned_curvature[item_ids]
        quartile_signals = item_signals[item_ids]
        quartiles.append(
            {
                "quartile": quartile,
                "n_items": int(len(item_ids)),
                "mean_curvature": float(quartile_curvature.mean()),
                "mean_negative_orc_need": float(quartile_signals[:, 0].mean()),
                "mean_two_hop_expansion": float(quartile_signals[:, 1].mean()),
                "mean_transition_entropy": float(quartile_signals[:, 2].mean()),
            }
        )
    return correlations, quartiles

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
    checkpoint = RQVAE_OUT_DIR / "rqvae_best.pth"
    raw_path = RQVAE_OUT_DIR / "sids_raw.npy"
    sid_path = RESULTS_ROOT / "dataset/Instruments/sids_for_hgrec.npy"
    json_path = RESULTS_ROOT / "item_sids.json"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {
                name: value.detach().cpu()
                for name, value in raw_module.state_dict().items()
            },
            "global_step": global_step,
            "local_optimizer_steps": local_steps,
            "run_mode": "LEARNABLE_CURVATURE",
            "version": "LearnableCurvature",
            "mechanism_name": experiment.MECHANISM_NAME,
            "quantization_geometry": "euclidean_tiger",
            "behavior_geometry": "poincare_pair_curvature_sqrt",
            "curvature_scope": "behavior_contrastive_loss_only",
            "curvature_source": "learnable_nonlinear_controller",
            "curvature_architecture": raw_module.controller.architecture(),
            "curvature_optimizer_lr": experiment.CONTROLLER_LR,
            "curvature_optimizer_weight_decay": 0.0,
            "curvature_mean_reg": experiment.CURVATURE_MEAN_REG,
            "curvature_std_reg": experiment.CURVATURE_STD_REG,
            "curvature_reference_mean": experiment.REFERENCE_CURVATURE_MEAN,
            "curvature_reference_std": experiment.REFERENCE_CURVATURE_STD,
            "sinkhorn_assignment_level": CODEBOOK_NUM - 1,
        },
        checkpoint,
    )
    with torch.no_grad():
        learned_curvature = raw_module.controller(
            raw_module.curvature_features
        ).detach().cpu().numpy()
    np.save(ITEM_CURVATURES_PATH, learned_curvature.astype(np.float32))
    torch.save(
        {
            "state_dict": {
                name: value.detach().cpu()
                for name, value in raw_module.controller.state_dict().items()
            },
            "architecture": raw_module.controller.architecture(),
        },
        ITEM_CONTROLLER_PATH,
    )
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(raw_path, tokens)
    sids = _extend_collisions(tokens)
    if len(np.unique(sids, axis=0)) != len(sids):
        raise RuntimeError("Collision extension failed to produce unique Stage3 SIDs")
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
        "snapshot_saved",
        version="LearnableCurvature",
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


def load_shared_initialization(
    raw_module, embeddings: np.ndarray, target_ids: np.ndarray
) -> None:
    if not SHARED_INITIALIZATION.is_file():
        raise FileNotFoundError(f"Missing matched TIGER baseline initialization: {SHARED_INITIALIZATION}")
    payload = torch.load(SHARED_INITIALIZATION, map_location="cpu")
    if payload.get("metadata") != _initialization_metadata(embeddings, target_ids):
        raise RuntimeError("Shared TIGER initialization metadata does not match this run")
    result = raw_module.load_state_dict(payload["state_dict"], strict=False)
    controller_prefix = "controller."
    missing = [key for key in result.missing_keys if not key.startswith(controller_prefix)]
    if missing != ["curvature_features"] or result.unexpected_keys:
        raise RuntimeError(
            "Shared TIGER initialization must differ only by the fixed curvature feature "
            f"buffer and the learnable controller; missing={result.missing_keys}, "
            f"unexpected={result.unexpected_keys}"
        )


def _launch_via_torchrun() -> None:
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
    print("[Iter54 learnable curvature controller] launching four-rank DDP", flush=True)
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
    item_signals = np.asarray(np.load(ITEM_SIGNALS_PATH), dtype=np.float32)
    if item_signals.shape != (len(embeddings), 3):
        raise ValueError("Saved train-only graph signals do not match the embedding item rows")
    if not np.isfinite(item_signals).all():
        raise ValueError("Saved graph-derived curvature inputs contain non-finite values")
    curvature_features = build_curvature_features(item_signals)
    all_embeddings = torch.from_numpy(embeddings)

    model = build_model(embeddings.shape[1], curvature_features).to(device)
    initialize_tiger_weights(model)
    load_shared_initialization(model, embeddings, target_ids)
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
    all_parameters = list(wrapped.parameters())
    controller_parameters = list(raw_module.controller.parameters())
    controller_parameter_ids = {id(parameter) for parameter in controller_parameters}
    rqvae_parameters = [
        parameter
        for parameter in all_parameters
        if id(parameter) not in controller_parameter_ids
    ]
    if len(rqvae_parameters) + len(controller_parameters) != len(all_parameters):
        raise RuntimeError("RQ-VAE and controller optimizer parameter groups overlap")
    optimizer = torch.optim.AdamW(
        rqvae_parameters, lr=LR, betas=ADAMW_BETAS, eps=ADAMW_EPS, weight_decay=WEIGHT_DECAY
    )
    controller_optimizer = torch.optim.AdamW(
        controller_parameters,
        lr=experiment.CONTROLLER_LR,
        betas=ADAMW_BETAS,
        eps=ADAMW_EPS,
        weight_decay=0.0,
    )

    _record(
        "train_start",
        run_mode="LEARNABLE_CURVATURE",
        version="LearnableCurvature",
        max_global_steps=MAX_GLOBAL_STEPS,
        eval_interval_steps=EVAL_INTERVAL_STEPS,
        seed=SEED,
        shared_initialization_loaded=True,
        shared_initialization_path=str(SHARED_INITIALIZATION),
        embedding_shape=list(embeddings.shape),
        train_target_items=len(target_ids),
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
        behavior_weight_max=BEHAVIOR_WEIGHT_MAX,
        behavior_ramp_steps=[20_000, 40_000],
        behavior_temperature=BEHAVIOR_TEMPERATURE,
        training_geometry="euclidean_tiger",
        behavior_geometry="poincare_pair_curvature_sqrt",
        curvature_scope="behavior_contrastive_loss_only",
        curvature_range=[experiment.CURVATURE_MIN, experiment.CURVATURE_MAX],
        curvature_architecture=raw_module.controller.architecture(),
        curvature_controller_lr=experiment.CONTROLLER_LR,
        curvature_optimizer_weight_decay=0.0,
        curvature_hidden=experiment.CONTROLLER_HIDDEN,
        curvature_bias_init=experiment.CONTROLLER_BIAS_INIT,
        curvature_mean_reg=experiment.CURVATURE_MEAN_REG,
        curvature_std_reg=experiment.CURVATURE_STD_REG,
        curvature_reference_mean=experiment.REFERENCE_CURVATURE_MEAN,
        curvature_signal_names=["negative_orc_need", "two_hop_expansion", "transition_entropy"],
        curvature_inputs=[str(ITEM_SIGNALS_PATH), str(ITEM_CONTROLLER_PATH)],
        dataset_kind="train_history_target_pairs",
    )

    local_steps = 0
    global_step = 0
    losses, recons, behavior_losses, curvature_losses = [], [], [], []
    start_time = time.time()
    wrapped.train()
    for epoch in range(1_000_000):
        sampler.set_epoch(epoch)
        for pairs in loader:
            pairs = pairs.to(device, non_blocking=True)
            source_ids, target_ids_batch = pairs[:, 0], pairs[:, 1]
            item_ids = torch.cat((source_ids, target_ids_batch), dim=0)
            batch = all_embeddings[item_ids.cpu()].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            controller_optimizer.zero_grad(set_to_none=True)
            (
                reconstructed,
                quant_loss,
                _,
                _,
                behavior_loss,
                batch_curvature,
                curvature_reg,
            ) = wrapped(batch, item_ids=item_ids, behavior_ids=(source_ids, target_ids_batch))
            behavior_weight = raw_module.get_behavior_weight()
            recon_loss = torch.nn.functional.mse_loss(reconstructed, batch)
            # The regularizer is added to the single backwarded loss so the
            # controller always receives a reduction.  It is a function of the
            # fixed feature buffer only, so RQ-VAE gradients are unchanged.
            loss = (
                recon_loss
                + quant_loss
                + behavior_weight * behavior_loss
                + curvature_reg
            )
            if not loss.requires_grad or loss.grad_fn is None or not torch.isfinite(loss):
                raise RuntimeError(f"Invalid total loss at global step {global_step}")
            loss.backward()
            if GRADIENT_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(rqvae_parameters, GRADIENT_CLIP_NORM)
                torch.nn.utils.clip_grad_norm_(
                    raw_module.controller.parameters(), GRADIENT_CLIP_NORM
                )
            optimizer.step()
            controller_optimizer.step()
            curvature_losses.append(float(curvature_reg.detach()))
            local_steps += 1
            step_count = torch.tensor(local_steps, device=device, dtype=torch.long)
            dist.all_reduce(step_count, op=dist.ReduceOp.SUM)
            global_step = int(step_count.item())
            raw_module.set_global_step(global_step)
            losses.append(float(loss.detach()))
            recons.append(float(recon_loss.detach()))
            behavior_losses.append(float(behavior_loss.detach()))

            if global_step % EVAL_INTERVAL_STEPS != 0 and global_step != MAX_GLOBAL_STEPS:
                continue
            if global_step > MAX_GLOBAL_STEPS:
                raise RuntimeError(f"Exceeded requested {MAX_GLOBAL_STEPS} steps: {global_step}")

            train_loss = torch.tensor(float(np.mean(losses)), device=device)
            train_recon = torch.tensor(float(np.mean(recons)), device=device)
            train_behavior = torch.tensor(float(np.mean(behavior_losses)), device=device)
            train_curvature = torch.tensor(float(np.mean(curvature_losses)), device=device)
            dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_recon, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_behavior, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_curvature, op=dist.ReduceOp.SUM)
            train_loss /= world_size
            train_recon /= world_size
            train_behavior /= world_size
            train_curvature /= world_size
            losses.clear()
            recons.clear()
            behavior_losses.clear()
            curvature_losses.clear()

            wrapped.eval()
            for layer in raw_module.rq.vq_layers:
                layer._skip_ddp_reduce = True
            dist.barrier()
            if rank == 0:
                with torch.no_grad():
                    tokens = raw_module.get_indices(all_embeddings.to(device)).cpu().numpy()
                    learned = raw_module.controller(
                        raw_module.curvature_features
                    ).detach().cpu().numpy()
                raw_unique = int(len(np.unique(tokens, axis=0)))
                usage_counts = [
                    np.bincount(tokens[:, level], minlength=CODEBOOK_SIZE[level]).astype(int).tolist()
                    for level in range(CODEBOOK_NUM)
                ]
                _record(
                    "train",
                    run_mode="LEARNABLE_CURVATURE",
                    global_step=global_step,
                    local_optimizer_steps=local_steps,
                    loss=float(train_loss.item()),
                    recon=float(train_recon.item()),
                    behavior_loss=float(train_behavior.item()),
                    behavior_weight=raw_module.get_behavior_weight(),
                    curvature_reg_loss=float(train_curvature.item()),
                    curvature_mean=float(learned.mean()),
                    curvature_std=float(learned.std()),
                    curvature_min=float(learned.min()),
                    curvature_max=float(learned.max()),
                    curvature_at_lower_bound=int((learned <= experiment.CURVATURE_MIN + 1e-3).sum()),
                    curvature_at_upper_bound=int((learned >= experiment.CURVATURE_MAX - 1e-3).sum()),
                    raw_unique=raw_unique,
                    raw_total=len(tokens),
                    collision=1.0 - raw_unique / len(tokens),
                    codebook_usage_counts=usage_counts,
                    codebook_used=[sum(count > 0 for count in counts) for counts in usage_counts],
                    elapsed_s=round(time.time() - start_time, 3),
                )
                if global_step == MAX_GLOBAL_STEPS:
                    correlations, quartiles = _curvature_interpretability(
                        learned, item_signals
                    )
                    _record(
                        "curvature_interpretability",
                        spearman_correlations=correlations,
                        curvature_quartiles=quartiles,
                    )
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
    _record(
        "train_end",
        run_mode="LEARNABLE_CURVATURE",
        global_step=global_step,
        local_optimizer_steps=local_steps,
        duration_s=round(time.time() - start_time, 3),
    )
    dist.destroy_process_group()


def launch_and_run() -> None:
    _launch_via_torchrun()
    main()
