"""Train a hyperbolic RQ-VAE in the Poincare ball (4-card DDP).

Pure hyperbolic quantization at a fixed per-level curvature: encoder,
Poincare codebook assignment via balanced Sinkhorn, residual subtraction in
the ball, and Möbius-style reconstruction. The Stage2 budget matches the
TIGER baseline exactly; descriptive SID metrics never stop training.
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
from model.model import behaviour_ranking_loss
import curvature_config as experiment


EMBEDDING_FILE = Path(experiment.EMBEDDING_FILE)
TRAIN_FILE = Path(experiment.TRAIN_FILE)
METRICS_PATH = Path(experiment.STAGE2_LOG_DIR / "training_metrics_hyperbolic.jsonl")
LOG_DIR = Path(experiment.STAGE2_LOG_DIR)
SNAPSHOT_STEPS = {int(experiment.MAX_GLOBAL_STEPS): "hyperbolic"}


# === TIGER hyperparameters; the step budget is TIGER-aligned in curvature_config ===
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

# Fixed Poincare curvature per quantization level. Identical across levels
# because the curvature curriculum was removed; only the level index varies.
LAYER_CURVATURES = (1.0, 1.0, 1.0)
# Working point of the tangent-space quantization, s = sqrt(c) * ||r||, per
# level. A level with a non-zero target overwrites the magnitude of its input to
# target / sqrt(c) before the Poincare map and maps the result back afterwards,
# so the encoder cannot shrink ||z|| to pull the level back towards the ball
# centre the way it cancels a plain multiplicative scale. Pinning the first
# level alone left the second and third working at s ~ 0.006, i.e. still in the
# linear region; all three are pinned so the whole stack quantizes on the same
# nonlinear shell. 0.0 leaves a level untouched.
#
# Depth of the working point has a peak, and the three points measured on this
# axis put it at 0.3: d_H/d_E rose 2.12 -> 2.22 -> 2.35 as s went 0.3 -> 0.4 ->
# 0.5, every Stage2 number improved monotonically, and the downstream recall
# fell 7.7% then 16.4%. So s = 0.3 is the best value *tried*, but only the
# deeper side of it has been explored. s = 0.2 tests the shallower side, where the
# metric factor 2/(1-s^2) is 2.08 against 2.0 at the origin, i.e. almost
# Euclidean with the ball's regularizing effect still present. If the peak is
# genuine, this should be worse than 0.3 and the axis is closed on both sides;
# if it is better, the whole curve is shifted and the geometry line is not
# exhausted.
#
# The uniform shell was measured rather than assumed, and it is the weakest
# part of the model. Walking the three levels under s = 0.2 gives per-level
# losses of 0.000957 (level 0) against 0.139 and 0.131 (levels 1 and 2), so
# the three-level average is ~90% decided by the two deep levels and level 0
# is barely trained: 79 of its 256 codes go unused. The cause is that the pin
# overwrites every level to the same radius, but the levels do not hold the
# same thing. Level 0 quantizes the encoder latent, whose natural norm is 2.07,
# so pinning it to 0.2 compresses it 10x and its coarse codes end up trivially
# bracketing the data. Levels 1 and 2 quantize residuals, which do live near
# 0.2, so the uniform shell suits them.
#
# So the radius is matched per level to what the level actually quantizes, but
# the encoder's own scale of 2.07 is not reachable: the working point is
# s = sqrt(c) * ||r||, and s must stay below 1 or the shell leaves the
# Poincare ball and the map clamps every direction back to the boundary. The
# legal range therefore tops out just under 1, and level 0 takes 0.9, which
# moves it from the near-linear region it was stuck in into the nonlinear part
# of the ball without leaving it. Levels 1 and 2 keep 0.2.
#
# Removing the shell everywhere was already tested and cost 12% (0.052752),
# and moving the ranking loss onto this shell cost 13.5% (0.051951), so both of
# those stay rejected; the asymmetry between the two objectives is deliberate,
# not an oversight.
LAYER_WORKING_RADII = (0.9, 0.2, 0.2)


# Pairwise hyperbolic ranking uses the same transition construction, in-batch
# negative permutation, and auxiliary weight as the previous effective
# behaviour-contrastive condition. The margin is the rounded median of
# d_H(A,B-) - d_H(A,B+) on a deterministic 1024-pair sample from the accepted
# parent, so about half the sampled constraints begin active.
BEHAVIOUR_LOSS_WEIGHT = 0.1
BEHAVIOUR_MARGIN = 0.4
ADAMW_BETA1 = 0.9
ADAMW_BASE_BETA2 = 0.999
ADAMW_EPS = 1e-8



# === DDP launcher 配置 (与 stage3 不同 master_port 避免冲突) ===
_LAUNCHER = {
    "torchrun":    "/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/torchrun",
    "script":      os.path.abspath(__file__),
    "nproc":       4,
    "master_port": 50202,
    "log":         str(LOG_DIR / "train_migrated.log"),
    "visible_dev": "0,1,2,3",
}

def configure_run(launcher_script: str) -> None:
    """Point the single hyperbolic run at its own log and snapshot paths."""
    global METRICS_PATH, LOG_DIR, SNAPSHOT_STEPS, MAX_GLOBAL_STEPS
    LOG_DIR = Path(experiment.STAGE2_LOG_DIR)
    MAX_GLOBAL_STEPS = int(experiment.MAX_GLOBAL_STEPS)
    SNAPSHOT_STEPS = {MAX_GLOBAL_STEPS: "hyperbolic"}
    METRICS_PATH = LOG_DIR / "training_metrics_hyperbolic.jsonl"
    _LAUNCHER["log"] = str(LOG_DIR / "train_hyperbolic_migrated.log")
    _LAUNCHER["script"] = os.path.abspath(launcher_script)


def _snapshot_paths(version: str) -> tuple[Path, Path, Path, Path]:
    root = Path(experiment.STAGE2_RESULT_DIR)
    return (
        root / "out/rqvae/instruments/rqvae_best.pth",
        root / "out/rqvae/instruments/sids_raw.npy",
        root / "dataset/Instruments/sids_for_hgrec.npy",
        root / "item_sids.json",
    )

def _save_snapshot(
    raw_module: RQVAE,
    raw_tokens: np.ndarray,
    version: str,
    global_step: int,
    local_optimizer_steps: int,
    codebook_sizes: list[int],
    rank: int,
) -> None:
    checkpoint_path, raw_path, sid_path, json_path = _snapshot_paths(version)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {
                name: value.detach().cpu()
                for name, value in raw_module.state_dict().items()
            },
            "global_step": global_step,
            "local_optimizer_steps": local_optimizer_steps,
            "geometry": "poincare_fixed_curvature",
            "version": version,
            "curvatures": raw_module.get_curvatures().detach().cpu().tolist(),
            "working_radii": list(raw_module.rq.get_working_radii()),
            "effective_epsilons": raw_module.rq.get_effective_epsilons(),
        },
        checkpoint_path,
    )
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(raw_path, raw_tokens)
    sid = _extend_collisions(raw_tokens, codebook_sizes)
    if len(np.unique(sid, axis=0)) != len(sid):
        raise RuntimeError(f"SID extension failed for version {version}")
    sid_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(sid_path, sid)
    payload = {
        str(item_id): [int(value) for value in row]
        for item_id, row in enumerate(sid)
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _record(
        rank, "snapshot_saved", version=version, global_step=global_step,
        checkpoint=str(checkpoint_path), raw_sids=str(raw_path),
        sids_for_hgrec=str(sid_path), item_sids=str(json_path),
        raw_unique=int(len(np.unique(raw_tokens, axis=0))),
        unique_after_extend=int(len(np.unique(sid, axis=0))),
    )


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


def _transition_pairs(train_frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Source and successor item ids from consecutive train-sequence events."""
    sources: list[int] = []
    successors: list[int] = []
    for history, target in zip(
        train_frame["seen_history"].to_numpy(),
        train_frame["target"].to_numpy(dtype=np.int64),
    ):
        if history is None or len(history) == 0:
            continue
        source = int(history[-1])
        if source < 0:
            raise ValueError("Transition source item is negative")
        sources.append(source)
        successors.append(int(target))
    if not sources:
        raise ValueError("Training data has no usable item transitions")
    return (
        np.asarray(sources, dtype=np.int64),
        np.asarray(successors, dtype=np.int64),
    )


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
        layer_curvatures=LAYER_CURVATURES,
        layer_working_radii=LAYER_WORKING_RADII,
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
        model = DDP(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=False,
        )

    train_dataset = TensorDataset(train_embeddings)
    loader_batch_size = BATCH_SIZE_PER_RANK
    train_sampler = (
        DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True,
            seed=SEED, drop_last=False,
        )
        if world_size > 1 else None
    )
    loader = DataLoader(
        train_dataset,
        batch_size=loader_batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )
    source_ids, successor_ids = _transition_pairs(train_frame)
    source_embeddings = all_embeddings[torch.from_numpy(source_ids)]
    successor_embeddings = all_embeddings[torch.from_numpy(successor_ids)]
    pair_dataset = TensorDataset(source_embeddings, successor_embeddings)
    pair_sampler = (
        DistributedSampler(
            pair_dataset, num_replicas=world_size, rank=rank, shuffle=True,
            seed=SEED, drop_last=False,
        )
        if world_size > 1 else None
    )
    pair_loader = DataLoader(
        pair_dataset,
        batch_size=loader_batch_size,
        shuffle=(pair_sampler is None),
        sampler=pair_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )
    pair_iter = iter(pair_loader)


    if OPTIMIZER.lower() != "adamw":
        raise ValueError("This trainer requires AdamW.")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        betas=(ADAMW_BETA1, ADAMW_BASE_BETA2),
        eps=ADAMW_EPS,
        weight_decay=WEIGHT_DECAY,
    )

    raw_module = model.module if isinstance(model, DDP) else model
    layer_curvatures = [
        float(layer.get_curvature().detach().item())
        for layer in raw_module.rq.vq_layers
    ]
    layer_working_radii = [float(v) for v in raw_module.rq.get_working_radii()]
    if rank == 0:
        print(
            f"[hyperbolic] cold_start=true max_global_steps={MAX_GLOBAL_STEPS} "
            f"batch_size_per_rank={loader_batch_size} "
            f"total_effective_batch={loader_batch_size * world_size} "
            f"c={[round(value, 6) for value in layer_curvatures]} "
            f"radius={layer_working_radii}",
            flush=True,
        )
        _record(
            rank, "train_start", geometry="poincare_fixed_curvature",
            cold_start=True,
            checkpoint_loaded=False,
            initialization="xavier_encoder_decoder_then_kmeans_codebooks",
            max_global_steps=MAX_GLOBAL_STEPS,
            eval_interval_steps=EVAL_INTERVAL_STEPS,
            loader_batch_size=loader_batch_size,
            total_effective_item_batch_size=loader_batch_size * world_size,
            dataset_kind="train_target_items",
            dataset_size=len(train_dataset),
            lr=LR, weight_decay=WEIGHT_DECAY, hidden_sizes=list(HIDDEN_SIZES),
            codebook_num=CODEBOOK_NUM, codebook_size=list(CODEBOOK_SIZE),
            codebook_dim=CODEBOOK_DIM, beta=BETA, vq_type=VQ_TYPE,
            sk_epsilon=SK_EPSILON, sk_iters=SK_ITERS, pca_dim=PCA_DIM,
            xavier_init=XAVIER_INIT, seed=SEED, all_items=len(all_embeddings),
            train_target_items=len(train_embeddings), embedding_shape=list(embeddings.shape),
            world_size=world_size,
            layer_curvatures=layer_curvatures,
            layer_working_radii=layer_working_radii,
            behaviour_loss_weight=BEHAVIOUR_LOSS_WEIGHT,
            behaviour_margin=BEHAVIOUR_MARGIN,
            behaviour_pairs=len(source_ids),
            snapshot_steps=SNAPSHOT_STEPS,
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
    saved_versions = set()
    local_optimizer_steps = 0
    global_step_sync = 0
    epoch = 0
    interval_loss_sum = 0.0
    interval_recon_sum = 0.0
    interval_updates = 0
    interval_behaviour_sum = 0.0
    last_progress_time = time.time()

    while global_step_sync < MAX_GLOBAL_STEPS:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if pair_sampler is not None:
            pair_sampler.set_epoch(epoch)
        model.train()
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            reconstructed, quant_loss, _, _ = model(batch)
            loss, recon_loss = raw_module.compute_loss(
                batch, reconstructed, quant_loss
            )
            try:
                pair_batch = next(pair_iter)
            except StopIteration:
                if pair_sampler is not None:
                    pair_sampler.set_epoch(epoch + 1)
                pair_iter = iter(pair_loader)
                pair_batch = next(pair_iter)
            pair_sources, pair_successors = (
                tensor.to(device, non_blocking=True) for tensor in pair_batch
            )
            encoded_source = raw_module.encoder(pair_sources)
            encoded_successor = raw_module.encoder(pair_successors)
            negatives = encoded_source[
                torch.randperm(encoded_source.shape[0], device=device)
            ]
            ranking_loss = behaviour_ranking_loss(
                encoded_source,
                encoded_successor,
                negatives,
                curvature=layer_curvatures[0],
                margin=BEHAVIOUR_MARGIN,
            )
            loss = loss + BEHAVIOUR_LOSS_WEIGHT * ranking_loss
            interval_behaviour_sum += float(ranking_loss.detach())
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"RQ-VAE loss became non-finite at distributed step {global_step_sync}"
                )
            loss.backward()
            if GRADIENT_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()

            local_optimizer_steps += 1
            step_tensor = torch.tensor(
                local_optimizer_steps, dtype=torch.int64, device=device
            )
            if world_size > 1:
                dist.all_reduce(step_tensor, op=dist.ReduceOp.SUM)
            global_step_sync = int(step_tensor.item())

            interval_loss_sum += float(loss.detach())
            interval_recon_sum += float(recon_loss.detach())
            interval_updates += 1

            now = time.time()
            if rank == 0 and now - last_progress_time >= 5:
                print(
                    f"[hyperbolic] global_step={global_step_sync}/"
                    f"{MAX_GLOBAL_STEPS} local_updates={local_optimizer_steps} "
                    f"loss={float(loss.detach()):.8f} "
                    f"recon={float(recon_loss.detach()):.8f}",
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

            loss_sum_tensor = torch.tensor(
                interval_loss_sum, dtype=torch.float64, device=device
            )
            recon_sum_tensor = torch.tensor(
                interval_recon_sum, dtype=torch.float64, device=device
            )
            updates_tensor = torch.tensor(
                interval_updates, dtype=torch.int64, device=device
            )
            behaviour_sum_tensor = torch.tensor(
                interval_behaviour_sum, dtype=torch.float64, device=device
            )
            if world_size > 1:
                dist.all_reduce(loss_sum_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(recon_sum_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(behaviour_sum_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(updates_tensor, op=dist.ReduceOp.SUM)
            denominator = max(int(updates_tensor.item()), 1)
            avg_loss = float(loss_sum_tensor.item()) / denominator
            avg_recon = float(recon_sum_tensor.item()) / denominator
            avg_behaviour = float(behaviour_sum_tensor.item()) / denominator
            interval_loss_sum = 0.0
            interval_recon_sum = 0.0
            interval_behaviour_sum = 0.0
            interval_updates = 0

            model.eval()
            for layer in raw_module.rq.vq_layers:
                layer._skip_ddp_reduce = True
            try:
                if rank == 0:
                    with torch.no_grad():
                        raw_tokens, layer_stats = raw_module.get_indices_with_stats(
                            all_embeddings.to(device)
                        )
                    raw_tokens = raw_tokens.cpu().numpy().astype(np.int64)
                    usage_counts = [
                        stat["usage_counts"].detach().cpu().tolist()
                        for stat in layer_stats
                    ]
                    codebook_used = [
                        sum(count > 0 for count in usage)
                        for usage in usage_counts
                    ]
                    assignment_entropy_nats = [
                        float(stat["assignment_entropy_nats"].item())
                        for stat in layer_stats
                    ]
                    assignment_entropy_normalized = [
                        entropy / float(np.log(len(usage)))
                        for entropy, usage in zip(assignment_entropy_nats, usage_counts)
                    ]
                    raw_unique = int(len(np.unique(raw_tokens, axis=0)))
                    collision_v = 1.0 - raw_unique / len(raw_tokens)
                    current_curvatures = (
                        raw_module.get_curvatures().detach().cpu().tolist()
                    )
                    current_epsilons = raw_module.rq.get_effective_epsilons()

                    print(
                        f"[hyperbolic] global_step={global_step_sync}/"
                        f"{MAX_GLOBAL_STEPS} loss={avg_loss:.8f} "
                        f"recon={avg_recon:.8f} "
                        f"c={[round(value, 6) for value in current_curvatures]} "
                        f"epsilon={[round(value, 7) for value in current_epsilons]} "
                        f"raw_unique={raw_unique}/{len(raw_tokens)} "
                        f"collision={collision_v:.6f} "
                        f"behaviour={avg_behaviour:.8f} "
                        f"codebook_used={codebook_used}",
                        flush=True,
                    )
                    _record(
                        rank, "train",
                        global_step=global_step_sync,
                        local_optimizer_steps=local_optimizer_steps,
                        loss=avg_loss, recon=avg_recon,
                        behaviour=avg_behaviour,
                        behaviour_loss_weight=BEHAVIOUR_LOSS_WEIGHT,
                        behaviour_margin=BEHAVIOUR_MARGIN,
                        current_curvatures=current_curvatures,
                        effective_epsilons=current_epsilons,
                        raw_unique=raw_unique, raw_total=len(raw_tokens),
                        collision=collision_v, codebook_usage_counts=usage_counts,
                        codebook_used=codebook_used,
                        assignment_entropy_nats=assignment_entropy_nats,
                        assignment_entropy_normalized=assignment_entropy_normalized,
                    )
                    version = SNAPSHOT_STEPS.get(global_step_sync)
                    if version is not None:
                        _save_snapshot(
                            raw_module, raw_tokens, version, global_step_sync,
                            local_optimizer_steps, codebook_sizes, rank,
                        )
                version = SNAPSHOT_STEPS.get(global_step_sync)
                if version is not None:
                    saved_versions.add(version)
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
        expected_versions = set(SNAPSHOT_STEPS.values())
        if saved_versions != expected_versions:
            raise RuntimeError(
                f"Missing Stage2 snapshots: expected={sorted(expected_versions)}, "
                f"saved={sorted(saved_versions)}"
            )
        _record(
            rank, "train_end",
            total_wall_time_s=round(time.time() - _T0, 3),
            global_step=global_step_sync,
            local_optimizer_steps=local_optimizer_steps,
            saved_versions=sorted(saved_versions),
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
    print(
        f"[hyperbolic] launching {_LAUNCHER['nproc']}-rank torchrun",
        flush=True,
    )
    with open(_LAUNCHER["log"], "w", encoding="utf-8") as fout:
        rc = subprocess.call(cmd, stdout=fout, stderr=subprocess.STDOUT, env=env)
    sys.exit(rc)


if __name__ == "__main__":
    _launch_via_torchrun()
    main()
