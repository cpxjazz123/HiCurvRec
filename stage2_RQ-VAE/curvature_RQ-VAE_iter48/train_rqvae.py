"""Train TIGER RQ-VAE with cyclic learnable Poincare curvature (4-card DDP).

Keep Iter46's architecture, initialization, Sinkhorn, and SID export, while
restoring Iter18's cyclic layer curvatures and curvature-conditioned AdamW.
Train the fixed 100k-step budget; descriptive SID metrics never stop training.
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
METRICS_PATH = Path(experiment.STAGE2_LOG_DIR / "training_metrics_A.jsonl")
LOG_DIR = Path(experiment.STAGE2_LOG_DIR)
RUN_MODE = "A"
SNAPSHOT_STEPS = {int(experiment.MAX_GLOBAL_STEPS): "A"}


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

C_CYCLIC_MIN = 0.05
C_CYCLIC_MAX = 1.5
C_CYCLIC_PERIOD = 100_000
LAYER_CURVATURE_NORMS = (0.001, 0.932889, 1.0)
CURVATURE_REG_WEIGHT = 0.005
BEHAVIOR_LOSS_WEIGHT = 0.20
BEHAVIOR_TEMPERATURE = 0.07
ADAMW_BETA1 = 0.9
ADAMW_BASE_BETA2 = 0.999
ADAMW_BETA2_SPAN = 0.009
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

def configure_run(mode: str, launcher_script: str) -> None:
    global RUN_MODE, METRICS_PATH, LOG_DIR, SNAPSHOT_STEPS, MAX_GLOBAL_STEPS
    supported_modes = {
        "A", "CURRICULUM", "BASELINE", "CURVATURE_ONLY", "BEHAVIOR_ONLY"
    }
    if mode not in supported_modes:
        raise ValueError(f"Unsupported Iter48 run mode: {mode}")
    RUN_MODE = mode
    LOG_DIR = Path(experiment.STAGE2_LOG_DIR)
    MAX_GLOBAL_STEPS = int(experiment.MAX_GLOBAL_STEPS)
    if mode == "A":
        SNAPSHOT_STEPS = {MAX_GLOBAL_STEPS: "A"}
        METRICS_PATH = LOG_DIR / "training_metrics_A.jsonl"
        _LAUNCHER["log"] = str(LOG_DIR / "train_A_migrated.log")
    elif mode == "CURRICULUM":
        SNAPSHOT_STEPS = {
            20_000: "B",
            40_000: "C",
            60_000: "D",
            100_000: "E",
        }
        METRICS_PATH = LOG_DIR / "training_metrics_curriculum.jsonl"
        _LAUNCHER["log"] = str(LOG_DIR / "train_curriculum_migrated.log")
    else:
        versions = {
            "BASELINE": "Baseline",
            "CURVATURE_ONLY": "CurvatureOnly",
            "BEHAVIOR_ONLY": "BehaviorOnly",
        }
        version = versions[mode]
        MAX_GLOBAL_STEPS = 40_000
        SNAPSHOT_STEPS = {MAX_GLOBAL_STEPS: version}
        METRICS_PATH = LOG_DIR / f"training_metrics_{version}.jsonl"
        _LAUNCHER["log"] = str(
            LOG_DIR / f"train_{version.lower()}_migrated.log"
        )
    _LAUNCHER["script"] = os.path.abspath(launcher_script)


def _snapshot_paths(version: str) -> tuple[Path, Path, Path, Path]:
    root = Path(experiment.STAGE2_RESULT_DIR)
    if version != "A":
        root = root / "versions" / version
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
            "run_mode": RUN_MODE,
            "version": version,
            "curvatures": raw_module.get_curvatures().detach().cpu().tolist(),
            "effective_epsilons": raw_module.rq.get_effective_epsilons(),
        },
        checkpoint_path,
    )
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(raw_path, raw_tokens)
    sid = _extend_collisions(raw_tokens, codebook_sizes)
    if len(np.unique(sid, axis=0)) != len(sid):
        raise RuntimeError(f"SID extension failed for Iter48 version {version}")
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

class TransitionDataset(torch.utils.data.Dataset):
    """Train-only history-to-target pairs for the behavior objective."""

    def __init__(self, embeddings: torch.Tensor, frame: pd.DataFrame):
        pairs = []
        for history, target in frame[["history", "target"]].itertuples(
            index=False, name=None
        ):
            if history is None or not isinstance(history, (list, tuple, np.ndarray)):
                continue
            if len(history) == 0:
                continue
            source_id = int(history[-1])
            target_id = int(target)
            if (
                source_id < 0
                or target_id < 0
                or source_id >= len(embeddings)
                or target_id >= len(embeddings)
            ):
                raise ValueError("Transition item ids do not match embedding rows")
            pairs.append((source_id, target_id))
        if not pairs:
            raise ValueError("The training data contains no valid history-target pairs")
        self.embeddings = embeddings
        self.pairs = torch.tensor(pairs, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.pairs.shape[0])

    def __getitem__(self, index: int):
        source_id, target_id = self.pairs[index]
        return (
            source_id,
            target_id,
            self.embeddings[source_id],
            self.embeddings[target_id],
        )


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
        c_cyclic_min=C_CYCLIC_MIN,
        c_cyclic_max=C_CYCLIC_MAX,
        c_cyclic_period=C_CYCLIC_PERIOD,
        layer_curvature_norms=LAYER_CURVATURE_NORMS,
        curvature_reg_weight=CURVATURE_REG_WEIGHT,
        behavior_loss_weight=BEHAVIOR_LOSS_WEIGHT,
        behavior_temperature=BEHAVIOR_TEMPERATURE,
        curriculum_enabled=RUN_MODE in {"CURRICULUM", "CURVATURE_ONLY"},
        behavior_curriculum_enabled=RUN_MODE in {"CURRICULUM", "BEHAVIOR_ONLY"},
        fixed_curvature=(
            1.0 if RUN_MODE in {"A", "BASELINE", "BEHAVIOR_ONLY"}
            else C_CYCLIC_MIN
        ),
    )



def build_curvature_conditioned_adamw(model):
    base_model = model.module if hasattr(model, "module") else model
    base_model.set_curriculum_step(0)
    layers = base_model.rq.vq_layers
    initial_curvatures = [
        float(layer.get_curvature().detach().item()) for layer in layers
    ]
    log_range = np.log(C_CYCLIC_MAX / C_CYCLIC_MIN)
    curvature_u = [
        float(
            np.clip(
                2.0 * np.log(max(curvature, C_CYCLIC_MIN) / C_CYCLIC_MIN)
                / log_range,
                0.0,
                1.0,
            )
        )
        for curvature in initial_curvatures
    ]
    layer_beta2 = [
        ADAMW_BASE_BETA2 - ADAMW_BETA2_SPAN * value
        for value in curvature_u
    ]
    if not all(
        np.isfinite(value) and 0.990 <= value <= ADAMW_BASE_BETA2
        for value in layer_beta2
    ):
        raise RuntimeError("Invalid curvature-conditioned AdamW beta2 values.")

    layer_parameters = [[] for _ in layers]
    scale_parameters = [[] for _ in layers]
    other_parameters = []
    for name, parameter in base_model.named_parameters():
        if not parameter.requires_grad:
            continue
        layer_index = next(
            (
                index
                for index in range(len(layers))
                if name.startswith(f"rq.vq_layers.{index}.")
            ),
            None,
        )
        if layer_index is None:
            other_parameters.append(parameter)
        elif name.endswith(".c_layer_scale"):
            scale_parameters[layer_index].append(parameter)
        else:
            layer_parameters[layer_index].append(parameter)
    if any(not parameters for parameters in layer_parameters):
        raise RuntimeError("Every RQ-VAE level needs its own AdamW parameter group.")

    optimizer_groups = []
    for index, (parameters, beta2) in enumerate(zip(layer_parameters, layer_beta2)):
        optimizer_groups.append(
            {
                "params": parameters,
                "lr": LR,
                "betas": (ADAMW_BETA1, beta2),
                "eps": ADAMW_EPS,
                "weight_decay": WEIGHT_DECAY,
                "curvature_layer_index": index,
            }
        )
        if scale_parameters[index]:
            optimizer_groups.append(
                {
                    "params": scale_parameters[index],
                    "lr": LR,
                    "betas": (ADAMW_BETA1, beta2),
                    "eps": ADAMW_EPS,
                    "weight_decay": 0.0,
                    "curvature_layer_index": index,
                    "parameter_role": "curvature_scale",
                }
            )
    if other_parameters:
        optimizer_groups.append(
            {
                "params": other_parameters,
                "lr": LR,
                "betas": (ADAMW_BETA1, ADAMW_BASE_BETA2),
                "eps": ADAMW_EPS,
                "weight_decay": WEIGHT_DECAY,
                "curvature_layer_index": None,
            }
        )
    return (
        torch.optim.AdamW(optimizer_groups),
        initial_curvatures,
        curvature_u,
        layer_beta2,
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

    if RUN_MODE == "A":
        train_dataset = TensorDataset(train_embeddings)
        loader_batch_size = BATCH_SIZE_PER_RANK
    else:
        train_dataset = TransitionDataset(all_embeddings, train_frame)
        loader_batch_size = BATCH_SIZE_PER_RANK // 2
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

    if OPTIMIZER.lower() != "adamw":
        raise ValueError("Iter48 requires AdamW.")
    if RUN_MODE in {"CURRICULUM", "CURVATURE_ONLY"}:
        optimizer, initial_curvatures, initial_curvature_u, layer_beta2 = (
            build_curvature_conditioned_adamw(model)
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LR,
            betas=(ADAMW_BETA1, ADAMW_BASE_BETA2),
            eps=ADAMW_EPS,
            weight_decay=WEIGHT_DECAY,
        )
        initial_curvatures = [1.0] * CODEBOOK_NUM
        initial_curvature_u = [None] * CODEBOOK_NUM
        layer_beta2 = [ADAMW_BASE_BETA2] * CODEBOOK_NUM

    raw_module = model.module if isinstance(model, DDP) else model
    raw_module.set_curriculum_step(0)
    if rank == 0:
        print(
            f"[Iter48 {RUN_MODE}] cold_start=true max_global_steps={MAX_GLOBAL_STEPS} "
            f"batch_size_per_rank={loader_batch_size} "
            f"effective_item_batch={BATCH_SIZE_PER_RANK} "
            f"total_effective_batch={BATCH_SIZE_PER_RANK * world_size} "
            f"initial_c={[round(value, 6) for value in initial_curvatures]} "
            f"layer_beta2={[round(value, 6) for value in layer_beta2]}",
            flush=True,
        )
        _record(
            rank, "train_start", run_mode=RUN_MODE, cold_start=True,
            checkpoint_loaded=False, initialization="xavier_encoder_decoder_then_kmeans_codebooks",
            max_global_steps=MAX_GLOBAL_STEPS,
            eval_interval_steps=EVAL_INTERVAL_STEPS,
            loader_batch_size=loader_batch_size,
            effective_item_batch_size=BATCH_SIZE_PER_RANK,
            total_effective_item_batch_size=BATCH_SIZE_PER_RANK * world_size,
            dataset_kind="train_target_items" if RUN_MODE == "A" else "train_history_target_pairs",
            dataset_size=len(train_dataset),
            lr=LR, weight_decay=WEIGHT_DECAY, hidden_sizes=list(HIDDEN_SIZES),
            codebook_num=CODEBOOK_NUM, codebook_size=list(CODEBOOK_SIZE),
            codebook_dim=CODEBOOK_DIM, beta=BETA, vq_type=VQ_TYPE,
            sk_epsilon=SK_EPSILON, sk_iters=SK_ITERS, pca_dim=PCA_DIM,
            xavier_init=XAVIER_INIT, seed=SEED, all_items=len(all_embeddings),
            train_target_items=len(train_embeddings), embedding_shape=list(embeddings.shape),
            world_size=world_size, curvature_min=C_CYCLIC_MIN,
            curvature_max=C_CYCLIC_MAX, curvature_period=C_CYCLIC_PERIOD,
            layer_curvature_norms=list(LAYER_CURVATURE_NORMS),
            initial_curvatures=initial_curvatures,
            initial_curvature_u=initial_curvature_u,
            beta2_by_layer=layer_beta2,
            curvature_regularization_weight=CURVATURE_REG_WEIGHT,
            behavior_loss_weight=BEHAVIOR_LOSS_WEIGHT,
            behavior_temperature=BEHAVIOR_TEMPERATURE,
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
    interval_behavior_sum = 0.0
    interval_updates = 0
    last_progress_time = time.time()

    while global_step_sync < MAX_GLOBAL_STEPS:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        for batch_values in loader:
            if RUN_MODE == "A":
                (batch,) = batch_values
                behavior_ids = None
                batch = batch.to(device, non_blocking=True)
            else:
                source_ids, target_ids, source_batch, target_batch = batch_values
                source_ids = source_ids.to(device, non_blocking=True)
                target_ids = target_ids.to(device, non_blocking=True)
                source_batch = source_batch.to(device, non_blocking=True)
                target_batch = target_batch.to(device, non_blocking=True)
                batch = torch.cat((source_batch, target_batch), dim=0)
                behavior_ids = (source_ids, target_ids)

            optimizer.zero_grad(set_to_none=True)
            reconstructed, quant_loss, _, _, behavior_loss = model(
                batch, behavior_ids=behavior_ids
            )
            loss, recon_loss = raw_module.compute_loss(
                batch, reconstructed, quant_loss, behavior_loss
            )
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
            raw_module.set_curriculum_step(global_step_sync)

            interval_loss_sum += float(loss.detach())
            interval_recon_sum += float(recon_loss.detach())
            interval_behavior_sum += float(behavior_loss.detach())
            interval_updates += 1

            now = time.time()
            if rank == 0 and now - last_progress_time >= 5:
                curvatures_now = raw_module.get_curvatures().detach().cpu().tolist()
                print(
                    f"[Iter48 {RUN_MODE}] global_step={global_step_sync}/"
                    f"{MAX_GLOBAL_STEPS} local_updates={local_optimizer_steps} "
                    f"loss={float(loss.detach()):.8f} "
                    f"recon={float(recon_loss.detach()):.8f} "
                    f"behavior={float(behavior_loss.detach()):.8f} "
                    f"behavior_weight={raw_module.rq.get_behavior_weight():.6f} "
                    f"c={[round(value, 6) for value in curvatures_now]}",
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
            behavior_sum_tensor = torch.tensor(
                interval_behavior_sum, dtype=torch.float64, device=device
            )
            updates_tensor = torch.tensor(
                interval_updates, dtype=torch.int64, device=device
            )
            if world_size > 1:
                dist.all_reduce(loss_sum_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(recon_sum_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(behavior_sum_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(updates_tensor, op=dist.ReduceOp.SUM)
            denominator = max(int(updates_tensor.item()), 1)
            avg_loss = float(loss_sum_tensor.item()) / denominator
            avg_recon = float(recon_sum_tensor.item()) / denominator
            avg_behavior = float(behavior_sum_tensor.item()) / denominator
            interval_loss_sum = 0.0
            interval_recon_sum = 0.0
            interval_behavior_sum = 0.0
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
                    curvature_reg_value = float(
                        raw_module.curvature_regularization().detach().item()
                    )
                    curvature_alpha = raw_module.rq.get_curriculum_alpha()
                    behavior_weight = raw_module.rq.get_behavior_weight()

                    print(
                        f"[Iter48 {RUN_MODE}] global_step={global_step_sync}/"
                        f"{MAX_GLOBAL_STEPS} loss={avg_loss:.8f} "
                        f"recon={avg_recon:.8f} behavior={avg_behavior:.8f} "
                        f"behavior_weight={behavior_weight:.6f} "
                        f"curvature_alpha={curvature_alpha:.6f} "
                        f"c={[round(value, 6) for value in current_curvatures]} "
                        f"epsilon={[round(value, 7) for value in current_epsilons]} "
                        f"curvature_reg={curvature_reg_value:.8f} "
                        f"raw_unique={raw_unique}/{len(raw_tokens)} "
                        f"collision={collision_v:.6f} "
                        f"codebook_used={codebook_used}",
                        flush=True,
                    )
                    _record(
                        rank, "train", run_mode=RUN_MODE,
                        global_step=global_step_sync,
                        local_optimizer_steps=local_optimizer_steps,
                        loss=avg_loss, recon=avg_recon, behavior_loss=avg_behavior,
                        behavior_weight=behavior_weight,
                        curvature_alpha=curvature_alpha,
                        current_curvatures=current_curvatures,
                        effective_epsilons=current_epsilons,
                        beta2_by_layer=layer_beta2,
                        curvature_regularization=curvature_reg_value,
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
            rank, "train_end", run_mode=RUN_MODE,
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
        f"[Iter48 {RUN_MODE}] launching {_LAUNCHER['nproc']}-rank torchrun",
        flush=True,
    )
    with open(_LAUNCHER["log"], "w", encoding="utf-8") as fout:
        rc = subprocess.call(cmd, stdout=fout, stderr=subprocess.STDOUT, env=env)
    sys.exit(rc)


if __name__ == "__main__":
    _launch_via_torchrun()
    main()
