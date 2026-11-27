"""Train RecBole3.0's TIGER RQ-VAE (4 卡 DDP) and export HG-Rec-compatible SIDs.

2026-09-22 改造 (CLAUDE.md §1 + §3):
- 参数全部硬编码为模块常量 (禁 argparse/CLI flag)
- 加 _launch_via_torchrun 自动 fork 4 卡 DDP (nproc=4, master_port=50202, 避免与 stage3 的 50201 冲突)
- 加全程 metrics JSONL 收集器 (每个 train/eval/ckpt_saved/train_end 一行)
- 输出路径硬编码到 results/stage2_RQ-VAE/TIGER_RQ-VAE/

2026-10-01 改造 (CLAUDE.md §2):
- 删除 PATIENCE / stale 计数与 early-stop 分支, 固定跑满 EPOCHS
- checkpoint 只在训练结束时保存一次 (final_epoch); 不再按 collision 最优
  覆盖, 中途评估仅记录, Stage3 只接受训练终点权重
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

_PROCESS_BOOT_MONOTONIC = time.perf_counter()

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from tqdm import tqdm

from distances import build_distance
from losses import build_loss
from model import RQVAE
from train_config import load_train_config


class _BatchedTensorDataset(TensorDataset):
    """Gather exactly the original sampler's tensor rows, including padding."""
    def __getitems__(self, indices):
        return tuple(tensor[indices] for tensor in self.tensors)


def _batch_passthrough(batch):
    return batch


# === 路径常量（无绝对路径；由文件自身向上找仓库标记推导）===
# Stage0 emits user-event parquet directly into results/stage0_build_parquet/.
def _repository_root() -> Path:
    """The repository root, found by walking up to the stage2_RQ-VAE directory.

    No absolute path and no parent counting: this file is built in the round
    workspace and copied into stage2_RQ-VAE/ before it trains, so the number of
    parents differs between the two locations while the marker does not.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if ((parent / "stage2_RQ-VAE").is_dir()
                and (parent / "tools/research/stage2/contract.py").is_file()):
            return parent
    raise RuntimeError("cannot locate the repository root above " + str(here))


REPO_ROOT = _repository_root()
# Source-only frozen TIGER ancestor plus the explicitly inherited Issue255
# mechanism patch; this version changes only the initial global module budget.
ARM_NAME = "TIGER_calibrated_kappa_rq"
RESULT_DIR = REPO_ROOT / "results" / "stage2_RQ-VAE" / ARM_NAME

EMBEDDING_FILE = REPO_ROOT / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
TRAIN_FILE     = REPO_ROOT / "results" / "stage0_build_parquet" / "train.parquet"
OUTPUT_SID     = RESULT_DIR / "sids_for_hgrec.npy"
OUTPUT_JSON    = RESULT_DIR / "item_sids.json"
CHECKPOINT     = RESULT_DIR / "rqvae_best.pth"
LOG_DIR        = REPO_ROOT / "stage2_RQ-VAE" / ARM_NAME / "logs"
METRICS_PATH   = LOG_DIR / "training_metrics.jsonl"


# === 超参来自 config/train_config.json（严格校验：缺键/多键/类型/范围）===
# 一个 JSON 文件、严格校验：缺键/多键/类型错/越界都在碰 GPU 之前 ConfigError。
# 路径与 ARM_NAME 不在配置里 —— 它们归框架 (CLAUDE.md 19b0/67)。
CONFIG_FILE = Path(__file__).resolve().with_name("config") / "train_config.json"
CONFIG = load_train_config(CONFIG_FILE)
# 损失与距离都由注册表给出；训练文件只负责导入并把它们接到模型上。
CRITERION = build_loss(CONFIG)
DISTANCE = build_distance(CONFIG)

EPOCHS              = CONFIG.EPOCHS
BATCH_SIZE_PER_RANK = CONFIG.BATCH_SIZE_PER_RANK     # 每卡 batch, 总 batch = 每卡 * world_size
LR                  = CONFIG.LR
WEIGHT_DECAY        = CONFIG.WEIGHT_DECAY
HIDDEN_SIZES        = CONFIG.HIDDEN_SIZES
CODEBOOK_NUM        = CONFIG.CODEBOOK_NUM
CODEBOOK_SIZE       = CONFIG.CODEBOOK_SIZE
CODEBOOK_DIM        = CONFIG.CODEBOOK_DIM
BETA                = CONFIG.BETA
VQ_TYPE             = CONFIG.VQ_TYPE
EMA_DECAY           = CONFIG.EMA_DECAY
SK_EPSILON          = CONFIG.SK_EPSILON
SK_ITERS            = CONFIG.SK_ITERS
PCA_DIM             = CONFIG.PCA_DIM
XAVIER_INIT         = CONFIG.XAVIER_INIT
WARMUP_EPOCHS       = CONFIG.WARMUP_EPOCHS
GRADIENT_CLIP_NORM  = CONFIG.GRADIENT_CLIP_NORM
OPTIMIZER           = CONFIG.OPTIMIZER
SEED                = CONFIG.SEED
NUM_WORKERS         = CONFIG.NUM_WORKERS
EVAL_INTERVAL       = CONFIG.EVAL_INTERVAL
# CLAUDE.md §2: Stage2 不设任何 early-stop — 固定跑满 EPOCHS,
# 候选是否采用完全交由下游 stage3 test_R@10 裁决.
# 保留 best-collision checkpoint 仅作为导出用的最后一次快照.




# === 2026-09-22: 全程训练 metrics JSONL 收集器 (rank 0 only) ===
_metrics_lock = threading.Lock()
_T0 = time.time()
RUN_ID = None
RUN_PURPOSE = "formal"
DIAGNOSTIC_OBSERVER = None
# Truncate-on-train_start: every fresh run starts with an empty metrics file
# so a single ``training_metrics.jsonl`` always represents exactly one run
# (no leftover rows from a prior failed attempt).
_metrics_truncated = False
def _record(rank, event, **fields):
    if rank != 0:
        return
    rec = {"event": event, "timestamp": time.time(), "wall_time_s": round(time.time() - _T0, 3),
           "run_id": RUN_ID, "purpose": RUN_PURPOSE}
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


def _calibration_batch(loader, device):
    """Peek the actual first batch, restoring both global and explicit RNGs."""
    generators={id(g):g for g in
        (loader.generator,getattr(loader.sampler,'generator',None)) if g is not None}
    states=[(generator,generator.get_state()) for generator in generators.values()]
    try:
        with torch.random.fork_rng(devices=[]):
            try:
                batch,item_ids=next(iter(loader))
            except StopIteration:
                raise ValueError("calibration requires a full per-rank batch") from None
    finally:
        for generator,state in states:
            generator.set_state(state)
    if len(item_ids)!=BATCH_SIZE_PER_RANK:
        raise ValueError("calibration requires a full per-rank batch")
    return batch.to(device,non_blocking=True),item_ids


def main(*, diagnostic_epochs: int | None = None) -> None:
    global RUN_ID, RUN_PURPOSE
    if diagnostic_epochs is not None and (
            type(diagnostic_epochs) is not int or not 0 < diagnostic_epochs <= EPOCHS):
        raise ValueError("diagnostic epoch bound must be within the configured budget")
    if diagnostic_epochs is None and DIAGNOSTIC_OBSERVER is not None:
        raise ValueError("a formal run cannot install diagnostic observers")
    RUN_PURPOSE = "diagnostic" if diagnostic_epochs is not None else "formal"
    # === DDP 初始化 ===
    if _single_card_mode():
        # One process, one rank, no process group: every collective below is
        # guarded by world_size > 1 and therefore skipped.
        rank, world_size, local_rank = 0, 1, 0
        global BATCH_SIZE_PER_RANK
        BATCH_SIZE_PER_RANK = SINGLE_CARD_BATCH
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
        print("[single card] one process, one rank, per-process batch %d"
              % BATCH_SIZE_PER_RANK, flush=True)
    elif "RANK" in os.environ and int(os.environ.get("RANK", -1)) >= 0:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if world_size == 1:
            # A one-rank group is the single-card case reached through torchrun:
            # keep the four-rank global batch so the optimisation does not move.
            BATCH_SIZE_PER_RANK = SINGLE_CARD_BATCH
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    identity = [RUN_ID or f"{ARM_NAME}_{time.time_ns()}"] if rank == 0 else [None]
    if world_size > 1:
        dist.broadcast_object_list(identity, src=0)
    RUN_ID = identity[0]
    workers = [{"rank": rank, "pid": os.getpid(), "local_rank": local_rank}]
    if world_size > 1:
        local_worker = workers[0]
        workers = [None] * world_size
        dist.all_gather_object(workers, local_worker)

    set_seed(SEED + rank)
    embeddings = maybe_apply_pca(load_embeddings(EMBEDDING_FILE), PCA_DIM, SEED)
    train_frame = pd.read_parquet(TRAIN_FILE)
    train_ids = np.unique(train_frame["target"].to_numpy(dtype=np.int64))
    if train_ids.size == 0 or train_ids.min() < 0 or train_ids.max() >= len(embeddings):
        raise ValueError("Training target item ids do not match the embedding rows")

    all_embeddings = torch.from_numpy(embeddings)
    train_embeddings = all_embeddings[torch.from_numpy(train_ids)]
    config = _tokenizer_config()
    model = RQVAE(config, in_dim=embeddings.shape[1], distance=DISTANCE).to(device)
    executed_provenance = model.provenance()
    if XAVIER_INIT:
        initialize_tiger_weights(model)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    train_dataset = _BatchedTensorDataset(train_embeddings, torch.from_numpy(train_ids))
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=SEED, drop_last=False) if world_size > 1 else None
    loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE_PER_RANK,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        collate_fn=_batch_passthrough,
        persistent_workers=False,  # 2026-09-24: 关闭避免每个 epoch 都 fork 一次
    )

    if OPTIMIZER.lower() == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = max(1, EPOCHS * len(loader))
    warmup_steps = max(0, WARMUP_EPOCHS * len(loader))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if total_steps <= warmup_steps:
            return 1.0
        return max(0.0, float(total_steps - step) / float(total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if rank == 0:
        print(f"[RecBole RQ-VAE] device={device} all_items={len(all_embeddings)} train_items={len(train_embeddings)} world_size={world_size}", flush=True)
        print(f"[RecBole RQ-VAE] config: epochs={EPOCHS} batch_size_per_rank={BATCH_SIZE_PER_RANK} total_batch={BATCH_SIZE_PER_RANK*world_size} lr={LR}", flush=True)
        _record(rank, "train_start", config_sha256=CONFIG.sha256, config_path=CONFIG.path, epochs=EPOCHS, batch_size_per_rank=BATCH_SIZE_PER_RANK, total_batch_size=BATCH_SIZE_PER_RANK*world_size, lr=LR, weight_decay=WEIGHT_DECAY, hidden_sizes=list(HIDDEN_SIZES), codebook_num=CODEBOOK_NUM, codebook_size=list(CODEBOOK_SIZE), codebook_dim=CODEBOOK_DIM, beta=BETA, vq_type=VQ_TYPE, ema_decay=EMA_DECAY, sk_epsilon=SK_EPSILON, sk_iters=SK_ITERS, pca_dim=PCA_DIM, xavier_init=XAVIER_INIT, warmup_epochs=WARMUP_EPOCHS, gradient_clip_norm=GRADIENT_CLIP_NORM, optimizer=OPTIMIZER, seed=SEED, num_workers=NUM_WORKERS, eval_interval=EVAL_INTERVAL, early_stop="disabled", all_items=len(all_embeddings), train_items=len(train_embeddings), embedding_shape=list(embeddings.shape), world_size=world_size)
        _record(rank, "execution_context", source_tree=str(Path(__file__).resolve().parent),
                epoch_budget=EPOCHS, diagnostic_stop_epoch=diagnostic_epochs,
                executed_provenance=executed_provenance, workers=workers,
                torch_version=torch.__version__, cuda_build=torch.version.cuda,
                gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU")

    # === codebook init: only rank 0 (codebook 不是 DDP 参数, 全 rank 共享) ===
    if rank == 0:
        print("[RecBole RQ-VAE] initializing codebooks with KMeans", flush=True)
    raw_module = model.module if isinstance(model, DDP) else model
    with torch.no_grad():
        raw_module.init_codebook(train_embeddings.to(device))
    if world_size > 1:
        dist.barrier()
    catalogue = all_embeddings.to(device)
    raw_module.freeze_tangent_scale(catalogue)
    sampler = raw_module.structural_sampler()
    if rank == 0:
        level_sample = sampler.draw(0, 0, world_size, track=False)
        level_scales = raw_module.freeze_level_scales(catalogue, structural_sample=level_sample)
        _record(rank, "level_scale_calibration", **level_scales)
    if world_size > 1:
        dist.broadcast(raw_module.level_scales, 0)
    raw_module._level_scales = tuple(float(value) for value in raw_module.level_scales)
    calibration_batch, calibration_ids = _calibration_batch(loader, device)
    calibration = raw_module.calibrate_curvature_weight(
        calibration_batch, calibration_ids, catalogue,
        structural_sample=sampler.draw(0, rank, world_size, track=False))
    _record(rank, "curvature_calibration", **calibration,
            tangent_scale=float(raw_module.tangent_scale),
            target_scale=float(raw_module.target_scale))
    if world_size > 1:
        dist.broadcast(raw_module.curvature_weight, 0)
        dist.broadcast(raw_module.level_scales, 0)
    raw_module._level_scales = tuple(float(value) for value in raw_module.level_scales)

    # Stage3 only consumes the final-epoch weights, so tracking the best
    # collision across epochs would be misleading; keep the last evaluation.
    final_collision = float("inf")
    final_raw_unique = 0
    codebook_sizes = list(config.codebook_size)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    _record(rank, "initialization_complete",
            initialization_seconds=time.perf_counter() - _PROCESS_BOOT_MONOTONIC)
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch - 1)
        model.train()
        sampler.begin_epoch()
        totals = torch.zeros(8, device=device)
        observed_losses = [] if DIAGNOSTIC_OBSERVER is not None else None
        for step, (batch, item_ids) in enumerate(loader):
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            sample = sampler.draw((epoch - 1) * len(loader) + step, rank, world_size)
            reconstructed, quant_parts, _, _, curvature_loss = model(
                batch, item_ids, catalogue_embeddings=catalogue, structural_sample=sample)
            original_loss, components = CRITERION(reconstructed, batch, quant_parts)
            recon_loss = components["reconstruction"]
            loss = raw_module.combined_loss(original_loss, curvature_loss)
            if not torch.isfinite(loss):
                raise RuntimeError(f"RQ-VAE loss became non-finite at epoch {epoch}")
            if observed_losses is not None:
                observed_losses.append(loss.detach())
            observe_first = DIAGNOSTIC_OBSERVER is not None and epoch == 1 and step == 0
            if observe_first:
                DIAGNOSTIC_OBSERVER("first_step_before_backward", model=raw_module,
                    batch=batch, item_ids=item_ids, catalogue=catalogue,
                    structural_sample=sample, original_loss=original_loss,
                    structural_loss=curvature_loss, combined_loss=loss)
            loss.backward()
            if observe_first:
                DIAGNOSTIC_OBSERVER("first_step_after_backward", model=raw_module)
            if GRADIENT_CLIP_NORM > 0:
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            else:
                gradient_norm = torch.linalg.vector_norm(torch.stack([
                    torch.linalg.vector_norm(p.grad) for p in model.parameters() if p.grad is not None]))
            optimizer.step()
            if observe_first:
                DIAGNOSTIC_OBSERVER("first_step_after_optimizer", model=raw_module)
            scheduler.step()
            # The per-level codebook gradient norms are measured from the
            # same backward pass; they are the per-layer unweighted norms.
            sampled_kappa = raw_module.last_structural_kappa.detach().mean()
            level_grads=[]
            for layer in raw_module.rq.vq_layers:
                grad=layer.embed.weight.grad
                level_grads.append(torch.zeros((),device=device) if grad is None
                                   else grad.detach().norm())
            totals += torch.cat((torch.stack((loss.detach(), recon_loss.detach(),
                                             curvature_loss.detach(), gradient_norm.detach())),
                                 torch.stack(level_grads), sampled_kappa[None]))
        if world_size > 1:
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        average = (totals / (len(loader) * world_size)).cpu().tolist()
        coverage = [sampler.receipt()]
        if world_size > 1:
            coverage = [None] * world_size
            dist.all_gather_object(coverage, sampler.receipt())
        avg_loss, avg_recon, avg_curvature, avg_gradient = average[:4]
        cur_lr = optimizer.param_groups[0]["lr"]
        raw_unique = collision_v = None
        if epoch % EVAL_INTERVAL == 0 or epoch == EPOCHS:
            model.eval()
            for layer in raw_module.rq.vq_layers:
                layer._skip_ddp_reduce = True
            try:
                if rank == 0:
                    with torch.no_grad():
                        raw_tokens = raw_module.get_indices(catalogue, infer_use_sk=False).cpu().numpy().astype(np.int64)
                    raw_unique = int(len(np.unique(raw_tokens, axis=0)))
                    collision_v = 1.0 - raw_unique / len(raw_tokens)
                    final_collision, final_raw_unique = collision_v, raw_unique
            finally:
                for layer in raw_module.rq.vq_layers:
                    layer._skip_ddp_reduce = False
            if world_size > 1:
                dist.barrier()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - epoch_start
        peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        _record(rank, "train", epoch=epoch, loss=avg_loss, recon=avg_recon,
                curvature_estimate=avg_curvature, gradient_norm_before_clip=avg_gradient,
                lr=cur_lr, epoch_time_s=elapsed, raw_unique=raw_unique, collision=collision_v,
                sampled_kappa_mean=average[7],
                level_codebook_gradient_norm=average[4:7], peak_allocated_bytes=peak,
                structural_coverage=coverage, cumulative_time_s=time.time() - _T0)
        if rank == 0:
            print(f"[RQ-VAE] epoch={epoch} loss={avg_loss:.8f} recon={avg_recon:.8f} "
                  f"curvature_estimate={avg_curvature:.6f} time={elapsed:.6f}s "
                  f"collision={collision_v}", flush=True)
        if world_size > 1:
            dist.barrier()
        end = time.perf_counter()
        _record(rank, "epoch_boundary", epoch=epoch, start=epoch_start, end=end,
                epoch_time_s=end - epoch_start, structural_coverage=coverage)
        if diagnostic_epochs is not None and epoch == diagnostic_epochs:
            if world_size > 1:
                dist.barrier()
            if DIAGNOSTIC_OBSERVER is not None:
                DIAGNOSTIC_OBSERVER("diagnostic_complete", model=raw_module, catalogue=catalogue,
                    completed_epoch=epoch, completed_losses=[float(value) for value in observed_losses])
            _record(rank, "diagnostic_end", completed_epochs=epoch, epoch_budget=EPOCHS,
                    checkpoint_written=False, sid_exported=False)
            if world_size > 1:
                dist.barrier()
                dist.destroy_process_group()
            return

    # === 训练结束: only rank 0 保存终点 checkpoint 并导出 SID ===
    # Stage3 只接受训练终点权重；不保留任何中间或"最优 collision"快照。
    if rank == 0:
        final_state = {
            name: value.detach().cpu().clone()
            for name, value in raw_module.state_dict().items()
        }
        CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": final_state,
                "epoch": EPOCHS,
                "collision": final_collision,
                "run_id": RUN_ID,
                "source_tree": str(Path(__file__).resolve().parent),
                "sampling": sampler.receipt(),
                "workers": workers,
                "selection": "final_epoch",
                "provenance": raw_module.provenance(expected=executed_provenance),
            },
            CHECKPOINT,
        )
        _record(
            rank, "ckpt_saved", epoch=EPOCHS, path=str(CHECKPOINT),
            reason="final_epoch", collision=final_collision,
            raw_unique=final_raw_unique,
        )
        raw_module.eval()
        # Skip DDP all_reduce inside VQLayer.forward during the export
        # inference pass — rank 1-3 are stuck in barrier, would deadlock.
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
        sid = _extend_collisions(raw_tokens, codebook_sizes)
        if len(np.unique(sid, axis=0)) != len(sid):
            raise RuntimeError("RecBole SID extension failed to make item codes unique")
        OUTPUT_SID.parent.mkdir(parents=True, exist_ok=True)
        np.save(OUTPUT_SID, sid)
        payload = {str(item_id): [int(value) for value in row] for item_id, row in enumerate(sid)}
        OUTPUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[RecBole RQ-VAE] exported raw_shape={raw_tokens.shape} sid_shape={sid.shape} unique={len(np.unique(sid, axis=0))} -> {OUTPUT_SID}", flush=True)
        _record(rank, "train_end", total_wall_time_s=round(time.time() - _T0, 3), final_collision=final_collision, checkpoint_selection="final_epoch", sid_shape=list(sid.shape), output_sid=str(OUTPUT_SID), output_json=str(OUTPUT_JSON), unique_after_extend=len(np.unique(sid, axis=0)))

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


#: The single-card contract of the frozen recipe. On a host with one visible
#: card the trainer runs in this process as the only rank instead of forking
#: four, and the per-process batch is the four-rank global batch, so the
#: optimisation regime does not move: the same six optimizer steps per epoch
#: over the same samples, only the order of the summation differs.
SINGLE_GPU_ENV = "GENREC_SINGLE_GPU"
SINGLE_CARD_BATCH = 4096




def _single_card_mode() -> bool:
    return os.environ.get(SINGLE_GPU_ENV) == "1"


def _launch_via_torchrun() -> None:
    """Use the admitted immutable pipeline; never fall back on refusal."""
    if int(os.environ.get("RANK", -1)) >= 0:
        return
    if _single_card_mode():
        # The framework asked for the one-process layout; main() initialises it.
        return
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from tools.stage2.server import load_shared
    outcome = load_shared().tool_stage2_start(str(Path(__file__).resolve().parent))
    if outcome.get("refused"):
        raise RuntimeError(outcome["refused"])
    if outcome.get("status") not in {"starting", "smoke_pending", "running"} and not outcome.get("started"):
        raise RuntimeError("the formal framework did not accept the run: " + str(outcome))
    print(json.dumps(outcome, ensure_ascii=False, default=str), flush=True)
    sys.exit(0)


if __name__ == "__main__":
    _launch_via_torchrun()
    main()
