"""Train the successor-branch Poincare RQ-VAE (4 卡 DDP) and export HG-Rec SIDs.

2026-10-24 (Issue #117 candidate tree, run_2d84ba77647b4008):
- 本树是 issue #117 的候选树 TIGER_c117_global_sinkhorn_poincare, 派生自
  TIGER_RQ-VAE, 只改「item semantic-ID 的几何与目标」:
  (1) 三层残差码本按平方双曲距离 D_c^2 分配 (c=1.0), 两臂共用固定投影半径
      R=0.9, arcosh 参数钳在 1.0 (不是 1+delta), flat 臂是同一实现的 c->0 极限;
  (2) 目标 = L_rec + 0.25*L_commit + 1.0*L_code, 其中 L_rec 以「初始化时冻结的
      内容目标」为重构对象, 因此编码器有真实的重构梯度路径, 不会塌缩;
  (3) 分配是 hard argmin, 梯度经显式 straight-through estimator 回传。
  码本容量、层数、隐层宽度、epoch、优化器、数据与评测全部与基线一致.
- 冻结协议: latent 32、3 层 x 256 码、3000 epochs、不做超参 sweep、不使用
  验证/测试/未来交互.

- 两处尚未与提案完全一致的已知偏差 (实现者记录, 不得当作已满足):
  (1) 硬标签由 auction + epsilon scaling + 单点改进修复给出, 不是经证明精确的
      平衡字典序运输最优解; 提案要求 "Solve the finite transportation problem
      exactly ... lexicographically smallest";
  (2) 转移项 L_tr 已接入: r[a,t,k] = (n_train(a,t,k) + p_pref[t,k]) / (n_train(a,t) + 1),
      n_train 只统计与 prefix2(t) 同前缀的后继, p_pref 按提案的 leave-one-out 平滑,
      边按 train 事件均匀采样 (无偏估计 (1/|T|) sum_T).

2026-09-22 改造 (CLAUDE.md §1 + §3):
- 参数全部硬编码为模块常量 (禁 argparse/CLI flag)
- 加 _launch_via_torchrun 自动 fork 4 卡 DDP (nproc=4, master_port=50202)
- 加全程 metrics JSONL 收集器

2026-10-01 改造 (CLAUDE.md §2):
- 删除 early-stop, 固定跑满 EPOCHS; checkpoint 只在训练终点保存一次
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
from tqdm import tqdm

from model import RQVAE


# === 硬编码路径常量 (CLAUDE.md §1) ===
# Stage0 emits user-event parquet directly into results/stage0_build_parquet/.
EMBEDDING_FILE = Path(
    "/home/wlia0047/ar57/wenyu/GeneRec/stage1_GeneEmbedding/output/sentence_t5.npy"  # stage1
)
TRAIN_FILE     = Path(
    "/home/wlia0047/ar57/wenyu/GeneRec/results/stage0_build_parquet/train.parquet"   # stage0
)
# === 本轮实验臂 (issue #117 候选树) ===
EXPERIMENT_ARM = "c117_global_sinkhorn_poincare"
# 产物目录名: MCP 的 tree_result_dir 以本声明为准, 因此它与树名一致, 不带 iter 编号.
RESULT_DIR_NAME = "TIGER_c117_global_sinkhorn_poincare"
_RESULT_DIR     = Path("/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE") / RESULT_DIR_NAME
OUTPUT_SID     = _RESULT_DIR / "sids_for_hgrec.npy"
OUTPUT_JSON    = _RESULT_DIR / "item_sids.json"
CHECKPOINT     = _RESULT_DIR / "rqvae_best.pth"
METRICS_PATH   = Path("/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_c117_global_sinkhorn_poincare/logs") / EXPERIMENT_ARM / "training_metrics.jsonl"
LOG_DIR        = Path("/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_c117_global_sinkhorn_poincare/logs") / EXPERIMENT_ARM


# === 硬编码超参 (原 argparse default 值, 保留原 TIGER 默认) ===
EPOCHS              = 3000
BATCH_SIZE_PER_RANK = 1024       # 每卡 batch, 总 batch = 1024 * 4 = 4096
LR                  = 1e-3
WEIGHT_DECAY        = 1e-4
HIDDEN_SIZES        = (512, 256, 128)
CODEBOOK_NUM        = 3
CODEBOOK_SIZE       = (256, 256, 256)
CODEBOOK_DIM        = 32
BETA                = 0.25
VQ_TYPE             = "vq"
EMA_DECAY           = 0.99
SK_EPSILON          = 0.003
SK_ITERS            = 50
PCA_DIM             = 0
XAVIER_INIT         = True
WARMUP_EPOCHS       = 50
GRADIENT_CLIP_NORM  = 1.0
OPTIMIZER           = "AdamW"
SEED                = 42
NUM_WORKERS         = 0  # 2026-09-24: 0 防 DDP fork storm (4 rank × 4 worker = 16 子进程在 barrier 后挂死)
EVAL_INTERVAL       = 50
# CLAUDE.md §2: Stage2 不设任何 early-stop — 固定跑满 EPOCHS,
# 候选是否采用完全交由下游 stage3 test_R@10 裁决.
# 保留 best-collision checkpoint 仅作为导出用的最后一次快照.


# === 冻结几何常量 (proposal: geometry_asymmetric_successor_branches_hyp) ===
CURVATURE_C     = 1.0    # 曲率 c; flat 臂是同一实现的 c->0 极限
GEOMETRY_RADIUS = 0.9    # 两臂共用的固定投影半径 R < 1


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
        # The only difference from the baseline host: the item semantic-ID
        # geometry is the proposal's curvature-parameterised Poincare ball and
        # the objective carries the train-only transition term.
        geometry="global_sinkhorn",
        curvature=CURVATURE_C,
        radius=GEOMETRY_RADIUS,
    )


# === 训练划分的转移 (train-only) 与前缀条件目标 r ===
def build_train_events(frame: pd.DataFrame) -> np.ndarray:
    """Directed item transitions of the train split, one row per observed event.

    The train split repeats each user's history at several prefix lengths, so
    only the longest prefix plus its target is the complete available TRAIN
    sequence (the same population the behaviour evidence uses).
    """
    latest: dict[int, tuple[int, ...]] = {}
    for user, history, target in frame[["user", "history", "target"]].itertuples(
            index=False, name=None):
        sequence = tuple(int(x) for x in history) + (int(target),)
        current = latest.get(int(user))
        if current is None or len(sequence) > len(current):
            latest[int(user)] = sequence
    events: list[tuple[int, int]] = []
    for key in sorted(latest):
        sequence = latest[key]
        for source, target in zip(sequence, sequence[1:]):
            events.append((source, target))
    return np.asarray(events, dtype=np.int64)


def transition_weights(events: np.ndarray, codes: torch.Tensor, item_index: dict,
                       successors: dict, n_codes: int = 256) -> tuple:
    """Prefix-conditional r vectors and the per-epoch group tables.

    p_pref[t,k] = (n_items(prefix2(t),k) - 1[k = q[t,3]] + 1)
                   / (n_items(prefix2(t)) - 1 + 256)
    r[a,t,k]    = (n_train(a,t,k) + p_pref[t,k]) / (n_train(a,t) + 1)
    """
    device = codes.device
    n_items, n_levels = codes.shape
    prefix2 = codes[:, :2].cpu().numpy()
    third = codes[:, 2].cpu().numpy()
    groups: dict = {}
    for item in range(n_items):
        key = (int(prefix2[item, 0]), int(prefix2[item, 1]))
        groups.setdefault(key, []).append(item)
    group_table = {}
    for key, members in groups.items():
        counts = np.zeros(n_codes, dtype=np.float64)
        for member in members:
            counts[third[member]] += 1.0
        group_table[key] = (counts, float(len(members)))
    return group_table


def _r_vectors(anchors, targets, codes, group_table, item_index, successors,
                n_codes: int = 256):
    """r[a,t,k] for the sampled edges, from the frozen labels and train counts."""
    prefix2 = codes[:, :2].cpu().numpy()
    third = codes[:, 2].cpu().numpy()
    out = np.zeros((len(anchors), n_codes), dtype=np.float32)
    for position, (anchor, target) in enumerate(zip(anchors, targets)):
        key = (int(prefix2[target, 0]), int(prefix2[target, 1]))
        counts, size = group_table.get(key, (np.zeros(256), 0.0))
        own = 1 if third[target] < 256 else 0
        p_pref = (counts - own + 1.0) / (size - 1.0 + float(n_codes))
        train_counts = np.zeros(n_codes, dtype=np.float64)
        total = 0.0
        for successor, count in successors.get(int(anchor), {}).items():
            successor = int(successor)
            if successor >= len(prefix2):
                continue
            if (int(prefix2[successor, 0]), int(prefix2[successor, 1])) != key:
                continue
            code = int(third[successor])
            train_counts[code] += float(count)
            total += float(count)
        out[position] = ((train_counts + p_pref) / (total + 1.0)).astype(np.float32)
    return out

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
    # Train-only directed transitions, the item index and the successor counts
    # the prefix-conditional r vectors are built from.
    events = build_train_events(train_frame)
    item_index = {int(item): position for position, item in enumerate(train_ids)}
    # Only transitions whose two ends are inside the Stage2 training item set can
    # be scored; the rest are dropped rather than indexed out of range.
    events = events[np.array([(int(a) in item_index) and (int(t) in item_index)
                               for a, t in events], dtype=bool)]
    # Vector lookup from raw item id to the row in the training item set.
    item_position = np.full(int(train_ids.max()) + 1, -1, dtype=np.int64)
    for position, item in enumerate(train_ids):
        item_position[int(item)] = position
    successors: dict = {}
    for anchor, target in events:
        if int(anchor) in item_index and int(target) in item_index:
            successors.setdefault(int(anchor), {})[int(target)] = \
                successors.setdefault(int(anchor), {}).get(int(target), 0) + 1
    edge_rng = np.random.default_rng(SEED + rank)
    config = _tokenizer_config()
    model = RQVAE(config, in_dim=embeddings.shape[1]).to(device)
    if XAVIER_INIT:
        initialize_tiger_weights(model)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # Each row carries its population index so the frozen assignment is
    # addressed correctly even when the sampler shuffles.
    train_dataset = TensorDataset(train_embeddings,
                                  torch.arange(len(train_embeddings), dtype=torch.long))
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
        _record(rank, "train_start", epochs=EPOCHS, batch_size_per_rank=BATCH_SIZE_PER_RANK, total_batch_size=BATCH_SIZE_PER_RANK*world_size, lr=LR, weight_decay=WEIGHT_DECAY, hidden_sizes=list(HIDDEN_SIZES), codebook_num=CODEBOOK_NUM, codebook_size=list(CODEBOOK_SIZE), codebook_dim=CODEBOOK_DIM, beta=BETA, vq_type=VQ_TYPE, ema_decay=EMA_DECAY, sk_epsilon=SK_EPSILON, sk_iters=SK_ITERS, pca_dim=PCA_DIM, xavier_init=XAVIER_INIT, warmup_epochs=WARMUP_EPOCHS, gradient_clip_norm=GRADIENT_CLIP_NORM, optimizer=OPTIMIZER, seed=SEED, num_workers=NUM_WORKERS, eval_interval=EVAL_INTERVAL, early_stop="disabled", all_items=len(all_embeddings), train_items=len(train_embeddings), embedding_shape=list(embeddings.shape), world_size=world_size)

    # === codebook init: only rank 0 (codebook 不是 DDP 参数, 全 rank 共享) ===
    if rank == 0:
        print("[RecBole RQ-VAE] initializing codebooks with KMeans", flush=True)
    raw_module = model.module if isinstance(model, DDP) else model
    with torch.no_grad():
        raw_module.init_codebook(train_embeddings.to(device))
    if world_size > 1:
        dist.barrier()

    # Stage3 only consumes the final-epoch weights, so tracking the best
    # collision across epochs would be misleading; keep the last evaluation.
    final_collision = float("inf")
    final_raw_unique = 0
    codebook_sizes = list(config.codebook_size)

    for epoch in range(1, EPOCHS + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch - 1)
        model.train()
        losses: list[float] = []
        recons: list[float] = []
        transitions: list[float] = []
        _epoch_t0 = time.time()
        # Once per epoch: recompute the frozen global assignment and the frozen
        # transport targets over the whole training population.
        raw_module.refresh_assignment(train_embeddings.to(device))
        level_codes = int(raw_module.rq.codebook_size[2])
        group_table = transition_weights(events, raw_module.frozen_codes, item_index,
                                         successors, level_codes)
        population_embeddings = train_embeddings.to(device)
        for batch, index in loader:
            batch = batch.to(device, non_blocking=True)
            index = index.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            picks = edge_rng.integers(0, len(events), size=batch.shape[0])
            anchors = events[picks, 0]
            targets = events[picks, 1]
            r_batch = _r_vectors(anchors, targets, raw_module.frozen_codes, group_table,
                                 item_index, successors, level_codes)
            target_rows = torch.from_numpy(item_position[targets]).to(device)
            edges = (train_embeddings.to(device)[target_rows],
                     torch.from_numpy(r_batch).to(device))
            loss, recon_loss, assign_loss, balance_loss, transition_loss = model(
                batch, index, edges, population=population_embeddings)
            # The global marginal term is computed once per epoch and enters the
            # objective exactly once, in the first batch of that epoch.
            if not torch.isfinite(loss):
                raise RuntimeError(f"RQ-VAE loss became non-finite at epoch {epoch}")
            loss.backward()
            if GRADIENT_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach()))
            recons.append(float(recon_loss.detach()))
            transitions.append(float(assign_loss.detach()) + float(transition_loss.detach()))
        epoch_time_s = round(time.time() - _epoch_t0, 3)

        # === DDP all_reduce: 跨 rank 求平均 loss / recon ===
        local_loss = torch.tensor(float(np.mean(losses)) if losses else 0.0, device=device)
        local_recon = torch.tensor(float(np.mean(recons)) if recons else 0.0, device=device)
        if world_size > 1:
            dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_recon, op=dist.ReduceOp.SUM)
        avg_loss = float(local_loss.item()) / max(world_size, 1)
        avg_recon = float(local_recon.item()) / max(world_size, 1)
        avg_transition = float(np.mean(transitions)) if transitions else 0.0
        cur_lr = optimizer.param_groups[0]["lr"]

        # === 非 eval 间隔: 只 log train 行 ===
        if epoch % EVAL_INTERVAL != 0 and epoch != EPOCHS:
            if rank == 0:
                print(f"[RQ-VAE] epoch={epoch} loss={avg_loss:.8f} recon={avg_recon:.8f} transition={avg_transition:.8f} lr={cur_lr:.3e} time={epoch_time_s}s", flush=True)
                _record(rank, "train", epoch=epoch, loss=avg_loss, recon=avg_recon, transition_loss=avg_transition, lr=cur_lr, epoch_time_s=epoch_time_s, cumulative_time_s=round(time.time() - _T0, 3))
            continue

        # === eval 间隔: only rank 0 跑 (避免 4 卡 N× 全集评估) ===
        model.eval()
        raw_unique = None
        collision_v = None
        try:
            if rank == 0:
                with torch.no_grad():
                    raw_tokens = raw_module.get_indices(
                        all_embeddings.to(device)
                    ).cpu().numpy().astype(np.int64)
                raw_unique = int(len(np.unique(raw_tokens, axis=0)))
                collision_v = 1.0 - raw_unique / len(raw_tokens)
                final_collision = collision_v
                final_raw_unique = raw_unique
                print(
                    f"[RQ-VAE] epoch={epoch} loss={avg_loss:.8f} recon={avg_recon:.8f} "
                    f"transition={avg_transition:.8f} lr={cur_lr:.3e} "
                    f"raw_unique={raw_unique}/{len(raw_tokens)} "
                    f"collision={collision_v:.6f} time={epoch_time_s}s",
                    flush=True,
                )
                _record(
                    rank, "train", epoch=epoch, loss=avg_loss, recon=avg_recon, transition_loss=avg_transition, lr=cur_lr,
                    epoch_time_s=epoch_time_s,
                    cumulative_time_s=round(time.time() - _T0, 3),
                    raw_unique=raw_unique, raw_total=len(raw_tokens), collision=collision_v,
                )
        finally:
            pass
        if world_size > 1:
            dist.barrier()

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
                "selection": "final_epoch",
            },
            CHECKPOINT,
        )
        _record(
            rank, "ckpt_saved", epoch=EPOCHS, path=str(CHECKPOINT),
            reason="final_epoch", collision=final_collision,
            raw_unique=final_raw_unique,
        )
        raw_module.eval()
        with torch.no_grad():
            raw_tokens = raw_module.get_indices(
                all_embeddings.to(device)
            ).cpu().numpy().astype(np.int64)
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
