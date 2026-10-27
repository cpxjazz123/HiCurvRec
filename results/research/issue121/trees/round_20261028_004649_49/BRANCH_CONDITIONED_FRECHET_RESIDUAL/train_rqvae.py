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

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import sparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from tqdm import tqdm

from model import RQVAE


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
        # The marker has to be unambiguous: `stage2_RQ-VAE` alone also names
        # results/stage2_RQ-VAE, which sits closer to a tree in the round
        # workspace and would make the root resolve to results/.
        if ((parent / "stage2_RQ-VAE").is_dir()
                and (parent / "results").is_dir()
                and (parent / "stage1_GeneEmbedding").is_dir()):
            return parent
    raise RuntimeError("cannot locate the repository root above " + str(here))


REPO_ROOT = _repository_root()
# The arm name is this tree's promoted name; a candidate copy changes only this line.
# The arm name is the name of the PROMOTED tree, and the promotion is the
# framework's: the candidate is selected under its idea id inside the round
# workspace and is copied to stage2_RQ-VAE/<TIGER_ + idea id> before it
# trains. The output-path preflight refuses a candidate whose RESULT_DIR is
# not its own promoted directory, so the promoted name is the one that belongs
# here; the round workspace keeps the idea id and is not where the outputs go.
ARM_NAME = "TIGER_BRANCH_CONDITIONED_FRECHET_RESIDUAL"
RESULT_DIR = REPO_ROOT / "results" / "stage2_RQ-VAE" / ARM_NAME

EMBEDDING_FILE = REPO_ROOT / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
TRAIN_FILE     = REPO_ROOT / "results" / "stage0_build_parquet" / "train.parquet"
OUTPUT_SID     = RESULT_DIR / "sids_for_hgrec.npy"
OUTPUT_JSON    = RESULT_DIR / "item_sids.json"
CHECKPOINT     = RESULT_DIR / "rqvae_best.pth"
LOG_DIR        = REPO_ROOT / "stage2_RQ-VAE" / ARM_NAME / "logs"
METRICS_PATH   = LOG_DIR / "training_metrics.jsonl"


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

# === 机制开关 (规格里的两个 code switch) ===
# behaviour_on=False: 所有物品共用一个全局分支与全局锚点, 曲率量化保持开启.
# geometry_on=False : 保留同一套分支标签, 但锚点取加权欧氏均值, 残差用普通减法.
# 唯一被训练的配置是 both_on; 另外两个只由 smoke test 行使其生效.
BEHAVIOR_ON         = True
GEOMETRY_ON         = True
# G 取基线第一层码本大小, 初始化种子取基线种子 (规格要求).
BRANCHES            = 256
# 每个物品的出向分布只保留计数最大的若干项, 聚类在截断后的稀疏向量上做.
CLUSTER_TOPK        = 32
# 每个 batch 物品抽一条出向边作为锚点的样本 (规格: batch 的采样训练边).
EDGES_PER_ITEM      = 1
# 导出时的分块大小: 切空间码向量是 (块 x 码本 x 维), 整批 24587 行放不进显存.
EXPORT_BLOCK        = 2048
# The single-card launch contract of the MCP: when this variable is 1 the
# trainer runs in this process instead of forking the four-rank job.
SINGLE_GPU_ENV   = "GENREC_SINGLE_GPU"
NUM_WORKERS         = 0  # 2026-09-24: 0 防 DDP fork storm (4 rank × 4 worker = 16 子进程在 barrier 后挂死)
EVAL_INTERVAL       = 50
# CLAUDE.md §2: Stage2 不设任何 early-stop — 固定跑满 EPOCHS,
# 候选是否采用完全交由下游 stage3 test_R@10 裁决.
# 保留 best-collision checkpoint 仅作为导出用的最后一次快照.


# === DDP launcher 配置 (与 stage3 不同 master_port 避免冲突) ===
_LAUNCHER = {
    # torchrun lives next to the interpreter that runs this file: no absolute
    # path, and the same interpreter the project pins in CLAUDE.md §2.
    "torchrun":    str(Path(sys.executable).with_name("torchrun")),
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


def _spherical_branches(profile, n_branches: int, seed: int, iterations: int = 20):
    """The deterministic spherical k-means over the successor distributions.

    The distance is the cosine distance, the centroids are the normalized
    means of their members, and the initialization is k-means++ with the
    baseline seed. The vectors are sparse over the successor items, so two
    items that go to different successors are different points even when their
    counts are identical.
    """
    from scipy import sparse

    n_items = profile.shape[0]
    generator = np.random.default_rng(seed)
    norms = np.sqrt(profile.multiply(profile).sum(axis=1)).A1
    has_profile = norms > 0
    if int(has_profile.sum()) < n_branches:
        raise RuntimeError("fewer items with outgoing transitions than branches")
    # k-means++ on the cosine distance
    chosen: list = []
    first = int(generator.choice(np.flatnonzero(has_profile)))
    chosen.append(first)
    closest = _cosine_to(profile, profile[first], np.flatnonzero(has_profile))
    for _ in range(n_branches - 1):
        weights = np.clip(1.0 - closest, 0.0, None)
        total = float(weights.sum())
        if total <= 0:
            candidate = int(generator.choice(np.flatnonzero(has_profile)))
        else:
            candidate = int(generator.choice(np.flatnonzero(has_profile),
                                            p=weights / weights.sum()))
        chosen.append(candidate)
        closer = _cosine_to(profile, profile[candidate], np.flatnonzero(has_profile))
        closest = np.maximum(closest, closer)
    centroids = profile[np.asarray(chosen)].tocsr()
    centroid_norms = np.sqrt(centroids.multiply(centroids).sum(axis=1)).A1
    assignment = np.zeros(n_items, dtype=np.int64)
    for _ in range(int(iterations)):
        similarity = (profile @ centroids.T).toarray()
        denominator = centroid_norms[None, :]
        similarity = np.divide(similarity, denominator,
                                out=np.zeros_like(similarity), where=denominator > 0)
        assignment = np.argmax(similarity, axis=1).astype(np.int64)
        updated = []
        for cluster in range(n_branches):
            members = np.flatnonzero(assignment == cluster)
            if members.size == 0:
                updated.append(centroids[cluster])
                continue
            mean = sparse.csr_matrix(profile[members].sum(axis=0))
            norm = float(np.sqrt((mean.multiply(mean)).sum()))
            if norm > 0:
                mean = mean / norm
            updated.append(sparse.csr_matrix(mean))
        centroids = sparse.vstack(updated).tocsr()
        centroid_norms = np.sqrt(centroids.multiply(centroids).sum(axis=1)).A1
    # An item without outgoing transitions has no direction: it joins the
    # branch whose centroid carries the most global mass.
    if not bool(has_profile.all()):
        fallback = int(np.argmax(centroid_norms))
        assignment[~has_profile] = fallback
    return assignment


def _cosine_to(profile, centroid, rows):
    """The cosine similarity of the given rows to one centroid."""
    from scipy import sparse

    column = sparse.csr_matrix(centroid.reshape(-1, 1))
    values = (profile[rows] @ column).toarray().ravel()
    norms = np.sqrt(profile[rows].multiply(profile[rows]).sum(axis=1)).A1
    centroid_norm = float(np.sqrt((centroid.multiply(centroid)).sum()))
    if centroid_norm <= 0:
        return np.zeros_like(values)
    return values / np.maximum(norms, 1e-12) / centroid_norm


def build_branch_structure(train_frame, n_items: int, seed: int):
    """训练期行为结构: 出向转移计数、稀疏出向分布、分支标签与边表.

    只使用训练划分里用户的时间有序序列 (history + target) 的相邻有向转移.
    每个物品的稀疏出向分布按 L2 归一化后, 用固定种子的球面 k-means (余弦距离)
    聚成 BRANCHES 个分支. 返回的边表按源物品排序, 供每个 batch 采样.
    """
    counts: dict = {}
    for history, target in zip(train_frame["history"].to_numpy(),
                               train_frame["target"].to_numpy(dtype=np.int64)):
        sequence = list(history) + [int(target)]
        for source, destination in zip(sequence, sequence[1:]):
            source = int(source)
            destination = int(destination)
            if source < 0 or destination < 0:
                continue
            if source >= n_items or destination >= n_items:
                continue
            counts[(source, destination)] = counts.get((source, destination), 0) + 1
    if not counts:
        raise RuntimeError("no training transition was counted")
    source_of = np.fromiter((key[0] for key in counts), dtype=np.int64,
                            count=len(counts))
    target_of = np.fromiter((key[1] for key in counts), dtype=np.int64,
                            count=len(counts))
    weight_of = np.fromiter(counts.values(), dtype=np.int64, count=len(counts))

    # 稀疏出向分布: 每个物品在它的后继物品上有一个分布, 计数最大的 CLUSTER_TOPK
    # 个后继保留, 再做 L2 归一化. 分布的坐标就是后继物品的身份, 所以聚类
    # 区分的是"跟谁一起被消费", 而不是"计数有多大".
    order = np.lexsort((-weight_of, source_of))
    source_sorted = source_of[order]
    target_sorted = target_of[order]
    weight_sorted = weight_of[order]
    boundaries = np.flatnonzero(np.diff(source_sorted)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(source_sorted)]))
    rows: list = []
    columns: list = []
    values: list = []
    for start, end in zip(starts, ends):
        item = int(source_sorted[start])
        local_targets = target_sorted[start:end]
        local_weights = weight_sorted[start:end].astype(np.float32)
        keep = np.argsort(-local_weights)[:CLUSTER_TOPK]
        local_targets = local_targets[keep]
        local_weights = local_weights[keep]
        norm = float(np.sqrt((local_weights ** 2).sum()))
        if norm <= 0:
            continue
        rows.append(np.full(len(local_targets), item, dtype=np.int64))
        columns.append(local_targets.astype(np.int64))
        values.append(local_weights / norm)
    if not rows:
        raise RuntimeError("no training transition survived the profile truncation")
    row_index = np.concatenate(rows)
    column_index = np.concatenate(columns)
    value_index = np.concatenate(values).astype(np.float32)
    profile = sparse.csr_matrix((value_index, (row_index, column_index)),
                                 shape=(n_items, n_items))

    branch_ids = _spherical_branches(profile, BRANCHES, seed)

    # 边表按源物品排序, item_edge_start 是 CSR 风格的偏移量.
    edge_order = np.argsort(source_of, kind="stable")
    edge_source = source_of[edge_order]
    edge_target = target_of[edge_order]
    edge_count = weight_of[edge_order]
    item_edge_start = np.zeros(n_items + 1, dtype=np.int64)
    np.add.at(item_edge_start, edge_source + 1, 1)
    item_edge_start = np.cumsum(item_edge_start)
    return branch_ids, edge_source, edge_target, edge_count, item_edge_start


def sample_training_edges(item_edge_start: np.ndarray, edge_target: np.ndarray,
                          edge_count: np.ndarray, item_ids: np.ndarray,
                          generator: np.random.Generator):
    """每个物品抽一条出向边, 返回 (边下标, 权重).

    抽样在物品自己的出向边里均匀进行, 边计数只作为锚点的权重 (规格:
    alpha_j = N_train(i->j)). 抽不到边的物品 (没有出向转移) 不贡献锚点.
    """
    starts = item_edge_start[item_ids]
    lengths = item_edge_start[item_ids + 1] - starts
    valid = lengths > 0
    pick = (generator.random(item_ids.shape[0]) * np.maximum(lengths, 1)).astype(np.int64)
    # An item without an outgoing edge has no edge to sample: its index stays
    # inside the table and its weight is zero, so it contributes to no anchor.
    safe = np.where(valid, starts + pick, 0)
    weights = np.where(valid, edge_count[safe], 0).astype(np.float32)
    return safe, weights, valid


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

    train_dataset = TensorDataset(train_embeddings, torch.from_numpy(train_ids))
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
    raw_module = model.module if isinstance(model, DDP) else model
    if rank == 0:
        print("[RecBole RQ-VAE] initializing codebooks with KMeans", flush=True)
    # The initialisation is a COLLECTIVE: every layer broadcasts its KMeans
    # centres. Calling it on rank 0 alone left the other ranks in a barrier
    # while rank 0 waited in a broadcast: a deadlock waiting to happen.
    with torch.no_grad():
        raw_module.init_codebook(train_embeddings.to(device))
    if world_size > 1:
        dist.barrier()

    # === 行为结构与机制上下文 ===
    # 只用训练划分的相邻有向转移; 分支数与初始化种子取基线的值.
    (branch_ids_np, edge_source_np, edge_target_np, edge_count_np,
     item_edge_start_np) = build_branch_structure(train_frame, len(all_embeddings), SEED)
    branch_ids_t = torch.from_numpy(branch_ids_np).to(device)
    edge_target_t = torch.from_numpy(edge_target_np).to(device)
    edge_count_t = torch.from_numpy(edge_count_np).to(device)
    edge_generator = np.random.default_rng(SEED)
    # 曲率: c = 1 / R^2, R 是基线编码器在训练开始时的输出半径 (实测一次, 不做扫描).
    with torch.no_grad():
        probe = raw_module.encoder(train_embeddings[:4096].to(device))
    curvature_radius = float(probe.norm(dim=-1).mean().item())
    curvature = 1.0 / max(curvature_radius, 1e-6) ** 2
    if rank == 0:
        print(f"[RecBole RQ-VAE] branches={BRANCHES} behaviour_on={BEHAVIOR_ON} "
              f"geometry_on={GEOMETRY_ON} radius={curvature_radius:.4f} curvature={curvature:.6f}",
              flush=True)

    def build_context(ids: torch.Tensor) -> dict:
        """细层的每步上下文: 分支标签、采样出的训练边、锚点所需的后继残差."""
        ids_np = ids.detach().cpu().numpy()
        indices, weights, _valid = sample_training_edges(
            item_edge_start_np, edge_target_np, edge_count_np, ids_np, edge_generator)
        targets = edge_target_t[torch.from_numpy(indices).to(device)]
        with torch.no_grad():
            target_encoded = raw_module.encoder(
                all_embeddings[targets.detach().cpu()].to(device))
        rows = torch.arange(ids.shape[0], device=device)
        return {
            "branch_ids": branch_ids_t[ids],
            "edge_sources": rows,
            "edge_targets": rows,
            "edge_weights": edge_count_t[torch.from_numpy(indices).to(device)].float(),
            "target_branches": branch_ids_t[targets],
            "n_branches": BRANCHES,
            "curvature": curvature,
            "behavior_on": BEHAVIOR_ON,
            "geometry_on": GEOMETRY_ON,
            "target_encoded": target_encoded,
        }

    def export_tokens(embeddings: torch.Tensor, ids: torch.Tensor) -> np.ndarray:
        """分块导出 SID: 切空间距离矩阵是 (块大小 x 码本大小), 整批会占显存."""
        blocks = []
        for start in range(0, int(ids.shape[0]), EXPORT_BLOCK):
            block_ids = ids[start:start + EXPORT_BLOCK]
            block = embeddings[start:start + EXPORT_BLOCK]
            tokens = raw_module.get_indices(block, infer_use_sk=False,
                                            fine_context=build_context(block_ids))
            blocks.append(tokens.cpu().numpy().astype(np.int64))
        return np.concatenate(blocks, axis=0)

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
        _epoch_t0 = time.time()
        for (batch, batch_ids) in loader:
            batch = batch.to(device, non_blocking=True)
            batch_ids = batch_ids.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            reconstructed, quant_loss, _, _ = model(batch, fine_context=build_context(batch_ids))
            # === DDP 包装下, 通过 .module 调 compute_loss (避免 DDP all_reduce 干扰) ===
            loss, recon_loss = raw_module.compute_loss(batch, reconstructed, quant_loss)
            if not torch.isfinite(loss):
                raise RuntimeError(f"RQ-VAE loss became non-finite at epoch {epoch}")
            loss.backward()
            if GRADIENT_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach()))
            recons.append(float(recon_loss.detach()))
        epoch_time_s = round(time.time() - _epoch_t0, 3)

        # === DDP all_reduce: 跨 rank 求平均 loss / recon ===
        local_loss = torch.tensor(float(np.mean(losses)) if losses else 0.0, device=device)
        local_recon = torch.tensor(float(np.mean(recons)) if recons else 0.0, device=device)
        if world_size > 1:
            dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_recon, op=dist.ReduceOp.SUM)
        avg_loss = float(local_loss.item()) / max(world_size, 1)
        avg_recon = float(local_recon.item()) / max(world_size, 1)
        cur_lr = optimizer.param_groups[0]["lr"]

        # === 非 eval 间隔: 只 log train 行 ===
        if epoch % EVAL_INTERVAL != 0 and epoch != EPOCHS:
            if rank == 0:
                print(f"[RQ-VAE] epoch={epoch} loss={avg_loss:.8f} recon={avg_recon:.8f} lr={cur_lr:.3e} time={epoch_time_s}s", flush=True)
                _record(rank, "train", epoch=epoch, loss=avg_loss, recon=avg_recon, lr=cur_lr, epoch_time_s=epoch_time_s, cumulative_time_s=round(time.time() - _T0, 3))
            continue

        # === eval 间隔: only rank 0 跑 (避免 4 卡 N× 全集评估) ===
        model.eval()
        # Skip DDP all_reduce inside VQLayer.forward when rank 0 runs eval alone:
        # rank 1-3 are stuck in barrier and won't join, would deadlock.
        for layer in raw_module.rq.vq_layers:
            layer._skip_ddp_reduce = True
        raw_unique = None
        collision_v = None
        try:
            if rank == 0:
                with torch.no_grad():
                    raw_tokens = export_tokens(
                        all_embeddings.to(device),
                        torch.arange(len(all_embeddings), device=device))
                raw_unique = int(len(np.unique(raw_tokens, axis=0)))
                collision_v = 1.0 - raw_unique / len(raw_tokens)
                final_collision = collision_v
                final_raw_unique = raw_unique
                print(
                    f"[RQ-VAE] epoch={epoch} loss={avg_loss:.8f} recon={avg_recon:.8f} "
                    f"lr={cur_lr:.3e} raw_unique={raw_unique}/{len(raw_tokens)} "
                    f"collision={collision_v:.6f} time={epoch_time_s}s",
                    flush=True,
                )
                _record(
                    rank, "train", epoch=epoch, loss=avg_loss, recon=avg_recon, lr=cur_lr,
                    epoch_time_s=epoch_time_s,
                    cumulative_time_s=round(time.time() - _T0, 3),
                    raw_unique=raw_unique, raw_total=len(raw_tokens), collision=collision_v,
                )
        finally:
            for layer in raw_module.rq.vq_layers:
                layer._skip_ddp_reduce = False
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
        # Skip DDP all_reduce inside VQLayer.forward during the export
        # inference pass — rank 1-3 are stuck in barrier, would deadlock.
        for layer in raw_module.rq.vq_layers:
            layer._skip_ddp_reduce = True
        try:
            with torch.no_grad():
                raw_tokens = export_tokens(
                    all_embeddings.to(device),
                    torch.arange(len(all_embeddings), device=device))
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


def _launch_via_torchrun() -> None:
    """2026-09-22: 与 stage3 同款, 但不同端口 (50202). 顶层调用时自动 fork 4 卡 DDP."""
    if int(os.environ.get("RANK", -1)) >= 0:
        return  # 已经在 torchrun 启动的子进程里
    if os.environ.get(SINGLE_GPU_ENV) == "1":
        # The frozen single-card branch: the trainer runs in this process.
        return
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
