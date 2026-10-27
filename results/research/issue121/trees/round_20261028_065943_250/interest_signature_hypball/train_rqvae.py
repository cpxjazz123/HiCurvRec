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
# The arm name is the name of the PROMOTED tree: the candidate is selected
# under its idea id inside the round workspace and is copied to
# stage2_RQ-VAE/<TIGER_ + idea id> before it trains. The output-path preflight
# refuses a candidate whose RESULT_DIR is not its own promoted directory.
ARM_NAME = "TIGER_interest_signature_hypball"
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
# 导出时的分块大小: 距离矩阵是 (块 x 码本), 整批 24587 行放不进显存.
EXPORT_BLOCK        = 2048
# The single-card launch contract of the MCP: when this variable is 1 the
# trainer runs in this process instead of forking the four-rank job.
SINGLE_GPU_ENV   = "GENREC_SINGLE_GPU"

# === 机制开关 (规格里的两个 code switch) ===
# behaviour_on=False: 行为位移为零, 所有第一层码都可用.
# geometry_on=False : 保留同样的 cohort 质量与父集合, 但 log/exp 退化为恒等映射,
#                     测地距离退化为平方欧氏距离.
# 唯一被训练的配置是 both_on; 另外两个只由 smoke test 行使其生效.
BEHAVIOR_ON         = True
GEOMETRY_ON         = True
# M 个兴趣 cohort = 第一层码本的十六分之一, 不做扫描. 若 M 等于码本大小, 每个父
# 只能分到一个码, 受限 argmin 被掩码强制, 曲率距离就没有任何决策可做, 机制不成立.
COHORT_COUNT        = (CODEBOOK_SIZE[0] if isinstance(CODEBOOK_SIZE, (list, tuple))
                       else CODEBOOK_SIZE) // 16
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


def _cohort_of_user(history, cohort_count: int) -> int:
    """The cohort of one training user: a stable hash of the ordered history.

    The hash is documented and deterministic: the ordered item ids are folded
    into a 64-bit value with FNV-1a and the cohort is that value modulo the
    cohort count. The same user always lands in the same cohort, on every host
    and in every run.
    """
    digest = 0xCBF29CE484222325
    for item in history:
        digest ^= int(item) & 0xFFFFFFFFFFFFFFFF
        digest = (digest * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return int(digest % int(cohort_count))


def build_interest_signatures(train_frame, n_items: int, cohort_count: int,
                              dim: int, seed: int, capacity: int):
    """The training-only interest signature of every item.

    A user is hashed into one of ``cohort_count`` cohorts by their ordered
    history. ``h_i[m]`` counts the training transitions that END at item i and
    come from a user of cohort m. The counts are smoothed by one per cohort and
    normalized, so no row is empty. The parent of an item is the cohort with
    the largest mass.

    The first-level codes are allocated over the nonempty parents in proportion
    to their item counts, by the largest remainder, and every nonempty parent
    gets at least one code. The behaviour displacement is the log map at the
    origin of the projected and normalized cohort-mass vector, which is a
    constant of the item.
    """
    from scipy import sparse

    users = train_frame["user"].to_numpy()
    histories = train_frame["history"].to_numpy()
    targets = train_frame["target"].to_numpy(dtype=np.int64)
    cohort_of_row = np.empty(len(users), dtype=np.int64)
    for row, history in enumerate(histories):
        cohort_of_row[row] = _cohort_of_user(history, cohort_count)

    rows: list = []
    columns: list = []
    values: list = []
    for row, history in enumerate(histories):
        sequence = list(history) + [int(targets[row])]
        cohort = int(cohort_of_row[row])
        for source, destination in zip(sequence, sequence[1:]):
            source = int(source)
            destination = int(destination)
            if source < 0 or destination < 0:
                continue
            if source >= n_items or destination >= n_items:
                continue
            rows.append(destination)
            columns.append(cohort)
            values.append(1.0)
    if not rows:
        raise RuntimeError("no training transition was counted")
    counts = sparse.coo_matrix((np.asarray(values, dtype=np.float64),
                                (np.asarray(rows, dtype=np.int64),
                                 np.asarray(columns, dtype=np.int64))),
                                shape=(n_items, cohort_count)).tocsr()
    # Add-one smoothing over every cohort: no row is empty.
    smoothed = counts + sparse.csr_matrix(
        np.ones((n_items, cohort_count), dtype=np.float64))
    mass = smoothed.sum(axis=1).A1
    signature = smoothed.multiply(1.0 / mass[:, None]).tocsr()

    parent = np.asarray(signature.argmax(axis=1)).ravel().astype(np.int64)
    nonempty = np.flatnonzero(np.bincount(parent, minlength=cohort_count) > 0)
    item_count = np.bincount(parent, minlength=cohort_count)

    # The allocation: proportional to the item counts, by the largest remainder,
    # at least one code per nonempty parent.
    # The capacity is the size of the first-level codebook, not the number of
    # cohorts: the cohorts are the parents and the codebook is what they share.
    capacity = int(capacity)
    share = item_count[nonempty].astype(np.float64)
    exact = share / max(float(share.sum()), 1.0) * capacity
    allocation = np.floor(exact).astype(np.int64)
    allocation = np.maximum(allocation, 1)
    remainder = capacity - int(allocation.sum())
    if remainder > 0:
        order = np.argsort(-(exact - np.floor(exact)))
        for index in order[:remainder]:
            allocation[index] += 1
    elif remainder < 0:
        order = np.argsort(exact - np.floor(exact))
        position = 0
        while remainder < 0:
            index = order[position % len(order)]
            if allocation[index] > 1:
                allocation[index] -= 1
                remainder += 1
            position += 1
    codes_of_parent: dict = {}
    cursor = 0
    for position, cohort in enumerate(nonempty):
        codes_of_parent[int(cohort)] = list(range(cursor, cursor + int(allocation[position])))
        cursor += int(allocation[position])

    # The fixed seeded orthonormal projection from the cohort space into the
    # latent dimension, and the behaviour displacement of every item.
    generator = np.random.default_rng(seed)
    # The projection has orthonormal ROWS, so it maps the cohort mass into the
    # latent dimension without changing its norm. The QR of the transpose gives
    # them: the reduced Q of a rectangular matrix would have the wrong shape.
    basis, _ = np.linalg.qr(generator.standard_normal((dim, cohort_count)))
    projection = basis.T
    dense = np.asarray(signature.todense(), dtype=np.float64)
    embedded = dense @ projection
    norms = np.linalg.norm(embedded, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    embedded = embedded / norms
    # The behaviour displacement is the log map at the origin of that vector,
    # so the vector has to stay strictly inside the ball.
    embedded = embedded * float(1.0 - 1e-5)
    return parent, embedded.astype(np.float32), codes_of_parent, nonempty


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

    # === 兴趣签名: cohort、父分配、行为位移 ===
    # 全部来自训练划分的相邻转移与用户历史, 不读验证或测试.
    (parent_ids, behaviour_shift, codes_of_parent,
     nonempty_parents) = build_interest_signatures(
        train_frame, len(all_embeddings), COHORT_COUNT, int(CODEBOOK_DIM), SEED,
        int(CODEBOOK_SIZE[0]))
    from model.hyperbolic import ball_log_map
    parent_t = torch.from_numpy(parent_ids).to(device)
    # The behaviour displacement is the log map at the origin of the projected
    # and normalized cohort-mass vector; the curvature is -1, so c = 1.
    shift_points = torch.from_numpy(behaviour_shift).to(device)
    shift_t = ball_log_map(torch.zeros_like(shift_points), shift_points, 1.0)
    parent_codes_t = torch.full((len(all_embeddings), int(CODEBOOK_SIZE[0])), -1,
                                dtype=torch.long, device=device)
    for cohort, codes in codes_of_parent.items():
        members = np.flatnonzero(parent_ids == cohort)
        for code in codes:
            parent_codes_t[members, code] = code
    parent_mask_t = parent_codes_t >= 0
    if rank == 0:
        allocated = sum(len(codes) for codes in codes_of_parent.values())
        print(f"[RecBole RQ-VAE] cohorts={COHORT_COUNT} nonempty_parents={len(nonempty_parents)} "
              f"allocated_codes={allocated} behaviour_on={BEHAVIOR_ON} geometry_on={GEOMETRY_ON}",
              flush=True)

    def build_first_level(ids: torch.Tensor) -> dict:
        """The first-level context of a batch: the behaviour shift and the parent."""
        shift = shift_t[ids]
        mask = parent_mask_t[ids]
        if not BEHAVIOR_ON:
            shift = torch.zeros_like(shift)
            mask = torch.ones_like(mask)
        return {
            "behaviour_shift": shift,
            "parent_mask": mask,
            "curvature": 1.0,
            "behavior_on": BEHAVIOR_ON,
            "geometry_on": GEOMETRY_ON,
        }

    def export_tokens(embeddings: torch.Tensor, ids: torch.Tensor) -> np.ndarray:
        blocks = []
        for start in range(0, int(ids.shape[0]), EXPORT_BLOCK):
            block_ids = ids[start:start + EXPORT_BLOCK]
            block = embeddings[start:start + EXPORT_BLOCK]
            tokens = raw_module.get_indices(block, infer_use_sk=False,
                                            first_level=build_first_level(block_ids))
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
            reconstructed, quant_loss, _, _ = model(batch, first_level=build_first_level(batch_ids))
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
