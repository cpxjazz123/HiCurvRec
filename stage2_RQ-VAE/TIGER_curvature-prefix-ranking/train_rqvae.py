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

import curvature_prefix
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
        # The marker must be unambiguous: ``results/`` also holds a
        # ``stage2_RQ-VAE`` directory, and a tree that runs from a workspace under
        # ``results/`` would otherwise take it for the repository root and write
        # to ``results/results/...``. The root is the directory that holds all
        # three of the project's top-level trees.
        if all((parent / marker).is_dir()
               for marker in ("stage2_RQ-VAE", "results", "stage1_GeneEmbedding")):
            return parent
    raise RuntimeError("cannot locate the repository root above " + str(here))


REPO_ROOT = _repository_root()
# The arm name is this tree's promoted name; a candidate copy changes only this line.
# Provenance: copied from stage2_RQ-VAE/TIGER_RQ-VAE (the frozen baseline)
# (the last complete mechanism, issue #121 baseline chain). This tree runs the
# curvature_prefix_ranking mechanism: AF3 cohesion on the TRAIN behaviour graph
# decides which neighbours should share coarse SID prefixes, and the ranking loss
# acts on the hyperbolic prefix-assignment probabilities of the residual quantizer.
ARM_NAME = "TIGER_curvature-prefix-ranking"
if ARM_NAME != "TIGER_curvature-prefix-ranking":
    raise RuntimeError(
        f"only the 'TIGER_curvature-prefix-ranking' mechanism may run in this tree; "
        f"got {ARM_NAME!r}")
RESULT_DIR = REPO_ROOT / "results" / "stage2_RQ-VAE" / ARM_NAME

EMBEDDING_FILE = REPO_ROOT / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
TRAIN_FILE     = REPO_ROOT / "results" / "stage0_build_parquet" / "train.parquet"
OUTPUT_SID     = RESULT_DIR / "sids_for_hgrec.npy"
OUTPUT_JSON    = RESULT_DIR / "item_sids.json"
CHECKPOINT     = RESULT_DIR / "rqvae_best.pth"
LOG_DIR        = RESULT_DIR / "logs"
METRICS_PATH   = LOG_DIR / "training_metrics.jsonl"


# === 单卡分支 (冻结契约的一部分) ===
# The frozen contract names the variable that tells the trainer to run on one
# card instead of forking four ranks.
SINGLE_GPU_ENV = "GENREC_SINGLE_GPU"

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

# === 候选机制: 后继吸引度加权的 Poincare Frechet 码本步 ===
# The two switches of the round. The arm that is trained has both on; the other
# two exist in the code and are not trained (Issue #121, one run per candidate).
USE_BEHAVIOUR = True
USE_GEOMETRY = True
#: The ranking acts on every level: the coarse prefix carries the cohesion and the
#: fine levels carry the separation that the historical record associates with
#: recall (L2 prefixes and L3 triples both correlate positively with R@10).
RANKING_LEVELS = (0, 1, 2)
#: Which graph curvature the cohesion weights are built from. ``af3`` counts the
#: triangles; ``forman`` is the triangle-free arm Issue #138 B requires.
CURVATURE_STAT = "af3"
# The step size of the Frechet step, a documented default that is never tuned.
# The new mechanism's coupling weight is not a tuned constant: the first training
# step measures the reconstruction loss and the mechanism loss on the same batch
# and fixes their ratio, so the two enter the objective at the same scale. The
# clamp is a numerical guard, not a search range.
MECHANISM_SCALE: float | None = None
MECHANISM_SCALE_BOUNDS = (1e-8, 1e8)
#: The mechanism may take at most this share of the reconstruction gradient at
#: the first step. A tenth of it was measured too weak: across a whole 3000-epoch
#: run the ranking value never left ln 2 and the exported L2 prefix count stayed
#: below the baseline's, so the behaviour signal never reached the assignment. A
#: quarter still leaves reconstruction dominant, and the collision trust region
#: rolls the partition back if the pressure crosses the budget.
MECHANISM_GRADIENT_SHARE = 0.25
#: The bucket-spread term enters at the same unit scale as the ranking; the
#: measured gradient budget fixes the coupling of the whole mechanism, so this is
#: a per-level unit weight, not a tuned knob.
SPREAD_WEIGHT = 1.0
#: The ball's radius must match the data. With curvature 1 the residuals sit at
#: 0.06-0.15 of the radius and the Poincare distance is exactly 2x the Euclidean
#: (measured at every level), so the curve only rescales the cost. The curvature
#: is measured once from the first batch's level-1 residual norms.
MECHANISM_CURVATURE: float | None = None
#: The coarse codebook follows the behaviour cohesion, but at most this fraction
#: of the distance to the weighted mean per step: a full step (the parent arm's
#: rule) collapsed the partition to 1,985 L2 prefixes.
CODEBOOK_STEP = 0.10
#: The co-cluster term enters at the unit scale; the measured gradient budget
#: sizes the mechanism as a whole.
CLUSTER_WEIGHT = 1.0
#: The coarse usage entropy, at the same unit scale: the co-cluster term alone
#: collapsed the exported L1 level from 254 codes to 105.
USAGE_WEIGHT = 1.0
#: Trust region on the SID partition: a training epoch whose measured collision
#: exceeds this budget is rolled back to the last accepted state. The mechanism
#: weight is NOT shrunk on a rollback: halving it drove the weight to 1e-19 in one
#: run, so the mechanism never trained while the rollbacks kept happening anyway.
#: The gate is 0.35; the measured eval-to-eval spread of a fixed recipe is about
#: 0.01, so the budget keeps one spread of margin (0.35 - 0.01).
COLLISION_BUDGET = 0.34
_ACCEPTED_STATE: dict | None = None
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


def _extend_collisions(tokens: np.ndarray, codebook_sizes: list[int],
                       priority: np.ndarray | None = None) -> np.ndarray:
    """Match RecBole3.0's extend path with one fixed extra SID level.

    The fourth token is the item's slot inside its coarse group. With
    ``priority`` the slot is the item's behaviour rank (highest cohesion first),
    so the same slot means the same behaviour position in every group; without it
    the slot is the item id order, which the Stage 3 model cannot learn.
    """
    if tokens.ndim != 2 or tokens.shape[1] != len(codebook_sizes):
        raise ValueError(f"Token shape {tokens.shape} disagrees with codebook sizes {codebook_sizes}")
    groups: dict[tuple[int, ...], list[int]] = {}
    for item_id, row in enumerate(tokens.tolist()):
        groups.setdefault(tuple(int(value) for value in row), []).append(item_id)
    offsets = np.cumsum([0, *codebook_sizes], dtype=np.int64)
    result = np.empty((len(tokens), tokens.shape[1] + 1), dtype=np.int64)
    for row, item_ids in groups.items():
        ordered = list(item_ids)
        if priority is not None:
            key = np.asarray(priority, dtype=np.float64)
            ordered.sort(key=lambda item: (-float(key[item]), item))
        for occurrence, item_id in enumerate(ordered):
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
        # The candidate's mechanism: the codebook of every level moves by the
        # successor-weighted Frechet step, and the same weights are inside the
        # commitment term. The two switches are the ablations of the round; only
        # The two switches of the round: they choose the curved or the Euclidean
        # cost inside the ranking loss. The SID assignment itself is the frozen
        # baseline's Euclidean rule either way.
        use_behaviour=USE_BEHAVIOUR,
        use_curvature=USE_GEOMETRY,
    )


def _training_edges(frame, n_items: int) -> np.ndarray:
    """The undirected item-item edges of the TRAIN split.

    Same consecutive-pair construction as the successor attraction: every
    session contributes ``(u, v)`` for consecutive items, and one undirected edge
    is kept per distinct pair. No validation or test interaction is read.
    """
    frame = frame.sort_values("user", kind="stable")
    last = frame.groupby("user", sort=False).tail(1)
    sources, targets = [], []
    for history, target in zip(last["history"].to_numpy(), last["target"].to_numpy()):
        sequence = list(history) + [int(target)]
        if len(sequence) < 2:
            continue
        sources.append(np.asarray(sequence[:-1], dtype=np.int64))
        targets.append(np.asarray(sequence[1:], dtype=np.int64))
    if not sources:
        return np.zeros((0, 2), dtype=np.int64)
    sources = np.concatenate(sources)
    targets = np.concatenate(targets)
    keep = ((sources >= 0) & (targets >= 0) & (sources < n_items)
            & (targets < n_items) & (sources != targets))
    return np.stack([sources[keep], targets[keep]], axis=1)


def _batch_positions(item_ids: torch.Tensor, wanted: torch.Tensor) -> torch.Tensor:
    """The row of every wanted item inside the batch, or -1."""
    order = torch.argsort(item_ids)
    sorted_ids = item_ids[order]
    position = torch.searchsorted(sorted_ids, wanted)
    inside = position < sorted_ids.numel()
    clipped = position.clamp_max(max(sorted_ids.numel() - 1, 0))
    matched = inside & (sorted_ids[clipped] == wanted)
    return torch.where(matched, order[clipped], torch.full_like(position, -1))


def main() -> None:
    global MECHANISM_SCALE, MECHANISM_CURVATURE, _ACCEPTED_STATE
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
    # The mechanism's behaviour side: one exact AF3 pass over the TRAIN edges.
    edge_table = curvature_prefix.build_edge_tables(
        _training_edges(train_frame, len(embeddings)), len(embeddings),
        statistic=CURVATURE_STAT)
    triple_lookup = curvature_prefix.TripleLookup(edge_table)
    if rank == 0:
        print(f"[RQ-VAE] AF3 table: {edge_table['af3'].size} edges, "
              f"median={edge_table['median']:.4f}, scale={edge_table['scale']:.4f}",
              flush=True)
        _record(rank, "mechanism_table", edges=int(edge_table["af3"].size),
                median=float(edge_table["median"]), scale=float(edge_table["scale"]),
                mechanism="curvature_prefix_ranking")
    model = RQVAE(config, in_dim=embeddings.shape[1]).to(device)
    if XAVIER_INIT:
        initialize_tiger_weights(model)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False, broadcast_buffers=False)

    # The ranking needs to know which item every embedding belongs to, so the
    # batch carries the global item id next to it.
    train_item_ids = torch.from_numpy(train_ids.astype(np.int64))

    class _ItemDataset(torch.utils.data.Dataset):
        """The training items: their embedding and their global id."""

        def __init__(self, embeddings: torch.Tensor, ids: torch.Tensor):
            self.embeddings = embeddings
            self.ids = ids

        def __len__(self) -> int:
            return int(self.embeddings.shape[0])

        def __getitem__(self, index: int):
            return self.embeddings[index], self.ids[index]

    train_dataset = _ItemDataset(train_embeddings, train_item_ids)
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
        optimizer = torch.optim.Adagrad([p for p in model.parameters() if p.requires_grad],
                                        lr=LR, weight_decay=WEIGHT_DECAY)
    else:
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                      lr=LR, weight_decay=WEIGHT_DECAY)
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
        # DDP broadcasts the parameters when it is constructed, which happened
        # before this call; the KMeans start is written after it, so the ranks
        # would keep different codebooks and assign different SIDs. Each rank
        # computes the same KMeans and the same Frechet step, but the start is
        # broadcast anyway so a rank whose KMeans converged elsewhere cannot
        # diverge on the first step.
        for layer in raw_module.rq.vq_layers:
            dist.broadcast(layer.get_code_embs().data, 0)
        dist.barrier()

    if USE_GEOMETRY:
        # The curvature matches the data (Issue #138 C): the ball radius is set by
        # the measured level-1 residual norms, and every level selects its code by
        # that Poincare cost in training and in the export pass alike.
        with torch.no_grad():
            for layer in raw_module.rq.vq_layers:
                layer._skip_ddp_reduce = True
            try:
                raw_module(torch.from_numpy(embeddings[:2048]).to(device))
            finally:
                for layer in raw_module.rq.vq_layers:
                    layer._skip_ddp_reduce = False
        MECHANISM_CURVATURE = curvature_prefix.data_curvature(
            raw_module.level_points()[0].norm(dim=-1))
        for layer in raw_module.rq.vq_layers:
            layer.curvature = MECHANISM_CURVATURE
        if rank == 0:
            print(f"[RQ-VAE] measured curvature K={MECHANISM_CURVATURE:.3f}", flush=True)
            _record(rank, "mechanism_curvature", value=MECHANISM_CURVATURE,
                    source="pre-training level-1 residuals")

    # Stage3 only consumes the final-epoch weights, so tracking the best
    # collision across epochs would be misleading; keep the last evaluation.
    final_collision = float("inf")
    final_raw_unique = None
    final_raw_unique = 0
    codebook_sizes = list(config.codebook_size)

    for epoch in range(1, EPOCHS + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch - 1)
        model.train()
        losses: list[float] = []
        recons: list[float] = []
        mechanism_terms: list[float] = []
        _epoch_t0 = time.time()
        for (batch, item_ids) in loader:
            batch = batch.to(device, non_blocking=True)
            item_ids = item_ids.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            reconstructed, quant_loss, _, tokens, reference = model(batch, item_ids)
            # === DDP 包装下, 通过 .module 调 compute_loss (避免 DDP all_reduce 干扰) ===
            loss, recon_loss = raw_module.compute_loss(reference, reconstructed, quant_loss)
            # === 候选机制: AF3 凝聚度引导的分层前缀排序 ===
            triples = triple_lookup.triples(item_ids.detach().cpu().numpy(),
                                            use_behaviour=USE_BEHAVIOUR,
                                            seed=curvature_prefix.RANDOM_TRIPLE_SEED + epoch)
            mechanism_value = 0.0
            points = raw_module.level_points()
            mechanism = None
            # The mechanism waits for an accepted partition: behaviour pressure on
            # an unstable quantizer collapses it (measured: L2 236, collision 0.877).
            if _ACCEPTED_STATE is not None:
                codebooks = raw_module.level_codebooks()
                mechanism = points[0].new_zeros(())
                if triples["anchor"].size:
                    anchor = torch.from_numpy(triples["anchor"]).to(device)
                    high = torch.from_numpy(triples["high"]).to(device)
                    low = torch.from_numpy(triples["low"]).to(device)
                    triple_weights = torch.from_numpy(triples["weight"]).to(device)
                    rows_anchor = _batch_positions(item_ids, anchor)
                    rows_high = _batch_positions(item_ids, high)
                    rows_low = _batch_positions(item_ids, low)
                    valid = (rows_anchor >= 0) & (rows_high >= 0) & (rows_low >= 0)
                    if bool(valid.any()):
                        mechanism = mechanism + curvature_prefix.curvature_prefix_ranking_loss(
                            points[0], points[1], points[2], codebooks,
                            rows_anchor[valid], rows_high[valid], rows_low[valid],
                            triple_weights[valid], levels=RANKING_LEVELS,
                            use_curvature=USE_GEOMETRY,
                            curvature=MECHANISM_CURVATURE or 1.0)
                        # The measured lever: the cohesive pair must land in ONE
                        # coarse code, not merely outrank the bridge pair.
                        anchor_levels = [points[0][rows_anchor[valid]]]
                        high_levels = [points[0][rows_high[valid]]]
                        mechanism = mechanism + CLUSTER_WEIGHT * curvature_prefix.prefix_cocluster_loss(
                            anchor_levels, high_levels, codebooks[:1], 0,
                            USE_GEOMETRY, MECHANISM_CURVATURE or 1.0)
                        anchor_levels = [points[0][rows_anchor[valid]],
                                         points[1][rows_anchor[valid]]]
                        high_levels = [points[0][rows_high[valid]],
                                       points[1][rows_high[valid]]]
                        mechanism = mechanism + CLUSTER_WEIGHT * curvature_prefix.prefix_cocluster_loss(
                            anchor_levels, high_levels, codebooks[:2], 1,
                            USE_GEOMETRY, MECHANISM_CURVATURE or 1.0)
                # Every item of the batch, not the sampled pairs: the coarse
                # bucket's fine codes are pushed to spread, weighted by the item's
                # mean AF3 cohesion.
                cohesion = torch.from_numpy(
                    triple_lookup.item_cohesion(item_ids.detach().cpu().numpy())
                ).float().to(device)
                buckets1 = tokens[:, 0]
                buckets2 = tokens[:, 0] * codebooks[1].shape[0] + tokens[:, 1]
                spread = (curvature_prefix.bucket_spread_loss(
                              points[1], codebooks[1], buckets1, cohesion, USE_GEOMETRY,
                              MECHANISM_CURVATURE or 1.0)
                          + curvature_prefix.bucket_spread_loss(
                              points[2], codebooks[2], buckets2, cohesion, USE_GEOMETRY,
                              MECHANISM_CURVATURE or 1.0))
                mechanism = mechanism + SPREAD_WEIGHT * spread
                mechanism = mechanism - USAGE_WEIGHT * curvature_prefix.usage_entropy(
                    points[0], codebooks[0], USE_GEOMETRY, MECHANISM_CURVATURE or 1.0)
            if mechanism is not None:
                if MECHANISM_SCALE is None:
                    parameters = [p for p in raw_module.parameters() if p.requires_grad]
                    mechanism_grads = torch.autograd.grad(
                        mechanism, parameters, retain_graph=True, allow_unused=True)
                    recon_grads = torch.autograd.grad(
                        recon_loss, parameters, retain_graph=True, allow_unused=True)
                    scale = curvature_prefix.gradient_budget_scale(
                        mechanism_grads, recon_grads, MECHANISM_GRADIENT_SHARE)
                    low_bound, high_bound = MECHANISM_SCALE_BOUNDS
                    MECHANISM_SCALE = min(max(scale, low_bound), high_bound)
                    _record(rank, "mechanism_scale", value=MECHANISM_SCALE,
                            recon=float(recon_loss.detach()),
                            mechanism=float(mechanism.detach()),
                            share=MECHANISM_GRADIENT_SHARE, epoch=epoch)
                loss = loss + MECHANISM_SCALE * mechanism
                mechanism_value = float(mechanism.detach())
            if not torch.isfinite(loss):
                raise RuntimeError(f"RQ-VAE loss became non-finite at epoch {epoch}")
            loss.backward()
            if GRADIENT_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()
            if _ACCEPTED_STATE is not None:
                # The coarse codebook is the mechanism's own object: it moves
                # towards the cohesion-weighted mean of the residuals assigned to
                # it, reduced across the ranks so the SID stays rank-independent.
                layer0 = raw_module.rq.vq_layers[0]
                cohesion_cpu = torch.from_numpy(
                    triple_lookup.item_cohesion(item_ids.detach().cpu().numpy())
                ).float().to(device)
                sums, wsum = curvature_prefix.cohesion_codebook_sums(
                    points[0].detach(), tokens[:, 0], cohesion_cpu,
                    layer0.get_code_embs().shape[0])
                if world_size > 1:
                    dist.all_reduce(sums, op=dist.ReduceOp.SUM)
                    dist.all_reduce(wsum, op=dist.ReduceOp.SUM)
                with torch.no_grad():
                    moved = curvature_prefix.apply_codebook_update(
                        layer0.get_code_embs().detach(), sums, wsum, CODEBOOK_STEP)
                    layer0.get_code_embs().data.copy_(moved)
            scheduler.step()
            losses.append(float(loss.detach()))
            recons.append(float(recon_loss.detach()))
            mechanism_terms.append(mechanism_value)
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
                print(f"[RQ-VAE] epoch={epoch} loss={avg_loss:.8f} recon={avg_recon:.8f} "
                      f"mech={float(np.mean(mechanism_terms)) if mechanism_terms else 0.0:.8f} "
                      f"lr={cur_lr:.3e} time={epoch_time_s}s", flush=True)
                _record(rank, "train", epoch=epoch, loss=avg_loss, recon=avg_recon,
                        mechanism=float(np.mean(mechanism_terms)) if mechanism_terms else 0.0,
                        mechanism_scale=MECHANISM_SCALE, lr=cur_lr,
                        epoch_time_s=epoch_time_s,
                        cumulative_time_s=round(time.time() - _T0, 3))
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
            accepted = torch.tensor(0, device=device)
            if rank == 0:
                with torch.no_grad():
                    raw_tokens = raw_module.get_indices(
                        all_embeddings.to(device), infer_use_sk=False,
                    ).cpu().numpy().astype(np.int64)
                counts = np.unique(raw_tokens, axis=0, return_counts=True)[1]
                raw_unique = int(counts.size)
                # The gate's definition: the share of items whose three-token
                # prefix is shared with another item (not 1 - unique/n, which
                # under-reports uneven buckets).
                collision_v = float(counts[counts > 1].sum()) / len(raw_tokens)
                decision = curvature_prefix.collision_budget_decision(
                    collision_v, COLLISION_BUDGET)
                accepted.fill_(1 if decision == "accept" else 0)
            # EVERY rank joins the collective. A broadcast issued by rank 0 alone
            # leaves the NCCL communicator out of step and corrupts the following
            # gradient all-reduces: the measured symptom was a partition that was
            # healthy at epoch 50 and collapsed right after the first eval.
            if world_size > 1:
                dist.broadcast(accepted, 0)
            decision = "accept" if int(accepted.item()) == 1 else "rollback"
            if decision == "accept":
                _ACCEPTED_STATE = {name: value.detach().cpu().clone()
                                   for name, value in raw_module.state_dict().items()}
                if rank == 0:
                    final_collision = collision_v
                    final_raw_unique = raw_unique
                    _record(rank, "collision_budget", decision="accept",
                            collision=float(collision_v), budget=COLLISION_BUDGET,
                            epoch=epoch, scale=MECHANISM_SCALE)
            else:
                if _ACCEPTED_STATE is not None:
                    raw_module.load_state_dict(_ACCEPTED_STATE)
                if rank == 0:
                    _record(rank, "collision_budget", decision="rollback",
                            collision=float(collision_v), budget=COLLISION_BUDGET,
                            epoch=epoch, scale=MECHANISM_SCALE)
            if rank == 0:
                print(
                    f"[RQ-VAE] epoch={epoch} loss={avg_loss:.8f} recon={avg_recon:.8f} "
                    f"mech={float(np.mean(mechanism_terms)) if mechanism_terms else 0.0:.8f} "
                    f"lr={cur_lr:.3e} raw_unique={raw_unique}/{len(raw_tokens)} "
                    f"collision={collision_v:.6f} time={epoch_time_s}s",
                    flush=True,
                )
                _record(
                    rank, "train", epoch=epoch, loss=avg_loss, recon=avg_recon, lr=cur_lr,
                    epoch_time_s=epoch_time_s,
                    cumulative_time_s=round(time.time() - _T0, 3),
                    raw_unique=raw_unique, raw_total=len(raw_tokens), collision=collision_v,
                    mechanism=float(np.mean(mechanism_terms)) if mechanism_terms else 0.0,
                    mechanism_scale=MECHANISM_SCALE,
                )
        finally:
            for layer in raw_module.rq.vq_layers:
                layer._skip_ddp_reduce = False
            model.train()
        if world_size > 1:
            dist.barrier()

    # === 训练结束: only rank 0 保存终点 checkpoint 并导出 SID ===
    # Stage3 只接受训练终点权重；不保留任何中间或"最优 collision"快照。
    if rank == 0:
        if _ACCEPTED_STATE is not None:
            raw_module.load_state_dict(_ACCEPTED_STATE)
            _record(rank, "accepted_state_restored",
                    collision=final_collision, epoch=EPOCHS)
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
                raw_tokens = raw_module.get_indices(
                    all_embeddings.to(device), infer_use_sk=False,
                    item_ids=torch.arange(len(embeddings), device=device),
                ).cpu().numpy().astype(np.int64)
        finally:
            for layer in raw_module.rq.vq_layers:
                layer._skip_ddp_reduce = False
        # The exported SID is the model's own assignment, unchanged: the owner's
        # collision gate must be satisfied by training, never by post-processing.
        cohesion = triple_lookup.item_cohesion(np.arange(len(embeddings)))
        sid = _extend_collisions(raw_tokens, codebook_sizes, priority=cohesion)
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
        # The frozen contract's single-card branch: one card, one process, the
        # four-card path untouched. The framework's smoke and its single-card
        # check both use it.
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = _LAUNCHER["visible_dev"]
    env["NCCL_IB_DISABLE"]     = "1"
    env["NCCL_P2P_DISABLE"]    = "1"
    env["NCCL_SHM_DISABLE"]    = "1"
    env["NCCL_TIMEOUT"]        = "3600"
    env["TORCH_NCCL_BLOCKING_WAIT"] = "1"
    # The four ranks are started by the framework's launcher, not by this file:
    # the port, the world size, the working directory and the log belong to the
    # frozen contract, so that no candidate has to re-implement them.
    sys.path.insert(0, str(REPO_ROOT))
    from tools.research.stage2 import contract as stage2_contract
    from tools.research.stage2 import launcher as stage2_launcher
    rc = stage2_launcher.launch(
        entry=_LAUNCHER["script"], result_dir=RESULT_DIR,
        port=stage2_contract.STAGE2_PORT, world_size=int(_LAUNCHER["nproc"]),
        python=_LAUNCHER["torchrun"].rsplit("/", 1)[0] + "/python3.9",
        env_extra={k: env[k] for k in ("CUDA_VISIBLE_DEVICES", "NCCL_IB_DISABLE",
                                       "NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE",
                                       "NCCL_TIMEOUT", "TORCH_NCCL_BLOCKING_WAIT")})
    sys.exit(rc)


if __name__ == "__main__":
    _launch_via_torchrun()
    main()
