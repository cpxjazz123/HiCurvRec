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
from model.af3_hierarchy import (
    CURVE,
    GradientBudget,
    af3_weights,
    balance_loss,
    gradient_norm,
    hierarchical_loss,
    soft_assignments,
)


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
        if (parent / "stage2_RQ-VAE").is_dir():
            return parent
    raise RuntimeError("cannot locate the repository root above " + str(here))


REPO_ROOT = _repository_root()
# The arm name is this tree's promoted name; a candidate copy changes only this line.
ARM_NAME = "TIGER_af3_hierarchy"
RESULT_DIR = REPO_ROOT / "results" / "stage2_RQ-VAE" / ARM_NAME

EMBEDDING_FILE = REPO_ROOT / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
TRAIN_FILE     = REPO_ROOT / "results" / "stage0_build_parquet" / "train.parquet"
OUTPUT_SID     = RESULT_DIR / "sids_for_hgrec.npy"
OUTPUT_JSON    = RESULT_DIR / "item_sids.json"
CHECKPOINT     = RESULT_DIR / "rqvae_best.pth"
LOG_DIR        = RESULT_DIR / "logs"
METRICS_PATH   = LOG_DIR / "training_metrics.jsonl"

# The frozen AF3 table of the training behaviour graph (issue #136's convention,
# exported once by tools/research/science/community_curvature/export_edge_af3.py).
# The arm reads it and never computes a curvature of its own: two copies of the
# same formula can drift, and then the arm would be supervised by a curvature
# nobody validated.
AF3_PATH = (REPO_ROOT / "results" / "research" / "issue121"
            / "community_curvature" / "train_edge_af3.npz")
# The held-out behaviour pairs: the same frozen set the Stage-2 validity check
# uses, so the training record reports the criterion on the same sample.
VALID_PAIRS = (REPO_ROOT / "results" / "stage3_T5Train" / "monotonicity"
               / "behavior_pairs_valid_s42.npz")


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

# === AF3 分层监督 ===
# 前 RAMP_START_EPOCH 个 epoch 只训练原目标; 其后线性引入, 到 RAMP_END_EPOCH
# 达到 1. 四个 λ 取同一值: 它们的相对量级由下面的梯度预算统一约束, 因此不需要
# 逐项调参 (CLAUDE.md §19g 要求常数可推导).
HIER_START_EPOCH    = 1500
HIER_END_EPOCH      = 3000
HIER_WEIGHT_LEVEL1  = 1.0
HIER_WEIGHT_LEVEL2  = 1.0
HIER_WEIGHT_LEVEL3  = 1.0
HIER_WEIGHT_BALANCE = 1.0
# 新项对原目标的梯度贡献上限 (编码器与码本分别检查, 取更严格缩放).
HIER_GRADIENT_SHARE = 0.10
# 每步采样的边/对数量. 每条产生两行, 因此每步额外编码 8 x 2 x N 行; N=192 时
# 约 3072 行, 与一次 batch 前向同量级.
HIER_EDGES_PER_STEP  = 192
HIER_PAIRS_PER_STEP  = 192
# 每 100 个 epoch 在导出的码上计算一次留出行为 AUC (判据 1).
AUC_INTERVAL        = 100
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


def prepare_af3_supervision(all_embeddings: torch.Tensor, n_items: int,
                             device: torch.device) -> dict:
    """The frozen AF3 table, read once, plus the sampling pools it drives.

    Everything comes from the training split: the AF3 table was exported from
    the train graph by the issue-#136 convention, and the pairs below are drawn
    from it. No validation or test interaction is read.
    """
    if not AF3_PATH.is_file():
        raise RuntimeError(
            "the frozen AF3 table is missing: " + str(AF3_PATH)
            + "; regenerate it with tools/research/science/community_curvature/"
              "export_edge_af3.py")
    payload = np.load(AF3_PATH)
    rows = torch.from_numpy(payload["rows"].astype(np.int64))
    cols = torch.from_numpy(payload["cols"].astype(np.int64))
    af3 = torch.from_numpy(payload["af3"].astype(np.float32))
    median = float(payload["median"])
    p90 = float(payload["p90"])

    weight = af3_weights(af3, median, p90)
    dense = weight > 0
    return {
        "rows": rows.to(device),
        "cols": cols.to(device),
        "af3_weight": weight.to(device),
        "dense_rows": rows[dense].to(device),
        "dense_cols": cols[dense].to(device),
        "dense_weight": weight[dense].to(device),
        "all_embeddings": all_embeddings,
        "n_items": n_items,
        "n_edges": int(rows.shape[0]),
        "n_dense": int(dense.sum()),
        "af3_median": median,
        "af3_p90": p90,
    }


def sample_hierarchical_pairs(state: dict, rng) -> dict:
    """Draw the four pools of pairs the losses consume.

    ``af3``: edges the curvature calls locally dense, weighted by how dense.
    ``positive``: behaviour edges, whatever their curvature.
    ``negative``: pairs that are NOT an edge; the level-2 term keeps only those
    that share a level-1 group, which is what makes it a refinement term.
    ``unrelated``: random pairs, for the level-3 hinge.
    """
    dense_rows = state["dense_rows"]
    n_dense = dense_rows.shape[0]
    af3_choice = torch.from_numpy(rng.integers(0, n_dense, size=HIER_EDGES_PER_STEP))
    af3_pairs = torch.stack([state["dense_rows"][af3_choice],
                             state["dense_cols"][af3_choice]], dim=1)
    af3_weight = state["dense_weight"][af3_choice]

    edge_choice = torch.from_numpy(rng.integers(0, state["rows"].shape[0],
                                                size=HIER_PAIRS_PER_STEP * 4))
    positive = torch.stack([state["rows"][edge_choice],
                            state["cols"][edge_choice]], dim=1)

    # A behaviour edge lands inside one level-1 group about 8% of the time, so
    # the pools are drawn wider than what the term keeps: the level-2 term needs
    # pairs that already share the group to have anything to refine.
    random_pairs = torch.from_numpy(rng.integers(0, state["n_items"],
                                                 size=(HIER_PAIRS_PER_STEP * 4, 2)))
    negative = random_pairs[:HIER_PAIRS_PER_STEP * 2]
    unrelated = random_pairs[HIER_PAIRS_PER_STEP * 2:]
    device = dense_rows.device
    return {"af3": af3_pairs.to(device), "af3_weight": af3_weight.to(device),
            "positive": positive.to(device), "negative": negative.to(device),
            "unrelated": unrelated.to(device)}


def evaluate_exported_behaviour(raw_module, tokens: torch.Tensor,
                                 device: torch.device) -> dict:
    """The held-out behaviour criterion, on the identifiers as exported.

    ``q`` is the sum of the three codewords at the exported codes — the same
    representation the Stage-3 arm will read — and the criterion is how well its
    cosine separates a held-out interaction pair from a random one. It is
    computed inside training so the record shows whether the mechanism is
    helping, instead of leaving it to a separate study.
    """
    pairs = np.load(VALID_PAIRS)
    left = torch.from_numpy(pairs["anchors"]).to(device)
    right = torch.from_numpy(pairs["positives"]).to(device)
    rng = np.random.default_rng(SEED)
    negative = torch.from_numpy(rng.integers(0, tokens.shape[0], size=left.shape[0])).to(device)
    with torch.no_grad():
        codes = tokens.to(device)
        quantised = torch.zeros(codes.shape[0], raw_module.rq.codebook_dim, device=device)
        for level, layer in enumerate(raw_module.rq.vq_layers):
            quantised = quantised + torch.nn.functional.embedding(
                codes[:, level], layer.get_code_embs())
        unit = torch.nn.functional.normalize(quantised, dim=-1)
        positive = (unit[left] * unit[right]).sum(dim=-1).cpu().numpy()
        unrelated = (unit[left] * unit[negative]).sum(dim=-1).cpu().numpy()
        random_left = torch.from_numpy(rng.integers(0, codes.shape[0], size=20000)).to(device)
        random_right = torch.from_numpy(rng.integers(0, codes.shape[0], size=20000)).to(device)
        background = (unit[random_left] * unit[random_right]).sum(dim=-1).mean().item()
    labels = np.concatenate([np.ones_like(positive), np.zeros_like(unrelated)])
    scores = np.concatenate([positive, unrelated])
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1)
    positives = labels.sum()
    auc = float((ranks[labels == 1].sum() - positives * (positives + 1) / 2)
                / (positives * (labels.size - positives)))
    return {"behaviour_auc": auc,
            "behaviour_positive_cos": float(positive.mean()),
            "behaviour_negative_cos": float(unrelated.mean()),
            "behaviour_background_cos": float(background)}


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

    train_dataset = TensorDataset(train_embeddings,
                                   torch.from_numpy(train_ids))
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

    # === AF3 分层监督: 冻结的 AF3 表与采样池 ===
    af3_state = prepare_af3_supervision(all_embeddings, len(all_embeddings), device)
    if rank == 0:
        _record(rank, "af3_prepared", edges=af3_state["n_edges"],
                dense_edges=af3_state["n_dense"],
                af3_median=af3_state["af3_median"], af3_p90=af3_state["af3_p90"],
                source=str(AF3_PATH))
        print("[AF3] edges=%d dense=%d median=%.1f p90=%.1f"
              % (af3_state["n_edges"], af3_state["n_dense"],
                 af3_state["af3_median"], af3_state["af3_p90"]), flush=True)

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
    budget = GradientBudget(share=HIER_GRADIENT_SHARE)
    sampling = np.random.default_rng(SEED)

    for epoch in range(1, EPOCHS + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch - 1)
        # 前 HIER_START_EPOCH 个 epoch 只训练原目标; 其后线性引入分层监督.
        ramp = min(1.0, max(0.0, (epoch - HIER_START_EPOCH)
                            / max(1, HIER_END_EPOCH - HIER_START_EPOCH)))
        model.train()
        losses: list[float] = []
        recons: list[float] = []
        hier_log: list[tuple] = []
        _epoch_t0 = time.time()
        for step, (batch, batch_ids) in enumerate(loader):
            batch = batch.to(device, non_blocking=True)
            batch_ids = batch_ids.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            reconstructed, quant_loss, _, _ = model(batch, batch_ids)
            # === DDP 包装下, 通过 .module 调 compute_loss (避免 DDP all_reduce 干扰) ===
            loss, recon_loss = raw_module.compute_loss(batch, reconstructed, quant_loss)
            if not torch.isfinite(loss):
                raise RuntimeError(f"RQ-VAE loss became non-finite at epoch {epoch}")

            if ramp > 0.0:
                # === AF3 分层监督 ===
                # 采样的边与对不在 batch 里, 先把它们的两个端点各编码一次.
                codebooks = [layer.get_code_embs() for layer in raw_module.rq.vq_layers]
                picked = sample_hierarchical_pairs(af3_state, sampling)
                endpoints = torch.cat([picked["af3"], picked["positive"],
                                       picked["negative"], picked["unrelated"]], dim=0)
                flat = af3_state["all_embeddings"][endpoints.reshape(-1).cpu()].to(device)
                latent = raw_module.encoder(flat)

                def pair_rows(offset_pairs: int, count: int) -> tuple[torch.Tensor, torch.Tensor]:
                    """The left/right endpoint rows of `count` pairs starting at
                    `offset_pairs`: pair k occupies the consecutive rows 2k, 2k+1."""
                    base = torch.arange(offset_pairs, offset_pairs + count, device=device) * 2
                    return base, base + 1

                n_af3 = int(picked["af3"].shape[0])
                n_pos = int(picked["positive"].shape[0])
                n_neg = int(picked["negative"].shape[0])
                n_unrel = int(picked["unrelated"].shape[0])
                af3_left, af3_right = pair_rows(0, n_af3)
                pos_left, pos_right = pair_rows(n_af3, n_pos)
                neg_left, neg_right = pair_rows(n_af3 + n_pos, n_neg)
                unrel_left, unrel_right = pair_rows(n_af3 + n_pos + n_neg, n_unrel)

                weights = {"level1": HIER_WEIGHT_LEVEL1, "level2": HIER_WEIGHT_LEVEL2,
                           "level3": HIER_WEIGHT_LEVEL3, "balance": HIER_WEIGHT_BALANCE}
                total, parts = hierarchical_loss(
                    latent, codebooks,
                    {
                        "af3_left": af3_left, "af3_right": af3_right,
                        "af3_weight": picked["af3_weight"],
                        "positive_left": pos_left, "positive_right": pos_right,
                        "negative_left": neg_left, "negative_right": neg_right,
                        "unrelated_left": unrel_left, "unrelated_right": unrel_right,
                    }, weights)
                if torch.isfinite(total):
                    encoder_parameters = list(raw_module.encoder.parameters())
                    base_grads = (gradient_norm(loss, encoder_parameters),
                                  gradient_norm(loss, codebooks))
                    new_grads = (gradient_norm(total, encoder_parameters),
                                 gradient_norm(total, codebooks))
                    budget.observe(base_encoder=base_grads[0], new_encoder=new_grads[0],
                                   base_codebook=base_grads[1], new_codebook=new_grads[1])
                    scale_tensor = torch.tensor([budget.scale], device=device)
                    if world_size > 1:
                        dist.all_reduce(scale_tensor, op=dist.ReduceOp.MIN)
                    scale = float(scale_tensor.item())
                    loss = loss + ramp * scale * total
                    hier_log.append((parts["level1"], parts["level2"], parts["level3"],
                                     parts["balance"], scale))
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
        hier_fields = {}
        if hier_log:
            hier_fields = {
                "hier_level1": float(np.mean([row[0] for row in hier_log])),
                "hier_level2": float(np.mean([row[1] for row in hier_log])),
                "hier_level3": float(np.mean([row[2] for row in hier_log])),
                "hier_balance": float(np.mean([row[3] for row in hier_log])),
                "hier_scale": float(np.mean([row[4] for row in hier_log])),
                "ramp": ramp,
            }

        # === 非 eval 间隔: 只 log train 行 ===
        if epoch % EVAL_INTERVAL != 0 and epoch != EPOCHS:
            if rank == 0:
                print(f"[RQ-VAE] epoch={epoch} loss={avg_loss:.8f} recon={avg_recon:.8f} lr={cur_lr:.3e} time={epoch_time_s}s", flush=True)
                _record(rank, "train", epoch=epoch, loss=avg_loss, recon=avg_recon, lr=cur_lr, epoch_time_s=epoch_time_s, cumulative_time_s=round(time.time() - _T0, 3), **hier_fields)
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
                    raw_tokens = raw_module.get_indices(
                        all_embeddings.to(device), infer_use_sk=False
                    ).cpu().numpy().astype(np.int64)
                raw_unique = int(len(np.unique(raw_tokens, axis=0)))
                collision_v = 1.0 - raw_unique / len(raw_tokens)
                final_collision = collision_v
                final_raw_unique = raw_unique
                # === 每层的码使用率 + 判据 1: 导出码上的留出行为 AUC ===
                # 用最终导出的三层码组装量化表示, 而不是训练期的软分配:
                # 判据必须量的就是那张将被 Stage3 消费的 SID.
                level_usage = [int(len(np.unique(raw_tokens[:, level])))
                               for level in range(raw_tokens.shape[1])]
                criterion = {}
                if epoch % AUC_INTERVAL == 0 and VALID_PAIRS.is_file():
                    unit = evaluate_exported_behaviour(
                        raw_module, torch.from_numpy(raw_tokens), device)
                    criterion = unit
                criterion["level_code_usage"] = level_usage
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
                    **criterion, **hier_fields,
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
