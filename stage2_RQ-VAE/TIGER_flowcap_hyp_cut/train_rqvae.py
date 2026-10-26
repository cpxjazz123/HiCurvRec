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
        if (parent / "stage2_RQ-VAE").is_dir():
            return parent
    raise RuntimeError("cannot locate the repository root above " + str(here))


REPO_ROOT = _repository_root()
# The arm name is this tree's promoted name; a candidate copy changes only this line.
ARM_NAME = "TIGER_flowcap_hyp_cut"
RESULT_DIR = REPO_ROOT / "results" / "stage2_RQ-VAE" / ARM_NAME

EMBEDDING_FILE = REPO_ROOT / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
TRAIN_FILE     = REPO_ROOT / "results" / "stage0_build_parquet" / "train.parquet"
OUTPUT_SID     = RESULT_DIR / "sids_for_hgrec.npy"
OUTPUT_JSON    = RESULT_DIR / "item_sids.json"
CHECKPOINT     = RESULT_DIR / "rqvae_best.pth"
LOG_DIR        = RESULT_DIR / "logs"
METRICS_PATH   = LOG_DIR / "training_metrics.jsonl"


# === 硬编码超参 (原 argparse default 值, 保留原 TIGER 默认) ===
EPOCHS              = 3000
BATCH_SIZE_PER_RANK = 1024       # 每卡 batch, 总 batch = 1024 * 4 = 4096
LR                  = 1e-3
WEIGHT_DECAY        = 1e-4
HIDDEN_SIZES        = (512, 256, 128)
CODEBOOK_NUM        = 3
CODEBOOK_SIZE       = (16, 96, 256)
CODEBOOK_DIM        = 32
BETA                = 0.25
VQ_TYPE             = "vq"
EMA_DECAY           = 0.99
SK_EPSILON          = 0.003
SK_ITERS            = 50
FLOW_WEIGHT         = 0.05
DISTILL_WEIGHT = 0.5        # behaviour-induced assignment distillation
COST_WEIGHT = 0.05           # expected curved assignment cost
HYPERBOLIC_CURVATURE = 0.5
BEHAVIOR_ON = True
LORENTZ_GEOMETRY_ON = True
SINGLE_GPU_ENV = "GENREC_SINGLE_GPU"
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


def balanced_assignments(logits: torch.Tensor, iterations: int = 20) -> torch.Tensor:
    """Entropic assignments with equal column mass over the supplied whole set."""
    q = torch.exp((logits - logits.max(dim=1, keepdim=True).values).clamp(min=-40, max=0)).T
    q = q / q.sum().clamp_min(1e-12)
    rows, cols = q.shape
    for _ in range(iterations):
        q = q / q.sum(dim=1, keepdim=True).clamp_min(1e-12)
        q = q / rows
        q = q / q.sum(dim=0, keepdim=True).clamp_min(1e-12)
        q = q / cols
    q = q * rows
    return q.T.clamp_min(1e-12)


def lorentz_points(v: torch.Tensor, curvature: float, enabled: bool) -> torch.Tensor:
    if not enabled:
        return v
    r = torch.linalg.vector_norm(v, dim=-1, keepdim=True).clamp_min(1e-8)
    root = curvature ** 0.5
    spatial = v * (torch.tanh(root * r) / (root * r))
    time = torch.sqrt((1.0 / curvature + spatial.square().sum(-1, keepdim=True)).clamp_min(1e-8))
    return torch.cat((time, spatial), dim=-1)


def curved_sq(a: torch.Tensor, b: torch.Tensor, curvature: float, enabled: bool) -> torch.Tensor:
    if not enabled:
        return (a - b).square().sum(-1)
    mink = -a[..., :1] * b[..., :1] + (a[..., 1:] * b[..., 1:]).sum(-1, keepdim=True)
    return (torch.acosh((curvature * -mink).clamp_min(1.0 + 1e-6)).square() / curvature).squeeze(-1)


def flow_capacity_loss(encoded, item_ids, all_embeddings, neighbors, raw_module, training_item_ids=None):
    """Sparse-edge flow loss with global training-catalog Sinkhorn capacities.

    A fresh full training-item assignment pass is used each step (exact global
    mass constraints for the current parameters; O(NK) memory/work). Edge scoring
    remains restricted to retained directed successors.
    """
    layers = raw_module.rq.vq_layers
    kappa = HYPERBOLIC_CURVATURE
    training_item_ids = item_ids if training_item_ids is None else training_item_ids
    catalog_ids = torch.as_tensor(training_item_ids, device=encoded.device, dtype=torch.long)
    catalog_x = raw_module.encoder(all_embeddings[catalog_ids.detach().cpu()].to(encoded.device))
    x1 = lorentz_points(catalog_x, kappa, LORENTZ_GEOMETRY_ON)
    c1 = lorentz_points(layers[0].get_code_embs(), kappa, LORENTZ_GEOMETRY_ON)
    p1 = balanced_assignments(-curved_sq(x1[:, None, :], c1[None, :, :], kappa, LORENTZ_GEOMETRY_ON) / 0.2)
    catalog_pos = {int(item): pos for pos, item in enumerate(catalog_ids.detach().cpu().tolist())}
    # Six child prototypes per parent: T2 prototypes are explicitly conditioned on T1.
    child = layers[1].get_code_embs().reshape(16, 6, -1)
    c2 = lorentz_points(layers[0].get_code_embs()[:, None, :] + child, kappa, LORENTZ_GEOMETRY_ON)
    resid = catalog_x[:, None, :] - layers[0].get_code_embs()[None, :, :]
    x2 = lorentz_points(resid, kappa, LORENTZ_GEOMETRY_ON)
    d1_sum = curved_sq(x1[:, None, :], c1[None, :, :], kappa, LORENTZ_GEOMETRY_ON)
    d2 = curved_sq(x2[:, :, None, :], c2[None, :, :, :], kappa, LORENTZ_GEOMETRY_ON)
    p2local = torch.stack([balanced_assignments(-d2[:, parent, :] / 0.2) for parent in range(16)], dim=1)
    p2 = p1[:, :, None] * p2local
    if not BEHAVIOR_ON or not neighbors:
        return encoded.sum() * 0.0
    positions = {int(item): pos for pos, item in enumerate(item_ids.tolist())}
    edges = [(positions[s], d, w) for s in positions for d, w in neighbors.get(s, ()) if d in catalog_pos]
    if not edges:
        return encoded.sum() * 0.0
    src = torch.tensor([e[0] for e in edges], device=encoded.device)
    dst = torch.tensor([e[1] for e in edges], device=encoded.device)
    weights = torch.tensor([e[2] for e in edges], device=encoded.device, dtype=encoded.dtype)
    # Successor assignments are selections from the same whole-catalog
    # constrained distributions; do not rebalance the sparse edge minibatch.
    # Expected Lorentz inner product is bilinear, so marginal assignment mass
    # can be reduced to one assignment-weighted prototype per edge. This avoids
    # the E x K x K geodesic-distance tensor. The edge score remains genuinely
    # curved: -<x,y>_L is the Lorentz cosine-hyperbolic argument (with the
    # constant curvature scale); assignments above are themselves based on
    # Lorentz geodesic distances. Cost is O(E*K*D), not O(E*K^2).
    def expected_lorentz_score(left: torch.Tensor, right: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        right_prototypes = right @ codes
        left_prototypes = left @ codes
        minkowski = -(left_prototypes[:, :1] * right_prototypes[:, :1]).squeeze(-1)
        minkowski = minkowski + (left_prototypes[:, 1:] * right_prototypes[:, 1:]).sum(-1)
        return -(minkowski * kappa).mul(weights).sum() / weights.sum().clamp_min(1e-12)

    src_global = torch.tensor([catalog_pos[int(item)] for item in item_ids.tolist()], device=encoded.device)
    dst_global = torch.tensor([catalog_pos[e[1]] for e in edges], device=encoded.device)
    flow = expected_lorentz_score(p1[src_global[src]], p1[dst_global], c1)
    flow = flow + expected_lorentz_score(p2[src_global[src]].reshape(-1, 96), p2[dst_global].reshape(-1, 96), c2.reshape(96, -1))
    # The balanced projection is row-normalised, so its own sum is a constant and the
    # score above reaches only the codebook. The model's own curved soft assignment has
    # to be pulled towards that behaviour-induced target for the encoder to learn the
    # coupling at all; without this term the measured gradient into `encoded` is zero.
    probs1 = torch.softmax(-curved_sq(x1[:, None, :], c1[None, :, :], kappa, LORENTZ_GEOMETRY_ON) / 0.2, dim=-1)
    distillation = -(p1.detach() * probs1.clamp_min(1e-9).log()).sum(-1).mean()
    # The expected curved assignment cost is a second gradient path that is non-zero at
    # initialisation, where both the balanced target and the softmax are still nearly
    # uniform and the distillation term alone has a vanishing derivative.
    expected_cost = (probs1 * d1_sum).sum(-1).mean()
    return flow + DISTILL_WEIGHT * distillation + COST_WEIGHT * expected_cost


@torch.no_grad()
def hardened_prefixes(encoded, raw_module):
    """Full-catalog balanced T1; within-parent balanced six-way T2."""
    layers = raw_module.rq.vq_layers
    kappa = HYPERBOLIC_CURVATURE
    c1 = layers[0].get_code_embs()
    x1 = lorentz_points(encoded, kappa, LORENTZ_GEOMETRY_ON)
    hc1 = lorentz_points(c1, kappa, LORENTZ_GEOMETRY_ON)
    p1 = balanced_assignments(-curved_sq(x1[:, None, :], hc1[None, :, :], kappa, LORENTZ_GEOMETRY_ON), 100)
    a1 = p1.argmax(-1)
    c2 = c1[:, None, :] + layers[1].get_code_embs().reshape(16, 6, -1)
    result = torch.empty((len(encoded), 2), dtype=torch.long, device=encoded.device)
    result[:, 0] = a1
    for parent in range(16):
        rows = (a1 == parent).nonzero().flatten()
        if not len(rows):
            continue
        residual = encoded[rows] - c1[parent]
        xx = lorentz_points(residual, kappa, LORENTZ_GEOMETRY_ON)
        cc = lorentz_points(c2[parent], kappa, LORENTZ_GEOMETRY_ON)
        probs = balanced_assignments(-curved_sq(xx[:, None, :], cc[None, :, :], kappa, LORENTZ_GEOMETRY_ON), 100)
        result[rows, 1] = probs.argmax(-1) + parent * 6
    return result


def residual_after_prefixes(encoded, prefixes, raw_module):
    """Subtract the exact parent-conditioned T1+T2 prototype used for assignment."""
    layers = raw_module.rq.vq_layers
    c1 = layers[0].get_code_embs()
    child = layers[1].get_code_embs().reshape(16, 6, -1)
    parent = prefixes[:, 0]
    global_t2 = prefixes[:, 1]
    local = global_t2 - parent * 6
    if torch.any((local < 0) | (local >= 6)):
        raise RuntimeError("T2 token is inconsistent with its selected T1 parent")
    selected = c1[parent] + child[parent, local]
    # Guard the exported token convention against accidentally dropping the parent.
    if not torch.equal(global_t2, parent * 6 + local):
        raise RuntimeError("T2 token does not identify the selected parent's child prototype")
    return encoded - selected


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

    train_dataset = TensorDataset(train_embeddings, torch.from_numpy(train_ids.astype(np.int64)))
    # Directed successors are counted only from training histories and targets.
    transitions: dict[int, dict[int, int]] = {}
    for row in train_frame.itertuples(index=False):
        seq = [int(v) for v in np.asarray(row.history).reshape(-1)] + [int(row.target)]
        for src, dst in zip(seq[:-1], seq[1:]):
            bucket = transitions.setdefault(src, {})
            bucket[dst] = bucket.get(dst, 0) + 1
    flow_neighbors = {}
    for src, edges in transitions.items():
        retained = sorted(edges.items(), key=lambda kv: -kv[1])[:16]
        total = float(sum(count for _, count in retained))
        if total:
            flow_neighbors[src] = tuple((dst, count / total) for dst, count in retained)
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
        _epoch_t0 = time.time()
        for batch, batch_item_ids in loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            reconstructed, quant_loss, _, _ = model(batch)
            # === DDP 包装下, 通过 .module 调 compute_loss (避免 DDP all_reduce 干扰) ===
            loss, recon_loss = raw_module.compute_loss(batch, reconstructed, quant_loss)
            # Coupled directed-flow/Lorentz pathway: soft assignments are computed
            # from hyperbolic prototype distances, then successor mass scores
            # the corresponding hyperbolic code-to-code geodesics.
            if FLOW_WEIGHT:
                loss = loss + FLOW_WEIGHT * flow_capacity_loss(raw_module.encoder(batch), batch_item_ids, all_embeddings, flow_neighbors, raw_module, train_ids)
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
                    encoded_all = raw_module.encoder(all_embeddings.to(device))
                    prefixes = hardened_prefixes(encoded_all, raw_module)
                    # Preserve baseline residual T3 quantizer after removing the new T1/T2 contribution.
                    residual = residual_after_prefixes(encoded_all, prefixes, raw_module)
                    t3 = raw_module.rq.vq_layers[2](residual, infer_use_sk=False)[3]
                    raw_tokens = torch.cat((prefixes, t3.reshape(-1, 1)), dim=1).cpu().numpy().astype(np.int64)
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
                encoded_all = raw_module.encoder(all_embeddings.to(device))
                prefixes = hardened_prefixes(encoded_all, raw_module)
                residual = residual_after_prefixes(encoded_all, prefixes, raw_module)
                t3 = raw_module.rq.vq_layers[2](residual, infer_use_sk=False)[3]
                raw_tokens = torch.cat((prefixes, t3.reshape(-1, 1)), dim=1).cpu().numpy().astype(np.int64)
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
