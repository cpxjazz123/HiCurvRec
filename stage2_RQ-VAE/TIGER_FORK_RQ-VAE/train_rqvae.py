"""Exp2: behaviour-fork-aware ball representation, on top of the repaired code.

Design principle, taken from the repair round rather than assumed: the ball MAP
helped (+10.8%) while the geodesic DISTANCE hurt (-19.2%). The ball's radial and
angular coordinates are therefore used as supervision targets, while the
discriminative contrastive term keeps plain Euclidean distance so it does not
inherit the metric's mis-ordering of near candidates.

Real purchase behaviour used (train transitions history[-1] -> target):

  * out-degree d_out(i), the number of distinct successors of item i. Observed
    distribution: median 7, p90 25, max 1295; 33.7% of items fork to >=10 and
    2.2% are terminal.
  * successor-set overlap, which is a rare and therefore strong signal: only
    2.9% of item pairs share any successor, so "shares a successor" identifies a
    behaviour branch while "does not" is uninformative (97.1% of pairs) and is
    deliberately NOT used as a separation criterion.

Euclidean limitation targeted: nothing in a raw latent fixes what its norm means,
so an item that leads to many others and an item that ends a purchase occupy no
distinguishable radial position, and the sharing structure has no dedicated
coordinate.

Hyperbolic property exploited: the ball gives a bounded radius (general items
near the origin, specific near the boundary) and directions that separate
angularly, so a fork becomes an angular spread at a fixed radius rather than an
unbounded magnitude.

Two additive terms, both parameter-free:

  L_radial = MSE(radius_fraction(i), rho*(i))
      rho* is a rank map of log1p(d_out) onto [RHO_LO, RHO_HI], decreasing: an
      item that leads to many others sits nearer the origin.
  L_angular = cosine contrastive, positives = pairs sharing >=1 successor
      items in the same behaviour branch get aligned directions; batch members
      with disjoint successor sets are the negatives.

Matched Euclidean control: identical loss FORMS, identical weights, identical
supervision, zero extra parameters in both arms; the control measures radius as
the raw latent norm rescaled to the same range and direction as the raw unit
vector. Only the coordinate system differs.

2026-09-22 改造 (CLAUDE.md §1 + §3):2026-09-22 改造 (CLAUDE.md §1 + §3):
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
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from tqdm import tqdm

from model import RQVAE
from hyperbolic import (
    ball_radius,
    pair_poincare_distance,
    poincare_distance,
    encode_ball,
    pair_poincare_distance,
    spatial_sq_distance,
    mobius_add,
    mobius_matvec,
)


# === 硬编码路径常量 (CLAUDE.md §1) ===
# Stage0 emits user-event parquet directly into results/stage0_build_parquet/.
EMBEDDING_FILE = Path(
    "/home/wlia0047/ar57/wenyu/GeneRec/stage1_GeneEmbedding/output/sentence_t5.npy"  # stage1
)
TRAIN_FILE     = Path(
    "/home/wlia0047/ar57/wenyu/GeneRec/results/stage0_build_parquet/train.parquet"   # stage0
)
# === 本轮实验臂 (改这一行切换 arm; 每个 arm 的提交状态即实际跑过的配置) ===
EXPERIMENT_ARM = "fork_radial_corr"
FORK_RADIAL_WEIGHT_DEFAULT = 0.05
_ARMS = {
    # Fork-structure term ablation. All arms keep the ball map, the Euclidean
    # discriminative term and the detached trace, so the variables are only the
    # two fork components and the metric used for grouping.
    #
    # fork_hyp_group / fork_euclid_group : full terms, grouping metric differs
    # fork_ang_only                      : radial weight 0, isolates the angular term
    # fork_radial_corr                   : radial term switched from absolute MSE to
    #                                      a scale-free correlation, because MSE on the
    #                                      absolute radius lost to latent-norm growth
    "fork_hyp_group":    {"group_distance": "poincare", "radial_mode": "mse",  "radial_weight": FORK_RADIAL_WEIGHT_DEFAULT},
    "fork_euclid_group": {"group_distance": "euclid",   "radial_mode": "mse",  "radial_weight": FORK_RADIAL_WEIGHT_DEFAULT},
    "fork_ang_only":     {"group_distance": "euclid",   "radial_mode": "off",  "radial_weight": 0.0},
    "fork_radial_corr":  {"group_distance": "euclid",   "radial_mode": "corr", "radial_weight": FORK_RADIAL_WEIGHT_DEFAULT},
}
_ARM = _ARMS[EXPERIMENT_ARM]
GROUP_DISTANCE = _ARM["group_distance"]      # metric used for the sibling term
RADIAL_MODE = _ARM["radial_mode"]            # "mse" | "corr" | "off"
FORK_RADIAL_WEIGHT = _ARM["radial_weight"]
CURVATURE = 1.0
TARGET_RADIUS_FRACTION = 0.7
RHO_LO = 0.35
RHO_HI = 0.92
FORK_ANGULAR_WEIGHT = 0.05
FORK_ANGULAR_TEMPERATURE = 0.1
FORK_SIBLINGS_PER_ITEM = 8
FORK_ANGULAR_MAX_PAIRS = 2048
DISTANCE_NORMALIZATION = "rms"
TEMPERATURE_CALIBRATION_ENABLED = True
TARGET_LOGIT_SPREAD = 5.0
GEOMETRY = "euclid"
USE_BALL = True
TRACE_GRADIENT = "detached"
_OUT = Path("/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_FORK_RQ-VAE") / EXPERIMENT_ARM
OUTPUT_SID     = _OUT / "sids_for_hgrec.npy"
OUTPUT_JSON    = _OUT / "item_sids.json"
CHECKPOINT     = _OUT / "rqvae_best.pth"
LOG_DIR        = Path("/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_FORK_RQ-VAE/logs") / EXPERIMENT_ARM
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
NUM_WORKERS         = 0  # 2026-09-24: 0 防 DDP fork storm (4 rank × 4 worker = 16 子进程在 barrier 后挂死)
EVAL_INTERVAL       = 50
# === 唯一新增机制: 历史 Iter48 (348aed182) 行为对比监督 ===
# 权重上限与温度沿用历史值; ramp 从 epoch RAMP_START_EPOCH 起线性升到
# BEHAVIOR_WEIGHT_MAX, 对应历史 20k/40k global step 的后半程。
BEHAVIOR_WEIGHT_MAX     = 0.20
BEHAVIOR_TEMPERATURE    = 0.07
RAMP_START_EPOCH        = 1500
PAIR_BATCH_SIZE_PER_RANK = 512   # 每卡 pair 数; source+target 单独一次前向
# === 实验 1 的两个开关 ===
# RESAMPLE_PAIRS=True 每 epoch 重建 pair 迭代器 (修复 cycle 缓存);
# False 复现基线的一次性 cycle 行为。
RESAMPLE_PAIRS          = True
# CLAUDE.md §2: Stage2 不设任何 early-stop — 固定跑满 EPOCHS,
# 候选是否采用完全交由下游 stage3 test_R@10 裁决.
# 保留 best-collision checkpoint 仅作为导出用的最后一次快照.


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


def behavior_weight(epoch: int) -> float:
    """iter48's delayed ramp, re-expressed in epochs over the 3000-epoch budget."""
    span = max(EPOCHS - RAMP_START_EPOCH, 1)
    alpha = min(max((epoch - RAMP_START_EPOCH) / span, 0.0), 1.0)
    return BEHAVIOR_WEIGHT_MAX * alpha


class TransitionPairs(torch.utils.data.Dataset):
    """The full ``history[-1] -> target`` stream, duplicates kept (iter48)."""

    def __init__(self, train_frame: pd.DataFrame) -> None:
        pairs: list[tuple[int, int]] = []
        for history, target in zip(
            train_frame["seen_history"].to_numpy(),
            train_frame["target"].to_numpy(dtype=np.int64),
        ):
            if history is None or len(history) == 0:
                continue
            pairs.append((int(history[-1]), int(target)))
        if not pairs:
            raise ValueError("Training parquet contains no history-target pairs")
        self.pairs = torch.tensor(pairs, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.pairs.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.pairs[index]


def _residuals_with_trace(
    raw_module: RQVAE, embeddings: torch.Tensor
) -> torch.Tensor:
    """Residual entering each quantization level, with a live gradient to the encoder.

    ``model/layers.py`` applies a straight-through estimator as
    ``x_q = x + (e - x).detach()``. Subtracting that from ``x`` cancels exactly:

        residual_1 = x - x_q = -(e - x).detach()

    so the residual is algebraically disconnected from ``x`` and the behaviour
    loss on levels 2 and 3 delivered *zero* gradient to the encoder. Measured on
    a real pair batch: level 1 gave an encoder gradient L1-norm of 46.4 with 8/8
    parameters non-zero, while levels 2 and 3 gave exactly 0.0 with 0/8, and the
    three-level mean loss carried exactly one third of the level-1 gradient.
    Two thirds of the intended supervision was inert.

    The fix is to build the trace the way the residual is actually defined --
    ``encoded - sum of the chosen codewords`` -- where each codeword is the raw
    lookup, detached because the codebook is not what the behaviour loss should
    move. Then ``d residual_k / d encoded = I`` and every level supervises the
    encoder.

    This touches only the trace used by the behaviour term. The quantizer, the
    reconstruction loss and the SID export keep using the original STE forward
    unchanged, so the RQ-VAE's own optimisation is untouched.
    """
    encoded = raw_module.encoder(embeddings)
    residual = encoded
    traces: list[torch.Tensor] = []
    for vq_layer in raw_module.rq.vq_layers:
        traces.append(residual)
        _, _, _, indices = vq_layer(residual)
        # Raw codeword lookup, detached: a constant offset, not a gradient path.
        codeword = vq_layer.embed_code(indices).detach()
        residual = residual - codeword
        if TRACE_GRADIENT == "detached":
            # The original iter48 / STE semantics: each level quantizes the
            # *detached* error of the level above, so only level 1 reaches the
            # encoder. Kept as an explicit arm because making levels 2 and 3
            # live turned out to collapse the codebook.
            residual = residual.detach()
    return torch.stack(traces[: raw_module.rq.codebook_num], dim=0)


def _geodesic_block(
    anchors: torch.Tensor,
    candidates: torch.Tensor,
    normalizer,
) -> torch.Tensor:
    """Distance matrix between two residual blocks under the arm's geometry.

    The whole point of this function is that anchors and candidates are mapped
    together, in ONE ``encode_ball`` call, so both halves are divided by the same
    normalizer and therefore live in the same ball. An earlier version mapped
    them in two separate calls, each with its own scale, which made the returned
    matrix meaningless as a distance. ``fix_euclid_raw`` skips the map entirely
    and measures the raw latents, which is the rebuilt baseline.

    ``normalizer`` is fixed before training starts, so it is a constant here and
    the map stays a pure function of the latent.
    """
    if USE_BALL:
        size = anchors.shape[0]
        mapped = encode_ball(torch.cat((anchors, candidates), dim=0), CURVATURE, normalizer)
        mapped_anchors, mapped_candidates = mapped[:size], mapped[size:]
    else:
        mapped_anchors, mapped_candidates = anchors, candidates
    if GEOMETRY == "poincare":
        return pair_poincare_distance(mapped_anchors, mapped_candidates, CURVATURE)
    # Floor before the sqrt. The same item can appear as an anchor in one pair
    # and as another pair's candidate, so two rows of this matrix describe the
    # identical mapped point and the squared distance is exactly 0 for 496 of
    # 262144 entries. sqrt'(0) is infinite, which propagates inf into the
    # the weights become NaN, and the level-3 Sinkhorn then raises inside
    # center_distance (whose guard actually fires on NaN, not on a constant
    # matrix). These rows are masked out of the cross-entropy anyway, so a
    # finite floor changes nothing except removing the inf.
    #
    # The Poincare branch never needed this because acosh is already floored at
    # 1 + ACOSH_EPS, which is why only the Euclidean arms crashed.
    return torch.sqrt(
        spatial_sq_distance(
            mapped_anchors[:, None, :], mapped_candidates[None, :, :]
        ).clamp_min(1e-12)
    )


def resolve_normalizer(raw_module: RQVAE, train_embeddings: torch.Tensor) -> float:
    """Fixed ball scale from training-set statistics, computed once.

    Maps the *median* latent norm to ``TARGET_RADIUS_FRACTION`` of the radius.

    The latent norm is heavy-tailed (median 0.0567, p99 0.164, max 0.418, i.e.
    max/median 7.4), so using the RMS or the mean as the base would put the bulk
    of the items well below the intended radius while a thin tail saturates at
    the boundary. Anchoring on the median gives a radius distribution with
    p10 0.620 / p50 0.700 / p90 0.843 / p99 0.987 and only 0.82% of items past
    0.99 R, which keeps the radial coordinate -- the dimension that carries
    hierarchy -- informative instead of collapsing it.

    The value is a constant for the whole run: it cannot depend on the batch, or
    identical items would receive different coordinates.
    """
    with torch.no_grad():
        norms = torch.linalg.vector_norm(raw_module.encoder(train_embeddings), dim=-1)
        median = float(torch.median(norms).clamp_min(1e-12))
    return float(np.arctanh(TARGET_RADIUS_FRACTION) / (CURVATURE ** 0.5) / median)


def behavior_contrastive_loss(
    residuals: torch.Tensor,
    source_ids: torch.Tensor,
    target_ids: torch.Tensor,
    normalizer,
    temperature: float | None = None,
) -> torch.Tensor:
    """iter48's three-level contrastive loss over residual vectors.

    ``residuals`` holds the pair batch with sources in the first half and
    targets in the second half, exactly as iter48 fed
    ``cat(source_emb, target_emb)`` through the quantizer.

    """
    batch_size = int(source_ids.shape[0])
    losses: list[torch.Tensor] = []
    for level in range(residuals.shape[0]):
        source = residuals[level, :batch_size]
        candidates = residuals[level, batch_size:]
        distances = _geodesic_block(source, candidates, normalizer)
        if DISTANCE_NORMALIZATION == "rms":
            rms = distances.detach().square().mean().sqrt().clamp_min(1e-8)
            distances = distances / rms
        logits = -distances / (BEHAVIOR_TEMPERATURE if temperature is None else temperature)
        # iter48's filter: drop candidates sharing the anchor's own target, then
        # keep only rows whose source and target differ.
        duplicate_targets = target_ids[:, None].eq(target_ids[None, :])
        duplicate_targets.fill_diagonal_(False)
        logits = logits.masked_fill(duplicate_targets, -torch.inf)
        valid = source_ids.ne(target_ids)
        if bool(valid.any()):
            labels = torch.arange(batch_size, device=source_ids.device)
            losses.append(F.cross_entropy(logits[valid], labels[valid]))
    if not losses:
        return residuals.sum() * 0.0
    return torch.stack(losses).mean()


def calibrate_temperature(
    raw_module: RQVAE,
    pair_batch: torch.Tensor,
    source_ids: torch.Tensor,
    target_ids: torch.Tensor,
    normalizer,
) -> float:
    """Solve for the temperature giving TARGET_LOGIT_SPREAD at initialisation.

    Same procedure for every arm, so the resolved value differs only because the
    arm's geometry produces a different distance distribution.
    """
    half = int(source_ids.shape[0])
    with torch.no_grad():
        residuals = _residuals_with_trace(raw_module, pair_batch)
        spreads = []
        for level in range(residuals.shape[0]):
            distances = _geodesic_block(
                residuals[level, :half], residuals[level, half:], normalizer
            )
            rms = distances.square().mean().sqrt().clamp_min(1e-8)
            spreads.append(distances / rms)
        stacked = torch.stack(spreads)  # (levels, half, half)
    low, high = 1e-3, 10.0
    for _ in range(60):
        mid = 0.5 * (low + high)
        logits = -stacked / mid
        spread = float((logits.max(-1).values - logits.min(-1).values).mean())
        if spread > TARGET_LOGIT_SPREAD:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def behaviour_out_degree(train_frame, n_items: int) -> np.ndarray:
    """Distinct successor count per item: the real fork size in the buy graph."""
    seen: list[set[int]] = [set() for _ in range(n_items)]
    for history, target in zip(
        train_frame["seen_history"].to_numpy(),
        train_frame["target"].to_numpy(dtype=np.int64),
    ):
        if history is None or len(history) == 0:
            continue
        seen[int(history[-1])].add(int(target))
    return np.asarray([len(v) for v in seen], dtype=np.int64)


def behaviour_siblings(train_frame, n_items: int, cap: int, seed: int) -> np.ndarray:
    """For each item, up to ``cap`` items sharing at least one successor.

    Successor-set overlap is the behaviour-branch signal. It is deliberately
    sparse: only 2.9% of item pairs share any successor, so an overlap is
    evidence of a shared branch while its absence carries almost no information
    and is not used.
    """
    successor: list[set[int]] = [set() for _ in range(n_items)]
    precursor: dict[int, list[int]] = {}
    for history, target in zip(
        train_frame["seen_history"].to_numpy(),
        train_frame["target"].to_numpy(dtype=np.int64),
    ):
        if history is None or len(history) == 0:
            continue
        source = int(history[-1])
        target = int(target)
        successor[source].add(target)
        precursor.setdefault(target, []).append(source)
    rng = np.random.default_rng(seed)
    siblings = np.full((n_items, cap), -1, dtype=np.int64)
    for item in range(n_items):
        if not successor[item]:
            continue
        # Anything that also leads to one of this item's successors is in the
        # same behaviour branch.
        pool: set[int] = set()
        for target in successor[item]:
            pool.update(precursor.get(target, ()))
        pool.discard(item)
        if not pool:
            continue
        candidates = sorted(pool)
        if len(candidates) > cap:
            picked = rng.choice(len(candidates), size=cap, replace=False)
            candidates = [candidates[i] for i in sorted(picked)]
        siblings[item, : len(candidates)] = candidates
    return siblings


def fork_geometry_terms(
    raw_module,
    batch_embeddings: torch.Tensor,
    row_ids: torch.Tensor,
    radius_target: torch.Tensor,
    siblings: torch.Tensor,
    normalizer,
    all_embeddings: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Radial hierarchy and angular branch-coherence terms for the fork arm.

    Both are parameter-free and act on whichever coordinate system the arm
    selected, so the ball arm and the raw control differ only in the coordinate
    system: same terms, same weights, same supervision, zero extra parameters.

    ``radius_target`` and ``siblings`` are indexed by the item universe, so they
    are gathered by ``row_ids``; sibling directions need embeddings for items
    that are not in this batch, which is why ``all_embeddings`` is passed in.
    The number of angular pairs is capped so the extra encoder forward stays
    bounded.
    """
    radius_target = radius_target[row_ids]
    batch_siblings = siblings[row_ids]
    latent = raw_module.encoder(batch_embeddings)

    # Both arms use the ball radius: the geometry switch for THIS experiment is
    # the metric inside the grouping term, not the coordinate system.
    z = encode_ball(latent, CURVATURE, normalizer)
    radius = (CURVATURE ** 0.5) * torch.linalg.vector_norm(z, dim=-1)
    radius = radius / ball_radius(CURVATURE)
    direction = latent / torch.linalg.vector_norm(
        latent, dim=-1, keepdim=True
    ).clamp_min(1e-12)

    # (A) radial hierarchy: behaviour forks nearer the origin.
    #
    # MSE on the absolute radius failed: the reconstruction objective drives the
    # latent norm up, the 0.05 weight could not oppose it, and every point
    # saturated at the boundary (rho_mean 1.0000 against a target mean of 0.617).
    # The mechanism's actual claim is an ORDERING, so the corrected form is a
    # scale-free correlation between radius and target, which cannot be
    # satisfied by pushing everything to a constant.
    if RADIAL_MODE == "off":
        radial_loss = torch.zeros((), device=radius.device)
    elif RADIAL_MODE == "corr":
        r_c = radius - radius.mean()
        t_c = radius_target - radius_target.mean()
        denom = (r_c.norm() * t_c.norm()).clamp_min(1e-8)
        radial_loss = 1.0 - (r_c * t_c).sum() / denom
    else:
        radial_loss = F.mse_loss(radius, radius_target)

    # (B) angular branch coherence: items sharing a successor align directions.
    valid = batch_siblings >= 0
    anchor_pos, sibling_pos = valid.nonzero(as_tuple=True)
    angular_loss = torch.zeros((), device=radius.device)
    positive_cosine = torch.zeros((), device=radius.device)
    n_pairs = int(anchor_pos.numel())
    if n_pairs > FORK_ANGULAR_MAX_PAIRS:
        keep = torch.randperm(n_pairs, device=radius.device)[:FORK_ANGULAR_MAX_PAIRS]
        anchor_pos, sibling_pos = anchor_pos[keep], sibling_pos[keep]
        n_pairs = int(anchor_pos.numel())
    if n_pairs > 0:
        sibling_ids = batch_siblings[anchor_pos, sibling_pos]
        sibling_latent = raw_module.encoder(
            all_embeddings[sibling_ids.cpu()].to(batch_embeddings.device)
        )
        anchor_dir = direction[anchor_pos]
        if GROUP_DISTANCE == "poincare":
            # Both sides mapped into the same ball by one call, so anchor and
            # sibling live in one space.
            z_anchor = encode_ball(
                latent[anchor_pos], CURVATURE, normalizer
            )
            z_sibling = encode_ball(sibling_latent, CURVATURE, normalizer)
            d_positive = poincare_distance(z_anchor, z_sibling, CURVATURE)
            d_all = pair_poincare_distance(
                z_anchor, encode_ball(latent, CURVATURE, normalizer), CURVATURE
            )
        else:
            z_anchor = encode_ball(latent[anchor_pos], CURVATURE, normalizer)
            z_all = encode_ball(latent, CURVATURE, normalizer)
            z_sibling = encode_ball(sibling_latent, CURVATURE, normalizer)
            d_positive = torch.linalg.vector_norm(z_anchor - z_sibling, dim=-1)
            d_all = torch.cdist(z_anchor, z_all, p=2)
        positive_cosine = -d_positive
        # The scale is fixed so the term cannot silently become a no-op or
        # dominate; distances are rescaled to unit RMS like the main term.
        rms = d_all.detach().square().mean().sqrt().clamp_min(1e-8)
        logits = -d_all / rms / FORK_ANGULAR_TEMPERATURE
        angular_loss = F.cross_entropy(logits, anchor_pos)
    stats = {
        "radius_mean": float(radius.detach().mean()),
        "radius_std": float(radius.detach().std()),
        "positive_distance_mean": float(positive_cosine.detach().mean()),
        "pairs_used": n_pairs,
        "target_radius_mean": float(radius_target.detach().mean()),
        "target_radius_std": float(radius_target.detach().std()),
    }
    return radial_loss, angular_loss, stats


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
    # iter48's full transition stream, drawn by its own sampler so the item
    # batch above keeps exactly the rows, order and Sinkhorn balancing the
    # baseline sees.
    pair_dataset = TransitionPairs(train_frame)
    # Behaviour-fork tables, built once from the train transitions.
    out_degree = behaviour_out_degree(train_frame, len(embeddings))
    log_degree = np.log1p(out_degree.astype(np.float64))
    order = np.argsort(np.argsort(log_degree))          # rank, ties broken stably
    rank_frac = order / max(len(order) - 1, 1)
    radius_target_np = RHO_HI - (RHO_HI - RHO_LO) * rank_frac
    radius_target = torch.from_numpy(radius_target_np.astype(np.float32))
    siblings_np = behaviour_siblings(
        train_frame, len(embeddings), FORK_SIBLINGS_PER_ITEM, SEED
    )
    fork_siblings = torch.from_numpy(siblings_np)
    config = _tokenizer_config()
    model = RQVAE(config, in_dim=embeddings.shape[1]).to(device)
    if XAVIER_INIT:
        initialize_tiger_weights(model)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # The second column carries each row's position in the item universe, which
    # is the index space the radius target and the sibling table use. TensorDataset
    # returns element i of every column, so the embedding batch is unchanged.
    train_row_ids = torch.arange(len(train_ids), dtype=torch.long)
    train_dataset = TensorDataset(train_embeddings, train_row_ids)
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

    # Independent sampler: a different seed offset keeps the pair stream from
    # being phase-locked to the item stream.
    pair_sampler = (
        DistributedSampler(
            pair_dataset, num_replicas=world_size, rank=rank,
            shuffle=True, seed=SEED + 1, drop_last=False,
        )
        if world_size > 1
        else None
    )
    pair_loader = DataLoader(
        pair_dataset,
        batch_size=PAIR_BATCH_SIZE_PER_RANK,
        shuffle=(pair_sampler is None),
        sampler=pair_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        drop_last=False,
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
        _record(rank, "train_start", epochs=EPOCHS, batch_size_per_rank=BATCH_SIZE_PER_RANK, total_batch_size=BATCH_SIZE_PER_RANK*world_size, lr=LR, weight_decay=WEIGHT_DECAY, hidden_sizes=list(HIDDEN_SIZES), codebook_num=CODEBOOK_NUM, codebook_size=list(CODEBOOK_SIZE), codebook_dim=CODEBOOK_DIM, beta=BETA, vq_type=VQ_TYPE, ema_decay=EMA_DECAY, sk_epsilon=SK_EPSILON, sk_iters=SK_ITERS, pca_dim=PCA_DIM, xavier_init=XAVIER_INIT, warmup_epochs=WARMUP_EPOCHS, gradient_clip_norm=GRADIENT_CLIP_NORM, optimizer=OPTIMIZER, seed=SEED, num_workers=NUM_WORKERS, eval_interval=EVAL_INTERVAL, early_stop="disabled", all_items=len(all_embeddings), train_items=len(train_embeddings), embedding_shape=list(embeddings.shape), world_size=world_size, mechanism="behaviour_geometry_repaired", experiment_arm=EXPERIMENT_ARM, geometry=GEOMETRY, curvature=CURVATURE, target_radius_fraction=TARGET_RADIUS_FRACTION, distance_normalization=DISTANCE_NORMALIZATION, resample_pairs=RESAMPLE_PAIRS, behavior_weight_max=BEHAVIOR_WEIGHT_MAX, behavior_temperature=BEHAVIOR_TEMPERATURE, temperature_calibration_enabled=TEMPERATURE_CALIBRATION_ENABLED, target_logit_spread=TARGET_LOGIT_SPREAD, behavior_ramp_start_epoch=RAMP_START_EPOCH, pair_batch_size_per_rank=PAIR_BATCH_SIZE_PER_RANK, transition_pairs=len(pair_dataset), pair_sampler_seed=SEED + 1, source="348aed182")

    # === codebook init: only rank 0 (codebook 不是 DDP 参数, 全 rank 共享) ===
    if rank == 0:
        print("[RecBole RQ-VAE] initializing codebooks with KMeans", flush=True)
    raw_module = model.module if isinstance(model, DDP) else model
    with torch.no_grad():
        raw_module.init_codebook(train_embeddings.to(device))
    if world_size > 1:
        dist.barrier()

    # Exp1 repair: one fixed normalizer from training-set statistics, so the
    # ball map cannot depend on batch membership and anchors and candidates share
    # a single ball. Constant for the whole run.
    ball_normalizer = resolve_normalizer(raw_module, train_embeddings.to(device))
    radius_target = radius_target.to(device)
    fork_siblings = fork_siblings.to(device)
    normalizer = torch.as_tensor(
        ball_normalizer, dtype=torch.float32, device=device
    )
    # Fix the arm's temperature with the same procedure for every arm.
    resolved_temperature = BEHAVIOR_TEMPERATURE
    calibrate_pairs = pair_dataset.pairs[:PAIR_BATCH_SIZE_PER_RANK].to(device)
    calibrate_batch = all_embeddings[
        torch.cat((calibrate_pairs[:, 0], calibrate_pairs[:, 1])).cpu()
    ].to(device)
    if TEMPERATURE_CALIBRATION_ENABLED:
        resolved_temperature = calibrate_temperature(
            raw_module, calibrate_batch, calibrate_pairs[:, 0], calibrate_pairs[:, 1],
            normalizer,
        )
    # Checksum of the initial codebooks: proves the arms being compared started
    # from the same quantization, which the multi-threaded KMeans did not
    # guarantee before.
    _init_parts = [
        layer.embed.weight.detach().cpu().numpy().tobytes()
        for layer in raw_module.rq.vq_layers
    ]
    _init_sha = hashlib.sha256(b"".join(_init_parts)).hexdigest()
    if rank == 0:
        _record(
            rank, "geometry_resolved", arm=EXPERIMENT_ARM, geometry=GEOMETRY,
            codebook_init_sha256=_init_sha,
            group_distance=GROUP_DISTANCE, radial_mode=RADIAL_MODE, curvature=CURVATURE, fork_radial_weight=FORK_RADIAL_WEIGHT, fork_angular_weight=FORK_ANGULAR_WEIGHT,
            target_radius_fraction=TARGET_RADIUS_FRACTION,
            ball_normalizer=ball_normalizer,
            resolved_temperature=resolved_temperature,
            target_logit_spread=TARGET_LOGIT_SPREAD,
        )

    # Stage3 only consumes the final-epoch weights, so tracking the best
    # collision across epochs would be misleading; keep the last evaluation.
    final_collision = float("inf")
    final_raw_unique = 0
    codebook_sizes = list(config.codebook_size)

    for epoch in range(1, EPOCHS + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch - 1)
        if pair_sampler is not None:
            pair_sampler.set_epoch(epoch - 1)
        # Rebuilt every epoch so the sampler's epoch seed actually takes effect.
        # The baseline built this once and replayed a frozen 664-batch pool
        # roughly 54 times; with RESAMPLE_PAIRS=False that behaviour is restored
        # exactly, which is what the "no_resample" control arm is for.
        if RESAMPLE_PAIRS or epoch == 1:
            pair_iter = iter(pair_loader)
        model.train()
        losses: list[float] = []
        recons: list[float] = []
        behaviors: list[float] = []
        fork_radials: list[float] = []
        fork_angulars: list[float] = []
        fork_radius_means: list[float] = []
        fork_pos_cosines: list[float] = []
        epoch_weight = behavior_weight(epoch)
        _epoch_t0 = time.time()
        for batch, row_index in loader:
            batch = batch.to(device, non_blocking=True)
            row_index = row_index.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            reconstructed, quant_loss, _, _ = model(batch)
            # === DDP 包装下, 通过 .module 调 compute_loss (避免 DDP all_reduce 干扰) ===
            loss, recon_loss = raw_module.compute_loss(batch, reconstructed, quant_loss)
            try:
                pairs = next(pair_iter)
            except StopIteration:
                pair_iter = iter(pair_loader)
                pairs = next(pair_iter)
            pairs = pairs.to(device, non_blocking=True)
            source_ids, target_ids = pairs[:, 0], pairs[:, 1]
            pair_batch = all_embeddings[
                torch.cat((source_ids, target_ids)).cpu()
            ].to(device, non_blocking=True)
            residuals = _residuals_with_trace(raw_module, pair_batch)
            behavior_loss = behavior_contrastive_loss(
                residuals, source_ids, target_ids, normalizer,
                resolved_temperature,
            )
            loss = loss + epoch_weight * behavior_loss
            # Behaviour-fork terms on the ball coordinates. They ramp with the
            # same epoch schedule as the behaviour term so the first half of
            # training still builds the reconstruction objective undisturbed.
            if epoch_weight > 0.0:
                # `batch` already holds this step's item embeddings and
                # `row_index` their positions in the item universe, which is the
                # space the radius target and the sibling table are indexed by.
                radial_term, angular_term, fork_stats = fork_geometry_terms(
                    raw_module, batch, row_index, radius_target, fork_siblings,
                    normalizer, all_embeddings,
                )
                loss = loss + epoch_weight * (
                    FORK_RADIAL_WEIGHT * radial_term
                    + FORK_ANGULAR_WEIGHT * angular_term
                )
                fork_radials.append(float(radial_term.detach()))
                fork_angulars.append(float(angular_term.detach()))
                fork_radius_means.append(fork_stats["radius_mean"])
                fork_pos_cosines.append(fork_stats["positive_distance_mean"])
            if not torch.isfinite(loss):
                raise RuntimeError(f"RQ-VAE loss became non-finite at epoch {epoch}")
            loss.backward()
            if GRADIENT_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach()))
            recons.append(float(recon_loss.detach()))
            behaviors.append(float(behavior_loss.detach()))
        epoch_time_s = round(time.time() - _epoch_t0, 3)

        # === DDP all_reduce: 跨 rank 求平均 loss / recon ===
        local_loss = torch.tensor(float(np.mean(losses)) if losses else 0.0, device=device)
        local_recon = torch.tensor(float(np.mean(recons)) if recons else 0.0, device=device)
        local_behavior = torch.tensor(float(np.mean(behaviors)) if behaviors else 0.0, device=device)
        if world_size > 1:
            dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_recon, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_behavior, op=dist.ReduceOp.SUM)
        avg_loss = float(local_loss.item()) / max(world_size, 1)
        avg_recon = float(local_recon.item()) / max(world_size, 1)
        avg_behavior = float(local_behavior.item()) / max(world_size, 1)
        cur_lr = optimizer.param_groups[0]["lr"]

        # === 非 eval 间隔: 只 log train 行 ===
        if epoch % EVAL_INTERVAL != 0 and epoch != EPOCHS:
            if rank == 0:
                print(f"[RQ-VAE] epoch={epoch} loss={avg_loss:.8f} recon={avg_recon:.8f} beh={avg_behavior:.6f} w={epoch_weight:.3f} lr={cur_lr:.3e} time={epoch_time_s}s", flush=True)
                _record(rank, "train", epoch=epoch, loss=avg_loss, recon=avg_recon, behavior_contrastive=avg_behavior, behavior_weight=epoch_weight, fork_radial=float(np.mean(fork_radials)) if fork_radials else 0.0, fork_angular=float(np.mean(fork_angulars)) if fork_angulars else 0.0, fork_radius_mean=float(np.mean(fork_radius_means)) if fork_radius_means else 0.0, fork_positive_distance=float(np.mean(fork_pos_cosines)) if fork_pos_cosines else 0.0, lr=cur_lr, epoch_time_s=epoch_time_s, cumulative_time_s=round(time.time() - _T0, 3))
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
                print(
                    f"[RQ-VAE] epoch={epoch} loss={avg_loss:.8f} recon={avg_recon:.8f} "
                    f"lr={cur_lr:.3e} raw_unique={raw_unique}/{len(raw_tokens)} "
                    f"collision={collision_v:.6f} time={epoch_time_s}s",
                    flush=True,
                )
                _record(
                    rank, "train", epoch=epoch, loss=avg_loss, recon=avg_recon, behavior_contrastive=avg_behavior, behavior_weight=epoch_weight, fork_radial=float(np.mean(fork_radials)) if fork_radials else 0.0, fork_angular=float(np.mean(fork_angulars)) if fork_angulars else 0.0, fork_radius_mean=float(np.mean(fork_radius_means)) if fork_radius_means else 0.0, fork_positive_distance=float(np.mean(fork_pos_cosines)) if fork_pos_cosines else 0.0, lr=cur_lr,
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
