"""TIGER + behaviour geometry (Exp2-5): Euclidean vs Poincare contrastive.

Baseline: TIGER_BEHAVIOR_RQ-VAE (test_R@10 = 0.061927). Sampling is fixed here
(per-epoch resampling; see TIGER_BEHAVIOR_SAMPLE_RQ-VAE). `model/` stays
byte-identical to TIGER, so the quantizer is untouched: only the distance used
inside the behaviour contrastive term changes, and Exp4/5 add a directional
transition operator on top of the same term.

One hardcoded arm table drives everything. Each arm's committed state is exactly
what ran, and each writes to its own results directory:

  geometry        "euclid" | "poincare"  -- the metric inside the contrastive term
  CURVATURE       positive c of the ball of curvature -c (Exp3 sweeps it)
  TANGENT_SCALE   fixed radius fraction of the ball image; see hyperbolic.py
  transition      "none" | "euclid_delta" | "mobius_add"  (Exp4 / Exp5)
  distance_normalization
                  "rms" | "none". Distance matrices are rescaled to unit RMS
                  before the temperature divide, identically for every arm, so
                  logit sharpness is matched and the arms differ only in the
                  ranking structure their metric induces. Without it a
                  Poincare distance (~2.5-3.8) and a Euclidean one (~0.5-1.6)
                  would enter softmax at different sharpness, which would
                  confound "which metric ranks successors better" with "which
                  metric happened to be scaled to a usable temperature".

The Euclidean control for Exp2 shares the ball map, the tangent scale, the
temperature, the mask and the batch: only `geometry` differs, so it is
parameter-matched by construction.

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
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from tqdm import tqdm

from model import RQVAE
from hyperbolic import (
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
EXPERIMENT_ARM = "exp4_euclid_delta"
_ARMS = {
    # Exp2: Poincare vs parameter-matched Euclidean, both on the same ball map.
    "exp2_poincare_c1":    {"geometry": "poincare", "curvature": 1.0, "transition": "none"},
    "exp2_euclid_matched": {"geometry": "euclid",   "curvature": 1.0, "transition": "none"},
    # Exp3: curvature sweep on whichever Exp2 geometry wins.
    "exp3_hyp_c0p1":       {"geometry": "poincare", "curvature": 0.1, "transition": "none"},
    "exp3_hyp_c0p5":       {"geometry": "poincare", "curvature": 0.5, "transition": "none"},
    "exp3_hyp_c5":         {"geometry": "poincare", "curvature": 5.0, "transition": "none"},
    # Exp4/5: directional A->B transition, Euclidean vs Mobius.
    "exp4_euclid_delta":   {"geometry": "euclid",   "curvature": 1.0, "transition": "euclid_delta"},
    "exp5_mobius_add":     {"geometry": "poincare", "curvature": 1.0, "transition": "mobius_add"},
}
_ARM = _ARMS[EXPERIMENT_ARM]
GEOMETRY = _ARM["geometry"]                 # "euclid" | "poincare"
CURVATURE = _ARM["curvature"]               # positive c of ball curvature -c
# Held CONSTANT across curvatures on purpose. The mean image radius fraction is
# tanh(sqrt(c) * TANGENT_SCALE), so fixing the tangent scale alone would let a
# curvature sweep also change how close to the boundary the points sit, and any
# Exp3 effect would be ambiguous between the two. Deriving the scale from a
# target radius fraction makes only `curvature` move.
TARGET_RADIUS_FRACTION = 0.7
TANGENT_SCALE = float(__import__("math").atanh(TARGET_RADIUS_FRACTION) / (CURVATURE ** 0.5))
TRANSITION = _ARM["transition"]             # "none" | "euclid_delta" | "mobius_add"
DISTANCE_NORMALIZATION = "rms"              # "rms" | "none"
# Temperature is CALIBRATED, not hand-set. The two geometries have different
# distance tails (Poincare p99 ~2.6x RMS, Euclidean ~1.7x), so one shared
# temperature would enter softmax at different sharpness and any difference
# between the arms would be partly a sharpness artefact. Instead every arm is
# calibrated by the same procedure: at initialisation, binary-search the
# temperature that makes the mean per-row logit spread equal TARGET_LOGIT_SPREAD.
# The resolved value is recorded in the metrics stream.
TEMPERATURE_CALIBRATION_ENABLED = True
TARGET_LOGIT_SPREAD = 5.0
_OUT = Path("/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_HYPERBOLIC_RQ-VAE") / EXPERIMENT_ARM
OUTPUT_SID     = _OUT / "sids_for_hgrec.npy"
OUTPUT_JSON    = _OUT / "item_sids.json"
CHECKPOINT     = _OUT / "rqvae_best.pth"
LOG_DIR        = Path("/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_HYPERBOLIC_RQ-VAE/logs") / EXPERIMENT_ARM
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
# FALSE_NEGATIVE_MASK=True 除重复 target 外, 再屏蔽同 anchor 的其它已知真实后继。
FALSE_NEGATIVE_MASK     = False  # Exp1 verdict: the FN mask raised matched AUC but not Stage3.
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
    """Per-level pre-quantization residuals, matching iter48's ResidualTraceRQLayer.

    ``model/`` stays byte-identical to the baseline, so the trace is produced by
    walking the unchanged quantizer stack rather than by patching it.
    """
    residual = raw_module.encoder(embeddings)
    traces = [residual]
    for vq_layer in raw_module.rq.vq_layers:
        quant, _, _, _ = vq_layer(residual)
        residual = residual - quant
        traces.append(residual)
    # iter48 records the residual *entering* each level; drop the final remainder.
    return torch.stack(traces[: raw_module.rq.codebook_num], dim=0)


def _successor_indicator(train_frame: pd.DataFrame, n_items: int):
    """Sparse indicator of every known ``anchor -> successor`` item pair.

    Used by the false-negative mask: a candidate that happens to be another real
    successor of the same anchor must not be pushed away. 323,633 distinct edges
    over 24,474 anchors, so a CSR of shape (n_items, n_items) is cheap.
    """
    from scipy.sparse import csr_matrix

    rows: list[int] = []
    cols: list[int] = []
    for history, target in zip(
        train_frame["seen_history"].to_numpy(),
        train_frame["target"].to_numpy(dtype=np.int64),
    ):
        if history is None or len(history) == 0:
            continue
        rows.append(int(history[-1]))
        cols.append(int(target))
    data = np.ones(len(rows), dtype=np.bool_)
    matrix = csr_matrix(
        (data, (np.asarray(rows), np.asarray(cols))),
        shape=(n_items, n_items),
        dtype=np.bool_,
    )
    matrix.sum_duplicates()
    matrix.data[:] = True
    return matrix


def _transition_shift(
    z_anchor: torch.Tensor, transition_weight: torch.Tensor
) -> torch.Tensor:
    """Move the anchor along the learned A->B direction for the arm's geometry.

    The transition models "where does a purchase at this anchor tend to land".
    Both arms carry exactly one 32x32 matrix, zero-initialised so the arm starts
    as the stateless baseline and learns the shift:

      euclid_delta  a' = a + a W^T                      (linear tangent shift)
      mobius_add    a' = a (+) mobius_matvec(W, a, c)    (ball translation)

    Same parameter count and same zero-init in both arms, so Exp4 vs Exp5
    compares the geometry of the shift, not its capacity.
    """
    if TRANSITION == "mobius_add":
        return mobius_add(
            z_anchor, mobius_matvec(transition_weight, z_anchor, CURVATURE), CURVATURE
        )
    return _inside_ball(z_anchor + z_anchor @ transition_weight.t())


def _inside_ball(x: torch.Tensor) -> torch.Tensor:
    limit = (1.0 - 1e-5) / (CURVATURE ** 0.5)
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.where(norm > limit, x * (limit / norm), x)


def _geodesic_block(
    anchors: torch.Tensor,
    candidates: torch.Tensor,
    transition_weight: torch.Tensor | None,
) -> torch.Tensor:
    """Distance matrix between two residual blocks under the arm's geometry.

    Both arms share the same ball map and the same tangent scale, so only the
    metric differs. ``transition`` adds the directional operator of Exp4/5 on
    top of the metric.
    """
    z_a = encode_ball(anchors, CURVATURE, TANGENT_SCALE)
    z_c = encode_ball(candidates, CURVATURE, TANGENT_SCALE)
    if TRANSITION != "none" and transition_weight is not None:
        z_a = _transition_shift(z_a, transition_weight)
    if GEOMETRY == "poincare":
        return pair_poincare_distance(z_a, z_c, CURVATURE)
    return torch.sqrt(
        spatial_sq_distance(z_a[:, None, :], z_c[None, :, :]).clamp_min(0.0)
    )


def behavior_contrastive_loss(
    residuals: torch.Tensor,
    source_ids: torch.Tensor,
    target_ids: torch.Tensor,
    successor_indicator=None,
    transition_weight: torch.Tensor | None = None,
    temperature: float | None = None,
) -> torch.Tensor:
    """iter48's three-level contrastive loss over residual vectors.

    ``residuals`` holds the pair batch with sources in the first half and
    targets in the second half, exactly as iter48 fed
    ``cat(source_emb, target_emb)`` through the quantizer.

    ``successor_indicator``, when given, additionally masks candidates that are
    other known successors of the same anchor (false negatives).
    """
    batch_size = int(source_ids.shape[0])
    losses: list[torch.Tensor] = []
    for level in range(residuals.shape[0]):
        source = residuals[level, :batch_size]
        candidates = residuals[level, batch_size:]
        distances = _geodesic_block(source, candidates, transition_weight)
        if DISTANCE_NORMALIZATION == "rms":
            rms = distances.detach().square().mean().sqrt().clamp_min(1e-8)
            distances = distances / rms
        logits = -distances / (BEHAVIOR_TEMPERATURE if temperature is None else temperature)
        # iter48's filter: drop candidates sharing the anchor's own target, then
        # keep only rows whose source and target differ.
        duplicate_targets = target_ids[:, None].eq(target_ids[None, :])
        duplicate_targets.fill_diagonal_(False)
        if successor_indicator is not None:
            rows = successor_indicator[source_ids.cpu().numpy()]
            blocked = torch.from_numpy(
                np.asarray(rows[:, target_ids.cpu().numpy()].todense())
            ).to(device=logits.device, dtype=torch.bool)
            # The diagonal is the true successor, never a false negative.
            blocked.fill_diagonal_(False)
            duplicate_targets = duplicate_targets | blocked
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
    transition_weight: torch.Tensor | None,
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
                residuals[level, :half], residuals[level, half:], transition_weight
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
    successor_indicator = (
        _successor_indicator(train_frame, len(embeddings))
        if FALSE_NEGATIVE_MASK
        else None
    )
    config = _tokenizer_config()
    model = RQVAE(config, in_dim=embeddings.shape[1]).to(device)
    if XAVIER_INIT:
        initialize_tiger_weights(model)
    if TRANSITION != "none":
        # Registered on the module (not on model/) so DDP and the optimizer pick
        # it up while `model/` stays byte-identical to TIGER. Zero-init makes the
        # arm start exactly at the stateless baseline.
        model.register_parameter(
            "transition_weight",
            nn.Parameter(
                torch.zeros(config.codebook_dim, config.codebook_dim, device=device)
            ),
        )
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    train_dataset = TensorDataset(train_embeddings)
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
        _record(rank, "train_start", epochs=EPOCHS, batch_size_per_rank=BATCH_SIZE_PER_RANK, total_batch_size=BATCH_SIZE_PER_RANK*world_size, lr=LR, weight_decay=WEIGHT_DECAY, hidden_sizes=list(HIDDEN_SIZES), codebook_num=CODEBOOK_NUM, codebook_size=list(CODEBOOK_SIZE), codebook_dim=CODEBOOK_DIM, beta=BETA, vq_type=VQ_TYPE, ema_decay=EMA_DECAY, sk_epsilon=SK_EPSILON, sk_iters=SK_ITERS, pca_dim=PCA_DIM, xavier_init=XAVIER_INIT, warmup_epochs=WARMUP_EPOCHS, gradient_clip_norm=GRADIENT_CLIP_NORM, optimizer=OPTIMIZER, seed=SEED, num_workers=NUM_WORKERS, eval_interval=EVAL_INTERVAL, early_stop="disabled", all_items=len(all_embeddings), train_items=len(train_embeddings), embedding_shape=list(embeddings.shape), world_size=world_size, mechanism="behaviour_geometry", experiment_arm=EXPERIMENT_ARM, geometry=GEOMETRY, curvature=CURVATURE, tangent_scale=TANGENT_SCALE, target_radius_fraction=TARGET_RADIUS_FRACTION, transition=TRANSITION, distance_normalization=DISTANCE_NORMALIZATION, resample_pairs=RESAMPLE_PAIRS, false_negative_mask=FALSE_NEGATIVE_MASK, behavior_weight_max=BEHAVIOR_WEIGHT_MAX, behavior_temperature=BEHAVIOR_TEMPERATURE, temperature_calibration_enabled=TEMPERATURE_CALIBRATION_ENABLED, target_logit_spread=TARGET_LOGIT_SPREAD, behavior_ramp_start_epoch=RAMP_START_EPOCH, pair_batch_size_per_rank=PAIR_BATCH_SIZE_PER_RANK, transition_pairs=len(pair_dataset), pair_sampler_seed=SEED + 1, source="348aed182")

    # === codebook init: only rank 0 (codebook 不是 DDP 参数, 全 rank 共享) ===
    if rank == 0:
        print("[RecBole RQ-VAE] initializing codebooks with KMeans", flush=True)
    raw_module = model.module if isinstance(model, DDP) else model
    with torch.no_grad():
        raw_module.init_codebook(train_embeddings.to(device))
    if world_size > 1:
        dist.barrier()

    transition_weight = (
        getattr(raw_module, "transition_weight", None)
        if TRANSITION != "none"
        else None
    )
    # Fix the arm's temperature with the same procedure for every arm.
    resolved_temperature = BEHAVIOR_TEMPERATURE
    if TEMPERATURE_CALIBRATION_ENABLED:
        calibrate_pairs = pair_dataset.pairs[:PAIR_BATCH_SIZE_PER_RANK].to(device)
        calibrate_batch = all_embeddings[
            torch.cat((calibrate_pairs[:, 0], calibrate_pairs[:, 1])).cpu()
        ].to(device)
        resolved_temperature = calibrate_temperature(
            raw_module, calibrate_batch, calibrate_pairs[:, 0], calibrate_pairs[:, 1],
            transition_weight,
        )
        _record(
            rank, "temperature_calibrated", resolved_temperature=resolved_temperature,
            target_logit_spread=TARGET_LOGIT_SPREAD, arm=EXPERIMENT_ARM,
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
        epoch_weight = behavior_weight(epoch)
        _epoch_t0 = time.time()
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=True)
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
                residuals, source_ids, target_ids, successor_indicator,
                transition_weight, resolved_temperature,
            )
            loss = loss + epoch_weight * behavior_loss
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
                _record(rank, "train", epoch=epoch, loss=avg_loss, recon=avg_recon, behavior_contrastive=avg_behavior, behavior_weight=epoch_weight, lr=cur_lr, epoch_time_s=epoch_time_s, cumulative_time_s=round(time.time() - _T0, 3))
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
                    rank, "train", epoch=epoch, loss=avg_loss, recon=avg_recon, behavior_contrastive=avg_behavior, behavior_weight=epoch_weight, lr=cur_lr,
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
