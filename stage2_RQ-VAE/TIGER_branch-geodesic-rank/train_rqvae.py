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

from model.hyperbolic import ball_distance, ball_exp_map
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
ARM_NAME = "TIGER_branch-geodesic-rank"
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
# The single-card launch contract of the MCP: when this variable is 1 the
# trainer runs in this process instead of forking the four-rank job.
SINGLE_GPU_ENV   = "GENREC_SINGLE_GPU"

# === 机制开关 (规格里的两个 code switch) ===
# behaviour_on=False: 行为亲和度的排序被打乱, 三元组的顺序不再来自真实重叠度,
#                     但曲率几何与计算量不变.
# geometry_on=False : exp_0/log_0 退化为恒等映射, 测地距离退化为平方欧氏距离,
#                     三元组身份、顺序、尺度与目标结构完全不变.
# 唯一被训练的配置是 both_on; 另外两个只由 smoke test 行使其生效.
BEHAVIOR_ON         = True
GEOMETRY_ON         = True
# 曲率固定为 -1 (c = 1), 与规格一致, 不做扫描.
GEOMETRIC_CURVATURE = 1.0
# 球映射的固定尺度: 中位潜向量范数映射到目标半径比例.
TARGET_RADIUS_FRACTION = 0.7
# 每个 batch 参与三元组的锚点上限与每个锚点的正例上限.
# The constants below are fixed engineering choices, not tuned constants: they
# bound the work of one step and they are not swept. The anchor cap bounds how
# many triplets one batch contributes, the partner cap bounds how many affinity
# partners of an item the sampler reads, and the radius fraction fixes where
# the median latent norm sits in the ball.
MAX_ANCHORS_PER_BATCH = 512
# 固定切片的大小, 距离尺度在它上面测量.
SCALE_PROBE_ITEMS       = 1024
# 每个物品保留的亲和伙伴上限, 采样只读这张表.
TOP_PARTNER_CAP         = 64
MAX_POSITIVES_PER_ANCHOR = 8
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


def build_successor_affinity(train_frame, n_items: int, seed: int):
    """The training-only successor overlap affinity between items.

    ``n_ij`` counts the adjacent directed transitions of the training
    sequences, ``p_i`` is the normalized outgoing row and ``pop_i`` the
    incoming popularity. The affinity of a pair that shares a successor is the
    sum of the overlap of the two outgoing distributions, corrected by the
    square root of the popularity factors. Pairs without a common successor
    carry no signal and are not kept.
    """
    from scipy import sparse

    rows: list = []
    columns: list = []
    values: list = []
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
            rows.append(source)
            columns.append(destination)
            values.append(1.0)
    if not rows:
        raise RuntimeError("no training transition was counted")
    counts = sparse.coo_matrix((np.asarray(values, dtype=np.float64),
                                (np.asarray(rows, dtype=np.int64),
                                 np.asarray(columns, dtype=np.int64))),
                               shape=(n_items, n_items)).tocsr()
    row_sum = np.asarray(counts.sum(axis=1)).ravel()
    row_sum[row_sum == 0.0] = 1.0
    # p_i(k) = n_ik / sum_k n_ik
    probability = counts.multiply(1.0 / row_sum[:, None]).tocsr()
    # pop_i = the number of transitions that END at i
    popularity = np.asarray(counts.sum(axis=0)).ravel()
    # f_i = 1 + log(1 + pop_i) and the denominator is sqrt(f_i * f_j).
    factor = 1.0 + np.log1p(popularity)
    factor[factor <= 0.0] = 1.0
    correction = np.sqrt(factor)

    # The overlap of two outgoing distributions: sum_k min(p_i(k), p_j(k)).
    # The candidate pairs are the items that share a successor, which is where
    # the overlap can be nonzero at all.
    probability = probability.tocsr()
    pair_source: list = []
    pair_target: list = []
    pair_value: list = []
    # The precursor set of every successor: the items that lead to it. The
    # fan-out is capped by a deterministic rule that keeps the RAREST
    # successors, because a successor many items lead to carries an overlap
    # that says little and a fan-out that costs a lot.
    precursor = counts.T.tocsr()
    # The fan-out cap is a memory bound, not a tuned constant. The rule is the
    # TOP-COUNT one of the specification: the successors with the largest
    # transition counts are the ones whose overlap can be formed.
    fanout_limit = 1024
    # The bound on the candidate set of one item: the overlap of every pair is
    # formed against it, so the dense intermediate stays of the size of the
    # bound and not of the item universe.
    candidate_limit = 512
    for item in range(n_items):
        successors = probability[item].indices
        if successors.size == 0:
            continue
        # The top-count rule: the successors with the largest counts come
        # first, and the cap keeps the overlap of every pair formable.
        counts = np.asarray(probability[item].todense()).ravel()[successors]
        rare = successors[np.argsort(-counts)[:fanout_limit]]
        # the items that also lead to one of this item's rare successors
        candidates = np.unique(precursor[rare].indices)
        candidates = candidates[candidates != item]
        if candidates.size == 0:
            continue
        if candidates.size > candidate_limit:
            # The candidate set of one item is bounded, so that the overlap of
            # every pair can be formed without a dense array of the size of the
            # item universe. The bound is a memory limit, fixed and not swept.
            candidates = candidates[:candidate_limit]
        # the overlap of the two outgoing distributions over the shared successors
        left = probability[item]
        right = probability[candidates]
        shared = np.intersect1d(left.indices, right.indices)
        if shared.size == 0:
            continue
        left_values = np.asarray(left.todense()).ravel()[shared]
        right_dense = right[:, shared].toarray()
        overlap = np.minimum(left_values[None, :], right_dense).sum(axis=1)
        keep = overlap > 0
        if not bool(keep.any()):
            continue
        pair_source.extend([item] * int(keep.sum()))
        pair_target.extend(candidates[keep].tolist())
        pair_value.extend((overlap[keep] / (correction[item] * correction[candidates[keep]])).tolist())
    if not pair_source:
        raise RuntimeError("no pair of items shares a successor")
    affinity = sparse.csr_matrix((np.asarray(pair_value, dtype=np.float64),
                                  (np.asarray(pair_source, dtype=np.int64),
                                   np.asarray(pair_target, dtype=np.int64))),
                                 shape=(n_items, n_items))
    # The affinity of the specification is symmetric, but the candidate
    # generation is directional: each item keeps its own capped candidate list,
    # so a pair can be present in one row only. The matrix is symmetrised, so
    # that the affinity of a pair does not depend on which of the two items
    # happened to keep the other one.
    affinity = affinity.maximum(affinity.T).tocsr()
    return affinity, probability, popularity


def top_affinity_partners(affinity, cap: int):
    """For every item, the cap partners with the largest affinity.

    The sampler reads this instead of the whole row: the rows of the popular
    items are long, and scanning them per step would make the sampling cost
    depend on the popularity rather than on the batch.
    """
    n_items = affinity.shape[0]
    partners = np.full((n_items, cap), -1, dtype=np.int64)
    weights = np.zeros((n_items, cap), dtype=np.float64)
    for item in range(n_items):
        start, end = affinity.indptr[item], affinity.indptr[item + 1]
        if end <= start:
            continue
        row_weights = affinity.data[start:end]
        row_columns = affinity.indices[start:end]
        if row_weights.size > cap:
            keep = np.argpartition(-row_weights, cap - 1)[:cap]
            row_weights = row_weights[keep]
            row_columns = row_columns[keep]
        order = np.argsort(-row_weights)
        partners[item, :order.size] = row_columns[order]
        weights[item, :order.size] = row_weights[order]
    return partners, weights


def resolve_ball_normalizer(raw_module, train_embeddings: torch.Tensor) -> float:
    """The fixed scale of the ball map, computed once from the training set.

    The median latent norm is mapped to TARGET_RADIUS_FRACTION of the radius.
    The value is a constant for the whole run: if it depended on the batch, the
    same item would receive different coordinates in different batches.
    """
    with torch.no_grad():
        norms = torch.linalg.vector_norm(raw_module.encoder(train_embeddings), dim=-1)
        median = float(torch.median(norms).clamp_min(1e-12))
    return float(np.arctanh(TARGET_RADIUS_FRACTION) / (GEOMETRIC_CURVATURE ** 0.5) / median)


def _affinity_of(affinity, row: int, column: int) -> float:
    """The affinity of one pair, read from the full matrix."""
    start, end = affinity.indptr[row], affinity.indptr[row + 1]
    if end <= start:
        return 0.0
    indices = affinity.indices[start:end]
    position = int(np.searchsorted(indices, column))
    if position < indices.size and int(indices[position]) == column:
        return float(affinity.data[start + position])
    return 0.0


def sample_rank_triplets(partners, partner_weights, affinity, batch_ids: np.ndarray,
                         generator: np.random.Generator,
                         max_anchors: int, max_positives: int):
    """The (anchor, positive, negative) triplets of one batch.

    The positive is a batch member sampled in proportion to the successor
    overlap affinity, and the negative is a batch member with a LOWER affinity
    to the same anchor, so the triplet always carries the ordering the
    behaviour signal states. Both are real items of the batch, so the ranking
    acts on the representations the quantizer will read.

    The returned indices are POSITIONS inside the batch, so the caller can
    index the encoded rows of the batch directly.

    The sampler reads the SPARSE row of the anchor and never materialises a
    dense block of the batch: a block of the batch size squared is quadratic
    work and a dense allocation on every step.
    """
    count = int(batch_ids.shape[0])
    empty = np.zeros(0, dtype=np.int64)
    if count < 2:
        return empty, empty.copy(), empty.copy()
    position = {int(item): index for index, item in enumerate(batch_ids)}
    anchors: list = []
    positives: list = []
    negatives: list = []
    for anchor in range(count):
        anchor_item = int(batch_ids[anchor])
        columns = partners[anchor_item]
        weights = partner_weights[anchor_item]
        # the batch members this anchor has an affinity with
        inside = [(position[int(column)], float(weight))
                  for column, weight in zip(columns, weights)
                  if int(column) >= 0 and int(column) in position]
        positive_pool = [(index, weight) for index, weight in inside if weight > 0.0]
        if not positive_pool:
            continue
        picks = generator.choice(len(positive_pool),
                                 size=min(max_positives, len(positive_pool)),
                                 replace=False,
                                 p=np.asarray([weight for _, weight in positive_pool]) /
                                 sum(weight for _, weight in positive_pool))
        for pick in picks:
            positive, positive_weight = positive_pool[pick]
            # every batch member with a lower affinity: the ones this anchor has
            # an affinity with and the ones it has none with, which are all zero
            # The negative is a batch member with a SMALLER affinity. The
            # members this anchor has an affinity with are known from the top
            # table; the affinity of a member the table does not list is read
            # from the full matrix, so that a truncated table never turns a
            # positive affinity into a zero one. The candidates are the row's
            # members with a lower weight and the members outside the row, and
            # the affinity of the chosen one is verified before it is used.
            inside_weights = {index: weight for index, weight in inside}
            lower = [(index, weight) for index, weight in inside_weights.items()
                     if weight < positive_weight]
            outside = [index for index in range(count)
                       if index != anchor and index not in inside_weights]
            if not lower and not outside:
                continue
            lower_weights = np.asarray([weight for _, weight in lower], dtype=np.float64)
            total = float(lower_weights.sum())
            for _attempt in range(4):
                if lower and (total > 0.0 or not outside):
                    choice = int(lower[int(generator.choice(len(lower), p=lower_weights / total))][0]) \
                        if total > 0.0 else int(lower[int(generator.integers(len(lower)))][0])
                else:
                    choice = int(outside[int(generator.integers(len(outside)))])
                true_weight = inside_weights.get(choice)
                if true_weight is None:
                    true_weight = float(_affinity_of(affinity, anchor_item, int(batch_ids[choice])))
                if true_weight < positive_weight:
                    break
            else:
                continue
            # The three arrays are appended together: the anchor belongs to the
            # triplet and not to the batch member that produced it.
            anchors.append(int(anchor))
            positives.append(int(positive))
            negatives.append(int(choice))
        if len(anchors) >= max_anchors:
            break
    return (np.asarray(anchors, dtype=np.int64),
            np.asarray(positives, dtype=np.int64),
            np.asarray(negatives, dtype=np.int64))


def behaviour_off_triplets(anchors: np.ndarray, positives: np.ndarray,
                            negatives: np.ndarray, generator: np.random.Generator):
    """The behaviour-off control of the specification.

    The triplets are the SAME ones: the same items, the same count, the same
    loss. What is permuted is the AFFINITY RANK, that is which of the two
    candidates of a triplet is the positive. The permutation of the two slots is
    independent, so the pairing is not moved around as a block either. The
    ordering the behaviour signal states is therefore destroyed, while
    everything else about the computation stays.
    """
    if anchors.size == 0:
        return None
    size = int(anchors.size)
    permuted_positives = positives[generator.permutation(size)]
    permuted_negatives = negatives[generator.permutation(size)]
    return anchors, permuted_positives, permuted_negatives


def ranking_loss_term(encoded: torch.Tensor, triplets, context: dict) -> torch.Tensor:
    """The softplus ranking loss on the squared geodesic distances.

    The context carries the two switches, the fixed ball scale and the frozen
    distance scale, so the geometry control is the same computation with the
    identity maps and the squared Euclidean distance.
    """
    from model.hyperbolic import ball_distance, ball_exp_map
    anchor_index, positive_index, negative_index = triplets
    if not bool(context.get("behavior_on", True)):
        # The behaviour-off control of the specification: the affinity RANK is
        # permuted, so the ordering the behaviour signal states is destroyed
        # while the triplets, their count and the loss stay the same.
        generator = context.get("generator")
        size = int(positive_index.shape[0])
        if generator is not None:
            order = torch.randperm(size, generator=generator, device=positive_index.device)
            positive_index = positive_index[order]
            order = torch.randperm(size, generator=generator, device=positive_index.device)
            negative_index = negative_index[order]
    geometry_on = bool(context.get("geometry_on", True))
    normalizer = float(context.get("normalizer", 1.0))
    curvature = float(context.get("curvature", GEOMETRIC_CURVATURE))
    if geometry_on:
        mapped = ball_exp_map(torch.zeros_like(encoded), encoded * normalizer, curvature)
        positive_distance = ball_distance(mapped[anchor_index], mapped[positive_index],
                                          curvature).square()
        negative_distance = ball_distance(mapped[anchor_index], mapped[negative_index],
                                          curvature).square()
    else:
        # The geometry-off control of the specification: the maps become the
        # identity and the metric becomes the squared Euclidean distance. The
        # SCALE stays the one of the curved path, so that the control differs
        # in the map and in the metric and in nothing else.
        mapped = encoded * normalizer
        positive_distance = (mapped[anchor_index] - mapped[positive_index]).square().sum(-1)
        negative_distance = (mapped[anchor_index] - mapped[negative_index]).square().sum(-1)
    scale_state = context.get("scale")
    scale = float(scale_state["value"]) if scale_state is not None else 1.0
    return torch.nn.functional.softplus((positive_distance - negative_distance) / scale).mean()


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

    # === 行为结构与球映射 ===
    # 全部来自训练划分的相邻转移, 不读验证或测试.
    (affinity, successor_probability, successor_popularity) = build_successor_affinity(
        train_frame, len(all_embeddings), SEED)
    affinity_csr = affinity.tocsr()
    partner_table, partner_weight_table = top_affinity_partners(affinity_csr, TOP_PARTNER_CAP)
    ball_normalizer = resolve_ball_normalizer(raw_module, train_embeddings.to(device))
    if rank == 0:
        print(f"[RecBole RQ-VAE] affinity_pairs={affinity_csr.nnz} "
              f"ball_normalizer={ball_normalizer:.6f} behaviour_on={BEHAVIOR_ON} "
              f"geometry_on={GEOMETRY_ON}", flush=True)
    rank_generator = np.random.default_rng(SEED + rank)
    rank_torch_generator = torch.Generator(device=device)
    rank_torch_generator.manual_seed(SEED + rank)
    # The distance scale is the median squared distance over a FIXED slice of
    # the training items, measured with the initial encoder and broadcast, so
    # that it is the same on every rank and does not depend on which shuffled
    # batch a rank happened to draw first.
    with torch.no_grad():
        probe_items = train_ids[:SCALE_PROBE_ITEMS]
        probe = raw_module.encoder(all_embeddings[torch.from_numpy(probe_items)].to(device))
        probe_mapped = ball_exp_map(torch.zeros_like(probe), probe * ball_normalizer,
                                    GEOMETRIC_CURVATURE)
        probe_distances = ball_distance(probe_mapped.unsqueeze(1),
                                       probe_mapped.unsqueeze(0),
                                       GEOMETRIC_CURVATURE).square()
        scale_value = float(probe_distances.median().clamp_min(1e-8).item())
    scale_tensor = torch.tensor([scale_value], device=device, dtype=torch.float32)
    if world_size > 1:
        dist.broadcast(scale_tensor, src=0)
    distance_scale: dict = {"value": float(scale_tensor.item())}
    ranking_rank_calibration: dict = {"mean": None}

    def rank_triplets(batch_ids: torch.Tensor):
        """The triplets of one batch, from the cached successor affinity."""
        ids = batch_ids.detach().cpu().numpy()
        anchors, positives, negatives = sample_rank_triplets(
            partner_table, partner_weight_table, affinity_csr, ids, rank_generator,
            MAX_ANCHORS_PER_BATCH, MAX_POSITIVES_PER_ANCHOR)
        if anchors.size == 0:
            return None
        anchor_index = torch.from_numpy(anchors).to(device)
        return (anchor_index,
                torch.from_numpy(positives).to(device),
                torch.from_numpy(negatives).to(device))

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
            reconstructed, quant_loss, _, _ = model(batch)
            # === DDP 包装下, 通过 .module 调 compute_loss (避免 DDP all_reduce 干扰) ===
            loss, recon_loss = raw_module.compute_loss(batch, reconstructed, quant_loss)
            triplets = rank_triplets(batch_ids)
            rank_context = {"geometry_on": GEOMETRY_ON, "behavior_on": BEHAVIOR_ON,
                            "normalizer": ball_normalizer, "scale": distance_scale,
                            "generator": rank_torch_generator}
            if ranking_rank_calibration["mean"] is None:
                # The calibration broadcast is UNCONDITIONAL: the triplet
                # sampling is local and a rank may have no triplet in this
                # batch, so a rank that entered the broadcast only when it had
                # one would wait for a rank that never comes. Every rank
                # measures a local value, zero when it has no triplet, and the
                # value of rank 0 is the one that counts.
                local_mean = 0.0
                if triplets is not None:
                    local_mean = float(ranking_loss_term(
                        raw_module.encoder(batch), triplets, rank_context
                    ).detach().clamp_min(1e-8).item())
                calibration = torch.tensor([local_mean], device=device, dtype=torch.float32)
                if world_size > 1:
                    dist.broadcast(calibration, src=0)
                value = float(calibration.item())
                # The divisor has to be POSITIVE: a first batch without a
                # triplet would otherwise freeze a near-zero calibration and
                # every later batch would divide its ranking term by it. When
                # rank 0 has no triplet, the term simply starts at its own
                # scale, which is the same convention the calibration states.
                ranking_rank_calibration["mean"] = value if value > 0.0 else 1.0
            if triplets is not None:
                encoded = raw_module.encoder(batch)
                ranking = ranking_loss_term(encoded, triplets, rank_context)
                loss = loss + ranking / ranking_rank_calibration["mean"]
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
