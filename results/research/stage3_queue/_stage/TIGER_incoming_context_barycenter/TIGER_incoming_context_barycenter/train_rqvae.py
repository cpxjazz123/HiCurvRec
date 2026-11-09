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
import math
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
ARM_NAME = "TIGER_incoming_context_barycenter"
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

# Incoming-context assignment: all statistics are built exclusively from the
# training parquet. Hyperbolic candidate scores affect the first-level VQ loss.
BEHAVIOUR_ON = True
GEOMETRY = "poincare"
CURVATURE = 1.0
CONTEXT_WEIGHT = 0.10
BALANCED_ASSIGNMENT = False  # iteration 4: the offset keeps the spread, the balance is off
CONTEXT_TEMPERATURE = 0.2
SINGLE_GPU_ENV = "GENREC_SINGLE_GPU"


def _stable_squared_poincare_from_arg(arg: torch.Tensor) -> torch.Tensor:
    """Evaluate acosh(1 + arg)^2 with its differentiable zero limit."""
    stable_arg = arg.clamp(min=0.0, max=1e6)
    # sqrt itself has an infinite derivative at zero. A tiny floor inside the
    # root prevents 0 * inf in autograd; its squared-distance bias is < 2e-12.
    root_arg = (stable_arg * 0.5).clamp_min(1e-12)
    value = 4.0 * torch.asinh(torch.sqrt(root_arg)).square()
    # Keep the graph connected at exact coincidence: a torch.where against a
    # constant zero selects a branch with no gradient, so a codepoint that sits
    # exactly on a barycentre would silently stop training that pair. Subtract
    # the value the floor adds instead, which is exactly zero at arg == 0 and
    # carries the derivative of the asinh form everywhere.
    floor_value = 4.0 * float(torch.asinh(torch.sqrt(torch.tensor(1e-12))).square())
    return (value - floor_value).clamp_min(0.0)


def incoming_assignment_loss(model: RQVAE, batch_ids: torch.Tensor,
                             bary: torch.Tensor, neg_bary: torch.Tensor,
                             embeddings: torch.Tensor,
                             behaviour_on: bool = True, geometry: str = "poincare",
                             concentration: torch.Tensor | None = None) -> torch.Tensor:
    """Contrast code candidates against the cached train-only predecessor barycentres.

    The barycentres are refreshed once per epoch from the current codebook (see
    ``refresh_context_barycentres``), so this function is one broadcast distance
    per candidate and stays vectorised over the batch. The behaviour signal and
    the curved contrast themselves are unchanged: only the per-item Python loop
    that recomputed a Frechet mean for every sample of every step was removed.
    """
    layer = model.rq.vq_layers[0]
    codes = layer.get_code_embs()
    x = model.encoder(embeddings[batch_ids])
    eps = 1e-5
    if not behaviour_on:
        # behaviour-off control: the plain reconstruction-consistent Euclidean
        # cost of the item's own latent. Nothing is detached, so the control
        # keeps the same gradient path as the coupled arm and differs only in
        # the behaviour/curvature term -- a detached control would compare two
        # different graphs instead of one graph with and without the term.
        control_cost = (x.square().sum(-1)[:, None] + codes.square().sum(-1)[None, :]
                        - 2.0 * (x @ codes.t())).clamp_min(0.0)
        return -control_cost / CONTEXT_TEMPERATURE

    def project(z):
        norm = z.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return z / norm * torch.tanh(norm * 0.5) * (1.0 - eps)

    def distance(a, b):
        # |a-b|^2 = |a|^2 + |b|^2 - 2 a.b: the broadcast form materialised a
        # (batch, codebook, dim) tensor and cost 7 s per epoch, the identity is
        # the same number and costs one matmul.
        if geometry == "euclidean":
            return (a.square().sum(-1)[:, None] + b.square().sum(-1)[None, :]
                    - 2.0 * (a @ b.t())).clamp_min(0.0)
        a, b = project(a), project(b)
        delta_sq = (a.square().sum(-1)[:, None] + b.square().sum(-1)[None, :]
                    - 2.0 * (a @ b.t())).clamp_min(0.0)
        an = a.norm(dim=-1).clamp(max=1 - eps)
        bn = b.norm(dim=-1).clamp(max=1 - eps)
        arg = (2 * delta_sq / ((1 - an.square())[:, None] * (1 - bn.square())[None, :])).clamp_min(0)
        # acosh(1 + arg) has an infinite derivative at arg=0.  The identity
        # acosh(1 + z) = 2 asinh(sqrt(z / 2)) has a finite derivative after
        # squaring: d^2/dz -> 2 as z -> 0.  Clamp only the upper range; preserve
        # the exact zero limit so coincident points have exactly zero distance.
        return _stable_squared_poincare_from_arg(arg)

    positive = bary[batch_ids]
    negative = neg_bary[batch_ids]
    # MODULATE, do not replace. The first iteration let the behaviour/curvature
    # cost replace the assignment entirely, so the argmax drifted to the few
    # codes nearest the barycentre and the first level collapsed into one bucket
    # holding 98.5% of the catalogue (cooccur_overlap_L3 0.0584 against the
    # 0.0756 baseline). The reconstruction-consistent squared Euclidean cost is
    # therefore the base term again and the geodesic contrast enters as a
    # zero-centred, unit-scale bias, weighted by the concentration gate and by
    # CONTEXT_WEIGHT: the Euclidean term still orders the codes, the behaviour
    # term only re-weights them, and no single bucket can absorb the catalogue.
    euclidean_cost = (x.square().sum(-1)[:, None] + codes.square().sum(-1)[None, :]
                      - 2.0 * (x @ codes.t())).clamp_min(0.0)
    behaviour_cost = distance(positive, codes) - 0.5 * distance(negative, codes)
    bias = behaviour_cost - behaviour_cost.mean(dim=-1, keepdim=True)
    bias = bias / bias.abs().mean(dim=-1, keepdim=True).clamp_min(eps)
    gate = (concentration[batch_ids].unsqueeze(-1) if concentration is not None
            else torch.ones_like(bias[:, :1]))
    # Iteration 4. Iterations 1-3 all let the shared context decide the
    # assignment, and the shared context has only as many outcomes as the
    # behaviour signal has regions (a few hundred) while a usable SID needs
    # about 20k triples -- so the partition collapsed no matter the weight or
    # the balancing. The context now enters as an OFFSET on the item's own
    # latent: the item is shifted towards its positive predecessor barycentre
    # and away from its co-consumption hard negative by a bounded fraction of
    # that direction, and the code is still chosen by the squared Euclidean
    # cost of the shifted latent. Two items with the same context keep their
    # own embedding and therefore keep separating, so the outcome count stays
    # at the item level while the behaviour signal still moves the assignment.
    displacement = (positive - x) - 0.5 * (negative - x)
    shifted = x + CONTEXT_WEIGHT * gate * displacement
    shifted_cost = (shifted.square().sum(-1)[:, None] + codes.square().sum(-1)[None, :]
                    - 2.0 * (shifted @ codes.t())).clamp_min(0.0)
    if BALANCED_ASSIGNMENT:
        scores = shifted_cost + CONTEXT_WEIGHT * gate * bias
    else:
        scores = shifted_cost
    return -scores / CONTEXT_TEMPERATURE


def build_context_tables(incoming_sparse, hard_negative, item_count: int, device):
    """Pad the train-only predecessor lists and hard negatives into dense tensors."""
    slots = 8
    index = torch.zeros((item_count, slots), dtype=torch.long)
    weight = torch.zeros((item_count, slots), dtype=torch.float32)
    mask = torch.zeros((item_count, slots), dtype=torch.bool)
    for item, values in incoming_sparse.items():
        if not 0 <= int(item) < item_count:
            continue
        values = values[:slots]
        total = float(sum(v for _, v in values)) or 1.0
        for slot, (predecessor, count) in enumerate(values):
            index[int(item), slot] = int(predecessor)
            weight[int(item), slot] = float(count) / total
            mask[int(item), slot] = True
    negative = torch.arange(item_count, dtype=torch.long)
    for item, value in hard_negative.items():
        if 0 <= int(item) < item_count:
            negative[int(item)] = int(value)
    return index.to(device), weight.to(device), mask.to(device), negative.to(device)


def refresh_context_barycentres(model, embeddings, index, weight, mask, negative, device,
                                geometry: str = "poincare"):
    """Per-epoch Frechet mean of the predecessors' codepoints, for every item.

    The predecessor identities are frozen (train-only); only the codepoints move,
    so the barycentres are recomputed once per epoch instead of once per sample
    per step. ``log0``/``project`` keep the mean inside the ball.
    """
    # unwrap DDP before reaching for the codebook: called with the wrapper this
    # raised "'DistributedDataParallel' object has no attribute 'rq'" on every
    # rank, and the one-rank smoke never exercised it
    raw = model.module if hasattr(model, "module") else model
    layer = raw.rq.vq_layers[0]
    eps = 1e-5

    def project(z):
        norm = z.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return z / norm * torch.tanh(norm * 0.5) * (1.0 - eps)

    def log_map(base, point):
        diff = mobius_add(-base, point)
        norm = diff.norm(dim=-1, keepdim=True).clamp_min(eps)
        bn = base.norm(dim=-1, keepdim=True).clamp(max=1.0 - eps)
        scale = 2.0 / (1.0 - bn.square()).clamp_min(eps)
        return scale * torch.atanh(norm.clamp(max=1.0 - eps)) * diff / norm

    def mobius_add(a, b):
        aa = a.square().sum(-1, keepdim=True); bb = b.square().sum(-1, keepdim=True)
        ab = (a * b).sum(-1, keepdim=True)
        numerator = (1 + 2 * ab + bb) * a + (1 - aa) * b
        denominator = (1 + 2 * ab + aa * bb).clamp_min(eps)
        return project(numerator / denominator)

    def exp_map(base, tangent):
        bn = base.norm(dim=-1, keepdim=True)
        tn = tangent.norm(dim=-1, keepdim=True).clamp_min(eps)
        scaled = torch.tanh((1.0 - bn.square()).clamp_min(eps) * tn / 2) * tangent / tn
        return mobius_add(base, scaled)

    with torch.no_grad():
        codes = layer.get_code_embs().detach()
        curved_codes = project(codes)
        flat = index.reshape(-1)
        unique, inverse = torch.unique(flat, return_inverse=True)
        encoded = raw.encoder(embeddings[unique])
        if geometry == "euclidean":
            region = torch.cdist(encoded, codes).argmin(-1)
        else:
            points = project(encoded)
            c = project(codes)
            delta = torch.cdist(points, c).square().clamp_min(0)
            pn = points.norm(dim=-1).clamp(max=1-eps); cn = c.norm(dim=-1).clamp(max=1-eps)
            d = torch.acosh(1 + (2 * delta / ((1-pn.square())[:, None] * (1-cn.square())[None, :]).clamp_min(eps)).clamp(0, 1e6))
            region = d.argmin(-1)
        regions = region[inverse].view_as(index)
        weights = (weight * mask.float()).unsqueeze(-1)
        def frechet_mean(reg):
            # geometry-off must remove the curvature from the barycentre too, not
            # only from the region choice: the Euclidean control averages the raw
            # codepoints and skips the Karcher iterations entirely.
            if geometry == "euclidean":
                pts = codes[reg]
                w = weights.expand_as(pts)
                return (w * pts).sum(1) / w.sum(1).clamp_min(eps)
            pts = curved_codes[reg]
            w = weights.expand_as(pts)
            mean = project((w * pts).sum(1) / w.sum(1).clamp_min(eps))
            for _ in range(8):
                tangent = log_map(mean[:, None, :].expand_as(pts), pts)
                step = (w * tangent).sum(1) / w.sum(1).clamp_min(eps)
                mean = exp_map(mean, 0.5 * step)
            return mean
        bary = frechet_mean(regions)
        negative_regions = regions[negative]
        neg_bary = frechet_mean(negative_regions)
        # Concentration is over predecessor *regions*, not predecessor IDs:
        # aggregate each item's normalized incoming mass by its detached region.
        # Include all first-codebook regions in the entropy normalization, even
        # when a particular item has zero mass in some regions.
        region_count = curved_codes.shape[0]
        probs = (weight * mask.float()).clamp_min(0)
        probs = probs / probs.sum(-1, keepdim=True).clamp_min(eps)
        region_mass = probs.new_zeros((probs.shape[0], region_count))
        region_mass.scatter_add_(1, regions.long(), probs)
        entropy_terms = torch.where(
            region_mass > 0,
            region_mass * region_mass.clamp_min(eps).log(),
            torch.zeros_like(region_mass),
        )
        entropy = -entropy_terms.sum(-1)
        concentration = (1.0 - entropy / math.log(max(region_count, 2))).clamp(0, 1)
        # An item with no observed training predecessor has an all-zero mask: its
        # region mass is empty, its entropy is zero, and the formula above would
        # call that "maximally concentrated" while its barycentre is the zero
        # vector. Such an item must fall back to the plain Euclidean assignment,
        # so the gate is forced to zero wherever no context exists.
        has_context = mask.any(dim=-1).to(concentration.dtype)
        concentration = concentration * has_context
    return bary.detach(), neg_bary.detach(), concentration.detach()


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
    # Adjacent history->target events define directed training-only counts.
    incoming: dict[int, dict[int, int]] = {}
    user_items: dict[int, set[int]] = {}
    transitioned: set[tuple[int, int]] = set()
    for row in train_frame.itertuples(index=False):
        target = int(row.target); history = [int(v) for v in row.history]
        user_items.setdefault(int(row.user), set()).add(target)
        if history:
            pred = history[-1]
            incoming.setdefault(target, {})[pred] = incoming.setdefault(target, {}).get(pred, 0) + 1
            transitioned.add((pred, target))
    incoming_sparse = {i: sorted(v.items(), key=lambda x: -x[1])[:8] for i,v in incoming.items()}
    # A high-overlap train co-consumption item with no observed incoming edge.
    # The hard negative is the train co-consumption item with the highest Jaccard
    # overlap over users that is *not* an observed predecessor of the item. The
    # overlap is read off the sparse item x item co-consumption matrix instead of
    # intersecting every pair of user sets, which was O(items^2) set work.
    import scipy.sparse as _sp
    rows_: list[int] = []
    cols_: list[int] = []
    data_: list[float] = []
    user_count = np.zeros(len(embeddings), dtype=np.float64)
    for items in user_items.values():
        ordered = sorted(int(v) for v in items if 0 <= int(v) < len(embeddings))
        for value in ordered:
            user_count[value] += 1.0
        for position, left in enumerate(ordered):
            for right in ordered[position + 1:]:
                rows_.append(left); cols_.append(right); data_.append(1.0)
                rows_.append(right); cols_.append(left); data_.append(1.0)
    overlap = _sp.csr_matrix((np.asarray(data_, dtype=np.float32),
                              (np.asarray(rows_), np.asarray(cols_))),
                             shape=(len(embeddings), len(embeddings)))
    overlap.sum_duplicates()
    hard_negative: dict[int, int] = {}
    for item in train_ids.tolist():
        start = overlap.indptr[item]; stop = overlap.indptr[item + 1]
        neighbours = overlap.indices[start:stop]
        counts = overlap.data[start:stop].astype(np.float64)
        if len(neighbours) == 0:
            continue
        union = user_count[item] + user_count[neighbours] - counts
        jaccard = counts / np.maximum(union, 1.0)
        allowed = np.array([other != item and (int(other), int(item)) not in transitioned
                            for other in neighbours])
        if not allowed.any():
            continue
        masked = np.where(allowed, jaccard, -1.0)
        best = int(np.argmax(masked))
        if masked[best] > 0:
            hard_negative[int(item)] = int(neighbours[best])
    config = _tokenizer_config()
    model = RQVAE(config, in_dim=embeddings.shape[1]).to(device)
    if XAVIER_INIT:
        initialize_tiger_weights(model)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    context_index, context_weight, context_mask, context_negative = build_context_tables(
        incoming_sparse, hard_negative, len(embeddings), device,
    )
    train_dataset = TensorDataset(torch.from_numpy(train_ids.copy()))
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
    # iteration 3: balance the coupled assignment on every level's quantizer
    for layer in raw_module.rq.vq_layers:
        layer.balanced_assignment = BALANCED_ASSIGNMENT
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
        if BEHAVIOUR_ON:
            context_bary, context_negative_bary, context_concentration = refresh_context_barycentres(
                model, all_embeddings.to(device), context_index, context_weight,
                context_mask, context_negative, device, GEOMETRY,
            )
        losses: list[float] = []
        recons: list[float] = []
        _epoch_t0 = time.time()
        for (batch_ids,) in loader:
            batch_ids = batch_ids.to(device, non_blocking=True)
            batch = all_embeddings[batch_ids.cpu()].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if BEHAVIOUR_ON:
                assignment_logits = incoming_assignment_loss(
                    raw_module, batch_ids, context_bary, context_negative_bary,
                    all_embeddings.to(device), True, GEOMETRY, context_concentration)
            else:
                assignment_logits = incoming_assignment_loss(
                    raw_module, batch_ids, None, None,
                    all_embeddings.to(device), False, GEOMETRY)
            reconstructed, quant_loss, _, _ = model(batch, first_assignment_logits=assignment_logits)
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
