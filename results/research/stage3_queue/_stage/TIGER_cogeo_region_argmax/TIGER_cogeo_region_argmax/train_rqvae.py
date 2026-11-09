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
        # The marker has to be unambiguous. ``stage2_RQ-VAE`` alone also names
        # results/stage2_RQ-VAE, which sits closer to a tree in the round
        # workspace and would make the repository root resolve to results/.
        if ((parent / "stage2_RQ-VAE").is_dir()
                and (parent / "results").is_dir()
                and (parent / "stage1_GeneEmbedding").is_dir()):
            return parent
    raise RuntimeError("cannot locate the repository root above " + str(here))


REPO_ROOT = _repository_root()
# The arm name is this tree's promoted name; a candidate copy changes only this line.
ARM_NAME = "TIGER_cogeo_region_argmax"
RESULT_DIR = REPO_ROOT / "results" / "stage2_RQ-VAE" / ARM_NAME

EMBEDDING_FILE = REPO_ROOT / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
TRAIN_FILE     = REPO_ROOT / "results" / "stage0_build_parquet" / "train.parquet"
OUTPUT_SID     = RESULT_DIR / "sids_for_hgrec.npy"
OUTPUT_JSON    = RESULT_DIR / "item_sids.json"
CHECKPOINT     = RESULT_DIR / "rqvae_best.pth"
# The metrics and the launcher log are arm artifacts as well: they belong under
# the arm's own result directory, not beside the tree it was promoted from.
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
COGEO_ENABLED       = True
COGEO_CURVATURE     = 1.0
COGEO_TEMPERATURE   = 1.0
COGEO_LAMBDA        = 1.0
COGEO_EPSILON       = 0.01
# The switch of the coupled assignment, as a source constant, so that an
# ablation is a real training arm and not only a comparison inside the smoke:
# "both_on" (the mechanism), "behaviour_off" (the behaviour terms are dropped)
# and "geometry_off" (the curve is replaced by the plain Euclidean distance).
COGEO_MODE          = "both_on"
# The behaviour term is a bounded adjustment of the item's own choice, never a
# replacement of it: the Euclidean distance of the item's latent to the code
# always orders the codes, and the behaviour term may move a code by at most
# this much.
COGEO_CLIP          = 1.0
# The weight of the hard negative context inside the bounded adjustment.
COGEO_ETA           = 0.5
# The weight of the soft constraint that keeps the first-level usage near the
# usage of the model without the mechanism. It is a pull, not a quota: the codes
# are not forced to be equally used.
# The usage pull is a nudge, not a quota: at 0.5 the per-code bias (about 7
# cost units between a used and an unused code) outweighed the Euclidean
# distances and drove the whole catalogue into one code.
COGEO_USAGE_WEIGHT  = 0.02
SINGLE_GPU_ENV      = "GENREC_SINGLE_GPU"
# CLAUDE.md §2: Stage2 不设任何 early-stop — 固定跑满 EPOCHS,
# 候选是否采用完全交由下游 stage3 test_R@10 裁决.
# 保留 best-collision checkpoint 仅作为导出用的最后一次快照.


# === DDP launcher 配置 (与 stage3 不同 master_port 避免冲突) ===
_LAUNCHER = {
    # torchrun lives next to the interpreter that runs this file: no absolute
    # path, and the same interpreter the project pins in CLAUDE.md §2.
    "torchrun":    str(Path(sys.executable).with_name("torchrun")),
    # A path relative to the repository root: the tree is copied into
    # stage2_RQ-VAE/ before it trains, so an absolute path to this file would
    # name the round workspace instead of the promoted tree.
    "script":      str(Path(__file__).resolve().relative_to(REPO_ROOT)),
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


class CoGeoAssignment:
    """Training-only sparse co-consumption profiles plus bounded directed radar."""
    def __init__(self, frame, n_items: int):
        from collections import defaultdict
        # The co-consumption and the transition counts are built with numpy:
        # the same pairs, counted the same way, without one dictionary
        # increment per pair (4.7 million of them, which cost six minutes of
        # start-up on every rank).
        # The behaviour signal is built from each user's longest training
        # sequence, so a pair that several prefixes of the same user share is
        # counted once per user instead of once per prefix.
        longest: dict[int, np.ndarray] = {}
        users = frame["user"] if "user" in frame.columns else range(len(frame))
        for user, hist, target in zip(users, frame["history"], frame["target"]):
            seq = np.asarray(hist, dtype=np.int64).reshape(-1)
            seq = np.append(seq, np.int64(target))
            seq = seq[(seq >= 0) & (seq < n_items)]
            current = longest.get(int(user))
            if current is None or seq.size > current.size:
                longest[int(user)] = seq
        keys: list[np.ndarray] = []
        trans_keys: list[np.ndarray] = []
        for seq in longest.values():
            if seq.size < 2:
                continue
            # every ordered pair of the sequence, both directions
            left, right = np.meshgrid(seq, seq, indexing="ij")
            keep = left != right
            keys.append(left[keep] * np.int64(n_items) + right[keep])
            # every adjacent pair of the sequence, not only the last one
            trans_keys.append(seq[:-1] * np.int64(n_items) + seq[1:])
        if keys:
            merged, counts = np.unique(np.concatenate(keys), return_counts=True)
            self.cooc_rows = (merged // n_items).astype(np.int64)
            self.cooc_cols = (merged % n_items).astype(np.int64)
            # the multiplicity of the pair, as the dictionary construction had
            self.cooc_counts = counts.astype(np.float32)
        else:
            self.cooc_rows = np.zeros(0, dtype=np.int64)
            self.cooc_cols = np.zeros(0, dtype=np.int64)
            self.cooc_counts = np.zeros(0, dtype=np.float32)
        if trans_keys:
            merged_t, counts_t = np.unique(np.concatenate(trans_keys), return_counts=True)
            self.trans_src = (merged_t // n_items).astype(np.int64)
            self.trans_dst = (merged_t % n_items).astype(np.int64)
            self.trans_val = counts_t.astype(np.float32)
        else:
            self.trans_src = np.zeros(0, dtype=np.int64)
            self.trans_dst = np.zeros(0, dtype=np.int64)
            self.trans_val = np.zeros(0, dtype=np.float32)
        self.transitions = defaultdict(float)
        for src, dst, value in zip(self.trans_src, self.trans_dst, self.trans_val):
            self.transitions[int(src), int(dst)] += float(value)
        # The hard negative of every item: the co-consumption neighbour with the
        # largest overlap that never precedes it. Both signals are frozen, so
        # this is computed once.
        predecessors: dict[int, set] = defaultdict(set)
        for src, dst in zip(self.trans_src, self.trans_dst):
            predecessors[int(dst)].add(int(src))
        # the user-set overlap is the Jaccard index, not the raw pair count:
        # count / (users_i + users_j - count)
        self.user_count = np.zeros(n_items, dtype=np.float64)
        for _sequence in longest.values():
            self.user_count[_sequence] += 1.0
        pair_count = self.cooc_counts.astype(np.float64)
        union = (self.user_count[self.cooc_rows] + self.user_count[self.cooc_cols]
                 - pair_count)
        overlap = pair_count / np.maximum(union, 1.0)
        order = np.lexsort((-overlap, self.cooc_rows))
        rows_sorted = self.cooc_rows[order]
        cols_sorted = self.cooc_cols[order]
        boundaries = np.searchsorted(rows_sorted, np.arange(n_items + 1))
        self.negative_index = np.arange(n_items, dtype=np.int64)
        for item in range(n_items):
            start, stop = int(boundaries[item]), int(boundaries[item + 1])
            blocked = predecessors.get(item, set())
            for position in range(start, stop):
                candidate = int(cols_sorted[position])
                if candidate != item and candidate not in blocked:
                    self.negative_index[item] = candidate
                    break
        self.n_items = n_items
        self.profiles = None
        self.prev_assignment = np.zeros(n_items, dtype=np.int64)
        # The frozen co-consumption signal as flat arrays and as one sparse
        # matrix. The per-item Python dictionaries stayed for the record, but
        # every hot path reads these: a per-epoch scatter over five million
        # Python-level increments and a per-step gather of a (codes, items,
        # slots) block were what made this arm an order of magnitude slower
        # than the baseline.
        self.cooc_vals = self.cooc_counts
        # Row-normalised, because the agreement term weighs the neighbours of
        # an item by their share.
        totals = np.zeros(n_items, dtype=np.float64)
        np.add.at(totals, self.cooc_rows, self.cooc_vals.astype(np.float64))
        totals[totals == 0.0] = 1.0
        self.cooc_normalised = (self.cooc_vals / totals[self.cooc_rows]).astype(np.float32)
        self.cooc_sparse = None      # built on the device at the first refresh
        # self.trans_src / self.trans_dst / self.trans_val already hold the
        # numpy construction of the same counts.
        self.P = {}
        self.gate = None
        self.bary = None
        self.neg_bary = None
        self.concentration = None
        self.usage_bias = None
        by_src = defaultdict(lambda: defaultdict(float))
        for (src, dst), count in self.transitions.items():
            by_src[src][dst] += count
        # P[r,k] represented sparsely as per predecessor and code.
        self.transition_rows = by_src
        self._build_padded_tables()

    def _build_padded_tables(self) -> None:
        """Pad the frozen neighbour and predecessor lists into dense tensors.

        The per-item Python loops these replace scanned every row of the
        co-consumption table and every training transition for every item of
        every batch. The padded tables are built once, from the same frozen
        signal, and a step becomes a gather and a weighted sum.
        """
        import numpy as _np
        from collections import defaultdict as _defaultdict
        predecessor_slots = 8
        incoming = _defaultdict(list)
        for (src, dst), count in self.transitions.items():
            incoming[int(dst)].append((int(src), float(count)))
        p_index = _np.zeros((self.n_items, predecessor_slots), dtype=_np.int64)
        p_weight = _np.zeros((self.n_items, predecessor_slots), dtype=_np.float32)
        p_mask = _np.zeros((self.n_items, predecessor_slots), dtype=bool)
        for item, pairs in incoming.items():
            if not 0 <= item < self.n_items:
                continue
            pairs = sorted(pairs, key=lambda pair: -pair[1])[:predecessor_slots]
            total = float(sum(value for _, value in pairs)) or 1.0
            for slot, (source, value) in enumerate(pairs):
                p_index[item, slot] = source
                p_weight[item, slot] = value / total
                p_mask[item, slot] = True
        self.predecessor_index = torch.as_tensor(p_index)
        self.predecessor_weight = torch.as_tensor(p_weight)
        self.predecessor_mask = torch.as_tensor(p_mask)
        self.P_tensor = None      # filled by refresh(), which knows the code count

    def _contexts(self, assignments, k: int, device):
        """The per-item context: the predecessor barycentre, the negative's, the gate.

        The context is a mean of codepoints, so items that share a context
        still separate by their own latent; and the gate is zero for an item
        without an observed predecessor, so an empty context never moves an
        assignment.
        """
        eps = 1e-5

        def project(values):
            norm = values.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            return values / norm * torch.tanh(norm * 0.5) * (1.0 - eps)

        def log0(values):
            norm = values.norm(dim=-1, keepdim=True).clamp(max=1.0 - eps)
            return 2.0 * torch.atanh(norm) * values / norm.clamp_min(eps)

        geometry = getattr(self, "mode", "both_on") == "geometry_off" and "euclidean" or "poincare"

        def to_ball(values):
            return values if geometry == "euclidean" else project(values)

        def from_ball(values):
            return values if geometry == "euclidean" else project(values)

        def tangent_of(values):
            return values if geometry == "euclidean" else log0(project(values))

        codes = self._codes
        tangent = tangent_of(codes)
        regions = torch.as_tensor(assignments, dtype=torch.long, device=device)
        source_code = tangent[regions[self.trans_src]]
        destination = torch.as_tensor(self.trans_dst, dtype=torch.long, device=device)
        weight = torch.as_tensor(self.trans_val, dtype=torch.float32, device=device)
        # The context of an item is the mean over ALL its observed predecessors,
        # not over a capped list of them: the transition edges are the signal.
        mass = torch.zeros((self.n_items, 1), dtype=torch.float32, device=device)
        mass.index_add_(0, destination, weight.unsqueeze(-1))
        bary = torch.zeros((self.n_items, tangent.shape[-1]), dtype=torch.float32, device=device)
        bary.index_add_(0, destination, source_code * weight.unsqueeze(-1))
        bary = from_ball(bary / mass.clamp_min(1e-9))
        # the same construction for the hard negative of the item
        negative = self.negative_index
        if not torch.is_tensor(negative):
            negative = torch.as_tensor(negative, dtype=torch.long, device=device)
        # the hard negative's own context, from the same construction
        negative_bary = bary[negative.clamp_min(0)]
        gate = (mass.squeeze(-1) > 0).float() * self.concentration
        return gate, bary, negative_bary

    def refresh(self, assignments, k: int, device):
        """The region profiles and the directed radar, rebuilt from the codes.

        Both are one ``bincount`` over the frozen signal: the profile of a code
        is the mean of the co-consumption rows of the items in it, and the radar
        of a source is the distribution of the codes of its successors.
        """
        self.prev_assignment = assignments.astype(np.int64, copy=False)
        # the predecessor tables live on the training device from here on
        self.predecessor_index = self.predecessor_index.to(device)
        self.predecessor_weight = self.predecessor_weight.to(device)
        self.predecessor_mask = self.predecessor_mask.to(device)
        self.negative_index = torch.as_tensor(self.negative_index, device=device)
        counts = np.bincount(assignments, minlength=k)
        if len(self.cooc_rows):
            codes = assignments[self.cooc_rows]
            flat = codes.astype(np.int64) * self.n_items + self.cooc_cols
            profiles = np.bincount(flat, weights=self.cooc_normalised,
                                   minlength=k * self.n_items).reshape(
                                       k, self.n_items).astype(np.float32)
        else:
            profiles = np.zeros((k, self.n_items), dtype=np.float32)
        profiles /= np.maximum(counts[:, None], 1)
        norms = np.linalg.norm(profiles, axis=1, keepdims=True)
        profiles /= np.maximum(norms, 1e-12)
        # (items, codes): the agreement term multiplies the item's own
        # neighbours with this table, so the item axis comes first.
        self.profiles = torch.as_tensor(np.ascontiguousarray(profiles.T), device=device)
        if len(self.trans_src):
            codes = assignments[self.trans_dst]
            flat = self.trans_src.astype(np.int64) * k + codes.astype(np.int64)
            dense_p = np.bincount(flat, weights=self.trans_val,
                                  minlength=self.n_items * k).reshape(
                                      self.n_items, k).astype(np.float32)
            totals = dense_p.sum(axis=1, keepdims=True)
            totals[totals == 0.0] = 1.0
            dense_p /= totals
        else:
            dense_p = np.zeros((self.n_items, k), dtype=np.float32)
        self.P_tensor = torch.as_tensor(dense_p, device=device)
        if getattr(self, "_codes", None) is None:
            self._codes = torch.zeros((k, 32), dtype=torch.float32, device=device)
        # The incoming-context concentration: the entropy of the predecessor
        # region mass, over the regions of the first level. It is zero where an
        # item has no observed predecessor.
        regions = torch.as_tensor(assignments, dtype=torch.long, device=device)
        predecessor_regions = regions[self.predecessor_index.clamp_min(0)]
        predecessor_weight = self.predecessor_weight * self.predecessor_mask.float()
        mass = torch.zeros((self.n_items, k), dtype=torch.float32, device=device)
        mass.scatter_add_(1, predecessor_regions, predecessor_weight)
        totals = mass.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        distribution = mass / totals
        entropy = -(distribution.clamp_min(1e-12) * distribution.clamp_min(1e-12).log()).sum(-1)
        concentration = 1.0 - entropy / float(np.log(max(k, 2)))
        has_context = (self.predecessor_mask.any(dim=-1)).float()
        self.concentration = (concentration * has_context).clamp(0.0, 1.0)
        if self.cooc_sparse is None:
            indices = torch.as_tensor(
                np.stack((self.cooc_rows, self.cooc_cols)), dtype=torch.long,
                device=device)
            values = torch.as_tensor(self.cooc_normalised, device=device)
            self.cooc_sparse = torch.sparse_coo_tensor(
                indices, values, (self.n_items, self.n_items)).coalesce()
        # The agreement of every item with every region, once per epoch: the
        # profiles only move when the assignment is refreshed, so this product
        # is done here instead of inside the training step.
        self.agree_all = torch.sparse.mm(self.cooc_sparse, self.profiles)
        # The context of every item: the Frechet mean of the codepoints its
        # predecessors are assigned to, the same for its hard negative, and the
        # gate that is zero where there is no predecessor at all.
        self.gate, self.bary, self.neg_bary = self._contexts(assignments, k, device)
        # The incoming transition mass of every item: the normalised counts of
        # the transitions that end in it, spread over the radar of their
        # sources. The whole incoming mass is used, not the largest few
        # predecessors.
        incoming = None
        if len(self.trans_src):
            totals = np.zeros(self.n_items, dtype=np.float32)
            np.add.at(totals, self.trans_dst, self.trans_val)
            totals[totals == 0.0] = 1.0
            weights = self.trans_val / totals[self.trans_dst]
            incoming = torch.sparse_coo_tensor(
                torch.as_tensor(np.stack((self.trans_dst, self.trans_src)),
                                dtype=torch.long, device=device),
                torch.as_tensor(weights, device=device),
                (self.n_items, self.n_items)).coalesce()
            self.directed_all = torch.sparse.mm(incoming, self.P_tensor)
        else:
            self.directed_all = torch.zeros((self.n_items, k), dtype=torch.float32,
                                            device=device)
        # the predecessor tables were moved to the device at the top of refresh
        self.predecessor_weight = self.predecessor_weight.to(device)
        self.predecessor_mask = self.predecessor_mask.to(device)

    def scores(self, ids, euclidean_dist, code_embs):
        """The coupled score: the item chooses, the context adjusts.

        The squared Euclidean distance from the item's own latent to the
        codewords is always the base of the score, so the item's representation
        decides which code it gets. The behaviour context is a bounded
        adjustment on top of it: the curved contrast between the positive
        predecessor context and the hard negative context, centred and clipped,
        and scaled by the incoming-context concentration, which is zero for an
        item without an observed predecessor.
        """
        if self.profiles is None:
            raise RuntimeError("co-geometry profiles must be initialized before assignment")
        ids_np = ids.detach().cpu().numpy().astype(np.int64)
        device = euclidean_dist.device
        z = getattr(self, "current_latent", None)
        if z is None or len(z) != len(ids_np):
            raise RuntimeError("missing latent context for Poincare assignment")
        c = COGEO_CURVATURE

        def project(values):
            norm = values.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            return values * torch.clamp((1.0 - 1e-5) / (norm * (c ** 0.5)), max=1.0)

        def geodesic(points):
            delta = (points[:, None, :] - y[None, :, :]).pow(2).sum(-1)
            nx = (points * points).sum(-1, keepdim=True)
            ny = (y * y).sum(-1).unsqueeze(0)
            arg = (2 * delta / ((1 - c * nx).clamp_min(1e-6) * (1 - c * ny).clamp_min(1e-6))).clamp_min(0)
            return torch.acosh((1 + arg).clamp_min(1 + 1e-7)) / (c ** 0.5)

        x, y = project(z), project(code_embs)
        mode = getattr(self, "mode", "both_on")
        # The item's own representation chooses the code: the squared Euclidean
        # distance to the codewords is always the base of the score.
        cost = euclidean_dist.clone()
        if mode in ("both_on", "geometry_off"):
            # The behaviour term compares the CANDIDATE CODE with the two
            # contexts: the positive predecessor context and the hard negative
            # context. It is not the distance from the item's own latent, which
            # is the base term already.
            positive = self.bary[ids_np]
            negative = self.neg_bary[ids_np]
            if mode == "geometry_off":
                # the Euclidean counterpart of both contexts, in the raw space:
                # no Poincare projection anywhere in the control
                raw_codes = code_embs
                hyp = (positive[:, None, :] - raw_codes[None, :, :]).pow(2).sum(-1)
                neg_hyp = (negative[:, None, :] - raw_codes[None, :, :]).pow(2).sum(-1)
            else:
                hyp = geodesic(project(positive))
                neg_hyp = geodesic(project(negative))
            behaviour = hyp.square() - COGEO_ETA * neg_hyp.square()
            behaviour = behaviour - behaviour.mean(dim=-1, keepdim=True)
            behaviour = behaviour.clamp(-COGEO_CLIP, COGEO_CLIP)
            gate = self.gate[ids_np].unsqueeze(-1)
            cost = cost + COGEO_LAMBDA * gate * behaviour
        # The soft usage constraint: a pull of the first-level usage towards the
        # usage of the model without the mechanism, not a quota. It belongs to
        # the mechanism, so the behaviour-off control does not apply it: that
        # control has to be exactly the plain Euclidean assignment.
        if self.usage_bias is not None and mode != "behaviour_off":
            cost = cost + self.usage_bias.unsqueeze(0)
        return -cost

    def __call__(self, ids, euclidean_dist, code_embs):
        """The hard choice of the coupled score."""
        return self.scores(ids, euclidean_dist, code_embs).argmax(-1)

    def compare_variants(self, ids, euclidean_dist, code_embs) -> dict:
        """How many items each variant moves away from the plain Euclidean choice.

        The three variants see the same batch and the same codebook, so the
        difference between them is the effect of the coupled score rather than
        of a different input.
        """
        plain = euclidean_dist.argmin(-1)
        out = {}
        choices = {}
        for mode in ("both_on", "behaviour_off", "geometry_off"):
            self.mode = mode
            chosen = self(ids, euclidean_dist, code_embs)
            out[mode] = int((chosen != plain).sum())
            choices[mode] = [int(value) for value in chosen.tolist()]
        self.mode = "both_on"
        out["items"] = int(len(plain))
        out["plain"] = [int(value) for value in plain.tolist()]
        out["choices"] = choices
        return out


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
    config = _tokenizer_config()
    model = RQVAE(config, in_dim=embeddings.shape[1]).to(device)
    if XAVIER_INIT:
        initialize_tiger_weights(model)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    # The unwrapped module is needed from here on: the first-level layer, the
    # initial encoding and the export all read the codebook and the encoder off
    # it. It used to be assigned further down, which made the trainer die with
    # an UnboundLocalError before the first batch.
    raw_module = model.module if isinstance(model, DDP) else model

    train_dataset = TensorDataset(train_embeddings, torch.from_numpy(train_ids.astype(np.int64)))
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
    cogeo = CoGeoAssignment(train_frame, len(all_embeddings)) if COGEO_ENABLED else None
    if cogeo is not None:
        cogeo.mode = COGEO_MODE
    first_layer = raw_module.rq.vq_layers[0]
    if cogeo is not None:
        with torch.no_grad():
            initial_encoded = raw_module.encoder(all_embeddings.to(device))
            # The hook is installed only once it can answer: it needs the region
            # profiles, and the first refresh is what builds them. Installing it
            # earlier leaves a window in which a forward would call a hook with
            # no profiles.
            initial_codes = first_layer._indices(first_layer._distances(initial_encoded), False).cpu().numpy()
            # the context codepoints come from the live first-level codebook
            cogeo._codes = first_layer.get_code_embs().detach()
            cogeo.refresh(initial_codes, first_layer.n_embed, device)
            # The soft usage constraint pulls the first-level usage towards the
            # usage of the model without the mechanism. It is a pull, not a quota:
            # the codes are not forced to be equally used.
            reference = np.bincount(initial_codes, minlength=first_layer.n_embed)
            cogeo.usage_reference = torch.as_tensor(
                reference.astype(np.float32) / max(reference.sum(), 1), device=device)
            first_layer.region_assignment = cogeo

    if rank == 0:
        print(f"[RecBole RQ-VAE] device={device} all_items={len(all_embeddings)} train_items={len(train_embeddings)} world_size={world_size}", flush=True)
        print(f"[RecBole RQ-VAE] config: epochs={EPOCHS} batch_size_per_rank={BATCH_SIZE_PER_RANK} total_batch={BATCH_SIZE_PER_RANK*world_size} lr={LR}", flush=True)
        _record(rank, "train_start", epochs=EPOCHS, batch_size_per_rank=BATCH_SIZE_PER_RANK, total_batch_size=BATCH_SIZE_PER_RANK*world_size, lr=LR, weight_decay=WEIGHT_DECAY, hidden_sizes=list(HIDDEN_SIZES), codebook_num=CODEBOOK_NUM, codebook_size=list(CODEBOOK_SIZE), codebook_dim=CODEBOOK_DIM, beta=BETA, vq_type=VQ_TYPE, ema_decay=EMA_DECAY, sk_epsilon=SK_EPSILON, sk_iters=SK_ITERS, pca_dim=PCA_DIM, xavier_init=XAVIER_INIT, warmup_epochs=WARMUP_EPOCHS, gradient_clip_norm=GRADIENT_CLIP_NORM, optimizer=OPTIMIZER, seed=SEED, num_workers=NUM_WORKERS, eval_interval=EVAL_INTERVAL, early_stop="disabled", all_items=len(all_embeddings), train_items=len(train_embeddings), embedding_shape=list(embeddings.shape), world_size=world_size)

    # === codebook init: only rank 0 (codebook 不是 DDP 参数, 全 rank 共享) ===
    if rank == 0:
        print("[RecBole RQ-VAE] initializing codebooks with KMeans", flush=True)
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
        for batch, batch_ids in loader:
            batch = batch.to(device, non_blocking=True)
            first_layer.region_item_ids = batch_ids
            optimizer.zero_grad(set_to_none=True)
            reconstructed, quant_loss, _, _ = model(batch)
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
        if cogeo is not None:
            with torch.no_grad():
                encoded = raw_module.encoder(all_embeddings.to(device))
                # Refresh the profiles from the assignment the coupled rule
                # itself produced, not from the plain Euclidean one: the hook
                # stays in place and is fed the whole catalogue and its latent.
                # The profiles therefore move with the coupled assignment, which
                # is what the proposal says they are built from.
                first_layer.region_item_ids = torch.arange(len(all_embeddings), device=device)
                cogeo.current_latent = encoded
                assignments = first_layer._indices(
                    first_layer._distances(encoded), False).cpu().numpy()
                # the codebook moved during the epoch, so the context codepoints
                # are refreshed with it
                cogeo._codes = first_layer.get_code_embs().detach()
                cogeo.refresh(assignments, first_layer.n_embed, device)
                usage = np.bincount(assignments, minlength=first_layer.n_embed)
                usage = usage.astype(np.float32) / max(usage.sum(), 1)
                reference = cogeo.usage_reference.cpu().numpy()
                bias = COGEO_USAGE_WEIGHT * (np.log(usage + 1e-6) - np.log(reference + 1e-6))
                cogeo.usage_bias = torch.as_tensor(bias, dtype=torch.float32, device=device)
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
                    # The assignment is item-ID aware: the export encodes every
                    # item, so the ids have to be the whole catalogue and not the
                    # last training batch.
                    first_layer.region_item_ids = torch.arange(
                        len(all_embeddings), device=device)
                    first_layer.region_item_ids = torch.arange(
                    len(all_embeddings), device=device)
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
        rc = subprocess.call(cmd, stdout=fout, stderr=subprocess.STDOUT, env=env,
                             cwd=str(REPO_ROOT))
    sys.exit(rc)


if __name__ == "__main__":
    _launch_via_torchrun()
    main()
