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

from distances import build_distance
from losses import build_loss
from model import RQVAE
from train_config import load_train_config
# Issue #287: the curvature-selected relative pair ranking on the L1 selected codeword.
# The frozen pair index and the frozen margin are Issue #274's; only the supervised distance
# changes (the L1 codeword the model actually selected instead of the sum of the three levels).
from pair_rank_loss import (LEVEL, PairIndex, hard_level_embedding, pair_ranking_diagnostics,
                            pair_ranking_loss)


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
        if ((parent / "stage2_RQ-VAE").is_dir()
                and (parent / "tools/research/stage2/contract.py").is_file()):
            return parent
    raise RuntimeError("cannot locate the repository root above " + str(here))


REPO_ROOT = _repository_root()
import sys as _sys
_sys.path.insert(0, str(REPO_ROOT))
# The arm name is this tree's promoted name; a candidate copy changes only this line.
ARM_NAME = "TIGER_l1_budget10_rq"
RESULT_DIR = REPO_ROOT / "results" / "stage2_RQ-VAE" / ARM_NAME

EMBEDDING_FILE = REPO_ROOT / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
TRAIN_FILE     = REPO_ROOT / "results" / "stage0_build_parquet" / "train.parquet"
OUTPUT_SID     = RESULT_DIR / "sids_for_hgrec.npy"
OUTPUT_JSON    = RESULT_DIR / "item_sids.json"
CHECKPOINT     = RESULT_DIR / "rqvae_best.pth"
LOG_DIR        = REPO_ROOT / "stage2_RQ-VAE" / ARM_NAME / "logs"
METRICS_PATH   = LOG_DIR / "training_metrics.jsonl"

# === Issue #288 R02: the frozen #274 pair index, the frozen margin and the raised gradient budget ===
# The pair index, the L1-codeword supervision and the margin are the Issue #287 R1 form (the only
# arm that passed P and C simultaneously on one export: P 0.083532, C 0.150933, N 0.009392).
# The single main change of R02 is the **gradient budget**: the weighted pair gradient may now reach
# 10 % of the reconstruction gradient per module instead of 2 %, because the measured L1 assignment
# change under the 2 % budget was only 0.85 % and the in-bucket behavioural purity has stayed ~0.35
# across five mechanisms (the pair term is too weak to reshape the partition).
# The pair index is selected offline from the real graph's sectional curvature and is read-only
# for the model: the curvature never enters the loss, the margin or a sample's weight.
PAIR_INDEX_FILE   = REPO_ROOT / "results" / "research" / "issue274" / "pairs_index.npz"
# The frozen margin, calibrated in advance on the frozen TIGER's training-pair distance scale.
PAIR_MARGIN       = 0.2529585361480713
# The number of ranking groups per optimizer step.
PAIR_GROUPS       = 256
# The gradient budget: the weighted pair gradient must stay below this share of the
# reconstruction gradient, module by module (the three codebooks).
PAIR_BUDGET       = 0.10
# The progressive enablement: the pair term ramps in over this many optimizer steps.
PAIR_RAMP_STEPS   = 200
# The number of items the first step probes for the real L1 effect (issue #287 section 3).
PROBE_ITEMS       = 4096


# === 超参来自 config/train_config.json（严格校验：缺键/多键/类型/范围）===
# 一个 JSON 文件、严格校验：缺键/多键/类型错/越界都在碰 GPU 之前 ConfigError。
# 路径与 ARM_NAME 不在配置里 —— 它们归框架 (CLAUDE.md 19b0/67)。
CONFIG_FILE = Path(__file__).resolve().with_name("config") / "train_config.json"
CONFIG = load_train_config(CONFIG_FILE)
# 损失与距离都由注册表给出；训练文件只负责导入并把它们接到模型上。
CRITERION = build_loss(CONFIG)
DISTANCE = build_distance(CONFIG)

EPOCHS              = CONFIG.EPOCHS
BATCH_SIZE_PER_RANK = CONFIG.BATCH_SIZE_PER_RANK     # 每卡 batch, 总 batch = 每卡 * world_size
LR                  = CONFIG.LR
WEIGHT_DECAY        = CONFIG.WEIGHT_DECAY
HIDDEN_SIZES        = CONFIG.HIDDEN_SIZES
CODEBOOK_NUM        = CONFIG.CODEBOOK_NUM
CODEBOOK_SIZE       = CONFIG.CODEBOOK_SIZE
CODEBOOK_DIM        = CONFIG.CODEBOOK_DIM
BETA                = CONFIG.BETA
VQ_TYPE             = CONFIG.VQ_TYPE
EMA_DECAY           = CONFIG.EMA_DECAY
SK_EPSILON          = CONFIG.SK_EPSILON
SK_ITERS            = CONFIG.SK_ITERS
PCA_DIM             = CONFIG.PCA_DIM
XAVIER_INIT         = CONFIG.XAVIER_INIT
WARMUP_EPOCHS       = CONFIG.WARMUP_EPOCHS
GRADIENT_CLIP_NORM  = CONFIG.GRADIENT_CLIP_NORM
OPTIMIZER           = CONFIG.OPTIMIZER
SEED                = CONFIG.SEED
NUM_WORKERS         = CONFIG.NUM_WORKERS
EVAL_INTERVAL       = CONFIG.EVAL_INTERVAL
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


def _pair_terms(raw_module, catalogue, index, rng, groups):
    """One step's pair-ranking term: the frozen pairs' L1 codewords and their hinge loss.

    The pair items' codes are selected with the same assignment the training forward uses
    (the Sinkhorn path in training mode); the selection is an argmin, so the codes are constants
    with respect to the encoder and the loss reaches the **L1 codebook only**.
    """
    selected = index.sample(rng, groups)
    items = index.items_of(selected)
    if items.size == 0:
        return None, None, None
    rows = index.rows_of(selected, items)
    with torch.no_grad():
        for layer in raw_module.rq.vq_layers:
            layer._skip_ddp_reduce = True
        try:
            tokens = raw_module.get_indices(catalogue[items])
        finally:
            for layer in raw_module.rq.vq_layers:
                layer._skip_ddp_reduce = False
    q = hard_level_embedding(tokens, raw_module.rq.vq_layers, LEVEL)
    return pair_ranking_loss(q, *rows, PAIR_MARGIN), q, rows


def _calibrate_pair_weight(raw_module, loader, catalogue, index, device, *,
                           budget=PAIR_BUDGET, batches=8):
    """Freeze lambda by the gradient budget on the real batches.

    The weighted pair-ranking gradient must stay below ``budget`` times the reconstruction
    gradient for every codebook module; the frozen weight is the largest admissible one, so the
    pair term can never dominate the reconstruction.  No loss value is used to pick the weight.
    The measurement uses the real training loss and the real batches, never a stand-in loss.

    Issue #287: the pair term reaches the **L1 codebook only** (the supervised level), so the
    budget is binding on L1 and the L2/L3 codebooks keep their reconstruction gradient untouched.
    The encoder receives no pair gradient at all, because the level selection is an argmin.
    """
    rng = np.random.default_rng(SEED)
    codebooks = tuple(layer.embed.weight for layer in raw_module.rq.vq_layers)
    encoder_params = tuple(raw_module.encoder.parameters())
    ratios = []
    cosines = []
    pair_norms = []
    recon_norms = []
    pair_encoder_norms = []
    recon_encoder_norms = []
    for batch_index, (batch,) in enumerate(loader):
        if batch_index >= batches:
            break
        batch = batch.to(device, non_blocking=True)
        reconstructed, quant_parts, _, _ = raw_module(batch)
        original, _components = CRITERION(reconstructed, batch, quant_parts)
        original_g = torch.autograd.grad(original, codebooks, retain_graph=True, allow_unused=True)
        original_g = tuple(torch.zeros_like(p) if g is None else g
                           for p, g in zip(codebooks, original_g))
        original_enc = torch.autograd.grad(original, encoder_params, retain_graph=True, allow_unused=True)
        original_enc = tuple(torch.zeros_like(p) if g is None else g
                             for p, g in zip(encoder_params, original_enc))
        pair_loss, _q, _rows = _pair_terms(raw_module, catalogue, index, rng, PAIR_GROUPS)
        if pair_loss is None:
            continue
        pair_g = torch.autograd.grad(pair_loss, codebooks, retain_graph=True, allow_unused=True)
        pair_g = tuple(torch.zeros_like(p) if g is None else g for p, g in zip(codebooks, pair_g))
        pair_enc = torch.autograd.grad(pair_loss, encoder_params, retain_graph=False, allow_unused=True)
        pair_enc = tuple(torch.zeros_like(p) if g is None else g for p, g in zip(encoder_params, pair_enc))
        pair_norms.append([float(torch.linalg.vector_norm(g)) for g in pair_g])
        recon_norms.append([float(torch.linalg.vector_norm(g)) for g in original_g])
        pair_encoder_norms.append(float(torch.linalg.vector_norm(
            torch.cat([g.reshape(-1) for g in pair_enc]))))
        recon_encoder_norms.append(float(torch.linalg.vector_norm(
            torch.cat([g.reshape(-1) for g in original_enc]))))
        for original_grad, pair_grad in zip(original_g, pair_g):
            original_norm = float(torch.linalg.vector_norm(original_grad))
            pair_norm = float(torch.linalg.vector_norm(pair_grad))
            if original_norm <= 0.0 or pair_norm <= 0.0:
                continue
            ratios.append(original_norm / pair_norm)
            cosines.append(float((original_grad.double() * pair_grad.double()).sum()
                                  / (original_norm * pair_norm)))
    if not ratios:
        raise RuntimeError('the pair-ranking calibration produced no measurable gradient')
    weight = float(budget * min(ratios))
    if not np.isfinite(weight) or weight <= 0:
        raise RuntimeError('the calibrated pair weight is not representable as a positive float')
    return weight, {
        'lambda': weight,
        'budget': float(budget),
        'measured_batches': len(ratios),
        'binding_ratio': float(min(ratios)),
        'median_ratio': float(np.median(ratios)),
        'reconstruction_cosine': float(np.mean(cosines)) if cosines else None,
        'margin': PAIR_MARGIN,
        'groups_per_step': PAIR_GROUPS,
        'ramp_steps': PAIR_RAMP_STEPS,
        'calibration_batch_items': int(PAIR_GROUPS * 4),
        'supervised_level': int(LEVEL),
        'pair_grad_norm_per_codebook': [round(float(np.mean([n[i] for n in pair_norms])), 8)
                                        for i in range(len(codebooks))],
        'recon_grad_norm_per_codebook': [round(float(np.mean([n[i] for n in recon_norms])), 8)
                                         for i in range(len(codebooks))],
        'pair_grad_norm_encoder': round(float(np.mean(pair_encoder_norms)), 8),
        'recon_grad_norm_encoder': round(float(np.mean(recon_encoder_norms)), 8),
    }


def main() -> None:
    # === DDP 初始化 ===
    if _single_card_mode():
        # One process, one rank, no process group: every collective below is
        # guarded by world_size > 1 and therefore skipped.
        rank, world_size, local_rank = 0, 1, 0
        global BATCH_SIZE_PER_RANK
        BATCH_SIZE_PER_RANK = SINGLE_CARD_BATCH
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
        print("[single card] one process, one rank, per-process batch %d"
              % BATCH_SIZE_PER_RANK, flush=True)
    elif "RANK" in os.environ and int(os.environ.get("RANK", -1)) >= 0:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if world_size == 1:
            # A one-rank group is the single-card case reached through torchrun:
            # keep the four-rank global batch so the optimisation does not move.
            BATCH_SIZE_PER_RANK = SINGLE_CARD_BATCH
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
    model = RQVAE(config, in_dim=embeddings.shape[1], distance=DISTANCE).to(device)
    if XAVIER_INIT:
        initialize_tiger_weights(model)
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
        _record(rank, "train_start", config_sha256=CONFIG.sha256, config_path=CONFIG.path, epochs=EPOCHS, batch_size_per_rank=BATCH_SIZE_PER_RANK, total_batch_size=BATCH_SIZE_PER_RANK*world_size, lr=LR, weight_decay=WEIGHT_DECAY, hidden_sizes=list(HIDDEN_SIZES), codebook_num=CODEBOOK_NUM, codebook_size=list(CODEBOOK_SIZE), codebook_dim=CODEBOOK_DIM, beta=BETA, vq_type=VQ_TYPE, ema_decay=EMA_DECAY, sk_epsilon=SK_EPSILON, sk_iters=SK_ITERS, pca_dim=PCA_DIM, xavier_init=XAVIER_INIT, warmup_epochs=WARMUP_EPOCHS, gradient_clip_norm=GRADIENT_CLIP_NORM, optimizer=OPTIMIZER, seed=SEED, num_workers=NUM_WORKERS, eval_interval=EVAL_INTERVAL, early_stop="disabled", all_items=len(all_embeddings), train_items=len(train_embeddings), embedding_shape=list(embeddings.shape), world_size=world_size)

    # === codebook init: only rank 0 (codebook 不是 DDP 参数, 全 rank 共享) ===
    if rank == 0:
        print("[RecBole RQ-VAE] initializing codebooks with KMeans", flush=True)
    raw_module = model.module if isinstance(model, DDP) else model
    with torch.no_grad():
        raw_module.init_codebook(train_embeddings.to(device))
    if world_size > 1:
        dist.barrier()

    # === Issue #274: the frozen pair index and the gradient-budget-calibrated pair weight ===
    pair_index = PairIndex.load(PAIR_INDEX_FILE)
    pair_catalogue = all_embeddings.to(device)
    pair_rng = np.random.default_rng(SEED + 1)
    pair_weight, pair_calibration = _calibrate_pair_weight(
        raw_module, loader, pair_catalogue, pair_index, device)
    if world_size > 1:
        # the codebooks are shared across the ranks, so the weight must be identical on every rank
        holder = torch.tensor([pair_weight], device=device)
        dist.all_reduce(holder, op=dist.ReduceOp.MIN)
        pair_weight = float(holder.item())
    pair_calibration['lambda'] = pair_weight
    if rank == 0:
        _record(rank, "pair_calibration", **pair_calibration)
        print(f"[RQ-VAE] pair-ranking weight={pair_weight:.6e} margin={PAIR_MARGIN:.6f} "
              f"groups={PAIR_GROUPS} ramp={PAIR_RAMP_STEPS}", flush=True)

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
        pair_epoch: list[dict] = []
        _epoch_t0 = time.time()
        for step, (batch,) in enumerate(loader):
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            reconstructed, quant_parts, _, _ = model(batch)
            # 唯一接入点：损失项与权重在 losses/ 里，这里只拿到总损失与分项。
            loss, components = CRITERION(reconstructed, batch, quant_parts)
            recon_loss = components["reconstruction"]
            # Issue #287: the relative pair ranking on the L1 selected codeword.  The weight is
            # frozen by the gradient budget and ramps in; the pairs come from the frozen #274 index.
            global_step = (epoch - 1) * len(loader) + step
            pair_scale = min(1.0, global_step / PAIR_RAMP_STEPS)
            pair_loss, pair_q, pair_rows = _pair_terms(
                raw_module, pair_catalogue, pair_index, pair_rng, PAIR_GROUPS)
            if pair_loss is not None:
                loss = loss + pair_weight * pair_scale * pair_loss
                pair_stats = pair_ranking_diagnostics(pair_q, *pair_rows, PAIR_MARGIN)
            else:
                pair_stats = None
            if not torch.isfinite(loss):
                raise RuntimeError(f"RQ-VAE loss became non-finite at epoch {epoch}")
            probe_before = None
            if epoch == 1 and step == 0:
                # Issue #287: the real L1 effect probe - the L1 assignments and the L1 codebook
                # before the first optimizer step, so the step's effect on the L1 prefix is measured.
                with torch.no_grad():
                    for layer in raw_module.rq.vq_layers:
                        layer._skip_ddp_reduce = True
                    try:
                        probe_before = {
                            'tokens': raw_module.get_indices(pair_catalogue[:PROBE_ITEMS]).clone(),
                            'l1_codebook': raw_module.rq.vq_layers[0].embed.weight.detach().clone(),
                        }
                    finally:
                        for layer in raw_module.rq.vq_layers:
                            layer._skip_ddp_reduce = False
            loss.backward()
            if GRADIENT_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()
            scheduler.step()
            if probe_before is not None:
                with torch.no_grad():
                    for layer in raw_module.rq.vq_layers:
                        layer._skip_ddp_reduce = True
                    try:
                        tokens_after = raw_module.get_indices(pair_catalogue[:PROBE_ITEMS])
                    finally:
                        for layer in raw_module.rq.vq_layers:
                            layer._skip_ddp_reduce = False
                l1_changed = float((probe_before['tokens'][:, 0] != tokens_after[:, 0]).float().mean())
                l1_moved = float(torch.linalg.vector_norm(
                    raw_module.rq.vq_layers[0].embed.weight.detach() - probe_before['l1_codebook']))
                probe = {'l1_assignment_change_rate': l1_changed, 'l1_codebook_step_norm': l1_moved,
                         'probe_items': int(PROBE_ITEMS), 'pair_weight': float(pair_weight),
                         'pair_scale': float(pair_scale)}
                if rank == 0:
                    _record(rank, "l1_probe", **probe)
                    print(f"[RQ-VAE] l1_probe change_rate={l1_changed:.6f} "
                          f"codebook_step_norm={l1_moved:.6e}", flush=True)
            losses.append(float(loss.detach()))
            recons.append(float(recon_loss.detach()))
            if pair_stats is not None:
                pair_epoch.append(pair_stats)
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
        pair_summary = {}
        if pair_epoch:
            pair_summary = {
                'pair_loss': float(np.mean([s['pair_loss'] for s in pair_epoch])),
                'pair_ranking_accuracy': float(np.mean([s['ranking_accuracy'] for s in pair_epoch])),
                'pair_margin_satisfaction_rate': float(
                    np.mean([s['margin_satisfaction_rate'] for s in pair_epoch])),
                'pair_mean_distance_gap': float(np.mean([s['mean_distance_gap'] for s in pair_epoch])),
                'pair_weight': float(pair_weight),
                'pair_scale': float(pair_scale),
                'pair_margin': PAIR_MARGIN,
            }

        # === 非 eval 间隔: 只 log train 行 ===
        if epoch % EVAL_INTERVAL != 0 and epoch != EPOCHS:
            if rank == 0:
                print(f"[RQ-VAE] epoch={epoch} loss={avg_loss:.8f} recon={avg_recon:.8f} lr={cur_lr:.3e} time={epoch_time_s}s", flush=True)
                _record(rank, "train", epoch=epoch, loss=avg_loss, recon=avg_recon, lr=cur_lr, epoch_time_s=epoch_time_s, cumulative_time_s=round(time.time() - _T0, 3), **pair_summary)
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
                    **pair_summary,
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
                "run_id": globals().get("RUN_ID") or f"{ARM_NAME}_{time.time_ns()}",
                "source_tree": str(Path(__file__).resolve().parent),
                "selection": "final_epoch",
                "provenance": raw_module.provenance(),
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


#: The single-card contract of the frozen recipe. On a host with one visible
#: card the trainer runs in this process as the only rank instead of forking
#: four, and the per-process batch is the four-rank global batch, so the
#: optimisation regime does not move: the same six optimizer steps per epoch
#: over the same samples, only the order of the summation differs.
SINGLE_GPU_ENV = "GENREC_SINGLE_GPU"
SINGLE_CARD_BATCH = 4096


def _visible_card_count() -> int:
    """How many cards this process can actually see."""
    try:
        return max(1, int(torch.cuda.device_count()))
    except Exception:  # noqa: BLE001 - a host without torch still has to run CPU
        return 0


def _single_card_mode() -> bool:
    return os.environ.get(SINGLE_GPU_ENV) == "1"


def _launch_via_torchrun() -> None:
    """Four ranks on a four-card host; this process alone when there is one card.

    The frozen recipe starts four ranks through the framework's launcher. On a
    host with a single visible card that launch cannot work: ranks 1 to 3 ask
    for ``cuda:1`` to ``cuda:3`` and torch aborts with 'invalid device ordinal'.
    A one-card host therefore runs in this process, with the four-rank global
    batch and no process group.
    """
    if int(os.environ.get("RANK", -1)) >= 0:
        return  # 已经在 torchrun 启动的子进程里
    if _visible_card_count() <= 1:
        os.environ[SINGLE_GPU_ENV] = "1"
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
