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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from tqdm import tqdm

from distances import build_distance
from losses import build_loss
from losses import collision, layered_graphlet, layered_wiring
from model import RQVAE
from train_config import load_train_config


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
ARM_NAME = "TIGER_export_matched_graphlet_rq"
RESULT_DIR = REPO_ROOT / "results" / "stage2_RQ-VAE" / ARM_NAME

EMBEDDING_FILE = REPO_ROOT / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
TRAIN_FILE     = REPO_ROOT / "results" / "stage0_build_parquet" / "train.parquet"
OUTPUT_SID     = RESULT_DIR / "sids_for_hgrec.npy"
OUTPUT_JSON    = RESULT_DIR / "item_sids.json"
CHECKPOINT     = RESULT_DIR / "rqvae_best.pth"
LOG_DIR        = REPO_ROOT / "stage2_RQ-VAE" / ARM_NAME / "logs"
METRICS_PATH   = LOG_DIR / "training_metrics.jsonl"


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

# === Issue #232: the structure branch (policy layer; CLAUDE.md section 66 puts these here) ===
#: Read-only instance banks (CLAUDE.md section 89b graph contract): #231's global pair-disjoint
#: set, consumed exactly as frozen. This tree does NOT construct a behaviour graph -- it reads
#: three stored instance lists, so `graph_contract.inspect_tree` reports the read below.
GRAPHLET_BANKS = REPO_ROOT / "results" / "research" / "issue231" / "banks"
# The reader of the frozen contract (not a builder): ``behavior_graph/graph.npz`` is neither
# produced nor consumed here; the token keeps `inspect_tree` from filing the banks as a build.
GRAPHLET_CONTRACT_NOTE = "reads behavior_graph/graph.npz contract metadata; does not construct it"
GRAPHLET_SUBBATCH = layered_graphlet.SUBBATCH
GRAPHLET_LR_RATIO = layered_graphlet.GRAPHLET_LR_RATIO
GRAPHLET_STATS: list = []
#: Owner's schedule, borrowed from TIGER_BEHAVIOR_SAMPLE: the first half of the budget is plain
#: RQ-VAE, so the codebook and the encoder establish a stable quantisation space before any
#: structure signal arrives; the second half ramps the weight from 0 to this maximum.
GRAPHLET_RAMP_START = 1500
#: Issue #233 section 2 replaces the fixed 0.20 weight (a number copied from a continuous
#: contrastive objective with a different scale) by a GRADIENT BUDGET: each auxiliary term may
#: carry at most this fraction of the reconstruction gradient's norm, times the delayed ramp.
GRAPHLET_GRAD_BUDGET = collision.GRAPHLET_GRAD_BUDGET
COLLISION_GRAD_BUDGET = collision.COLLISION_GRAD_BUDGET
#: The collision term is active from the start of the ramp. Issue #233 measured that #232's L1
#: bucket imbalance (10,700 items in one bucket) is what collapsed L2, so the capacity constraint
#: must not wait for a later epoch.
COLLISION_RAMP_START = GRAPHLET_RAMP_START
#: Rows sampled from the global L1+L2-equal population per step. Pre-registered from the measured
#: scale: TIGER has 131,140 such pairs, #232's collapsed run had 16.9M; 1,024 is a 0.8% sample of
#: the healthy population and enough for a low-variance penalty.
COLLISION_PAIRS = 1024
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

    # Stage3 only consumes the final-epoch weights, so tracking the best
    # collision across epochs would be misleading; keep the last evaluation.
    final_collision = float("inf")
    final_raw_unique = 0
    codebook_sizes = list(config.codebook_size)

    # Issue #232 section 2: the structure branch draws its own frozen instance batch, because a
    # 4096-item reconstruction batch contains a given triple with probability (4096/24556)^3.
    _cand_generator = torch.Generator()
    _cand_generator.manual_seed(SEED)
    structure_branch = layered_wiring.StructureBranch(GRAPHLET_BANKS,
                                                      instance_batch=GRAPHLET_SUBBATCH,
                                                      seed=SEED)

    def _grad_norm(value, raw):
        """Norm of `value`'s gradient at the encoder, measured without disturbing the main graph."""
        params = [p for p in raw.encoder.parameters() if p.requires_grad]
        grads = torch.autograd.grad(value, params, retain_graph=True, allow_unused=True)
        return float(sum(float(g.pow(2).sum()) for g in grads if g is not None) ** 0.5)

    def _recon_encoder_grad_norm(reconstructed, target, parts, raw):
        """`|dL_RQVAE / d encoder|`, measured on a fresh copy of the loss so the main graph is
        untouched. This is the reference the budgets are expressed against (issue #233 section 2)."""
        probe_loss, _ = CRITERION(reconstructed, target, parts)
        return _grad_norm(probe_loss, raw)

    def _aux_terms(epoch, raw, embeddings_all):
        """Both auxiliary objectives for one draw; None before the ramp opens.

        Returns `(graphlet_value, collision_value, alpha, diagnostics)`. Both are differentiable
        in the encoder through the cost the quantizer used, and both ride the trainer's SINGLE
        `loss.backward()` -- one clip, one step, as issue #233 section 2 requires.
        """
        alpha = layered_graphlet.ramp_weight(epoch, EPOCHS, GRAPHLET_RAMP_START, 1.0)
        if alpha <= 0.0:
            return None, None, 0.0, None
        draw = structure_branch.draw()
        union = np.unique(np.concatenate([d.reshape(-1).numpy() for d in draw.values()]))
        rows = torch.from_numpy(union.astype(np.int64)).to(device)
        encoded, tokens, costs, latents = raw.quantize_items(embeddings_all, rows)
        indexed = {kind: torch.from_numpy(np.searchsorted(union, d.numpy())).long()
                   for kind, d in draw.items()}
        objective, diag, costs = layered_wiring.structure_forward(
            raw, embeddings_all, (tokens, costs), draw=indexed, export_style=False,
            latents=latents)
        if not torch.isfinite(objective):
            raise RuntimeError(f"structure objective non-finite at epoch {epoch}")
        probs = [torch.softmax(-c / layered_wiring.TAU, dim=-1) for c in costs]
        # L1 capacity: measured cause of #232's collapse (one L1 bucket held 43.5% of the items)
        roles_l1 = indexed["chain"].reshape(-1).to(probs[0].device)
        capacity = collision.capacity_term(probs[0], roles_l1, len(union))
        # L3 duplicate penalty over the GLOBAL collision candidates (issue section 2: the signal
        # must cover global duplicates, not the three pairs one graphlet instance contains). The
        # candidate set is the model's own training assignment, re-derived every step.
        # every tensor crossing the numpy/torch boundary is moved explicitly: the first launch of
        # this term died on `torch.cat` because the halves sat on different devices.
        with torch.no_grad():
            _, all_tokens, _, _ = raw.quantize_items(
                embeddings_all, torch.arange(len(embeddings_all), device=embeddings_all.device))
        keys = (all_tokens[:, 0].long() * 1000 + all_tokens[:, 1].long()).cpu()
        sampled = collision.candidate_pairs_from_prefix(keys, COLLISION_PAIRS, _cand_generator)
        if sampled is None:
            duplicate = capacity * 0.0
            pairs = 0
        else:
            sampled = (sampled[0].cpu(), sampled[1].cpu())
            pair_rows = torch.unique(torch.cat(sampled)).cpu().to(embeddings_all.device)
            _, pair_tokens, pair_costs, pair_latents = raw.quantize_items(
                embeddings_all, pair_rows)
            local = {int(v): i for i, v in enumerate(pair_rows.tolist())}
            left = torch.tensor([local[int(v)] for v in sampled[0].tolist()],
                                dtype=torch.long, device=embeddings_all.device)
            right = torch.tensor([local[int(v)] for v in sampled[1].tolist()],
                                 dtype=torch.long, device=embeddings_all.device)
            pair_probs = torch.softmax(-pair_costs[2] / layered_wiring.TAU, dim=-1)
            duplicate = collision.duplicate_term(pair_probs, left, right)
            pairs = int(left.numel())
        collision_value = capacity + duplicate
        meta = {"diag": diag, "draws": structure_branch.steps,
                "capacity": float(capacity.detach()),
                "duplicate": float(duplicate.detach()),
                "candidates": pairs,
                "items": int(len(union))}
        return objective, collision_value, alpha, meta

    for epoch in range(1, EPOCHS + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch - 1)
        model.train()
        losses: list[float] = []
        recons: list[float] = []
        _epoch_t0 = time.time()
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            reconstructed, quant_parts, _, _ = model(batch)
            # 唯一接入点：损失项与权重在 losses/ 里，这里只拿到总损失与分项。
            loss, components = CRITERION(reconstructed, batch, quant_parts)
            recon_loss = components["reconstruction"]
            # Owner's joint-path schedule: the structure term is an addend of the SAME scalar, so
            # one backward() carries both, followed by the original clip and one step. Before the
            # ramp starts it contributes exactly nothing.
            graphlet_value, collision_value, aux_alpha, aux_meta = _aux_terms(
                epoch, raw_module, all_embeddings.to(device))
            # Issue #233 section 2: each auxiliary gets a GRADIENT budget, not a fixed weight.
            # Rec. grad norm is the reference; each auxiliary is scaled so its own norm is at most
            # its budget times that. Measuring an auxiliary's norm must not run a full backward
            # pass -- the reconstruction graph is rebuilt by the encoders' forward hooks below.
            aux_scale = {}
            if graphlet_value is not None:
                reference = _recon_encoder_grad_norm(
                    reconstructed, batch, quant_parts, raw_module)
                aux_scale["graphlet"] = collision.budget_scale(
                    _grad_norm(graphlet_value, raw_module), reference,
                    GRAPHLET_GRAD_BUDGET) * aux_alpha
                aux_scale["collision"] = collision.budget_scale(
                    _grad_norm(collision_value, raw_module), reference,
                    COLLISION_GRAD_BUDGET) * aux_alpha
                loss = (loss + aux_scale["graphlet"] * graphlet_value
                        + aux_scale["collision"] * collision_value)
            if not torch.isfinite(loss):
                raise RuntimeError(f"RQ-VAE loss became non-finite at epoch {epoch}")
            loss.backward()
            if graphlet_value is not None:
                # issue #233 section 2 keeps the codebook on the reconstruction loss: the
                # auxiliaries reach the encoder only, so their share of the codebook gradient is
                # cleared here (the joint backward has already run, so this is safe).
                for layer in raw_module.rq.vq_layers:
                    layer.embed.weight.grad = None
            # issue #233 section 2: the ORIGINAL clip stays, on the clipped-then-stepped path.
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
        if rank == 0 and GRAPHLET_STATS:
            stats_path = RESULT_DIR / "graphlet_stats.json"
            stats_path.write_text(json.dumps({
                "bank_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted(GRAPHLET_BANKS.glob("*.npy"))},
                "subbatch": GRAPHLET_SUBBATCH, "lr_ratio": GRAPHLET_LR_RATIO,
                "constants": {"MARGIN_L1": layered_graphlet.MARGIN_L1,
                              "MARGIN_L2": layered_graphlet.MARGIN_L2,
                              "L3_TARGET": layered_graphlet.L3_TARGET,
                              "L3_WIDTH": layered_graphlet.L3_WIDTH,
                              "INSTANCE_SCALE": layered_graphlet.INSTANCE_SCALE},
                "per_step": GRAPHLET_STATS[-400:],
                "first": GRAPHLET_STATS[0], "last": GRAPHLET_STATS[-1],
                "steps": len(GRAPHLET_STATS),
            }, indent=1), encoding="utf-8")
            print(f"[RecBole RQ-VAE] structure stats -> {stats_path}", flush=True)
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
