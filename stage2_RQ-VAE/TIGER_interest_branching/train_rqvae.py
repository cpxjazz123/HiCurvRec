"""TIGER + iter48 behaviour contrastive with corrected pair sampling.

Baseline for this program is TIGER_BEHAVIOR_RQ-VAE (test_R@10 = 0.061927).
That arm fed the contrastive term from ``pair_cycle = cycle(pair_loader)``
built once before the epoch loop. ``itertools.cycle`` materialises its first
pass and replays it forever, so the sampler's ``set_epoch`` was a no-op, only
``ceil(339519 / 512) = 664`` distinct pair batches per rank were ever drawn, and
the ~36,000 draws of the active epochs wrapped that frozen pool about 54 times
with an identical negative pool every time.

This tree fixes that and adds one optional mechanism, selected by hardcoded
constants so each arm's committed state is exactly what ran:

  RESAMPLE_PAIRS      rebuild the pair iterator every epoch (True for both arms;
                      an arm with this False is the baseline's behaviour)
  FALSE_NEGATIVE_MASK drop candidates that are themselves known successors of
                      the anchor, not just repeats of the anchor's own target.
                      The baseline masked only duplicate targets, so a second
                      real successor was still pushed away as a negative.
  EXPERIMENT_ARM      "resample" or "resample_fn_mask"; drives the output paths

Everything else is unchanged: ``model/`` stays byte-identical to TIGER, and the
3000-epoch budget, deduplicated item batch, AdamW warmup + linear decay,
codebook sizes, KMeans init and seed 42 all stay as the baseline fixes them.
The delayed ramp stays as the baseline's: 0 at epoch 1500, 0.2 at epoch 3000.

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
from scripts.crowding_losses import MatchedParameters, crowding_loss
from scripts.crowding_metrics import (assemble_losses, crowding_ramp_weight,
                                      crowding_record, crowding_train_fields)

import crowding_losses2 as _crowd2

from arm_geometry import ARM_CONFIGS as _BASE_GEOMETRY_ARM_CONFIGS
from geom_losses import compose_loss as _geometry_compose
from scripts.capacity_mask import install_level_capacity
from crowding_diagnostics import crowding_report, full_item_recon_mse
# === M5 唯一机制: L1 桶内的协同兴趣分叉 (L2 角向惩罚) ===
from interest_branching import (BRANCH_K, BRANCH_STRENGTH, install_interest_branch,
                                interest_directions, _trace_with_parents)
from geom_losses import geometry_loss as _geometry_loss
from quant_rank_config import GROUP_CONFIGS as _QUANT_RANK_GROUP_CONFIGS
from quant_rank_loss import compose_loss as _quant_rank_compose
from quant_rank_loss import quantized_representation as _quantized_representation
from quant_rank_loss import quantized_ranking_loss as _quant_rank_loss
from quant_rank_loss import quantized_ranking_loss_margin as _quant_rank_margin_loss
import anti_collapse_losses as _ac
from anti_collapse_config import (ALIGN_WEIGHT as _AC_ALIGN_WEIGHT,
                                  DIVERSITY_WEIGHT as _AC_DIVERSITY_WEIGHT,
                                  GUMBEL_TAU as _AC_GUMBEL_TAU, ARM_FLAGS)
from anti_collapse_losses import compose as _ac_compose

_AC_ALIGN = _ac.align_loss
_AC_DIVERSITY = _ac.diversity_loss
_AC_GUMBEL = _ac.gumbel_quantized_representation

_GEOMETRY_ARM_CONFIGS = {
    **_BASE_GEOMETRY_ARM_CONFIGS,
    "B_E": dict(_BASE_GEOMETRY_ARM_CONFIGS["E0"]),
    "B_H": dict(_BASE_GEOMETRY_ARM_CONFIGS["H1"]),
    "B_E_nocb": dict(_BASE_GEOMETRY_ARM_CONFIGS["E0"]),
    "MARGIN": dict(_BASE_GEOMETRY_ARM_CONFIGS["E0"]),
    "LETTER": dict(_BASE_GEOMETRY_ARM_CONFIGS["E0"]),
    "LETTER_nodiv": dict(_BASE_GEOMETRY_ARM_CONFIGS["E0"]),
    "DIGER": dict(_BASE_GEOMETRY_ARM_CONFIGS["E0"]),
    # E5: continued training from the BASELINE checkpoint, with a frozen copy of that
    # checkpoint's codebooks as an anchor. ANCHOR_CTRL differs from the other three
    # only in that its anchor coefficient is zero.
    "ANCHOR_CTRL": dict(_BASE_GEOMETRY_ARM_CONFIGS["E0"]),
    "ANCHOR_W001": dict(_BASE_GEOMETRY_ARM_CONFIGS["E0"]),
    "ANCHOR_W01": dict(_BASE_GEOMETRY_ARM_CONFIGS["E0"]),
    "ANCHOR_W1": dict(_BASE_GEOMETRY_ARM_CONFIGS["E0"]),
}
from scripts.hyp_losses import (ARM_CONFIGS as _HYP_ARM_CONFIGS,
                                install_nearest_assignment as _install_hyp_assignment,
                                mechanism_loss as _hyp_mechanism_loss)

# === 硬编码路径常量 (CLAUDE.md §1) ===
# Stage0 emits user-event parquet directly into results/stage0_build_parquet/.
EMBEDDING_FILE = Path(
    "/home/wlia0047/ar57/wenyu/GeneRec/stage1_GeneEmbedding/output/sentence_t5.npy"  # stage1
)
TRAIN_FILE     = Path(
    "/home/wlia0047/ar57/wenyu/GeneRec/results/stage0_build_parquet/train.parquet"   # stage0
)
# ============================================================================
# 出处: 本树由 stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE 复制而来 (2026-10-15),
# 属于「码字拥挤 (Codeword Crowding)」实验序列的机制 M5。
# 相对 Crowding-only 对照 (TIGER_CROWDING_CTRL) 的唯一改动 = multi-interest hyperbolic branching inside each L1 bucket；
# 共同层 (L1 容量掩码 256->128 + 拥挤诊断) 逐字复制自对照树, 不得改动。
# ============================================================================
# === 本轮实验臂 (改这一行切换 arm; 每个 arm 的提交状态即实际跑过的配置) ===
EXPERIMENT_ARM = "interest_branching"   # 唯一机制: multi-interest hyperbolic branching inside each L1 bucket
# === 唯一机制 (CLAUDE.md §1: 只有 interest_branching, 没有可选臂) ===
# interest_branching 是本树唯一允许的机制: 共同层 + multi-interest hyperbolic branching inside each L1 bucket。
if EXPERIMENT_ARM != "interest_branching":
    raise RuntimeError(
        f"only the 'interest_branching' mechanism may run in this tree; got {EXPERIMENT_ARM!r}"
    )
_ANTI_COLLAPSE_FLAGS = ARM_FLAGS.get(EXPERIMENT_ARM)
# Seed replication arms. Exp1a beat the R48 baseline by only ~1.7 unpaired
# standard errors, so the same configuration is re-run under other seeds to see
# whether the gain survives. Every other constant is untouched.
# 输出目录不再来自查表：见下面的「路径自动解析」，由 TREE_NAME + EXPERIMENT_ARM 推导。
_CROWD_LEVEL_SETTINGS = {
    "w1": {"strength": 0.05, "tau": 0.20, "depth": 1, "k": 5, "concentration": 1.0},
    "w2": {"strength": 0.10, "tau": 0.10, "depth": 1, "k": 20, "concentration": 4.0},
    "w3": {"strength": 0.20, "tau": 0.05, "depth": 2, "k": 50, "concentration": 16.0},
}
_ARM_CROWDING = {
    f"crowd_{m}_w{level}": (m, level - 1, dict(_CROWD_LEVEL_SETTINGS[f"w{level}"], mechanism=m))
    for m in ("posattract", "couser", "simweight", "density", "softassign", "prefixshare",
              "random", "scale", "parammatch") for level in (1, 2, 3)
}

# === Round 2: positive/negative competition crowding (crowding_losses2) ===
# The six mechanisms and three controls are additive to Exp1a's behaviour term.
# Strength ladder is fixed up front at 0.5 / 2.0 / 8.0 (fourfold steps, 16x
# span) and is NOT retuned against any measured metric table.
CROWDING2_MECHANISM = ""      # "" disables; else a name from _crowd2.MECHANISMS / _crowd2.CONTROLS
CROWDING2_LEVEL = 2           # 1 -> 0.5, 2 -> 2.0, 3 -> 8.0
CROWDING2_SUCCESSORS = 8      # padded successor slots per anchor (GPU-vectorised)
CROWDING2_NEGATIVES = 4       # in-batch matched negatives per anchor
CROWDING2_ANCHORS = 512       # anchors per step; keeps the additive term cheap
CROWDING2_RAMP_START = 21     # begin after the same warmup the behaviour term uses

_ARM_BEHAVIOR_WEIGHT = {
    "bw0":   0.0,
    "bw001": 0.002,
    "bw01":  0.02,
}

# E5. Continued training from the BASELINE checkpoint, with a frozen copy of that
# checkpoint's codebooks pulled toward the current ones by an L2 penalty. The four arms
# differ only in the coefficient, so ANCHOR_CTRL -- same checkpoint, same budget, same
# MARGIN behavioural loss, coefficient zero -- isolates the anchoring itself.
# ANCHOR arms load this tree's own weights; the path is derived further down from
# TREE_NAME so a copy of this tree never points at another tree's checkpoint.
_ARM_ANCHOR_WEIGHT = {
    "ANCHOR_CTRL": 0.0,
    "ANCHOR_W001": 0.01,
    "ANCHOR_W01":  0.1,
    "ANCHOR_W1":   1.0,
}
_ANCHOR_WEIGHT = _ARM_ANCHOR_WEIGHT.get(EXPERIMENT_ARM, 0.0)
_ANCHOR_ACTIVE = EXPERIMENT_ARM in _ARM_ANCHOR_WEIGHT

_ARM_CROWDING2 = {
    f"c2_{m}_w{level}": (m, level)
    for m in ("poslocal", "multipos", "couserw", "hncontrast", "prefixbal", "margincomp",
              "random", "scale", "behonly") for level in (1, 2, 3)
}
_CROWDING2_CONFIG = _ARM_CROWDING2.get(EXPERIMENT_ARM) if CROWDING2_MECHANISM == "" else None
if CROWDING2_MECHANISM:
    _CROWDING2_CONFIG = (CROWDING2_MECHANISM, CROWDING2_LEVEL)
elif EXPERIMENT_ARM in _ARM_CROWDING2:
    _CROWDING2_CONFIG = _ARM_CROWDING2[EXPERIMENT_ARM]
else:
    _CROWDING2_CONFIG = None

# Fixed before the epoch loop; the sampler reads these.
_C2_WIDTH = CROWDING2_SUCCESSORS
_C2_NEG = CROWDING2_NEGATIVES

# === Capacity freeze: all future arms retain this representation capacity. ===
_CAPACITY_ARMS = frozenset((
    "cap_d8", "cap_d16", "cap_d64", "cap_d128", "cap_l1c128",
    "cap_l1c512", "cap_d16_l1c512", "cap_d64_l1c128", "cap_d32_wide",
))
if EXPERIMENT_ARM in _CAPACITY_ARMS:
    raise RuntimeError(
        f"Capacity freeze is active: capacity arm {EXPERIMENT_ARM!r} is closed "
        "and cannot be selected."
    )
# === 路径自动解析 ===
# 树名取自本文件所在目录，仓库根是唯一固定常量。复制本树改名后，输出/日志/ckpt
# 会自己跟着新目录名走，不必改任何一行路径，也不会把产物写回源树。
# 用 __file__ 的原样拼写（不 resolve）：仓库同时以 /home 与 /fs04 可见，日志统一保留
# 调用方给的拼写；相对启动时退回 cwd（MCP 启动即以树目录为 cwd）。
_THIS_FILE = Path(__file__)
TREE_DIR  = (_THIS_FILE.parent if _THIS_FILE.is_absolute() else Path.cwd())
TREE_NAME = TREE_DIR.name
REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
OUT_ROOT  = REPO_ROOT / "results/stage2_RQ-VAE"
_ARM_SEED = {"resample_s43": 43, "resample_s44": 44}.get(EXPERIMENT_ARM, 42)
_OUT = OUT_ROOT / TREE_NAME / EXPERIMENT_ARM
OUTPUT_SID     = _OUT / "sids_for_hgrec.npy"
OUTPUT_JSON    = _OUT / "item_sids.json"
CHECKPOINT     = _OUT / "rqvae_best.pth"
_ANCHOR_BASELINE_CHECKPOINT = str(CHECKPOINT)
LOG_DIR        = REPO_ROOT / "stage2_RQ-VAE" / TREE_NAME / "logs" / EXPERIMENT_ARM
METRICS_PATH   = LOG_DIR / "training_metrics.jsonl"
DIRECTIONAL_MODE = {"resample_proj_sym": "sym", "resample_proj_dir": "dir"}.get(EXPERIMENT_ARM, "none")
_ARM_ACTIVE_LEVELS = {"resample_l1only": (0,), "resample_l2only": (1,), "resample_l3only": (2,)}
ACTIVE_LEVELS = _ARM_ACTIVE_LEVELS.get(EXPERIMENT_ARM, (0, 1, 2))
# Restored 2026-10-13: this definition was deleted by the geometry wiring edit and the
# training loop still reads it at line ~759, so every arm died with a NameError.
_HYP_CONFIG = _HYP_ARM_CONFIGS.get(EXPERIMENT_ARM)
# Restored 2026-10-13: the geometry-wiring edit deleted this module-level line while
# the training loop still reads it at line ~771, so every arm died with a NameError.
_CROWD_CONFIG = _ARM_CROWDING.get(EXPERIMENT_ARM)

_GEOMETRY_CONFIG = _GEOMETRY_ARM_CONFIGS.get(EXPERIMENT_ARM)

def _resolve_geometry_arm(arm: str) -> dict:
    """Resolve a geometry arm against the trainer's frozen shared settings."""
    if arm not in _GEOMETRY_ARM_CONFIGS:
        raise KeyError(f"{arm!r} is not a geometry arm")
    shared = (
        CODEBOOK_DIM, CODEBOOK_SIZE, HIDDEN_SIZES, EPOCHS, LR, OPTIMIZER,
        BATCH_SIZE_PER_RANK, BETA, BEHAVIOR_WEIGHT_MAX,
        BEHAVIOR_TEMPERATURE, RAMP_START_EPOCH, PAIR_BATCH_SIZE_PER_RANK,
        RESAMPLE_PAIRS, FALSE_NEGATIVE_MASK, SEED,
    )
    return {
        "shared": shared,
        "geometry": _GEOMETRY_ARM_CONFIGS[arm],
        "output_dir": str(OUT_ROOT / TREE_NAME / arm),
    }

def _resolve_quant_arm(arm: str) -> dict:
    group_a_arm, group = {"B_E": ("E0", "B"), "B_H": ("H1", "B"), "B_E_nocb": ("E0", "B"),
                          "MARGIN": ("E0", "B"), "LETTER": ("E0", "B"),
                          "LETTER_nodiv": ("E0", "B"), "DIGER": ("E0", "B")}[arm]
    baseline = _resolve_geometry_arm(group_a_arm)
    return {
        **baseline,
        "group_a_geometry_arm": group_a_arm,
        "level_weights": _QUANT_RANK_GROUP_CONFIGS[group]["level_weights"],
        "quantized_ranking": True,
    }


def _compose_quantized_loss(
    behavior_loss: torch.Tensor, quantized_term: torch.Tensor
) -> torch.Tensor:
    return _quant_rank_compose(behavior_loss, quantized_term)


def _quantized_ranking_loss(
    residuals: torch.Tensor,
    token_ids: torch.Tensor,
    model: nn.Module,
    anchors: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
    *,
    weight: float = 0.2,
) -> torch.Tensor:
    raw_module = model.module if isinstance(model, DDP) else model
    # E5 keeps MARGIN's behavioural loss exactly; its own variable is the codebook
    # anchor, so the form here has to match MARGIN's rather than drift to softplus.
    margin_arms = {"MARGIN", "ANCHOR_CTRL", "ANCHOR_W001", "ANCHOR_W01", "ANCHOR_W1"}
    ranking_loss = _quant_rank_margin_loss if EXPERIMENT_ARM in margin_arms else _quant_rank_loss
    return ranking_loss(
        residuals, token_ids, raw_module.rq.vq_layers,
        anchors, positives, negatives,
        arm={"B_E": "E0", "B_H": "H1", "B_E_nocb": "E0", "MARGIN": "E0", "LETTER": "E0",
             "LETTER_nodiv": "E0", "DIGER": "E0",
             "ANCHOR_CTRL": "E0", "ANCHOR_W001": "E0", "ANCHOR_W01": "E0", "ANCHOR_W1": "E0"}.get(EXPERIMENT_ARM, EXPERIMENT_ARM),
        curvature=1.0, weight=weight,
        level_weights=_QUANT_RANK_GROUP_CONFIGS["B"]["level_weights"],
        detach_codebook=(EXPERIMENT_ARM == "B_E_nocb"),
    )
def _anti_collapse_term(residuals, token_ids, model, quantized_term):
    raw_module = model.module if isinstance(model, DDP) else model
    flags = _ANTI_COLLAPSE_FLAGS
    if flags is None:
        return torch.zeros((), device=residuals.device)
    total = torch.zeros((), device=residuals.device)
    if flags["use_gumbel"]:
        q = _AC_GUMBEL(residuals, raw_module.rq.vq_layers, _AC_GUMBEL_TAU)
    else:
        q = _quantized_representation(residuals, token_ids, raw_module.rq.vq_layers)
    if flags["use_align"]:
        total = total + _AC_ALIGN_WEIGHT * _AC_ALIGN(residuals, q)
    if flags["use_diversity"]:
        total = total + _AC_DIVERSITY(
            residuals, raw_module.rq.vq_layers, _AC_GUMBEL_TAU)
    return total




def _geometry_loss_for_pairs(
    arm: str,
    latent: torch.Tensor,
    source_ids: torch.Tensor,
    codebook: torch.Tensor,
    *,
    weight: float | None = None,
) -> torch.Tensor:
    """Apply geometry to packed source, positive and matched-negative item rows."""
    batch_size = int(source_ids.shape[0])
    anchors = torch.arange(batch_size, device=latent.device)
    positives = anchors + batch_size
    negatives = positives + batch_size
    geometry_arm = {"B_E": "E0", "B_H": "H1", "B_E_nocb": "E0", "MARGIN": "E0", "LETTER": "E0",
                    "LETTER_nodiv": "E0", "DIGER": "E0",
             "ANCHOR_CTRL": "E0", "ANCHOR_W001": "E0", "ANCHOR_W01": "E0", "ANCHOR_W1": "E0"}.get(arm, arm)
    arm_weight = _GEOMETRY_ARM_CONFIGS[arm]["weight"] if weight is None else weight
    return _geometry_loss(
        geometry_arm, latent, codebook, anchors, positives, negatives,
        curvature=1.0, weight=arm_weight,
    )


def _matched_geometry_negative_ids(
    source_ids: torch.Tensor,
    target_ids: torch.Tensor,
    bucket_padded_t: torch.Tensor,
    item_bucket_t: torch.Tensor,
) -> torch.Tensor:
    """Sample a distinct, real negative from each source's popularity bucket."""
    buckets = item_bucket_t[source_ids]
    candidates = bucket_padded_t[buckets]
    valid = (
        (candidates >= 0)
        & (candidates != source_ids.unsqueeze(1))
        & (candidates != target_ids.unsqueeze(1))
    )
    sizes = valid.sum(dim=1)
    slots = (torch.rand(len(source_ids), device=source_ids.device) *
             sizes.float()).long()
    positions = valid.cumsum(dim=1)
    selected = (positions > slots.unsqueeze(1)).to(torch.int64).argmax(dim=1)
    return candidates.gather(1, selected.unsqueeze(1)).squeeze(1)




def _compose_geometry_loss(behavior_loss: torch.Tensor, geometry_term: torch.Tensor) -> torch.Tensor:
    return _geometry_compose(behavior_loss, geometry_term)


def _geometry_gradient_norms(geometry_term: torch.Tensor, model: nn.Module) -> dict[str, float]:
    """Measure encoder and VQ-codebook gradients from this geometry term only."""
    raw_module = model.module if isinstance(model, DDP) else model
    encoder_parameters = tuple(raw_module.encoder.parameters())
    codebook_parameters = tuple(
        layer.get_code_embs() for layer in raw_module.rq.vq_layers
    )
    gradients = torch.autograd.grad(
        geometry_term, encoder_parameters + codebook_parameters,
        retain_graph=True, allow_unused=True,
    )

    def norm(values):
        present = [gradient.detach().square().sum() for gradient in values if gradient is not None]
        if not present:
            return 0.0
        return float(torch.sqrt(torch.stack(present).sum()))

    split = len(encoder_parameters)
    return {
        "encoder": norm(gradients[:split]),
        "codebook": norm(gradients[split:]),
    }


def _quantized_gradient_norms(quantized_term: torch.Tensor, model: nn.Module) -> dict[str, float]:
    raw_module = model.module if isinstance(model, DDP) else model
    encoder_parameters = tuple(raw_module.encoder.parameters())
    codebook_parameters = tuple(layer.get_code_embs() for layer in raw_module.rq.vq_layers)
    gradients = torch.autograd.grad(
        quantized_term, encoder_parameters + codebook_parameters,
        retain_graph=True, allow_unused=True,
    )

    def norm(values):
        present = [gradient.detach().square().sum() for gradient in values if gradient is not None]
        return float(torch.sqrt(torch.stack(present).sum())) if present else 0.0

    split = len(encoder_parameters)
    return {
        "encoder": norm(gradients[:split]),
        "codebook_l1": norm(gradients[split:split + 1]),
        "codebook_l2": norm(gradients[split + 1:split + 2]),
        "codebook_l3": norm(gradients[split + 2:]),
    }


# === 硬编码超参 (原 argparse default 值, 保留原 TIGER 默认) ===
EPOCHS              = 3000
BATCH_SIZE_PER_RANK = 1024       # 每卡 batch, 总 batch = 1024 * 4 = 4096
LR                  = 1e-3
WEIGHT_DECAY        = 1e-4
HIDDEN_SIZES        = (512, 256, 128)
CODEBOOK_NUM        = 3
CODEBOOK_SIZE       = (256, 256, 256)
CODEBOOK_DIM        = 32
assert CODEBOOK_DIM == 32, "Capacity freeze requires CODEBOOK_DIM=32"
assert len(CODEBOOK_SIZE) == 3, "Capacity freeze requires exactly three RQ-VAE levels"
assert CODEBOOK_SIZE == (256, 256, 256), "Capacity freeze requires CODEBOOK_SIZE=(256,256,256)"
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
SEED                = _ARM_SEED
NUM_WORKERS         = 0  # 2026-09-24: 0 防 DDP fork storm (4 rank × 4 worker = 16 子进程在 barrier 后挂死)
EVAL_INTERVAL       = 50
# === 共同层 (crowding by capacity): L1 只保留前 128 个可选码字, 其余距离为 +inf ===
# 所有拥挤实验树逐字相同; 机制树不得改动本常量。
CROWDING_L1_ALLOWED = 128
# 本树唯一的额外机制; 对照树为 "none"。写入 train_start 便于事后核对。
CROWDING_MECHANISM = "interest_branching"
BEHAVIOR_WEIGHT_MAX     = 0.20
# Behaviour-weight sweep arms. The diagnostic found the behaviour term's encoder gradient
# is 179x to 601x the reconstruction gradient's, so this sweeps that magnitude. Everything
# else is identical to the resample baseline. The override MUST sit after the assignment
# above; putting it earlier silently does nothing.
if EXPERIMENT_ARM in _ARM_BEHAVIOR_WEIGHT:
    BEHAVIOR_WEIGHT_MAX = _ARM_BEHAVIOR_WEIGHT[EXPERIMENT_ARM]
BEHAVIOR_TEMPERATURE    = 0.07
RAMP_START_EPOCH        = 1500
PAIR_BATCH_SIZE_PER_RANK = 512
RESAMPLE_PAIRS          = True

FALSE_NEGATIVE_MASK     = EXPERIMENT_ARM == "resample_fn_mask"
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
_HYP_METRIC_FIELDS = {}
_metrics_truncated = False
def _record(rank, event, **fields):
    if rank != 0:
        return
    rec = {"event": event, "timestamp": time.time(), "wall_time_s": round(time.time() - _T0, 3)}
    if event == "train" and _HYP_CONFIG is not None:
        fields.update(_HYP_METRIC_FIELDS)
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

    M5: delegated to ``interest_branching._trace_with_parents`` so the trace hands
    the parent indices down exactly like ``RQLayer.forward`` does; the mechanism
    would otherwise train the behaviour term on a differently assigned path.
    """
    return _trace_with_parents(raw_module, embeddings)


def _residuals_and_tokens_with_trace(
    raw_module: RQVAE, embeddings: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return residual traces and token IDs from the same real RQ layer calls."""
    return _trace_with_parents(raw_module, embeddings, with_tokens=True)


def _successor_counts(train_frame: pd.DataFrame, n_items: int):
    """Successor CSR with real co-occurrence counts, plus item popularity.

    Returns (indptr, indices, edge_counts, popularity, n_users) as numpy arrays.
    ``indptr`` is the CSR row pointer over item ids; ``indices`` holds the
    successor ids sorted within each row; ``edge_counts[i]`` is the number of
    users exhibiting exactly that ``indptr[r] -> indices[i]`` transition, which
    is the co-occurrence count the PMI weights are computed from. Per-anchor
    successor counts are derivable as ``indptr[r+1] - indptr[r]``.
    """
    rows: list[int] = []
    cols: list[int] = []
    bought: list[np.ndarray] = []
    for user, history, target in zip(
        train_frame["user"].to_numpy(),
        train_frame["history"].to_numpy(),
        train_frame["target"].to_numpy(dtype=np.int64),
    ):
        if history is not None and len(history) > 0:
            rows.append(int(history[-1]))
            cols.append(int(target))
            bought.append(np.asarray(history, dtype=np.int64))
        bought.append(np.asarray([target], dtype=np.int64))
    rows_a = np.asarray(rows, dtype=np.int64)
    cols_a = np.asarray(cols, dtype=np.int64)
    popularity = np.bincount(np.concatenate(bought), minlength=n_items).astype(np.float64)
    order = np.lexsort((cols_a, rows_a))
    rows_s, cols_s = rows_a[order], cols_a[order]
    uniq_rows, _starts, row_counts = np.unique(rows_s, return_index=True, return_counts=True)
    indptr = np.zeros(n_items + 1, dtype=np.int64)
    indptr[uniq_rows + 1] = row_counts
    indptr = np.cumsum(indptr)
    indices = cols_s
    # Run-length encode identical consecutive transitions to get per-edge counts.
    edge_id = rows_s * np.int64(n_items) + cols_s
    change = np.empty(len(edge_id), dtype=bool)
    change[0] = True
    change[1:] = edge_id[1:] != edge_id[:-1]
    run_starts = np.flatnonzero(change)
    run_lengths = np.diff(np.append(run_starts, len(edge_id)))
    # Expand back to one count per successor position so it aligns with `indices`.
    edge_counts = np.repeat(run_lengths.astype(np.float64), run_lengths)
    return indptr, indices, edge_counts, popularity, int(train_frame["user"].nunique())


def _popularity_deciles(popularity: np.ndarray, n_buckets: int = 10):
    """Rank items into popularity buckets; used to match negatives by popularity.

    Returns (bucket_padded, bucket_counts, item_bucket) where ``bucket_padded``
    is an (n_buckets, max_size) int64 array of item ids padded with -1, and
    ``item_bucket`` maps each item id to its bucket.
    """
    rank = np.argsort(np.argsort(popularity, kind="stable"), kind="stable")
    bucket = np.minimum((rank * n_buckets) // len(popularity), n_buckets - 1)
    groups = [np.flatnonzero(bucket == b).astype(np.int64) for b in range(n_buckets)]
    max_size = max(1, max(len(g) for g in groups))
    padded = np.full((n_buckets, max_size), -1, dtype=np.int64)
    counts = np.zeros(n_buckets, dtype=np.int64)
    for b, g in enumerate(groups):
        counts[b] = len(g)
        if len(g):
            padded[b, : len(g)] = g
    return padded, counts, bucket.astype(np.int64)


def _crowding2_sampler(indptr, indices, bucket_padded_t, item_bucket_t,
                       pmi_per_edge, n_anchor, device, generator):
    """GPU-vectorised successor-set sample: B anchors x S successors + matched negatives.

    Everything is one gather over precomputed index tensors; there is no Python
    loop over anchors, successors or negatives (CLAUDE.md section 15, item 8).
    PMI weights come from the observed transition counts, so shared-user
    evidence - not adjacency alone - drives the couserw attraction.
    """
    ptr = torch.as_tensor(indptr, device=device)
    per_anchor = ptr[1:] - ptr[:-1]
    eligible = torch.nonzero(per_anchor >= 1, as_tuple=True)[0]
    n_eligible = eligible.numel()
    # Sample a fixed number of anchors, not the whole catalogue: this term is
    # additive to the behaviour loss and must not dominate the step cost.
    b = min(int(n_anchor), int(n_eligible))
    pick = eligible[torch.randint(n_eligible, (b,), device=device, generator=generator)]
    start = ptr[pick]
    width = per_anchor[pick].float().unsqueeze(1)
    slot = (torch.rand(b, _C2_WIDTH, device=device, generator=generator) * width).long()
    slot = torch.minimum(slot, per_anchor[pick].unsqueeze(1) - 1)
    flat = torch.as_tensor(indices, device=device)
    succ = flat[start.unsqueeze(1) + slot]
    mask = torch.arange(_C2_WIDTH, device=device).unsqueeze(0) < per_anchor[pick].unsqueeze(1)

    # Popularity-matched negatives: sample from the anchor's own popularity bucket,
    # which is what makes them hard rather than uniformly easy. The bucket map is
    # precomputed once, so nothing is re-sorted per batch.
    bucket = item_bucket_t[pick]
    width = bucket_padded_t.shape[1]
    sizes = (bucket_padded_t != -1).sum(dim=1)[bucket].unsqueeze(1)
    neg_slot = (torch.rand(b, _C2_NEG, device=device, generator=generator) * sizes.float()).long()
    neg_slot = torch.minimum(neg_slot, sizes - 1)
    flat_pools = bucket_padded_t.reshape(-1)
    negatives = flat_pools[bucket.unsqueeze(1) * width + neg_slot]

    pmi_edges = torch.as_tensor(pmi_per_edge, device=device, dtype=torch.float32)
    pmi = pmi_edges[(start.unsqueeze(1) + slot).reshape(-1)].view(b, _C2_WIDTH) * mask
    return pick, succ, mask, negatives, pmi


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


def _contrastive_ce(
    logits: torch.Tensor,
    source_ids: torch.Tensor,
    target_ids: torch.Tensor,
    successor_indicator=None,
) -> torch.Tensor | None:
    """Apply the shared duplicate/false-negative mask and positive-label CE."""
    batch_size = int(source_ids.shape[0])
    duplicate_targets = target_ids[:, None].eq(target_ids[None, :])
    duplicate_targets.fill_diagonal_(False)
    if successor_indicator is not None:
        rows = successor_indicator[source_ids.cpu().numpy()]
        blocked = torch.from_numpy(
            np.asarray(rows[:, target_ids.cpu().numpy()].todense())
        ).to(device=logits.device, dtype=torch.bool)
        blocked.fill_diagonal_(False)
        duplicate_targets = duplicate_targets | blocked
    logits = logits.masked_fill(duplicate_targets, -torch.inf)
    valid = source_ids.ne(target_ids)
    if not bool(valid.any()):
        return None
    labels = torch.arange(batch_size, device=source_ids.device)
    return F.cross_entropy(logits[valid], labels[valid])


class DirectionalHeads(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Keep Linear's constructor from advancing the global RNG used by the
        # encoder and sklearn KMeans; initialize tensors from this local stream.
        generator = torch.Generator().manual_seed(SEED + 7919)
        with torch.random.fork_rng(devices=[]):
            self.f_source = nn.Linear(CODEBOOK_DIM, CODEBOOK_DIM)
            self.f_target = nn.Linear(CODEBOOK_DIM, CODEBOOK_DIM)
        with torch.no_grad():
            for layer in (self.f_source, self.f_target):
                bound = 1.0 / CODEBOOK_DIM**0.5
                layer.weight.uniform_(-bound, bound, generator=generator)
                layer.bias.uniform_(-bound, bound, generator=generator)

    def project_source(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.f_source(F.normalize(x, dim=-1)), dim=-1)

    def project_target(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.f_target(F.normalize(x, dim=-1)), dim=-1)

    def n_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def behavior_contrastive_loss(
    residuals: torch.Tensor,
    source_ids: torch.Tensor,
    target_ids: torch.Tensor,
    successor_indicator=None,
) -> torch.Tensor:
    """iter48's three-level contrastive loss over residual vectors."""
    batch_size = int(source_ids.shape[0])
    losses: list[torch.Tensor] = []
    for level in ACTIVE_LEVELS:
        source = residuals[level, :batch_size]
        candidates = residuals[level, batch_size:]
        logits = -torch.cdist(source, candidates, p=2) / BEHAVIOR_TEMPERATURE
        loss = _contrastive_ce(logits, source_ids, target_ids, successor_indicator)
        if loss is not None:
            losses.append(loss)
    if not losses:
        return residuals.sum() * 0.0
    return torch.stack(losses).mean()


def directional_residual_loss(
    residuals: torch.Tensor,
    source_ids: torch.Tensor,
    target_ids: torch.Tensor,
    heads: DirectionalHeads,
    symmetric: bool,
    successor_indicator=None,
) -> torch.Tensor:
    """Contrast each source projection with target projections per residual level."""
    batch_size = int(source_ids.shape[0])
    losses: list[torch.Tensor] = []
    for level in range(residuals.shape[0]):
        source = residuals[level, :batch_size]
        candidates = residuals[level, batch_size:]
        logits = heads.project_source(source) @ heads.project_target(candidates).T
        if symmetric:
            reverse = heads.project_source(candidates) @ heads.project_target(source).T
            logits = 0.5 * (logits + reverse)
        logits = logits / BEHAVIOR_TEMPERATURE
        loss = _contrastive_ce(logits, source_ids, target_ids, successor_indicator)
        if loss is not None:
            losses.append(loss)
    if not losses:
        return residuals.sum() * 0.0
    return torch.stack(losses).mean()

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

    geometry_bucket_padded_t = geometry_item_bucket_t = None
    if _GEOMETRY_CONFIG is not None:
        _, _, _, geometry_popularity, _ = _successor_counts(
            train_frame, len(embeddings)
        )
        geometry_buckets, _, geometry_item_bucket = _popularity_deciles(
            geometry_popularity
        )
        geometry_bucket_padded_t = torch.as_tensor(geometry_buckets, device=device)
        geometry_item_bucket_t = torch.as_tensor(geometry_item_bucket, device=device)
    all_embeddings = torch.from_numpy(embeddings)
    train_embeddings = all_embeddings[torch.from_numpy(train_ids)]
    # iter48's full transition stream, drawn by its own sampler so the item
    # batch above keeps exactly the rows, order and Sinkhorn balancing the
    # baseline sees.
    pair_dataset = TransitionPairs(train_frame)
    # 共同层: 拥挤诊断用的 anchor->successor 流 (与训练采样无关，只作测量).
    _transition_pairs_array = pair_dataset.pairs.numpy().astype(np.int64)
    successor_indicator = (
        _successor_indicator(train_frame, len(embeddings))
        if FALSE_NEGATIVE_MASK
        else None
    )
    _hyp_degrees = None
    if _HYP_CONFIG is not None:
        _hyp_degrees = np.bincount(
            pair_dataset.pairs[:, 0].numpy(), minlength=len(embeddings)
        ).astype(np.float32)
    config = _tokenizer_config()
    model = RQVAE(config, in_dim=embeddings.shape[1]).to(device)
    if XAVIER_INIT:
        initialize_tiger_weights(model)
    if _CROWD_CONFIG and _CROWD_CONFIG[0] == "parammatch":
        model.add_module("crowding_control", MatchedParameters(CODEBOOK_SIZE[_CROWD_CONFIG[1]] * CODEBOOK_DIM).to(device))
    if _ANCHOR_ACTIVE:
        # Warm start from the baseline. The anchor is only meaningful if the codebooks
        # begin where the baseline left them, so the checkpoint load comes first and
        # the frozen copy is taken from what was just loaded.
        anchor_state = torch.load(_ANCHOR_BASELINE_CHECKPOINT, map_location="cpu")
        model.load_state_dict(anchor_state["state_dict"])
        model.register_buffer(
            "codebook_anchor",
            torch.stack([
                layer.get_code_embs().detach().clone() for layer in model.rq.vq_layers
            ]).to(device),
        )
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
    raw_module = model.module if isinstance(model, DDP) else model
    if _HYP_CONFIG is not None and _HYP_CONFIG["mechanism"] in {"assign", "joint"}:
        _install_hyp_assignment(
            raw_module.rq.vq_layers[0], _HYP_CONFIG["curvature"],
            _HYP_CONFIG["distance"] == "euclidean",
        )
    # === 共同层: L1 容量掩码。必须装在所有其它 _distances 补丁之后，因为本包装器
    # 捕获当时已安装的距离函数，从而与测地分配等机制叠加而不是互相覆盖。===
    install_level_capacity(raw_module.rq.vq_layers[0], CROWDING_L1_ALLOWED)
    directional_heads = None
    if DIRECTIONAL_MODE != "none":
        directional_heads = DirectionalHeads().to(device)
        raw_module.directional_heads = directional_heads

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
    # === Round 2 crowding precompute (successor CSR, popularity-matched negative
    # pools, PMI weights). All numpy, all once, no per-batch Python loops. ===
    c2_state = None
    if _CROWDING2_CONFIG is not None:
        _indptr, _indices, _edge_counts, _pop, _n_users = _successor_counts(
            train_frame, len(all_embeddings))
        # PMI(a,p) = log(C_ap * C / (C_a * C_p)) over the observed transition graph,
        # clipped at zero. C_ap is the per-edge user count, C_a the outgoing row
        # count, C_p the number of users who ever bought p.
        _pmi = np.zeros(len(_indices), dtype=np.float64)
        np.log(_edge_counts, out=_pmi, where=_edge_counts > 0)
        _pmi += np.log(max(_n_users, 1)) - np.log(np.maximum(_pop[_indices], 1.0))
        _pmi = np.maximum(_pmi, 0.0)
        _buckets, _bucket_sizes, _item_bucket = _popularity_deciles(_pop)
        _buckets_t = torch.as_tensor(_buckets, device=device)
        _item_bucket_t = torch.as_tensor(_item_bucket, device=device)
        c2_gen = torch.Generator(device=device).manual_seed(SEED + 104729 + rank)
        c2_state = (_indptr, _indices, _buckets_t, _item_bucket_t, _pmi, c2_gen)

    parameters = list(model.parameters())
    if OPTIMIZER.lower() == "adagrad":
        optimizer = torch.optim.Adagrad(parameters, lr=LR, weight_decay=WEIGHT_DECAY)
    else:
        optimizer = torch.optim.AdamW(parameters, lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = max(1, EPOCHS * len(loader))
    warmup_steps = max(0, WARMUP_EPOCHS * len(loader))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if total_steps <= warmup_steps:
            return 1.0
        return max(0.0, float(total_steps - step) / float(total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if _CROWD_CONFIG is None:
        crowding_metadata = None
    else:
        crowding_mechanism, crowding_level, crowding_parameters = _CROWD_CONFIG
        crowding_metadata = crowding_record(
            {**crowding_parameters, "mechanism": crowding_mechanism},
            EPOCHS,
            RAMP_START_EPOCH,
            CODEBOOK_SIZE[crowding_level] * CODEBOOK_DIM if crowding_mechanism == "parammatch" else 0,
        )
    if rank == 0:
        print(f"[RecBole RQ-VAE] device={device} all_items={len(all_embeddings)} train_items={len(train_embeddings)} world_size={world_size}", flush=True)
        print(f"[RecBole RQ-VAE] config: epochs={EPOCHS} batch_size_per_rank={BATCH_SIZE_PER_RANK} total_batch={BATCH_SIZE_PER_RANK*world_size} lr={LR}", flush=True)
        _record(rank, "train_start", epochs=EPOCHS, batch_size_per_rank=BATCH_SIZE_PER_RANK, total_batch_size=BATCH_SIZE_PER_RANK*world_size, lr=LR, weight_decay=WEIGHT_DECAY, hidden_sizes=list(HIDDEN_SIZES), codebook_num=CODEBOOK_NUM, codebook_size=list(CODEBOOK_SIZE), codebook_dim=CODEBOOK_DIM, beta=BETA, vq_type=VQ_TYPE, ema_decay=EMA_DECAY, sk_epsilon=SK_EPSILON, sk_iters=SK_ITERS, pca_dim=PCA_DIM, xavier_init=XAVIER_INIT, warmup_epochs=WARMUP_EPOCHS, gradient_clip_norm=GRADIENT_CLIP_NORM, optimizer=OPTIMIZER, seed=SEED, num_workers=NUM_WORKERS, eval_interval=EVAL_INTERVAL, early_stop="disabled", all_items=len(all_embeddings), train_items=len(train_embeddings), embedding_shape=list(embeddings.shape), world_size=world_size, mechanism="iter48_residual_contrastive_corrected_sampling", experiment_arm=EXPERIMENT_ARM, active_levels=list(ACTIVE_LEVELS), resample_pairs=RESAMPLE_PAIRS, false_negative_mask=FALSE_NEGATIVE_MASK, behavior_weight_max=BEHAVIOR_WEIGHT_MAX, behavior_temperature=BEHAVIOR_TEMPERATURE, behavior_ramp_start_epoch=RAMP_START_EPOCH, pair_batch_size_per_rank=PAIR_BATCH_SIZE_PER_RANK, transition_pairs=len(pair_dataset), pair_sampler_seed=SEED + 1, source="348aed182", directional_mode=DIRECTIONAL_MODE, directional_head_parameters=0 if directional_heads is None else directional_heads.n_parameters(), crowding_l1_allowed=CROWDING_L1_ALLOWED, crowding_mechanism=CROWDING_MECHANISM, crowding=crowding_metadata)

    # === codebook init: only rank 0 (codebook 不是 DDP 参数, 全 rank 共享) ===
    if rank == 0:
        print("[RecBole RQ-VAE] initializing codebooks with KMeans", flush=True)
    with torch.no_grad():
        raw_module.init_codebook(train_embeddings.to(device))
    if world_size > 1:
        dist.barrier()

    # === M5 唯一机制: L2 分配按 L1 桶的协同兴趣方向分叉 ===
    # 方向必须从 KMeans 初始分配 + 训练转移图读出, 所以装在这里; 桶的测量用容量掩码
    # 约束下的 L1 分配 (真实拥挤 regime)。只包装 L2, L1/L3 规则与对照完全一致。
    _branch_directions = torch.zeros(
        int(CODEBOOK_SIZE[0]), BRANCH_K, int(CODEBOOK_DIM), device=device
    )
    if rank == 0:
        with torch.no_grad():
            _branch_directions = interest_directions(
                raw_module, all_embeddings.to(device), _transition_pairs_array,
                allowed=CROWDING_L1_ALLOWED,
            )
    if world_size > 1:
        dist.broadcast(_branch_directions, 0)
    install_interest_branch(
        raw_module.rq.vq_layers[1], raw_module.rq.vq_layers[0], _branch_directions
    )
    if rank == 0:
        _record(
            rank, "interest_buckets", epoch=0,
            buckets_with_axis=int((_branch_directions.abs().sum(-1).sum(-1) > 0).sum()),
            k=BRANCH_K, strength=BRANCH_STRENGTH,
        )

    # Stage3 only consumes the final-epoch weights, so tracking the best
    # collision across epochs would be misleading; keep the last evaluation.
    final_collision = float("inf")
    final_raw_unique = 0
    codebook_sizes = list(config.codebook_size)
    crowd_graph = None
    if _CROWD_CONFIG and _CROWD_CONFIG[0] in {"couser", "simweight", "density", "random"}:
        crowd_graph = np.load("/home/wlia0047/ar57/wenyu/GeneRec/results/crowding/graph_s42.npz")

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
        crowds: list[float] = []
        geometry_values: list[float] = []
        geometry_encoder_norms: list[float] = []
        geometry_codebook_norms: list[float] = []
        quantized_values: list[float] = []
        quantized_encoder_norms: list[float] = []
        quantized_codebook_l1_norms: list[float] = []
        quantized_codebook_l2_norms: list[float] = []
        quantized_codebook_l3_norms: list[float] = []
        hyperbolic_weights = []
        epoch_weight = behavior_weight(epoch)
        crowd_config_for_ramp = None if _CROWD_CONFIG is None else {
            **_CROWD_CONFIG[2], "mechanism": _CROWD_CONFIG[0]
        }
        epoch_crowding_weight = crowding_ramp_weight(
            epoch, crowd_config_for_ramp, RAMP_START_EPOCH, EPOCHS
        )
        epoch_hyp_alpha = epoch_weight / BEHAVIOR_WEIGHT_MAX if BEHAVIOR_WEIGHT_MAX > 0.0 else 0.0
        if _HYP_CONFIG is not None and _HYP_CONFIG["mechanism"] in {"assign", "joint"}:
            raw_module.rq.vq_layers[0]._hyp_assignment_active = epoch_hyp_alpha > 0.0
        _epoch_t0 = time.time()
        hyperbolic_losses = []
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            reconstructed, quant_loss, _, _ = model(batch)
            # === DDP 包装下, 通过 .module 调 compute_loss (避免 DDP all_reduce 干扰) ===
            base_loss, recon_loss = raw_module.compute_loss(batch, reconstructed, quant_loss)
            behavior_loss = torch.zeros((), device=device)
            crowd_value = torch.zeros((), device=device)
            crowd_metric = torch.zeros((), device=device)
            crowding_weight = epoch_crowding_weight
            hyp_value = torch.zeros((), device=device)
            hyp_weight = 0.0
            geometry_value = torch.zeros((), device=device)
            geometry_encoder_norm = 0.0
            geometry_codebook_norm = 0.0
            quantized_term = torch.zeros((), device=device)
            anti_collapse_term = torch.zeros((), device=device)
            anchor_term = torch.zeros((), device=device)
            anchor_fields: dict[str, float] = {}
            if epoch_weight > 0.0:
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
                if DIRECTIONAL_MODE != "none":
                    behavior_loss = directional_residual_loss(
                        residuals, source_ids, target_ids, directional_heads,
                        DIRECTIONAL_MODE == "sym", successor_indicator,
                    )
                else:
                    behavior_loss = behavior_contrastive_loss(
                        residuals, source_ids, target_ids, successor_indicator
                    )
                if _GEOMETRY_CONFIG is not None:
                    geometry_negative_ids = _matched_geometry_negative_ids(
                        source_ids, target_ids, geometry_bucket_padded_t,
                        geometry_item_bucket_t,
                    )
                    geometry_item_ids = torch.cat(
                        (source_ids, target_ids, geometry_negative_ids)
                    )
                    geometry_item_batch = all_embeddings[
                        geometry_item_ids.cpu()
                    ].to(device, non_blocking=True)
                    geometry_latent = raw_module.encoder(geometry_item_batch)
                    geometry_value = _geometry_loss_for_pairs(
                        EXPERIMENT_ARM, geometry_latent, source_ids,
                        raw_module.rq.vq_layers[0].get_code_embs(),
                    )
                    geometry_norms = _geometry_gradient_norms(geometry_value, raw_module)
                    geometry_encoder_norm = geometry_norms["encoder"]
                    geometry_codebook_norm = geometry_norms["codebook"]
                    if EXPERIMENT_ARM in {"B_E", "B_H", "B_E_nocb", "MARGIN", "LETTER", "LETTER_nodiv", "DIGER", "ANCHOR_CTRL", "ANCHOR_W001", "ANCHOR_W01", "ANCHOR_W1"}:
                        quant_residuals, quant_tokens = _residuals_and_tokens_with_trace(
                            raw_module, geometry_item_batch,
                        )
                        pair_count = int(source_ids.shape[0])
                        quant_anchors = torch.arange(pair_count, device=device)
                        quant_positives = quant_anchors + pair_count
                        quant_negatives = quant_positives + pair_count
                        quantized_term = _quantized_ranking_loss(
                            quant_residuals, quant_tokens, raw_module,
                            quant_anchors, quant_positives, quant_negatives,
                            weight=0.2 * epoch_hyp_alpha,
                        )
                        anti_collapse_term = _anti_collapse_term(
                            quant_residuals, quant_tokens, model, quantized_term,
                        )
                        quant_norms = _quantized_gradient_norms(
                            quantized_term, raw_module,
                        )
                        quantized_encoder_norms.append(quant_norms["encoder"])
                        quantized_codebook_l1_norms.append(quant_norms["codebook_l1"])
                        quantized_codebook_l2_norms.append(quant_norms["codebook_l2"])
                        quantized_codebook_l3_norms.append(quant_norms["codebook_l3"])
                    if _ANCHOR_ACTIVE:
                        # The frozen baseline codebook, pulled toward the current one.
                        # This is the only counterweight the ranking term has: at
                        # coefficient 0.2 it supplies about 56x the 1.18e-04 the VQ
                        # commitment term gives L1, so a coefficient of zero here is a
                        # genuine control rather than a formality.
                        anchor_gap = torch.stack([
                            (layer.get_code_embs()
                             - raw_module.codebook_anchor[level]).pow(2).sum()
                            for level, layer in enumerate(raw_module.rq.vq_layers)
                        ])
                        anchor_term = anchor_gap.sum() * (_ANCHOR_WEIGHT * epoch_hyp_alpha)
                        anchor_fields = {
                            "anchor_loss": float(anchor_term.detach()),
                            "anchor_drift_l1": float(anchor_gap[0].detach()),
                            "anchor_drift_l2": float(anchor_gap[1].detach()),
                            "anchor_drift_l3": float(anchor_gap[2].detach()),
                        }
                if _CROWD_CONFIG is not None:
                    mechanism, crowd_level, crowd_config = _CROWD_CONFIG
                    n_pairs = int(source_ids.shape[0])
                    crowd_target_ids = target_ids
                    if mechanism == "couser":
                        graph_neighbors = crowd_graph["neighbors"]
                        slots = source_ids.cpu().numpy() % crowd_config["k"]
                        selected = graph_neighbors[source_ids.cpu().numpy(), slots]
                        selected = np.where(selected >= 0, selected, target_ids.cpu().numpy())
                        crowd_target_ids = torch.as_tensor(selected, device=device, dtype=torch.long)
                    crowd_pair_batch = all_embeddings[
                        torch.cat((source_ids, crowd_target_ids)).cpu()
                    ].to(device, non_blocking=True)
                    local_pairs = torch.stack((torch.arange(n_pairs, device=device),
                                               torch.arange(n_pairs, 2 * n_pairs, device=device)), dim=1)
                    latent = raw_module.encoder(crowd_pair_batch)
                    graph_weights = torch.ones(n_pairs, device=device)
                    graph_degrees = torch.ones(len(latent), device=device)
                    if crowd_graph is not None:
                        graph_degrees = torch.as_tensor(crowd_graph["degree"], device=device)[
                            torch.cat((source_ids, crowd_target_ids))]
                        if mechanism == "simweight":
                            neigh = crowd_graph["neighbors"][source_ids.cpu().numpy()]
                            pair_match = neigh == target_ids.cpu().numpy()[:, None]
                            slots = pair_match.argmax(axis=1)
                            found = pair_match.any(axis=1)
                            raw_weights = crowd_graph["weights"][source_ids.cpu().numpy(), slots]
                            graph_weights = torch.as_tensor(np.where(found, raw_weights, 0.0), device=device)
                    random_local = torch.stack((torch.arange(n_pairs, device=device),
                                               torch.randperm(n_pairs, device=device) + n_pairs), dim=1)
                    crowd_value = crowding_loss(
                        mechanism, latent, local_pairs, strength=crowd_config["strength"],
                        codebook=(tuple(layer.get_code_embs() for layer in raw_module.rq.vq_layers[:crowd_config["depth"]])
                                  if mechanism == "prefixshare" else raw_module.rq.vq_layers[crowd_level].get_code_embs()),
                        tau=crowd_config["tau"], depth=crowd_config["depth"],
                        concentration=crowd_config["concentration"],
                        pair_weights=graph_weights,
                        degrees=graph_degrees,
                        negatives=torch.roll(local_pairs[:, 1], 1),
                        random_pairs=random_local,
                        param_block=getattr(raw_module, "crowding_control", None),
                    )
                    crowd_metric = crowd_value.detach()
            loss = assemble_losses(
                base_loss, behavior_loss, epoch_weight, crowd_value, crowding_weight,
                _CROWD_CONFIG is not None,
            )
            if _GEOMETRY_CONFIG is not None:
                loss = _compose_geometry_loss(loss, geometry_value)
            if EXPERIMENT_ARM in {"B_E", "B_H", "B_E_nocb", "MARGIN", "LETTER", "LETTER_nodiv", "DIGER", "ANCHOR_CTRL", "ANCHOR_W001", "ANCHOR_W01", "ANCHOR_W1"}:
                loss = _compose_quantized_loss(loss, quantized_term)
                loss = _ac_compose(loss, anti_collapse_term)
            if _ANCHOR_ACTIVE:
                loss = loss + anchor_term
            crowd2_value = torch.zeros((), device=device)
            if c2_state is not None and epoch > CROWDING2_RAMP_START and epoch_weight > 0.0:
                mech2, level2 = _CROWDING2_CONFIG
                (_indptr, _indices, _buckets_t, _item_bucket_t,
                 _pmi2, _c2_gen) = c2_state
                pick, succ, succ_mask, negs, pmi_w = _crowding2_sampler(
                    _indptr, _indices, _buckets_t, _item_bucket_t,
                    _pmi2, CROWDING2_ANCHORS, device, _c2_gen,
                )
                # One embedding lookup for anchors and their successors together;
                # z therefore indexes anchors at 0..B-1 and successors at B..B+S-1.
                mech_ids = torch.cat((pick, succ.reshape(-1)))
                mech_emb = all_embeddings[mech_ids.cpu()].to(device, non_blocking=True)
                mech_z = raw_module.encoder(mech_emb)
                mech_res = _residuals_with_trace(raw_module, mech_emb)
                b_anchor = mech_z.shape[0] // (1 + _C2_WIDTH)
                anchor_idx = torch.arange(b_anchor, device=device)
                pos_idx = b_anchor + torch.arange(_C2_WIDTH, device=device).unsqueeze(0)
                pos_idx = pos_idx.expand(b_anchor, _C2_WIDTH)
                neg_idx = torch.roll(anchor_idx, 1).unsqueeze(1).expand(-1, _C2_NEG)
                crowd2 = _crowd2.mechanism_loss2(
                    mech2, mech_z, anchor_idx, pos_idx, succ_mask, neg_idx,
                    torch.stack([layer.get_code_embs()
                                for layer in raw_module.rq.vq_layers], dim=0),
                    residual_levels=mech_res,
                    pmi_weights=pmi_w,
                    random_pairs=torch.stack((anchor_idx, torch.roll(anchor_idx, 7)), dim=1),
                    behavior_loss=behavior_loss,
                    strength=_crowd2.STRENGTHS[level2 - 1],
                )
                crowd2_value = next(iter(crowd2["losses"].values())).detach()
                # Additive composition: Exp1a's behaviour term is untouched, and
                # additive_total returns that identical tensor at zero weight.
                loss = _crowd2.additive_total(loss, crowd2)
                crowds.append(float(crowd2_value))
            if _HYP_CONFIG is not None and epoch_weight > 0.0:
                hyp_latent = raw_module.encoder(pair_batch)
                hyp_positive = torch.arange(len(source_ids), device=device)
                hyp_degree = torch.as_tensor(
                    _hyp_degrees[source_ids.cpu().numpy()], device=device,
                )
                hyp_codebooks = (
                    tuple(layer.get_code_embs() for layer in raw_module.rq.vq_layers)
                    if _HYP_CONFIG["mechanism"] == "perlayer"
                    else raw_module.rq.vq_layers[0].get_code_embs()
                )
                hyp_value = _hyp_mechanism_loss(
                    _HYP_CONFIG["mechanism"], hyp_latent, hyp_codebooks,
                    hyp_positive, _HYP_CONFIG["curvature"], _HYP_CONFIG["strength"],
                    hyp_degree, euclidean=_HYP_CONFIG["distance"] == "euclidean",
                )
                hyp_weight = _HYP_CONFIG["strength"] * epoch_hyp_alpha
                loss = loss + hyp_value * hyp_weight
            if not torch.isfinite(loss):
                raise RuntimeError(f"RQ-VAE loss became non-finite at epoch {epoch}")
            # The parameter-matched control's block is used by NOTHING except the crowding
            # term, and assemble_losses() above overwrites `loss` unconditionally, so the
            # zero-weighted edge has to be added AFTER it and immediately before backward.
            # DDP refuses a parameter that never receives a gradient; multiplying by 0.0
            # keeps the value identical (the block is finite) and only satisfies the reducer.
            _crowd_control = getattr(raw_module, "crowding_control", None)
            if _crowd_control is not None:
                loss = loss + 0.0 * sum(
                    parameter.square().mean() for parameter in _crowd_control.parameters()
                )
            loss.backward()
            if directional_heads is not None and epoch_weight > 0.0 and world_size > 1:
                for parameter in directional_heads.parameters():
                    if parameter.grad is not None:
                        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
                        parameter.grad /= world_size
            if GRADIENT_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach()))
            recons.append(float(recon_loss.detach()))
            behaviors.append(float(behavior_loss.detach()))
            geometry_values.append(float(geometry_value.detach()))
            geometry_encoder_norms.append(geometry_encoder_norm)
            geometry_codebook_norms.append(geometry_codebook_norm)
            quantized_values.append(float(quantized_term.detach()))
            crowds.append(float(crowd_metric.detach()))
            if _HYP_CONFIG is not None:
                hyperbolic_losses.append(float(hyp_value.detach()))
                hyperbolic_weights.append(hyp_weight)
        epoch_time_s = round(time.time() - _epoch_t0, 3)

        # === DDP all_reduce: 跨 rank 求平均 loss / recon ===
        local_loss = torch.tensor(float(np.mean(losses)) if losses else 0.0, device=device)
        local_recon = torch.tensor(float(np.mean(recons)) if recons else 0.0, device=device)
        local_behavior = torch.tensor(float(np.mean(behaviors)) if behaviors else 0.0, device=device)
        local_crowding = torch.tensor(float(np.mean(crowds)) if crowds else 0.0, device=device)
        local_geometry = torch.tensor(float(np.mean(geometry_values)) if geometry_values else 0.0, device=device)
        local_geometry_encoder_grad = torch.tensor(float(np.mean(geometry_encoder_norms)) if geometry_encoder_norms else 0.0, device=device)
        local_geometry_codebook_grad = torch.tensor(float(np.mean(geometry_codebook_norms)) if geometry_codebook_norms else 0.0, device=device)
        local_quantized_metrics = torch.tensor([
            float(np.mean(quantized_values)) if quantized_values else 0.0,
            float(np.mean(quantized_encoder_norms)) if quantized_encoder_norms else 0.0,
            float(np.mean(quantized_codebook_l1_norms)) if quantized_codebook_l1_norms else 0.0,
            float(np.mean(quantized_codebook_l2_norms)) if quantized_codebook_l2_norms else 0.0,
            float(np.mean(quantized_codebook_l3_norms)) if quantized_codebook_l3_norms else 0.0,
        ], device=device)
        if world_size > 1:
            dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_recon, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_behavior, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_crowding, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_geometry, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_geometry_encoder_grad, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_geometry_codebook_grad, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_quantized_metrics, op=dist.ReduceOp.SUM)
        avg_loss = float(local_loss.item()) / max(world_size, 1)
        avg_recon = float(local_recon.item()) / max(world_size, 1)
        avg_behavior = float(local_behavior.item()) / max(world_size, 1)
        avg_crowding = float(local_crowding.item()) / max(world_size, 1)
        avg_geometry = float(local_geometry.item()) / max(world_size, 1)
        avg_geometry_encoder_grad = float(local_geometry_encoder_grad.item()) / max(world_size, 1)
        avg_geometry_codebook_grad = float(local_geometry_codebook_grad.item()) / max(world_size, 1)
        avg_quantized_metrics = local_quantized_metrics.div(max(world_size, 1)).tolist()
        anchor_metric_fields = {}
        if _ANCHOR_ACTIVE and anchor_fields:
            anchor_local = torch.tensor(
                [float(anchor_fields[key]) for key in sorted(anchor_fields)],
                device=device,
            )
            if world_size > 1:
                dist.all_reduce(anchor_local, op=dist.ReduceOp.SUM)
                anchor_local = anchor_local / max(world_size, 1)
            anchor_metric_fields = {
                key: float(value)
                for key, value in zip(sorted(anchor_fields), anchor_local)
            }
        quantized_metric_fields = {}
        if EXPERIMENT_ARM in {"B_E", "B_H", "B_E_nocb", "MARGIN", "LETTER", "LETTER_nodiv", "DIGER", "ANCHOR_CTRL", "ANCHOR_W001", "ANCHOR_W01", "ANCHOR_W1"} and epoch_weight > 0.0:
            quantized_metric_fields = {
                "quantized_ranking_loss": avg_quantized_metrics[0],
                "quantized_encoder_grad_norm": avg_quantized_metrics[1],
                "quantized_codebook_l1_grad_norm": avg_quantized_metrics[2],
                "quantized_codebook_l2_grad_norm": avg_quantized_metrics[3],
                "quantized_codebook_l3_grad_norm": avg_quantized_metrics[4],
            }
        _HYP_METRIC_FIELDS.clear()
        if _HYP_CONFIG is not None:
            _HYP_METRIC_FIELDS.update({
                "hyperbolic_loss": float(np.mean(hyperbolic_losses)),
                "hyperbolic_weight": float(np.mean(hyperbolic_weights)),
            })
        cur_lr = optimizer.param_groups[0]["lr"]

        # === 非 eval 间隔: 只 log train 行 ===
        if epoch % EVAL_INTERVAL != 0 and epoch != EPOCHS:
            if rank == 0:
                print(f"[RQ-VAE] epoch={epoch} loss={avg_loss:.8f} recon={avg_recon:.8f} beh={avg_behavior:.6f} geometry={avg_geometry:.6f} geometry_encoder_grad_norm={avg_geometry_encoder_grad:.8g} geometry_codebook_grad_norm={avg_geometry_codebook_grad:.8g} quantized_metrics={quantized_metric_fields} w={epoch_weight:.3f} lr={cur_lr:.3e} time={epoch_time_s}s", flush=True)
                _record(rank, "train", epoch=epoch, loss=avg_loss, recon=avg_recon, behavior_contrastive=avg_behavior, behavior_weight=epoch_weight, geometry_loss=avg_geometry, geometry_encoder_grad_norm=avg_geometry_encoder_grad, geometry_codebook_grad_norm=avg_geometry_codebook_grad, lr=cur_lr, epoch_time_s=epoch_time_s, cumulative_time_s=round(time.time() - _T0, 3), **crowding_train_fields(_CROWD_CONFIG, avg_crowding, epoch_crowding_weight), **quantized_metric_fields, **anchor_metric_fields)
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
                # 共同层: 每次评估记录一次拥挤诊断 (全量 item 的 SID + 训练转移流).
                _record(
                    rank, "crowding", epoch=epoch,
                    **crowding_report(
                        raw_tokens, _transition_pairs_array,
                        np.random.default_rng(SEED + 31),
                        recon_mse=full_item_recon_mse(raw_module, all_embeddings.to(device)),
                    ),
                )
                print(
                    f"[RQ-VAE] epoch={epoch} loss={avg_loss:.8f} recon={avg_recon:.8f} "
                    f"geometry={avg_geometry:.6f} geometry_encoder_grad_norm={avg_geometry_encoder_grad:.8g} "
                    f"geometry_codebook_grad_norm={avg_geometry_codebook_grad:.8g} "
                    f"quantized_metrics={quantized_metric_fields} "
                    f"lr={cur_lr:.3e} raw_unique={raw_unique}/{len(raw_tokens)} "
                    f"collision={collision_v:.6f} time={epoch_time_s}s",
                    flush=True,
                )
                _record(
                    rank, "train", epoch=epoch, loss=avg_loss, recon=avg_recon, behavior_contrastive=avg_behavior, behavior_weight=epoch_weight, lr=cur_lr,
                    epoch_time_s=epoch_time_s,
                    geometry_loss=avg_geometry,
                    geometry_encoder_grad_norm=avg_geometry_encoder_grad,
                    geometry_codebook_grad_norm=avg_geometry_codebook_grad,
                    cumulative_time_s=round(time.time() - _T0, 3),
                    **crowding_train_fields(_CROWD_CONFIG, avg_crowding, epoch_crowding_weight),
                    **quantized_metric_fields, **anchor_metric_fields,
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
