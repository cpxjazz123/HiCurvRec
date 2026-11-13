"""Train RecBole3.0's TIGER RQ-VAE (4 卡 DDP) and export HG-Rec-compatible SIDs.

2026-09-22 改造 (CLAUDE.md §1 + §3):
- 参数全部硬编码为模块常量 (禁 argparse/CLI flag)
- 加 _launch_via_torchrun 自动 fork 4 卡 DDP (nproc=4, master_port=50202, 避免与 stage3 的 50201 冲突)
- 加全程 metrics JSONL 收集器 (每个 train/eval/ckpt_saved/train_end 一行)
- 输出路径硬编码到 results/stage2_RQ-VAE/TIGER_af3_curvature_factor_mass_rq/

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

# `eval/af3_item_potential.npz` is read-only evidence for this arm (AF3 formula 14).
import af3_item_potential
# The ONE u -> p map (math.json `share_map`); the trainer calls it exactly once.
import factor_share_geometry
from scipy.stats import rankdata


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
ARM_NAME = "TIGER_af3_curvature_factor_mass_rq"
RESULT_DIR = REPO_ROOT / "results" / "stage2_RQ-VAE" / ARM_NAME

EMBEDDING_FILE = REPO_ROOT / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
TRAIN_FILE     = REPO_ROOT / "results" / "stage0_build_parquet" / "train.parquet"
OUTPUT_SID     = RESULT_DIR / "sids_for_hgrec.npy"
OUTPUT_JSON    = RESULT_DIR / "item_sids.json"
CHECKPOINT     = RESULT_DIR / "rqvae_best.pth"
LOG_DIR        = REPO_ROOT / "stage2_RQ-VAE" / ARM_NAME / "logs"
METRICS_PATH   = LOG_DIR / "training_metrics.jsonl"

# === 机制接线 (AF3 product manifold) ===
# The metric comes from the registry by name; the Euclidean entry stays the default,
# so this line is the ONE place that selects the arm's distance. It is NOT a config key:
# config/train_config.json must stay byte-identical to the frozen baseline's.
DISTANCE_NAME = "factor_share"
# The wiring switch (`both_on` main arm / `behaviour_off` / `geometry_off`). A degenerate
# check flips this one line; it never enters the config and never moves a hyperparameter.
SHARE_MODE = "both_on"
#: the frozen full-table potential, relative to this candidate tree.
POTENTIAL_FILE = Path(__file__).resolve().with_name("eval") / "af3_item_potential.npz"


def load_frozen_table(path: Path | None = None) -> dict:
    """Read the frozen table once; this file never recomputes it.

    `af3_item_potential.load` verifies the archive digest and every FIELDS entry, so a
    tampered or truncated artefact fails loud before anything touches the GPU.
    """
    target = POTENTIAL_FILE if path is None else Path(path)
    table = af3_item_potential.load(target)
    item_ids = np.asarray(table["item_ids"], dtype=np.int64)
    if item_ids.size == 0:
        raise RuntimeError("the frozen AF3 potential table is empty: " + str(target))
    if not np.array_equal(item_ids, np.arange(item_ids.size, dtype=np.int64)):
        raise RuntimeError("the frozen AF3 table must cover a dense item-id space "
                           "[0, N); got a sparse/gapped id column")
    u = np.asarray(table["u"], dtype=np.float64)
    if u.shape != item_ids.shape or not np.isfinite(u).all():
        raise RuntimeError("the frozen AF3 share column u is malformed")
    if float(u.min()) < 0.0 or float(u.max()) > 1.0:
        raise RuntimeError("the frozen AF3 share column u leaves [0, 1]")
    return table


def shares_for(item_ids, table: dict) -> torch.Tensor:
    """`u_i` for exactly these item ids; a missing id is a hard error, never a default."""
    ids = np.asarray(item_ids, dtype=np.int64).reshape(-1)
    if ids.size == 0:
        raise RuntimeError("no item ids: the per-item share cannot be indexed")
    table_ids = np.asarray(table["item_ids"], dtype=np.int64)
    missing = np.setdiff1d(ids, table_ids, assume_unique=False)
    if missing.size:
        raise RuntimeError("%d item ids are missing from the frozen AF3 table, e.g. %s"
                           % (int(missing.size), missing[:8].tolist()))
    u = np.asarray(table["u"], dtype=np.float64)
    return torch.from_numpy(u[ids].astype(np.float32))


def frozen_share(table: dict) -> torch.Tensor:
    """`p_i = u_i` for the WHOLE table: `share(u, 'struct')` called once and frozen.

    `math.json` `share_map` says `p_i = u_i` is the ONE mapping and that it lives in
    `factor_share_geometry.share`; this is the trainer's single call to it, made before
    any training step and frozen into a plain `(N,)` tensor. The model then only
    INDEXES that tensor with the batch's item ids -- it never re-derives the map.
    """
    u = np.asarray(table["u"], dtype=np.float64)
    values = torch.from_numpy(u.astype(np.float32))
    share = factor_share_geometry.share(values, "struct")
    if tuple(share.shape) != tuple(values.shape):
        raise RuntimeError("share() changed the table length: %s -> %s"
                           % (tuple(values.shape), tuple(share.shape)))
    share = share.detach().to(dtype=torch.float32).clone()
    # p_i = u_i is an identity map; anything else means the arm's share map moved.
    if not torch.allclose(share, values, rtol=0.0, atol=1e-6):
        raise RuntimeError("p_i = u_i is no longer the identity; check "
                           "factor_share_geometry.share")
    return share


def _spearman(left, right):
    """Rank correlation of two equally long series; None when either side is constant."""
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if a.size != b.size or a.size < 2:
        raise RuntimeError("Spearman needs two series of the same length >= 2: %d vs %d"
                           % (a.size, b.size))
    if float(np.ptp(a)) == 0.0 or float(np.ptp(b)) == 0.0:
        return None
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def potential_report(table: dict, share: torch.Tensor | None = None) -> dict:
    """The read-only readings of AF3 formula `metrics`, computed from the table.

    `p` is NOT recomputed here: the caller passes the SAME frozen `(N,)` vector the
    quantiser receives (`frozen_share`), so the rank correlations below measure the
    vector that actually enters the geometry, not `u` sorted against itself.
    """
    u = np.asarray(table["u"], dtype=np.float64)
    rank = np.asarray(table["rank"], dtype=np.float64)
    potential = np.asarray(table["potential"], dtype=np.float64)
    missing = np.asarray(table["is_missing"], dtype=bool)
    num_items = int(u.size)
    p = (np.asarray(frozen_share(table), dtype=np.float64)
         if share is None else np.asarray(share, dtype=np.float64).reshape(-1))
    if p.shape != u.shape:
        raise RuntimeError("the logged share has %d entries but the table has %d"
                           % (int(p.size), int(u.size)))
    s_hi, s_lo = np.sqrt(p), np.sqrt(1.0 - p)
    graph_mean_rank = float(np.asarray(table["rank_of_graph_mean"]).reshape(-1)[0])
    p_miss = (graph_mean_rank - 0.5) / float(num_items)
    return {
        "items": num_items,
        "missing": int(missing.sum()),
        "p_percentiles": {str(q): float(np.percentile(p, q)) for q in (0, 1, 25, 50, 75, 99, 100)},
        "s_hi_percentiles": {str(q): float(np.percentile(s_hi, q)) for q in (0, 50, 100)},
        "s_lo_percentiles": {str(q): float(np.percentile(s_lo, q)) for q in (0, 50, 100)},
        # p against u: the mapping is monotone by definition, so ~+1 here only holds if
        # the vector the quantiser gets is really the mapped one.
        "p_u_spearman": _spearman(p, u),
        # p against the structural potential: catches a rank map that lost ties or the
        # graph-mean rows (None when the potential table has no spread).
        "p_potential_spearman": _spearman(p, potential),
        "rank_of_graph_mean": graph_mean_rank,
        "p_miss": float(p_miss),
        # math.json `|p_miss - SHARE_REF|`: the reference is the geometry constant, not a
        # literal, so the reading cannot drift away from `factor_share_geometry`.
        "p_miss_minus_share_ref": float(abs(p_miss - factor_share_geometry.SHARE_REF)),
        "u_min": float(u.min()),
        "u_max": float(u.max()),
        "rank_min": float(rank.min()),
        "rank_max": float(rank.max()),
        "vq_type": str(CONFIG.VQ_TYPE),
    }


def energy_conservation_report(table: dict, hidden: torch.Tensor,
                               item_ids: torch.Tensor) -> dict:
    """`metrics.energy_conservation_rel_error`, measured on real encoder output.

    `math.json` `energy_split`: the product side is NOT scaled by `p`, so charting the
    two hyperbolic tangent blocks and reading them back with `logmap0` has to return
    the same vectors elementwise, whatever `p_i` is. `energy_rel_error_max` is a
    measurement -- 0 only while the point stays inside its ball, so a clipped point
    shows up instead of being assumed away -- and `energy_p_corr` reads the rank
    correlation between the product side's tangent norm and the share instead of
    hard-coding it to 0 (it flips sign when `p` is replaced by `1 - p`, which is what
    makes the reading a test).
    """
    geometry = factor_share_geometry
    values = torch.as_tensor(hidden, dtype=torch.float32).detach()
    if values.dim() != 2 or int(values.shape[1]) != int(geometry.D_TOTAL):
        raise ValueError("hidden must be (B, %d), got shape %s"
                         % (geometry.D_TOTAL, tuple(values.shape)))
    ids = torch.as_tensor(item_ids, dtype=torch.long).reshape(-1)
    p_all = frozen_share(table)
    if int(ids.numel()) != int(values.shape[0]):
        raise ValueError("one share per row is required: %d ids vs %d rows"
                         % (int(ids.numel()), int(values.shape[0])))
    if int(ids.max()) >= int(p_all.numel()):
        raise RuntimeError("an item id is outside the frozen share table")
    p = p_all.to(dtype=torch.float64)[ids]
    h1 = values[:, geometry.D_E:geometry.D_E + geometry.D_H1]
    h2 = values[:, geometry.D_E + geometry.D_H1:]
    x_hi, x_lo = geometry.item_chart_head(values)
    back = geometry.log_at_origin(x_hi, x_lo).to(torch.float64)
    target = torch.cat([h1, h2], dim=-1).to(torch.float64)
    norm_back = back.norm(dim=-1)
    norm_target = target.norm(dim=-1)
    rel = (norm_back - norm_target).abs() / norm_target.clamp_min(1e-12)
    # A point pinned at the project() boundary is NOT a rounded chart; report it.
    radius_hi = 1.0 / float(geometry.C_HI) ** 0.5
    radius_lo = 1.0 / float(geometry.C_LO) ** 0.5
    clipped = ((x_hi.to(torch.float64).norm(dim=-1) >= 0.999 * radius_hi)
               | (x_lo.to(torch.float64).norm(dim=-1) >= 0.999 * radius_lo))
    return {
        "items": int(values.shape[0]),
        "energy_rel_error_max": float(rel.max()),
        "energy_p_corr": _spearman(p.numpy(), norm_back.numpy()),
        "energy_clipped": int(clipped.sum()),
    }



def _quantiles(values, quantiles=(0, 1, 25, 50, 75, 99, 100)) -> dict:
    """Percentiles of a finite 1-D readout; a non-finite value is a hard error."""
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    if data.size == 0 or not bool(np.isfinite(data).all()):
        raise RuntimeError("a percentile readout needs finite values (got %d, finite=%s)"
                           % (data.size, bool(np.isfinite(data).all())))
    return {str(q): float(np.percentile(data, q)) for q in quantiles}


def _probe_ids(item_ids, count: int, p_all: torch.Tensor, name: str) -> torch.Tensor:
    """The item id of each probed row; the share is indexed by it, never by position.

    `item_ids=None` means the caller probed item ids `0 .. count-1` (the order a share
    table is built in), which is what the unit fixtures do. The trainer passes the real
    ids, because `train_item_ids[:probe]` is NOT `arange(probe)` once an item has no
    training interaction -- reading `p_all[:count]` there would pair every row with
    another item's `p_i`.
    """
    if count <= 0:
        raise RuntimeError("%s has no probed row to measure" % name)
    if item_ids is None:
        ids = torch.arange(count, dtype=torch.long)
    else:
        ids = torch.as_tensor(item_ids, dtype=torch.long).reshape(-1)
        if int(ids.numel()) != count:
            raise RuntimeError("%s needs one item id per probed row (%d rows, %d ids)"
                               % (name, count, int(ids.numel())))
    if int(ids.min()) < 0 or int(ids.max()) >= int(p_all.numel()):
        raise RuntimeError("%s got an item id outside the frozen share table of %d items "
                           "(ids %s)" % (name, int(p_all.numel()), ids[:8].tolist()))
    return ids


def codeword_report(table: dict, hidden: torch.Tensor, codebooks,
                    share: torch.Tensor | None = None, sample: int = 256,
                    item_ids: torch.Tensor | None = None) -> dict:
    """`metrics.codeword_scale_energy` + `metrics.clip_nan_count`, per level.

    Also reports the one log line `4_method` asks for and no formula defines:
    the relative error of the two codeword-side scaling factors' squared sum
    `sqrt(p)^2 + sqrt(1-p)^2` against 1.

    Everything is measured on the frozen codebook and the encoder output handed in:
    `clip_count` counts codeword points that `project` would pull back to
    `(1 - BALL_EPS) / sqrt(c)` (the saturation evidence `constants.c_hi.why`
    requires), `nan_count` counts non-finite chart points, and the power fit reads
    the exponents of the c -> 0 expansion terms (`||x||^2` ~ p^0, the cross term ~
    p^{1/2}, `||tau||^2` ~ p^1).
    """
    geometry = factor_share_geometry
    contract = geometry.contract()
    values = torch.as_tensor(hidden, dtype=torch.float64).detach()
    if values.dim() != 2 or int(values.shape[1]) != int(geometry.D_TOTAL):
        raise ValueError("hidden must be (B, %d), got shape %s"
                         % (geometry.D_TOTAL, tuple(values.shape)))
    levels = ([codebooks] if torch.is_tensor(codebooks) else list(codebooks))
    if not levels:
        raise ValueError("codeword_report needs at least one codebook")
    codes = [torch.as_tensor(item, dtype=torch.float64).detach() for item in levels]
    for code in codes:
        if code.dim() != 2 or int(code.shape[1]) != int(geometry.D_TOTAL):
            raise ValueError("a codebook must be (K, %d), got shape %s"
                             % (geometry.D_TOTAL, tuple(code.shape)))
    p_all = (frozen_share(table) if share is None
             else torch.as_tensor(share, dtype=torch.float64).reshape(-1))
    count = min(int(sample), int(values.shape[0]))
    # FAIL (step 2, I1): the row's own `p_i`, resolved by item id. `p_all[:count]`
    # paired row j with item j's share instead of item `item_ids[j]`'s.
    ids = _probe_ids(item_ids, count, p_all, "codeword_report")
    theta = geometry.share(p_all[ids].to(torch.float64), "struct").to(torch.float64)
    s_hi, s_lo = torch.sqrt(theta), torch.sqrt(1.0 - theta)
    # `4_method` log: s_hi^2 + s_lo^2 is 1 up to rounding; a non-zero value here would
    # mean the two scalings no longer complement each other.
    scale_rel = float(((s_hi.pow(2) + s_lo.pow(2)) - 1.0).abs().max())
    per_level = {}
    for level, code in enumerate(codes):
        tau_h1 = code[:, geometry.D_E:geometry.D_E + geometry.D_H1]
        tau_h2 = code[:, geometry.D_E + geometry.D_H1:]
        g_hi = s_hi.view(-1, 1, 1) * tau_h1.unsqueeze(0)
        g_lo = s_lo.view(-1, 1, 1) * tau_h2.unsqueeze(0)
        v_hi = contract.expmap0(g_hi.reshape(-1, geometry.D_H1), geometry.C_HI)
        v_lo = contract.expmap0(g_lo.reshape(-1, geometry.D_H2), geometry.C_LO)
        radius_hi = float(geometry.C_HI) ** -0.5
        radius_lo = float(geometry.C_LO) ** -0.5
        limit_hi = (1.0 - float(contract.BALL_EPS)) * radius_hi
        limit_lo = (1.0 - float(contract.BALL_EPS)) * radius_lo
        per_level["level%d" % level] = {
            "v_hi_norm_percentiles": _quantiles(v_hi.norm(dim=-1).flatten()),
            "v_lo_norm_percentiles": _quantiles(v_lo.norm(dim=-1).flatten()),
            "g_hi_over_radius_percentiles":
                _quantiles(g_hi.norm(dim=-1).flatten() / radius_hi),
            "g_lo_over_radius_percentiles":
                _quantiles(g_lo.norm(dim=-1).flatten() / radius_lo),
            "clip_count": int((v_hi.norm(dim=-1) >= limit_hi * (1.0 - 1e-9)).sum())
                          + int((v_lo.norm(dim=-1) >= limit_lo * (1.0 - 1e-9)).sum()),
            "nan_count": int((~torch.isfinite(v_hi)).sum())
                         + int((~torch.isfinite(v_lo)).sum()),
        }
    # c -> 0 expansion exponents, read on level 0's codeword Euclidean/hyperbolic blocks
    code = codes[0]
    rE, h1, _ = geometry.split_tangent(values[:count].to(torch.float64))
    tau_h1 = code[:, geometry.D_E:geometry.D_E + geometry.D_H1]
    picks = torch.arange(count) % int(code.shape[0])
    tau_pick = tau_h1[picks]
    cross = (-2.0 * (h1 * tau_pick).sum(-1)) * s_hi
    tau_term = tau_h1[picks].pow(2).sum(-1) * theta
    log_p = torch.log(theta).numpy()

    def _slope(term: torch.Tensor):
        keep = (term != 0) & torch.isfinite(term)
        y = torch.log(term[keep].abs()).numpy()
        x = log_p[keep.numpy()]
        if x.size < 3 or float(np.ptp(x)) == 0.0:
            return None
        return float(np.polyfit(x, y, 1)[0])

    slope_cross = _slope(cross)
    slope_tau = _slope(tau_term)
    residual = (None if slope_cross is None or slope_tau is None
                else max(abs(slope_cross - 0.5), abs(slope_tau - 1.0)))
    return {
        "items": count,
        "scale_sq_sum_rel_error_max": scale_rel,
        "scale_ratio_p_spearman": _spearman(theta.numpy(), (s_hi / s_lo).numpy()),
        "power_slope_cross": slope_cross,
        "power_slope_tau": slope_tau,
        "power_slope_residual": residual,
        "levels": per_level,
    }


def assignment_report(model, embeddings, item_ids, probe: int = 256) -> dict:
    """`metrics.argmin_flip_vs_behaviour_off` + `metrics.sid_diff_rate`.

    Same frozen weights, two forwards: the main arm and `behaviour_off`. Flips are
    counted per level (the argmin levels and the Sinkhorn level separately) and the
    SID difference is the rate of items whose whole token tuple moved. All zeros mean
    `u -> p` is not wired (`switches.behaviour_off`).
    """
    raw = model.module if isinstance(model, DDP) else model
    count = min(int(probe), int(embeddings.shape[0]))
    if count <= 0:
        raise ValueError("assignment_report needs at least one probed row")
    ids = torch.as_tensor(item_ids, dtype=torch.long).reshape(-1)[:count]
    batch = torch.as_tensor(embeddings)[:count]
    device = next(raw.parameters()).device
    was_training = raw.training
    previous = raw.share_mode
    # Single-rank diagnostics: the unused-code all_reduce would deadlock the other ranks.
    layers = list(raw.rq.vq_layers)
    saved_flags = [getattr(layer, "_skip_ddp_reduce", False) for layer in layers]
    for layer in layers:
        layer._skip_ddp_reduce = True
    raw.train()
    try:
        with torch.no_grad():
            torch.manual_seed(0)
            main_tokens = raw(batch.to(device), ids.to(device))[3].cpu()
            raw.share_mode = "behaviour_off"
            torch.manual_seed(0)
            off_tokens = raw(batch.to(device), ids.to(device))[3].cpu()
    finally:
        raw.share_mode = previous
        raw.train(was_training)
        for layer, flag in zip(layers, saved_flags):
            layer._skip_ddp_reduce = flag
    if main_tokens.shape != off_tokens.shape:
        raise RuntimeError("the two forwards returned different token shapes")
    flips = main_tokens != off_tokens
    names = [("level%d_sinkhorn" % level) if getattr(layer, "use_sk", False)
             else ("level%d_argmin" % level)
             for level, layer in enumerate(layers)]
    return {
        "items": count,
        "argmin_flip_vs_behaviour_off":
            {name: int(flips[:, level].sum()) for level, name in enumerate(names)},
        "argmin_flip_rate": float(flips.float().mean()),
        "sid_diff_rate": float(flips.any(dim=1).float().mean()),
    }


def geometry_readout(table: dict, hidden: torch.Tensor, codebook,
                     share: torch.Tensor | None = None, sample: int = 256,
                     item_ids: torch.Tensor | None = None) -> dict:
    """`metrics.factor_cost_vs_share`, `euclid_limit_rel_error`,
    `beta_effective_scale`, `batch_equivariance`, `dimension_control`.

    One read-only snapshot on frozen weights: nothing here reaches the loss, the
    gradient or the assignment.
    """
    geometry = factor_share_geometry
    contract = geometry.contract()
    code = torch.as_tensor(codebook, dtype=torch.float64).detach()
    values = torch.as_tensor(hidden, dtype=torch.float64).detach()
    if code.dim() != 2 or int(code.shape[1]) != int(geometry.D_TOTAL):
        raise ValueError("codebook must be (K, %d)" % (geometry.D_TOTAL,))
    if values.dim() != 2 or int(values.shape[1]) != int(geometry.D_TOTAL):
        raise ValueError("hidden must be (B, %d)" % (geometry.D_TOTAL,))
    p_all = (frozen_share(table) if share is None
             else torch.as_tensor(share, dtype=torch.float64).reshape(-1))
    count = min(int(sample), int(values.shape[0]))
    # FAIL (step 2, I2): same indexing fix as `codeword_report` -- the probed row keeps
    # its own `p_i`, so `factor_cost_vs_share`, `euclid_limit_rel_error` and the
    # `beta_effective_scale` read the frozen p the quantiser actually received.
    ids = _probe_ids(item_ids, count, p_all, "geometry_readout")
    theta = geometry.share(p_all[ids].to(torch.float64), "struct").to(torch.float64)
    latent = values[:count]
    D_main = geometry.product_cost(latent, code, theta, mode="both_on")
    index = torch.argmin(D_main, dim=-1)
    rE, _, _ = geometry.split_tangent(latent)
    x_hi, x_lo = geometry.item_chart_head(latent)
    v_hi, v_lo = geometry.codeword_chart(code, theta)
    hyperbolic = (contract.squared_distance(x_hi.unsqueeze(1), v_hi, geometry.C_HI)
                  + contract.squared_distance(x_lo.unsqueeze(1), v_lo, geometry.C_LO))
    row_hyp = hyperbolic.gather(1, index.view(-1, 1)).squeeze(1)
    # c -> 0 limit (scaling KEPT) against the baseline expansion (scaling dropped)
    saved = (geometry.C_HI, geometry.C_LO)
    try:
        geometry.C_HI, geometry.C_LO = 0.0, 0.0
        limit = geometry.product_cost(latent, code, theta, mode="both_on")
    finally:
        geometry.C_HI, geometry.C_LO = saved
    baseline = geometry.euclidean_expansion(latent, code)
    euclid_rel = float(((limit - baseline).abs() / baseline.abs().clamp_min(1e-12)).max())
    # beta effective scale: the geometric parts against the baseline MSE magnitude
    parts = geometry.quant_parts(rE, x_hi, x_lo, v_hi, v_lo, code[:, :geometry.D_E], index)
    baseline_mse = float((latent - code[index]).pow(2).mean())
    denominator = max(baseline_mse, 1e-12)
    # batch equivariance: a permutation of the batch permutes D and p, nothing else
    order = torch.arange(count - 1, -1, -1)
    D_perm = geometry.product_cost(latent[order], code, theta[order], mode="both_on")
    # dimension control: swap the two hyperbolic blocks with p pinned at SHARE_REF
    reference = torch.full_like(theta, geometry.SHARE_REF)
    D_ref = geometry.product_cost(latent, code, reference, mode="both_on")
    swap = torch.cat([rE, latent[:, geometry.D_E + geometry.D_H1:],
                      latent[:, geometry.D_E:geometry.D_E + geometry.D_H1]], dim=-1)
    swap_code = torch.cat([code[:, :geometry.D_E], code[:, geometry.D_E + geometry.D_H1:],
                           code[:, geometry.D_E:geometry.D_E + geometry.D_H1]], dim=-1)
    D_swap = geometry.product_cost(swap, swap_code, reference, mode="both_on")
    return {
        "items": count,
        "factor_cost_vs_share_spearman": _spearman(theta.numpy(), row_hyp.numpy()),
        "euclid_limit_rel_error_max": euclid_rel,
        "beta_effective_scale": {
            "commitment_over_baseline_mse": float(parts[0]) / denominator,
            "codebook_over_baseline_mse": float(parts[1]) / denominator,
        },
        "batch_equivariance_max_abs_diff": float((D_perm - D_main[order]).abs().max()),
        "dimension_control_rel_change": float(
            ((D_swap - D_ref).abs() / D_ref.abs().clamp_min(1e-12)).mean()),
    }


# === 超参来自 config/train_config.json（严格校验：缺键/多键/类型/范围）===
# 一个 JSON 文件、严格校验：缺键/多键/类型错/越界都在碰 GPU 之前 ConfigError。
# 路径与 ARM_NAME 不在配置里 —— 它们归框架 (CLAUDE.md 19b0/67)。
CONFIG_FILE = Path(__file__).resolve().with_name("config") / "train_config.json"
CONFIG = load_train_config(CONFIG_FILE)
# 损失与距离都由注册表给出；训练文件只负责导入并把它们接到模型上。
CRITERION = build_loss(CONFIG)
# 机制距离从注册表按名字取；默认仍是 "euclidean"，所以这一行是本轮唯一的选用点。
DISTANCE = build_distance(CONFIG, name=DISTANCE_NAME)

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

    # === AF3 全表势：开训前只读 + 覆盖训练集校验（缺一份就直接 RuntimeError）===
    potential_table = load_frozen_table()
    # p_i = u_i, computed ONCE through factor_share_geometry.share and frozen; the
    # model below only indexes this (N,) tensor with the batch's item ids.
    share_vector = frozen_share(potential_table)
    # Coverage of the training ids: a missing id is a hard error, never a default.
    shares_for(train_ids, potential_table)
    potential_readings = potential_report(potential_table, share_vector)
    if potential_readings["vq_type"] != "vq":
        raise RuntimeError("the factor_share arm requires vq_type='vq', got %r"
                           % (potential_readings["vq_type"],))
    if np.asarray(potential_table["item_ids"]).size <= int(train_ids.max()):
        raise RuntimeError("the frozen AF3 table does not cover the embedding id space "
                           "(table %d items, max train id %d)"
                           % (int(np.asarray(potential_table["item_ids"]).size),
                              int(train_ids.max())))

    all_embeddings = torch.from_numpy(embeddings)
    all_item_ids = torch.arange(len(all_embeddings), dtype=torch.long)
    train_embeddings = all_embeddings[torch.from_numpy(train_ids)]
    train_item_ids = torch.from_numpy(train_ids.astype(np.int64))
    config = _tokenizer_config()
    if rank == 0:
        print("[AF3 potential] %s digest=%s items=%d missing=%d u=[%.6f, %.6f] "
              "p_miss=%.6f |p_miss-SHARE_REF|=%.6f p_u_spearman=%s p_potential_spearman=%s"
              % (POTENTIAL_FILE.name, potential_table["digest"],
                 potential_readings["items"], potential_readings["missing"],
                 potential_readings["u_min"], potential_readings["u_max"],
                 potential_readings["p_miss"],
                 potential_readings["p_miss_minus_share_ref"],
                 potential_readings["p_u_spearman"],
                 potential_readings["p_potential_spearman"]), flush=True)
        print("[AF3 potential] p percentiles=%s share_mode=%s distance=%s vq_type=%s"
              % (potential_readings["p_percentiles"], SHARE_MODE, DISTANCE_NAME,
                 potential_readings["vq_type"]), flush=True)
        _record(rank, "af3_potential", path=str(POTENTIAL_FILE),
                digest=potential_table["digest"], share_mode=SHARE_MODE,
                distance=DISTANCE_NAME, **potential_readings)
    model = RQVAE(config, in_dim=embeddings.shape[1], distance=DISTANCE,
                  shares=share_vector, share_mode=SHARE_MODE).to(device)
    if XAVIER_INIT:
        initialize_tiger_weights(model)
    # === 商品侧模长与 p 无关的实测读数（math.json energy_conservation_rel_error）===
    # 只读、只在 rank 0 算一次：对同一批编码器输出验证 chart 头不带 p，且该读数
    # 不是恒等式（越出球的点会在这里显形，见 energy_clipped）。
    if rank == 0:
        probe = min(256, int(train_embeddings.shape[0]))
        probe_ids = torch.arange(probe, dtype=torch.long)
        with torch.no_grad():
            probe_hidden = model.encoder(train_embeddings[probe_ids].to(device)).cpu()
        energy_readings = energy_conservation_report(
            potential_table, probe_hidden, train_item_ids[probe_ids])
        print("[AF3 energy] product-side rel_error_max=%.3e p_corr=%s clipped=%d/%d"
              % (energy_readings["energy_rel_error_max"], energy_readings["energy_p_corr"],
                 energy_readings["energy_clipped"], energy_readings["items"]), flush=True)
        _record(rank, "energy_conservation", **energy_readings)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    train_dataset = TensorDataset(train_embeddings, train_item_ids)
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
        raw_module.init_codebook(train_embeddings.to(device), train_item_ids.to(device))
    if world_size > 1:
        dist.barrier()

    # === 只读接线读数（math.json `metrics`）: 同一份冻结权重、不重训 ===
    # codeword_scale_energy / clip_nan_count / 两放缩因子平方和相对误差（4_method 日志项）
    # + argmin_flip_vs_behaviour_off / sid_diff_rate + 其余只读统计。
    # 全部只在 rank 0 、只在初始化后的快照上算一次；不进损失、不进梯度。
    if rank == 0:
        probe = min(256, int(train_embeddings.shape[0]))
        probe_ids = torch.arange(probe, dtype=torch.long)
        probe_embeddings = train_embeddings[probe_ids]
        probe_item_ids = train_item_ids[probe_ids]
        probe_codebooks = [layer.get_code_embs().detach().cpu()
                           for layer in raw_module.rq.vq_layers]
        with torch.no_grad():
            probe_hidden = raw_module.encoder(probe_embeddings.to(device)).cpu()
        codeword_readings = codeword_report(potential_table, probe_hidden,
                                           probe_codebooks, share_vector,
                                           item_ids=probe_item_ids)
        print("[AF3 codeword] scale_sq_sum_rel_error=%.3e clip=%d nan=%d "
              "power_slope=(cross=%s, tau=%s, residual=%s)"
              % (codeword_readings["scale_sq_sum_rel_error_max"],
                 sum(level["clip_count"] for level in codeword_readings["levels"].values()),
                 sum(level["nan_count"] for level in codeword_readings["levels"].values()),
                 codeword_readings["power_slope_cross"],
                 codeword_readings["power_slope_tau"],
                 codeword_readings["power_slope_residual"]), flush=True)
        _record(rank, "codeword_scale", **codeword_readings)
        assignment_readings = assignment_report(raw_module, probe_embeddings, probe_item_ids)
        print("[AF3 assignment] flips=%s sid_diff_rate=%.6f"
              % (assignment_readings["argmin_flip_vs_behaviour_off"],
                 assignment_readings["sid_diff_rate"]), flush=True)
        _record(rank, "assignment_wiring", **assignment_readings)
        geometry_readings = geometry_readout(potential_table, probe_hidden,
                                             probe_codebooks[-1], share_vector,
                                             item_ids=probe_item_ids)
        print("[AF3 metrics] %s" % (geometry_readings,), flush=True)
        _record(rank, "geometry_readout", **geometry_readings)

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
        for batch, item_ids in loader:
            batch = batch.to(device, non_blocking=True)
            item_ids = item_ids.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            reconstructed, quant_parts, _, _ = model(batch, item_ids)
            # 唯一接入点：损失项与权重在 losses/ 里，这里只拿到总损失与分项。
            loss, components = CRITERION(reconstructed, batch, quant_parts)
            recon_loss = components["reconstruction"]
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
                        all_embeddings.to(device), all_item_ids.to(device),
                        infer_use_sk=False,
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
                    all_embeddings.to(device), all_item_ids.to(device),
                    infer_use_sk=False,
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
