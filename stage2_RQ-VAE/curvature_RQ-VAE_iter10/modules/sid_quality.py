"""SID 质量评估与早停门控.

在 stage2 RQ-VAE 训练循环的每个 ckpt 落盘点, 把当前模型对整个 item_emb.npy
推理得到 sids.npy, 再用与 baseline 同样的 4 项指标 (HitRate@K=50 / 3-token
SID Gini / 每层 Gini) 与 baseline 对比, 决定是否立即终止训练.

R2 (CLAUDE.md): baseline = TIGER_RQ-VAE/sids_for_hgrec_recbole.npy
  - HitRate@K=50 = 0.7165
  - Full SID Gini = 0.0672
  - Layer 0 Gini = 0.3598, Layer 1 = 0.2467, Layer 2 = 0.1656
  - L0-L1 unique pairs = 13340 (baseline 前三列)
  - H(L1|L0) = 5.6115581472 bit (baseline 前三列)

早停门控 (硬阈值, 不允许 fallback):
  - HitRate@K=50 < baseline * 0.5       → 立即终止 (语义已崩溃)
  - 任一层 Gini > 0.90                  → 立即终止 (单层 collapse)
  - Full SID Gini > 0.50                → 立即终止 (整体坍缩)
  - L0-L1 unique pairs < baseline * 0.5  → 立即终止 (跨层组合 collapse)
  - H(L1|L0) < baseline * 0.5            → 立即终止 (L1 被 L0 决定)

任何指标缺测 (NaN/Inf/为空) 直接 raise, 不允许 fallback.
"""
from __future__ import annotations

import json
import subprocess
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict

import numpy as np
import torch


# === Baseline (R2: TIGER_RQ-VAE/sids_for_hgrec_recbole.npy, 2026-09-15) ===
BASELINE_HITRATE_K50 = 0.7165
BASELINE_FULL_GINI = 0.0672
BASELINE_PER_LAYER_GINI = [0.3598, 0.2467, 0.1656]
BASELINE_L01_UNIQUE_PAIRS = 13_340
BASELINE_H_L1_GIVEN_L0 = 5.611558147195798

# === 早停门控 (硬阈值, 无 fallback) ===
EARLY_STOP_HITRATE_MIN_RATIO = 0.5   # HitRate < baseline * 0.5 → 终止
EARLY_STOP_LAYER_GINI_MAX = 0.90     # 任一层 Gini > 0.90 → 终止 (layer collapse)
EARLY_STOP_FULL_GINI_MAX = 0.50      # Full SID Gini > 0.50 → 终止 (整体坍缩)
EARLY_STOP_JOINT_MIN_RATIO = 0.5     # 联合结构指标低于 baseline * ratio → 终止


@dataclass
class SidMetrics:
    hitrate_k50: float
    full_gini: float
    per_layer_gini: list[float]
    n_unique_full: int
    n_items: int
    l01_unique_pairs: int
    h_l1_given_l0: float

    def to_dict(self) -> Dict:
        return asdict(self)


def build_sids_for_corpus(model, item_emb_path: str, device: torch.device) -> np.ndarray:
    """在 eval 模式下对整个 item_emb.npy 推理, 产出 (N, 3) int32 sids."""
    if model is None:
        raise ValueError("build_sids_for_corpus: model 不能为空")
    arr = np.load(item_emb_path).astype(np.float32)
    n_items, input_dim = arr.shape
    inner = model.module if hasattr(model, "module") else model
    expected = inner.input_dim
    if input_dim != expected:
        raise ValueError(
            f"build_sids_for_corpus: item_emb dim={input_dim}, 模型期望 {expected}"
        )

    was_training = inner.training
    inner.eval()
    sids_chunks: list[np.ndarray] = []
    chunk_size = 1024
    try:
        with torch.no_grad():
            for start in range(0, n_items, chunk_size):
                end = min(start + chunk_size, n_items)
                chunk = torch.from_numpy(arr[start:end]).to(device)
                quantized = inner.get_semantic_ids(chunk)
                ids = quantized.sem_ids.detach().to("cpu", torch.int32).numpy()
                sids_chunks.append(ids)
    finally:
        if was_training:
            inner.train()

    sids = np.concatenate(sids_chunks, axis=0)
    if sids.shape != (n_items, 3):
        raise RuntimeError(
            f"build_sids_for_corpus: 产出 SID 形状 {sids.shape}, 期望 ({n_items}, 3)"
        )
    return sids


def _gini_from_counts(counts: np.ndarray) -> float:
    if counts.size == 0:
        raise ValueError("_gini_from_counts: counts 为空")
    if not np.isfinite(counts).all():
        raise ValueError(f"_gini_from_counts: counts 含 NaN/Inf, {counts}")
    s = np.sort(counts.astype(np.float64))
    n = s.size
    total = s.sum()
    if total <= 0:
        raise ValueError("_gini_from_counts: 总和为 0, 无法计算 Gini")
    index = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(index * s) - (n + 1) * total) / (n * total))


def evaluate_sid_quality(sids: np.ndarray) -> SidMetrics:
    """直接用 numpy 计算 4 项指标, 不走 /tmp/sid_metrics_any.py (避免依赖 parquet).

    与 /tmp/sid_metrics_any.py 的 baseline 计算口径一致 (HitRate@K=50 用
    embedding 余弦 + items 历史共现邻居, 不在训练循环里, 此处只计算 SID 自身
    的 Gini / 唯一数; HitRate 用 placeholder 占位, 由调用方比较时跳过).
    """
    if not isinstance(sids, np.ndarray):
        raise ValueError("evaluate_sid_quality: sids 不是 ndarray")
    if sids.ndim != 2 or sids.shape[1] != 3:
        raise ValueError(f"evaluate_sid_quality: 期望 (N, 3), 实际 {sids.shape}")
    if not np.isfinite(sids).all():
        raise ValueError("evaluate_sid_quality: sids 含 NaN/Inf")

    n_items = sids.shape[0]
    if n_items < 10:
        raise ValueError(f"evaluate_sid_quality: N={n_items} 太少")

    tuples = [tuple(row) for row in sids.tolist()]
    full_counts = np.array(list(Counter(tuples).values()), dtype=np.float64)
    full_gini = _gini_from_counts(full_counts)

    per_layer = []
    for layer_index in range(3):
        layer_counts = np.array(
            list(Counter(sids[:, layer_index].tolist()).values()), dtype=np.float64
        )
        per_layer.append(_gini_from_counts(layer_counts))

    l0_counts = Counter(sids[:, 0].tolist())
    l01_counts = Counter(zip(sids[:, 0].tolist(), sids[:, 1].tolist()))
    h_l1_given_l0 = -sum(
        (count / n_items) * np.log2(count / l0_counts[l0])
        for (l0, _l1), count in l01_counts.items()
    )
    if not np.isfinite(h_l1_given_l0) or h_l1_given_l0 < 0:
        raise ValueError("evaluate_sid_quality: H(L1|L0) 无效")

    return SidMetrics(
        hitrate_k50=float("nan"),  # 此函数不计算 HitRate, 由调用方外部算
        full_gini=full_gini,
        per_layer_gini=per_layer,
        n_unique_full=int(full_counts.size),
        n_items=int(n_items),
        l01_unique_pairs=int(len(l01_counts)),
        h_l1_given_l0=float(h_l1_given_l0),
    )


def evaluate_sid_quality_full(sids: np.ndarray, item_emb_path: str) -> SidMetrics:
    """调 /tmp/sid_metrics_any.py (subprocess) 拿 HitRate@K=50 + Gini 全套.

    走 subprocess 而非 import 是为了复用既有 baseline 口径, 避免重写 HitRate.
    """
    metrics = evaluate_sid_quality(sids)
    label = f"step{sids.shape[0]}"
    tmp_sid_path = Path("/tmp") / f"_sid_quality_{label}.npy"
    np.save(tmp_sid_path, sids)
    cmd = [
        "/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9",
        "/tmp/sid_metrics_any.py",
        str(tmp_sid_path),
        label,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(
            f"evaluate_sid_quality_full: /tmp/sid_metrics_any.py 退出码 "
            f"{proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    parsed = _parse_sid_metrics_stdout(proc.stdout)
    metrics.hitrate_k50 = parsed["hitrate_k50"]
    return metrics


def _parse_sid_metrics_stdout(stdout: str) -> Dict[str, float]:
    """从 /tmp/sid_metrics_any.py 的 print 中解析 HitRate@K=50 和 Gini."""
    hitrate = None
    full_gini = None
    layer_gini = []
    for line in stdout.splitlines():
        if "HitRate @ K=50:" in line:
            try:
                hitrate = float(line.rsplit(":", 1)[1].strip().split()[0])
            except (ValueError, IndexError) as exc:
                raise ValueError(
                    f"_parse_sid_metrics_stdout: 无法解析 HitRate 行: {line!r}"
                ) from exc
        if "Gini (full SID occupancy):" in line:
            try:
                full_gini = float(line.rsplit(":", 1)[1].strip())
            except (ValueError, IndexError) as exc:
                raise ValueError(
                    f"_parse_sid_metrics_stdout: 无法解析 Full Gini 行: {line!r}"
                ) from exc
        if "layer " in line and "Gini=" in line:
            try:
                value = float(line.rsplit("Gini=", 1)[1].strip())
                layer_gini.append(value)
            except (ValueError, IndexError) as exc:
                raise ValueError(
                    f"_parse_sid_metrics_stdout: 无法解析 layer Gini 行: {line!r}"
                ) from exc
    if hitrate is None:
        raise ValueError("_parse_sid_metrics_stdout: 找不到 HitRate@K=50 行")
    if full_gini is None:
        raise ValueError("_parse_sid_metrics_stdout: 找不到 Full Gini 行")
    if len(layer_gini) != 3:
        raise ValueError(
            f"_parse_sid_metrics_stdout: layer Gini 数 {len(layer_gini)} ≠ 3"
        )
    return {"hitrate_k50": hitrate, "full_gini": full_gini, "per_layer_gini": layer_gini}


def should_early_stop(metrics: SidMetrics) -> tuple[bool, str]:
    """返回 (是否终止, 原因)."""
    if not isinstance(metrics, SidMetrics):
        raise ValueError("should_early_stop: metrics 类型无效")
    if not np.isfinite(metrics.full_gini):
        raise ValueError("should_early_stop: full_gini 必须为有限数")
    if len(metrics.per_layer_gini) != 3 or not np.isfinite(
        np.asarray(metrics.per_layer_gini, dtype=np.float64)
    ).all():
        raise ValueError("should_early_stop: per_layer_gini 必须包含 3 个有限数")
    if metrics.n_items < 1 or metrics.n_unique_full < 1:
        raise ValueError("should_early_stop: n_items 和 n_unique_full 必须为正数")
    if metrics.l01_unique_pairs < 1 or not np.isfinite(metrics.h_l1_given_l0):
        raise ValueError("should_early_stop: 联合 SID 指标必须有效")
    if np.isinf(metrics.hitrate_k50):
        raise ValueError("should_early_stop: hitrate_k50 不得为 Inf")

    reasons = []
    if metrics.l01_unique_pairs < (
        BASELINE_L01_UNIQUE_PAIRS * EARLY_STOP_JOINT_MIN_RATIO
    ):
        reasons.append(
            f"l01_unique_pairs={metrics.l01_unique_pairs} < "
            f"baseline*ratio={BASELINE_L01_UNIQUE_PAIRS * EARLY_STOP_JOINT_MIN_RATIO:.0f}"
        )
    if metrics.h_l1_given_l0 < (
        BASELINE_H_L1_GIVEN_L0 * EARLY_STOP_JOINT_MIN_RATIO
    ):
        reasons.append(
            f"H(L1|L0)={metrics.h_l1_given_l0:.4f} < "
            f"baseline*ratio={BASELINE_H_L1_GIVEN_L0 * EARLY_STOP_JOINT_MIN_RATIO:.4f}"
        )

    if not np.isfinite(metrics.hitrate_k50):
        # HitRate 不可用时仍检查 Full/Layer Gini 以及跨层联合结构。
        if metrics.full_gini > EARLY_STOP_FULL_GINI_MAX:
            reasons.append(
                f"full_gini={metrics.full_gini:.4f} > {EARLY_STOP_FULL_GINI_MAX}"
            )
        for layer_index, value in enumerate(metrics.per_layer_gini):
            if value > EARLY_STOP_LAYER_GINI_MAX:
                reasons.append(
                    f"layer{layer_index}_gini={value:.4f} > {EARLY_STOP_LAYER_GINI_MAX}"
                )
    elif metrics.hitrate_k50 < BASELINE_HITRATE_K50 * EARLY_STOP_HITRATE_MIN_RATIO:
        reasons.append(
            f"hitrate_k50={metrics.hitrate_k50:.4f} < "
            f"baseline*ratio={BASELINE_HITRATE_K50 * EARLY_STOP_HITRATE_MIN_RATIO:.4f}"
        )
    elif metrics.full_gini > EARLY_STOP_FULL_GINI_MAX:
        reasons.append(
            f"full_gini={metrics.full_gini:.4f} > {EARLY_STOP_FULL_GINI_MAX}"
        )
    else:
        for layer_index, value in enumerate(metrics.per_layer_gini):
            if value > EARLY_STOP_LAYER_GINI_MAX:
                reasons.append(
                    f"layer{layer_index}_gini={value:.4f} > {EARLY_STOP_LAYER_GINI_MAX}"
                )
                break

    return bool(reasons), "; ".join(reasons)


def format_metrics(metrics: SidMetrics) -> str:
    layers = "/".join(f"{v:.4f}" for v in metrics.per_layer_gini)
    return (
        f"hitrate@50={metrics.hitrate_k50:.4f} "
        f"full_gini={metrics.full_gini:.4f} "
        f"per_layer=[{layers}] "
        f"unique={metrics.n_unique_full}/{metrics.n_items} "
        f"l01_pairs={metrics.l01_unique_pairs} "
        f"H_l1_given_l0={metrics.h_l1_given_l0:.4f}"
    )


def write_quality_report(
    out_path: str,
    step: int,
    metrics: SidMetrics,
    early_stopped: bool,
    reason: str,
) -> None:
    payload = {
        "step": step,
        "metrics": metrics.to_dict(),
        "early_stopped": early_stopped,
        "reason": reason,
        "baseline": {
            "hitrate_k50": BASELINE_HITRATE_K50,
            "full_gini": BASELINE_FULL_GINI,
            "per_layer_gini": BASELINE_PER_LAYER_GINI,
            "l01_unique_pairs": BASELINE_L01_UNIQUE_PAIRS,
            "h_l1_given_l0": BASELINE_H_L1_GIVEN_L0,
        },
    }
    Path(out_path).write_text(json.dumps(payload, indent=2))
