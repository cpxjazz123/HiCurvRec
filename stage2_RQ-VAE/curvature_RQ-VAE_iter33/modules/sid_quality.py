"""In-process SID-derived descriptive metrics; never a training or selection gate.

The checkpoint reporting path computes occupancy, diversity, and conditional
entropy statistics directly from three-token SIDs. These metrics describe the
exported codes only; Stage2 always runs to its configured step count unless the
training or SID export itself is invalid.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict

import numpy as np
import torch


# === SID-derived descriptive baselines; none is a gate ===
# Retained only for reports that compare descriptive occupancy statistics.
BASELINE_FULL_GINI = 0.0672
BASELINE_PER_LAYER_GINI = [0.3598, 0.2467, 0.1656]
BASELINE_L01_UNIQUE_PAIRS = 13_340
BASELINE_H_L1_GIVEN_L0 = 5.611558147195798

# SID-derived metrics are descriptive only. No outer or inner gate is defined.


@dataclass
class SidMetrics:
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
    # 2026-09-24: baseline curvature_RQ-VAE 的 VQLayer.forward 没调 distributed.all_reduce,
    # 因此 build_sids_for_corpus 在 rank 0 单跑时无需 _skip_ddp_reduce wrap.
    # 但 TIGER_RQ-VAE (有 attention) 需要此 wrap; 此 helper 在 modules/sid_quality.py 共享时两边都 safe.
    vq_layers = getattr(inner, "vq_layers", getattr(inner, "layers", []))
    skip_attr = "_skip_ddp_reduce"
    saved = [getattr(layer, skip_attr, False) for layer in vq_layers]
    for layer in vq_layers:
        setattr(layer, skip_attr, True)
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
        for layer, prev in zip(vq_layers, saved):
            setattr(layer, skip_attr, prev)
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
    """Compute SID-derived occupancy, diversity, and conditional entropy."""
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
        full_gini=full_gini,
        per_layer_gini=per_layer,
        n_unique_full=int(full_counts.size),
        n_items=int(n_items),
        l01_unique_pairs=int(len(l01_counts)),
        h_l1_given_l0=float(h_l1_given_l0),
    )


def should_early_stop(metrics: SidMetrics) -> tuple[bool, str]:
    return (False, "")


def format_metrics(metrics: SidMetrics) -> str:
    layers = "/".join(f"{v:.4f}" for v in metrics.per_layer_gini)
    return (
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
            "full_gini": BASELINE_FULL_GINI,
            "per_layer_gini": BASELINE_PER_LAYER_GINI,
            "l01_unique_pairs": BASELINE_L01_UNIQUE_PAIRS,
            "h_l1_given_l0": BASELINE_H_L1_GIVEN_L0,
        },
    }
    Path(out_path).write_text(json.dumps(payload, indent=2))
