"""将 curvature RQ-VAE 的 3-token SID 导出为 stage3 输入 (TIGER collision extension 对齐版).

与 stage2_RQ-VAE/TIGER_RQ-VAE/train_rqvae.py 的 _extend_collisions 保持同一算法
(match RecBole3.0's extend path with one fixed extra SID level):
  - 按 3-token 分组, collision 组内第 occurrence 个 item 分配 extension token
    = offsets[-1] + occurrence, 其中 offsets = cumsum([0, *codebook_sizes]);
  - 无冲突 item 的 extension token 恒为 offsets[-1] (= 768, 与 TIGER baseline 一致);
  - 4-token 产物必须全表唯一, 否则直接 raise (不允许 fallback).

产物:
  1. curvature_config.SIDS_NPY  (N, 4) int64  — stage2 正式 stage3 输入
  2. stage3 dataset item_sids_recbole.json  — {str(item_id): [t0, t1, t2, ext]}

启动 (无 CLI 参数, 路径全部硬编码):
  python3 scripts/export_sids_for_stage3.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
STAGE2_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(STAGE2_DIR))

from curvature_config import SIDS_NPY  # noqa: E402

# === 硬编码路径与常量 (项目规则: 禁止 CLI 参数) ===
SOURCE_SIDS_NPY = Path(
    "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE/"
    "out/rqvae/instruments/sids_final.npy"
)
STAGE3_JSON = Path(
    "/fs04/ar57/wenyu/GeneRec/stage3_T5Train/dataset/Amazon_2023_Instruments/"
    "item_sids_recbole.json"
)
CODEBOOK_SIZES = [256, 256, 256]  # N_LAYERS=3, CODEBOOK_SIZE=256
N_ITEMS = 24587


def _extend_collisions(tokens: np.ndarray, codebook_sizes: list[int]) -> np.ndarray:
    """对齐 TIGER _extend_collisions: 追加一个固定 extension level 消解 collision."""
    if tokens.ndim != 2 or tokens.shape[1] != len(codebook_sizes):
        raise ValueError(
            f"Token shape {tokens.shape} 与 codebook sizes {codebook_sizes} 不一致"
        )
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


def main() -> None:
    if not SOURCE_SIDS_NPY.is_file():
        raise FileNotFoundError(f"源 SID 不存在: {SOURCE_SIDS_NPY}")
    raw = np.load(SOURCE_SIDS_NPY, allow_pickle=False)
    if raw.shape != (N_ITEMS, len(CODEBOOK_SIZES)):
        raise ValueError(
            f"源 SID 形状 {raw.shape} 与预期 ({N_ITEMS}, {len(CODEBOOK_SIZES)}) 不一致"
        )
    if not np.isfinite(raw).all() or raw.min() < 0:
        raise ValueError("源 SID 含非法值 (NaN/负数)")

    tokens = raw.astype(np.int64)
    n_unique_3t = len(np.unique(tokens, axis=0))
    sid = _extend_collisions(tokens, CODEBOOK_SIZES)

    # 硬校验: 4-token 必须全表唯一 (与 TIGER 导出一致, 失败即 raise)
    if len(np.unique(sid, axis=0)) != len(sid):
        raise RuntimeError("collision extension 未能使 item codes 唯一")
    offsets_last = int(np.cumsum([0, *CODEBOOK_SIZES])[-1])
    n_colliding = len(sid) - n_unique_3t
    print(
        f"[export] 3-token unique={n_unique_3t}/{len(sid)} "
        f"(collision rows={n_colliding}) -> extension 值域 "
        f"[{sid[:, -1].min()}, {sid[:, -1].max()}] (default={offsets_last})"
    )

    # 产物 1: (N, 4) int64 npy
    sids_npy_path = Path(SIDS_NPY)
    sids_npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(sids_npy_path, sid)
    print(f"[export] wrote {sids_npy_path}: shape={sid.shape}")

    # 产物 2: stage3 RecBole json (key 0..N-1, 每行 4 个 int)
    payload = {
        str(item_id): [int(value) for value in row] for item_id, row in enumerate(sid)
    }
    STAGE3_JSON.parent.mkdir(parents=True, exist_ok=True)
    STAGE3_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[export] wrote {STAGE3_JSON}: n_items={len(payload)}")

    # 回读校验
    back = json.loads(STAGE3_JSON.read_text(encoding="utf-8"))
    keys = sorted(int(key) for key in back)
    if keys != list(range(len(sid))):
        raise ValueError("stage3 json key 不是 0..N-1")
    widths = {len(row) for row in back.values()}
    if widths != {len(CODEBOOK_SIZES) + 1}:
        raise ValueError(f"stage3 json 行宽异常: {widths}")
    if any([int(v) for v in sid[i]] != back[str(i)] for i in (0, 1, len(sid) - 1)):
        raise ValueError("stage3 json 与 npy 内容不一致")
    reload = np.load(sids_npy_path, allow_pickle=False)
    if not np.array_equal(reload, sid):
        raise ValueError("SIDS_NPY 回读不一致")
    print("[export] SID_WIRING_PASS: 4-token extension npy + stage3 json 校验通过")


if __name__ == "__main__":
    main()
