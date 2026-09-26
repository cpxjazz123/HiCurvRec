"""将 curvature RQ-VAE 的 3-token SID 导出为 TIGER-compatible 4-token 输入.

扩展算法与 stage2_RQ-VAE/TIGER_RQ-VAE/train_rqvae.py 保持一致:
  - collision 组内第 occurrence 个 item 的 extension token = 768 + occurrence;
  - 无冲突 item 的 extension token 恒为 768;
  - 4-token SID 必须全表唯一, 否则直接失败.

产物:
  1. curvature_config.SIDS_NPY: (N, 4) int64;
  2. curvature_config.ITEM_SIDS_JSON: TIGER-compatible stage2 SID mapping.

启动 (无 CLI 参数, 路径全部硬编码):
  python3 scripts/export_sids_for_stage3.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
STAGE2_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(STAGE2_DIR))

from curvature_config import (  # noqa: E402
    ITEM_SIDS_JSON,
    RAW_SIDS_NPY,
    SIDS_NPY,
)

# === 硬编码路径与常量 (项目规则: 禁止 CLI 参数) ===
SOURCE_SIDS_NPY = Path(RAW_SIDS_NPY)
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
        raise FileNotFoundError(f"源 3-token SID 不存在: {SOURCE_SIDS_NPY}")
    raw = np.load(SOURCE_SIDS_NPY, allow_pickle=False)
    if raw.shape != (N_ITEMS, len(CODEBOOK_SIZES)):
        raise ValueError(
            f"源 SID 形状 {raw.shape} 与预期 "
            f"({N_ITEMS}, {len(CODEBOOK_SIZES)}) 不一致"
        )
    if not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(f"源 SID 必须为整数数组, 实际 dtype={raw.dtype}")

    tokens = raw.astype(np.int64, copy=False)
    codebook_limits = np.asarray(CODEBOOK_SIZES, dtype=np.int64)
    if np.any(tokens < 0) or np.any(tokens >= codebook_limits):
        raise ValueError("源 SID token 超出对应 codebook 的取值范围")

    source_path = SOURCE_SIDS_NPY.resolve()
    sids_npy_path = Path(SIDS_NPY)
    if source_path == sids_npy_path.resolve():
        raise ValueError("RAW_SIDS_NPY 与 SIDS_NPY 必须分离以保证导出可重复")

    n_unique_3t = len(np.unique(tokens, axis=0))
    sid = _extend_collisions(tokens, CODEBOOK_SIZES)
    if len(np.unique(sid, axis=0)) != len(sid):
        raise RuntimeError("collision extension 未能使 item codes 唯一")
    offsets_last = int(np.cumsum([0, *CODEBOOK_SIZES])[-1])
    n_colliding = len(sid) - n_unique_3t
    print(
        f"[export] 3-token unique={n_unique_3t}/{len(sid)} "
        f"(collision rows={n_colliding}) -> extension 值域 "
        f"[{sid[:, -1].min()}, {sid[:, -1].max()}] (default={offsets_last})"
    )

    sids_npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(sids_npy_path, sid)
    print(f"[export] wrote {sids_npy_path}: shape={sid.shape}")

    payload = {
        str(item_id): [int(value) for value in row] for item_id, row in enumerate(sid)
    }
    json_path = Path(ITEM_SIDS_JSON)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    back = json.loads(json_path.read_text(encoding="utf-8"))
    keys = sorted(int(key) for key in back)
    if keys != list(range(len(sid))):
        raise ValueError(f"{json_path}: JSON key 不是 0..N-1")
    widths = {len(row) for row in back.values()}
    if widths != {len(CODEBOOK_SIZES) + 1}:
        raise ValueError(f"{json_path}: JSON 行宽异常: {widths}")
    if any(
        [int(value) for value in sid[i]] != back[str(i)]
        for i in (0, 1, len(sid) - 1)
    ):
        raise ValueError(f"{json_path}: JSON 与 SID 内容不一致")
    print(f"[export] wrote {json_path}: n_items={len(back)}")

    reloaded = np.load(sids_npy_path, allow_pickle=False)
    if not np.array_equal(reloaded, sid):
        raise ValueError("SIDS_NPY 回读不一致")
    print("[export] SID_WIRING_PASS: 4-token extension npy + JSON 校验通过")


if __name__ == "__main__":
    main()
