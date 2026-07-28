#!/usr/bin/env python3
"""Task #163 Phase A — Prepare per-layer codeword assignments from Task #156 baseline.

Loads Task #156 RQ-VAE default (Euclidean) codeword path:
  HG-Rec/dataset/Instruments/Instruments_t5_rqvae_code_default.npy

Splits into 4 layer groups:
  L0 = codeword[:, 0]
  L1 = (codeword[:, 0], codeword[:, 1])
  L2 = (codeword[:, 0], codeword[:, 1], codeword[:, 2])
  dedup = codeword[:, 3]  (Stage 2 SID inference 追加列, 本任务不直接用)

Saves per-layer assignments + group keys → $CLAUDE_JOB_DIR/tmp/task163/

Output:
  task163_codeword_L0.npy       (N,) int64     L0 codeword per item
  task163_codeword_L1.npy       (N,) int64     L1 prefix-hash per item
  task163_codeword_L2.npy       (N,) int64     L2 prefix-hash per item
  task163_phase_a_summary.json  distribution stats per layer

This script is the input to Phase B (ORC subgraph measurement).
"""

import os
import sys
import json
import hashlib

import numpy as np

CODEWORD_PATH = (
    "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/"
    "Instruments_t5_rqvae_code_default.npy"
)
TMP_DIR = os.environ.get(
    "CLAUDE_JOB_TMP",
    f"/home/wlia0047/.claude/jobs/{os.environ.get('CLAUDE_JOB_ID', 'a1f6b58b')}/tmp/task163",
)
os.makedirs(TMP_DIR, exist_ok=True)


def prefix_hash(codes: np.ndarray) -> int:
    """Stable hash for tuple of codewords (for grouping)."""
    h = hashlib.md5()
    h.update(codes.astype(np.int64).tobytes())
    return int(h.hexdigest()[:16], 16)


def main() -> int:
    if not os.path.exists(CODEWORD_PATH):
        raise FileNotFoundError(
            f"Task #156 codeword not found: {CODEWORD_PATH}. "
            "Run Stage 2 SID inference first (Task #156 launcher). "
            "DO NOT fallback to other codewords (variable contamination)."
        )

    codeword = np.load(CODEWORD_PATH)
    print(f"[Phase A] Loaded {CODEWORD_PATH}")
    print(f"[Phase A] Shape: {codeword.shape}, dtype: {codeword.dtype}")

    if codeword.ndim != 2 or codeword.shape[1] != 4:
        raise ValueError(
            f"Expected (N, 4) shape from Task #156 code_default, got {codeword.shape}. "
            "Shape mismatch means wrong artifact loaded. R9 strict."
        )
    if codeword.dtype not in (np.int32, np.int64):
        raise ValueError(
            f"Expected int codeword (int32/int64), got {codeword.dtype}. "
            "Float codeword means it's an embedding not a code assignment."
        )

    N = codeword.shape[0]
    L0 = codeword[:, 0].astype(np.int64)
    L1 = codeword[:, 1].astype(np.int64)
    L2 = codeword[:, 2].astype(np.int64)
    dedup = codeword[:, 3].astype(np.int64)

    print(f"[Phase A] N = {N} items")
    print(f"[Phase A] L0 unique = {len(np.unique(L0))}, L1 unique = {len(np.unique(L1))}, "
          f"L2 unique = {len(np.unique(L2))}, dedup unique = {len(np.unique(dedup))}")

    np.save(os.path.join(TMP_DIR, "task163_codeword_L0.npy"), L0)
    np.save(os.path.join(TMP_DIR, "task163_codeword_L1.npy"), L1)
    np.save(os.path.join(TMP_DIR, "task163_codeword_L2.npy"), L2)

    summary = {
        "n_items": int(N),
        "l0_unique_count": int(len(np.unique(L0))),
        "l1_unique_count": int(len(np.unique(L1))),
        "l2_unique_count": int(len(np.unique(L2))),
        "dedup_unique_count": int(len(np.unique(dedup))),
        "l0_value_range": [int(L0.min()), int(L0.max())],
        "l1_value_range": [int(L1.min()), int(L1.max())],
        "l2_value_range": [int(L2.min()), int(L2.max())],
        "l1_prefix_count": int(32 * 64),  # 2048 L0×L1 combos (most empty)
        "l2_prefix_count": int(32 * 64 * 256),  # 524288 combos (mostly empty)
    }
    with open(os.path.join(TMP_DIR, "task163_phase_a_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[Phase A] Saved to {TMP_DIR}/")
    print(f"[Phase A] L0 group sizes (min/median/max items):")
    unique, counts = np.unique(L0, return_counts=True)
    print(f"   min={counts.min()}, median={np.median(counts):.0f}, max={counts.max()}")
    print(f"[Phase A] L1 (codeword column 1) group sizes:")
    unique, counts = np.unique(L1, return_counts=True)
    print(f"   unique={len(unique)}, min={counts.min()}, median={np.median(counts):.0f}, max={counts.max()}")

    print(f"[Phase A] DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
