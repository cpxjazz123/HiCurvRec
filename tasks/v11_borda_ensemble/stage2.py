#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v9 Stage 2: 验证 v15 历史 Stage 2 产物存在 + SHA256 匹配.

R30 合规: 所有 v9 配置顶部硬编码.
R34 合规: stage2.py 必须存在 (即使是复用 wrapper).
"""
import hashlib
from pathlib import Path

V15_STAGE2_DIR = Path("/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep")
V15_CKPT = V15_STAGE2_DIR / "hrqvae_kappa_sync.ckpt"
V15_SID = V15_STAGE2_DIR / "sid_output.npy"

EXPECTED_SID_SHA = "5f8331cc462c867f40a40bf4378760c27c4abb41db74b5843fb91380f2274f07"

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    print(f"[v9/stage2] checking {V15_STAGE2_DIR}")
    assert V15_CKPT.exists(), f"missing {V15_CKPT}"
    assert V15_SID.exists(), f"missing {V15_SID}"

    sid_sha = sha256_file(V15_SID)
    print(f"[v9/stage2] sid_output.npy SHA256={sid_sha}")
    if sid_sha != EXPECTED_SID_SHA:
        raise ValueError(f"SID SHA mismatch: {sid_sha} != {EXPECTED_SID_SHA}")

    print(f"[v9/stage2] v15 history Stage 2 产物 OK (c=[1.35, 6.00, 4.39], util=1.0)")
    print(f"[v9/stage2] DONE (no training)")

if __name__ == "__main__":
    main()
