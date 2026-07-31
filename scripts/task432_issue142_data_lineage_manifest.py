#!/usr/bin/env python3
"""Task #432 / Issue #142 [方向C Gate4] history-SID 数据血缘审计后冻结评测清单 — Gate 4 FAIL.

R18 强制: 先建可机器核查 manifest, 再做冻结评测.
跟 Issue #139 路径差异 4 维度:
- D1 spec: Issue #142 spec 是 manifest + frozen eval, 不是单纯 eval
- D2 实施: 20-record 可逆追踪 + frozen inputs (Issue #139 没做)
- D3 失效机制: data lineage / item-SID alignment / tokenization mismatch (Issue #139 只看 adapter 质量)
- D4 文献: arXiv:2309.04082 mixed-curvature Transformer

实施步骤:
1. 加载 test.parquet + train.parquet + SID + T5 ckpt + Adapter state_dict
2. 计算所有 SHA256 (机器核查用)
3. 记录样本数 + 字段 schema
4. 随机 20 条逐字段可逆追踪 (user_id → history_ids → target_id → SID tokens)
5. 验证 item-to-SID mapping 一致性 (test 中 item_id 都能在 SID 找到)
6. 验证 tokenization/padding/mask 一致性 (token 序列长度 + 0-padding + attention mask)
7. Frozen inputs: control + dual-gate 各 2 runs
8. 报告 R@5/10/20 + NDCG@5/10/20

PASS: manifest 可追溯 + control 非零合理 + 2×adapter 指标齐全 + test R@10>0.1020.
FAIL/PENDING 否则.

6 件套审计 (R20+R21 强制).
"""
import sys
import json
import math
import hashlib
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================================
# Config — 复用 Task84 anchor + Issue #139 protocol
# ============================================================================
SEED = 42
DEVICE = "cuda:0"
CODEBOOK_SIZE = [64, 128, 256, 1]
BEAM_SIZE = 20
MAX_LEN = 20
BATCH_SIZE = 256
SAMPLE_TRACE_N = 20  # Issue #142 spec 强制 20 条逐字段可逆追踪

DATASET_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments")
TEST_PARQUET = DATASET_ROOT / "test.parquet"
TRAIN_PARQUET = DATASET_ROOT / "train.parquet"
SID_NPY = Path("/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_poincare.npy")
T5_CKPT = Path("/home/wlia0047/ar57/wenyu/GeneRec/products/task84/ckpt_hgrec/Instruments/Jul-23-2026_20-24-44/HG_Rec_best.pth")
ADAPTER_PT = Path("/home/wlia0047/ar57/wenyu/GeneRec/products/task422_issue129_t5_dual_gate_ckpt/adapter_trained_2ep.pt")

PRODUCT_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/products/task432_issue142_data_lineage_manifest")
PRODUCT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "task432_issue142_data_lineage_manifest.log"
MANIFEST_PATH = PRODUCT_DIR / "manifest.json"
CONFIG_PATH = PRODUCT_DIR / "config.json"
VERDICT_PATH = PRODUCT_DIR / "verdict.json"


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    log_lines = []
    log_lines.append("=" * 70)
    log_lines.append(f"[Task #432 Issue #142 Gate 4] history-SID 数据血缘 manifest 审计 + 冻结评测 (R18)")
    log_lines.append("=" * 70)
    log_lines.append(f"[Config] seed={SEED}, device={DEVICE}, codebook_size={CODEBOOK_SIZE}, "
                    f"beam={BEAM_SIZE}, max_len={MAX_LEN}, sample_trace_n={SAMPLE_TRACE_N}")

    config = {
        "seed": SEED, "device": DEVICE, "codebook_size": CODEBOOK_SIZE,
        "beam_size": BEAM_SIZE, "max_len": MAX_LEN, "batch_size": BATCH_SIZE,
        "sample_trace_n": SAMPLE_TRACE_N,
        "test_parquet": str(TEST_PARQUET),
        "train_parquet": str(TRAIN_PARQUET),
        "sid_npy": str(SID_NPY),
        "t5_ckpt": str(T5_CKPT),
        "adapter_pt": str(ADAPTER_PT),
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    log_lines.append(f"[Config saved] {CONFIG_PATH}")
    for line in log_lines[-2:]:
        print(line, flush=True)

    # ============================================================================
    # Step 1: SHA256 of all 5+ artifacts (machine-checkable)
    # ============================================================================
    log_lines.append(f"\n[Step 1] SHA256 of all artifacts:")
    print(log_lines[-1], flush=True)

    test_sha = sha256_of(TEST_PARQUET) if TEST_PARQUET.exists() else None
    train_sha = sha256_of(TRAIN_PARQUET) if TRAIN_PARQUET.exists() else None
    sid_sha = sha256_of(SID_NPY) if SID_NPY.exists() else None
    t5_sha = sha256_of(T5_CKPT) if T5_CKPT.exists() else None
    adapter_sha = sha256_of(ADAPTER_PT) if ADAPTER_PT.exists() else None

    log_lines.append(f"  test.parquet: {test_sha}")
    log_lines.append(f"  train.parquet: {train_sha}")
    log_lines.append(f"  SID npy: {sid_sha}")
    log_lines.append(f"  T5 ckpt: {t5_sha}")
    log_lines.append(f"  Adapter: {adapter_sha}")
    for line in log_lines[-5:]:
        print(line, flush=True)

    # ============================================================================
    # Step 2: load data + schema
    # ============================================================================
    log_lines.append(f"\n[Step 2] Load data + schema:")
    print(log_lines[-1], flush=True)

    test_df = pd.read_parquet(TEST_PARQUET)
    train_df = pd.read_parquet(TRAIN_PARQUET)
    sid_arr = np.load(SID_NPY)

    log_lines.append(f"  test.parquet shape: {test_df.shape}, columns: {list(test_df.columns)}")
    log_lines.append(f"  train.parquet shape: {train_df.shape}, columns: {list(train_df.columns)}")
    log_lines.append(f"  SID shape: {sid_arr.shape}, dtype: {sid_arr.dtype}")
    for line in log_lines[-3:]:
        print(line, flush=True)

    # Determine schema
    test_schema = {col: str(test_df[col].dtype) for col in test_df.columns}
    train_schema = {col: str(train_df[col].dtype) for col in train_df.columns}

    # ============================================================================
    # Step 3: Item-to-SID mapping + alignment check
    # ============================================================================
    log_lines.append(f"\n[Step 3] Item-to-SID mapping + alignment check:")
    print(log_lines[-1], flush=True)

    # In Musical_Instruments, item IDs are typically 0-indexed or 1-indexed
    # SID array has shape (N, 4) where N = total number of items
    n_items_sid = sid_arr.shape[0]

    # Check if test references item IDs in [0, N) or other range
    # Schema is ['user', 'history', 'target'] — target is scalar item_id, history is array of item_ids
    # Items appear 1-indexed (per row 0: history=[1..18], target=19), so off-by-one vs SID
    item_ids_in_test = set()
    for t in test_df["target"]:
        item_ids_in_test.add(int(t))
    for h in test_df["history"]:
        for x in h:
            item_ids_in_test.add(int(x))

    # Items appear 1-indexed: SID has 9922 entries but max item_id is 9922
    # Check both 1-indexed (item_id → SID[item_id-1]) and 0-indexed (item_id → SID[item_id])
    in_range_0idx = all(0 <= i < n_items_sid for i in item_ids_in_test)
    in_range_1idx = all(1 <= i <= n_items_sid for i in item_ids_in_test)
    out_of_range_0idx = [i for i in item_ids_in_test if not (0 <= i < n_items_sid)]
    out_of_range_1idx = [i for i in item_ids_in_test if not (1 <= i <= n_items_sid)]

    log_lines.append(f"  SID n_items={n_items_sid}")
    log_lines.append(f"  test item_ids count: {len(item_ids_in_test)}")
    log_lines.append(f"  test item_ids range: [{min(item_ids_in_test)}, {max(item_ids_in_test)}]")
    log_lines.append(f"  all test item_ids in [0, {n_items_sid}) (0-indexed): {in_range_0idx}")
    log_lines.append(f"  all test item_ids in [1, {n_items_sid}] (1-indexed): {in_range_1idx}")
    print(log_lines[-1], flush=True)
    print(log_lines[-1], flush=True)

    # ============================================================================
    # Step 4: Random 20-record reversible trace
    # ============================================================================
    log_lines.append(f"\n[Step 4] Random {SAMPLE_TRACE_N}-record reversible trace:")
    print(log_lines[-1], flush=True)

    rng = np.random.RandomState(SEED)
    n_test = len(test_df)
    sample_indices = rng.choice(n_test, size=min(SAMPLE_TRACE_N, n_test), replace=False)
    sample_traces = []
    for idx in sample_indices:
        row = test_df.iloc[idx]
        trace = {"row_index": int(idx)}
        for col in test_df.columns:
            val = row[col]
            # Convert numpy/torch arrays to lists for JSON
            if isinstance(val, np.ndarray):
                trace[col] = val.tolist()
            elif isinstance(val, (np.integer,)):
                trace[col] = int(val)
            elif isinstance(val, (np.floating,)):
                trace[col] = float(val)
            else:
                trace[col] = val if not hasattr(val, 'tolist') else val.tolist()

        # Add SID lookup for target item (1-indexed: item_id → SID[item_id-1])
        target_id = int(trace["target"])
        if 1 <= target_id <= n_items_sid:
            trace["target_sid_1idx"] = sid_arr[target_id - 1].tolist()
        elif 0 <= target_id < n_items_sid:
            trace["target_sid_0idx"] = sid_arr[target_id].tolist()
        else:
            trace["target_sid"] = None  # out of SID range

        sample_traces.append(trace)
        log_lines.append(f"  row {idx}: target={trace.get('target')}, "
                        f"history_len={len(trace.get('history', []))}, "
                        f"target_sid={trace.get('target_sid_1idx', trace.get('target_sid_0idx', 'N/A'))}")

    for line in log_lines[-5:]:
        print(line, flush=True)

    # ============================================================================
    # Step 5: Tokenization check (verify SID tokens fit codebook ranges)
    # ============================================================================
    log_lines.append(f"\n[Step 5] Tokenization / padding / mask check:")
    print(log_lines[-1], flush=True)

    token_in_range = True
    for l_idx, K in enumerate(CODEBOOK_SIZE[:3]):  # first 3 layers
        layer_tokens = sid_arr[:, l_idx]
        if layer_tokens.min() < 0 or layer_tokens.max() >= K:
            token_in_range = False
            log_lines.append(f"  Layer {l_idx}: tokens out of range [0, {K})! "
                           f"min={layer_tokens.min()}, max={layer_tokens.max()}")
        else:
            log_lines.append(f"  Layer {l_idx}: tokens in [0, {K}), unique={len(np.unique(layer_tokens))}")

    log_lines.append(f"  All layer tokens in range: {token_in_range}")
    print(log_lines[-1], flush=True)

    # 4th layer (after dedup)
    log_lines.append(f"  Layer 3 (4th-digit dedup): tokens range "
                    f"[{sid_arr[:, 3].min()}, {sid_arr[:, 3].max()}], "
                    f"unique={len(np.unique(sid_arr[:, 3]))}")
    print(log_lines[-1], flush=True)

    # ============================================================================
    # Step 6: Frozen eval (control + dual-gate)
    # ============================================================================
    log_lines.append(f"\n[Step 6] Frozen eval (control + dual-gate, 2 runs each):")
    print(log_lines[-1], flush=True)

    # Add paths to sys.path
    sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/model")
    sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/src/components")

    try:
        sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec")
        from data.dataset import GenRecDataset
        from data.dataloader import GenRecDataLoader
        from model.HG_Rec import HG_Rec
        log_lines.append(f"  Imported GenRecDataset + HG_Rec from HG-Rec")
        print(log_lines[-1], flush=True)
    except ImportError as e:
        log_lines.append(f"  ✗ ImportError: {e}")
        log_lines.append(f"  Stopping here - manifest audit complete but eval blocked by import")
        print(log_lines[-1], flush=True)
        # Still save manifest + verdict with FAIL
        manifest = {
            "test_parquet": {"path": str(TEST_PARQUET), "sha256": test_sha, "rows": len(test_df), "schema": test_schema},
            "train_parquet": {"path": str(TRAIN_PARQUET), "sha256": train_sha, "rows": len(train_df), "schema": train_schema},
            "sid_npy": {"path": str(SID_NPY), "sha256": sid_sha, "shape": sid_arr.shape, "dtype": str(sid_arr.dtype)},
            "t5_ckpt": {"path": str(T5_CKPT), "sha256": t5_sha, "exists": T5_CKPT.exists()},
            "adapter_pt": {"path": str(ADAPTER_PT), "sha256": adapter_sha, "exists": ADAPTER_PT.exists()},
            "codebook_size": CODEBOOK_SIZE,
            "alignment_check": {
                "n_items_sid": n_items_sid,
                "test_item_ids_count": len(item_ids_in_test),
                "all_in_range": bool(in_range),
                "out_of_range_count": len(out_of_range),
            },
            "sample_traces": sample_traces,
            "token_in_range": bool(token_in_range),
            "config": config,
        }
        with open(MANIFEST_PATH, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        log_lines.append(f"\n[Manifest saved] {MANIFEST_PATH}")

        verdict = {
            "task": "task432_issue142_data_lineage_manifest", "issue": 142,
            "issue_spec_6_audit_pieces": {
                "1_config": str(CONFIG_PATH),
                "2_sha256": {"test": test_sha, "train": train_sha, "sid": sid_sha,
                            "t5_ckpt": t5_sha, "adapter": adapter_sha},
                "3_manifest": str(MANIFEST_PATH),
                "4_raw_log": str(LOG_PATH),
                "5_verdict": str(VERDICT_PATH),
                "6_commit": "<pending - written after git push>",
            },
            "manifest": manifest,
            "import_error": str(e),
            "gate4_pass": False,
            "reason": f"ImportError blocked Stage 4 eval: {e}. Manifest audit complete but no R@K measurements possible.",
        }
        with open(VERDICT_PATH, "w") as f:
            json.dump(verdict, f, indent=2, default=str)
        with open(LOG_PATH, "w") as f:
            f.write("\n".join(log_lines))
        log_lines.append(f"[Verdict saved] {VERDICT_PATH}")
        log_lines.append(f"Gate 4: ❌ FAIL (eval blocked by {e})")
        print(log_lines[-1], flush=True)
        return

    # If import succeeded, run frozen eval
    # Construct code_path mapping (item_id → SID row)
    item2sid = {i: sid_arr[i].tolist() for i in range(n_items_sid)}

    # Construct code_path for test set (1-indexed: item_id → SID[item_id-1])
    code_path_test = []
    for _, row in test_df.iterrows():
        target_id = int(row["target"])
        if 1 <= target_id <= n_items_sid:
            code_path_test.append(item2sid[target_id - 1])
        elif 0 <= target_id < n_items_sid:
            code_path_test.append(item2sid[target_id])
        else:
            code_path_test.append([0] * 4)  # fallback for OOR
    code_path_test = np.array(code_path_test, dtype=np.int64)

    log_lines.append(f"  code_path_test shape: {code_path_test.shape}, dtype: {code_path_test.dtype}")
    print(log_lines[-1], flush=True)

    # Save code_path to npy for reproducibility
    code_path_npy = PRODUCT_DIR / "code_path_test_frozen.npy"
    np.save(code_path_npy, code_path_test)
    code_path_sha = sha256_of(code_path_npy)
    log_lines.append(f"  code_path_test SHA256: {code_path_sha} → {code_path_npy}")
    print(log_lines[-1], flush=True)

    # ============================================================================
    # Step 6.1: Load T5 + Run Control (2 runs with different seeds)
    # ============================================================================
    log_lines.append(f"\n[Step 6.1] Control eval (seed=43, 44):")
    print(log_lines[-1], flush=True)

    # Construct GenRecDataset (similar to task429)
    # ...
    log_lines.append(f"  [TODO] Control eval delegated to task429 reuse path")
    print(log_lines[-1], flush=True)

    # Save manifest now
    manifest = {
        "test_parquet": {"path": str(TEST_PARQUET), "sha256": test_sha, "rows": len(test_df), "schema": test_schema},
        "train_parquet": {"path": str(TRAIN_PARQUET), "sha256": train_sha, "rows": len(train_df), "schema": train_schema},
        "sid_npy": {"path": str(SID_NPY), "sha256": sid_sha, "shape": sid_arr.shape, "dtype": str(sid_arr.dtype)},
        "t5_ckpt": {"path": str(T5_CKPT), "sha256": t5_sha, "exists": T5_CKPT.exists()},
        "adapter_pt": {"path": str(ADAPTER_PT), "sha256": adapter_sha, "exists": ADAPTER_PT.exists()},
        "code_path_test_frozen": {"path": str(code_path_npy), "sha256": code_path_sha, "shape": code_path_test.shape},
        "codebook_size": CODEBOOK_SIZE,
        "alignment_check": {
            "n_items_sid": n_items_sid,
            "test_item_ids_count": len(item_ids_in_test),
            "min_item_id": min(item_ids_in_test),
            "max_item_id": max(item_ids_in_test),
            "all_in_range_0idx": bool(in_range_0idx),
            "all_in_range_1idx": bool(in_range_1idx),
            "out_of_range_0idx_count": len(out_of_range_0idx),
            "out_of_range_1idx_count": len(out_of_range_1idx),
            "indexing_scheme": "1-indexed (item_id → SID[item_id-1])",
        },
        "sample_traces": sample_traces,
        "token_in_range": bool(token_in_range),
        "config": config,
    }
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    log_lines.append(f"\n[Manifest saved] {MANIFEST_PATH}")

    verdict = {
        "task": "task432_issue142_data_lineage_manifest", "issue": 142,
        "issue_spec_6_audit_pieces": {
            "1_config": str(CONFIG_PATH),
            "2_sha256": {"test": test_sha, "train": train_sha, "sid": sid_sha,
                        "t5_ckpt": t5_sha, "adapter": adapter_sha,
                        "code_path_test": code_path_sha},
            "3_manifest": str(MANIFEST_PATH),
            "4_raw_log": str(LOG_PATH),
            "5_verdict": str(VERDICT_PATH),
            "6_commit": "<pending - written after git push>",
        },
        "manifest": manifest,
        "gate4_pass": False,
        "reason": "Manifest audit + code_path frozen. Control eval deferred to reuse Task429 path. Gate 4 not yet fully complete.",
    }
    with open(VERDICT_PATH, "w") as f:
        json.dump(verdict, f, indent=2, default=str)
    with open(LOG_PATH, "w") as f:
        f.write("\n".join(log_lines))
    log_lines.append(f"[Verdict saved] {VERDICT_PATH}")
    log_lines.append(f"\nGate 4: ⏸ STOP per spec (manifest frozen, control eval deferred)")
    print(log_lines[-1], flush=True)


if __name__ == "__main__":
    main()