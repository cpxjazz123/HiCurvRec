"""Issue #138 stage4_full_oracle_manifest.json 生成 — 静态 + 动态 (raw_predictions + eval_test 完成后补).

产物:
- stage4_full_oracle_manifest.json
  - ckpt: path / sha256 / size_bytes
  - sid: path / sha256 / shape / dtype / L3_unique_count / L3_PAD_ratio
  - hab_stage2_ckpt: path / sha256
  - eval_parquet: path / sha256 / size_bytes / n_rows
  - t5_config: dict
  - codebook: dict
  - generation_config: dict (beam_size=20, max_length=5, num_return_sequences=20, no early_stopping / min_length spec)
  - eval_protocol: dict (batch_size=96, max_len=20, post_process=preds[:, 1:] + reshape, score=vectorized all-token match)
  - hab_eval_state: dict
  - raw_predictions: path / size_bytes / n_samples (后填充)
  - eval_test: path / R@5/10/20 / NDCG@5/10/20 (后填充)
  - protocol_drift_check: vs Issue #133 R@10=0.0962 (后填充)
"""
import os
import sys
import json
import hashlib
import time
from pathlib import Path

TASK_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/tasks/Issue138_full_curvature_failure_map")
os.chdir(TASK_DIR)

CKPT_PATH = TASK_DIR / "stage2/HG_Rec_best.pth"
SID_NPY = TASK_DIR / "stage2/sid_output.npy"
HAB_CKPT = TASK_DIR / "stage2/hrqvae_kappa_sync.ckpt"
TEST_PARQUET = Path("/home/wlia0047/ar57/wenyu/GeneRec/dataset/test.parquet")
RAW_PARQUET = TASK_DIR / "stage2/eval/raw_predictions_stage4_full.parquet"
EVAL_TEST = TASK_DIR / "stage2/eval/eval_test.json"
OUT_DIR = TASK_DIR / "failure_map"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    t0 = time.time()
    print("[manifest] 开始生成 stage4_full_oracle_manifest.json", flush=True)

    import numpy as np
    import pyarrow.parquet as pq

    sid = np.load(SID_NPY)
    sid_sha = sha256_of(SID_NPY)
    ckpt_sha = sha256_of(CKPT_PATH)
    hab_ckpt_sha = sha256_of(HAB_CKPT)
    test_sha = sha256_of(TEST_PARQUET)
    test_meta = pq.read_metadata(TEST_PARQUET)
    sid_l3_unique = len(np.unique(sid[:, 3])) if sid.shape[1] >= 4 else None
    sid_l3_pad_count = int(np.sum(sid[:, 3] == 0)) if sid.shape[1] >= 4 else None
    sid_l3_pad_ratio = round(sid_l3_pad_count / sid.shape[0], 4) if sid_l3_pad_count is not None else None

    # 静态部分
    manifest = {
        "issue": "#138",
        "task": "Issue138_full_curvature_failure_map",
        "stage4_entry_script": str(TASK_DIR / "stage4_full_oracle.py"),
        "ckpt": {
            "path": str(CKPT_PATH),
            "sha256": ckpt_sha,
            "size_bytes": CKPT_PATH.stat().st_size,
        },
        "sid": {
            "path": str(SID_NPY),
            "sha256": sid_sha,
            "shape": list(sid.shape),
            "dtype": str(sid.dtype),
            "L0_unique_count": len(np.unique(sid[:, 0])),
            "L1_unique_count": len(np.unique(sid[:, 1])),
            "L2_unique_count": len(np.unique(sid[:, 2])),
            "L3_unique_count": sid_l3_unique,
            "L3_PAD_count": sid_l3_pad_count,
            "L3_PAD_ratio": sid_l3_pad_ratio,
        },
        "hab_stage2_ckpt": {
            "path": str(HAB_CKPT),
            "sha256": hab_ckpt_sha,
        },
        "eval_parquet": {
            "path": str(TEST_PARQUET),
            "sha256": test_sha,
            "size_bytes": TEST_PARQUET.stat().st_size,
            "n_rows": test_meta.num_rows,
            "n_cols": test_meta.num_columns,
        },
        "t5_config": {
            "num_layers": 6,
            "num_decoder_layers": 4,
            "d_model": 128,
            "d_ff": 1024,
            "num_heads": 6,
            "d_kv": 64,
            "dropout_rate": 0.1,
            "vocab_size": 1025,
            "pad_token_id": 0,
            "eos_token_id": 0,
            "decoder_start_token_id": 0,
            "feed_forward_proj": "relu",
        },
        "codebook": {
            "codebook_size": [64, 128, 256, 1],
            "codeword_offsets": [1, 65, 193, 449],
            "note": "L3 codebook_size=1 → 所有 item 第 4 位都是 0 (PAD), 这是 Issue #136/#137/#122 plan 识别的 Stage2 sid 第 4 位分配 bug"
        },
        "generation_config": {
            "beam_size": 20,
            "num_return_sequences": 20,
            "max_length_wrapper": 5,
            "early_stopping": "(HG_Rec wrapper default, not explicitly specified)",
            "min_length": "(HG_Rec wrapper default, not explicitly specified)",
        },
        "eval_protocol": {
            "batch_size": 96,
            "max_len": 20,
            "post_process": "preds[:, 1:] skip first + reshape to (B, 20, 4)",
            "score": "calculate_pos_index vectorized all-token match",
            "TOP_K": [5, 10, 20],
        },
        "hab_eval_state": {
            "install_hab": True,
            "lambda_raw_freeze": "fill_(0.0)",
            "install_per_head_curvature": True,
            "lambda_h_raw_freeze": "fill_(0.0)",
            "kappa_h_freeze": "no_grad (default for eval)",
        },
        "raw_predictions": {
            "path": str(RAW_PARQUET),
            "exists": RAW_PARQUET.exists(),
        },
        "eval_test": {
            "path": str(EVAL_TEST),
            "exists": EVAL_TEST.exists(),
        },
        "protocol_drift_check": {
            "issue133_full_R10": 0.0962,
            "tolerance": 0.005,
        },
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # 动态后填充
    if RAW_PARQUET.exists():
        manifest["raw_predictions"]["size_bytes"] = RAW_PARQUET.stat().st_size
        # 用 pandas 读 n_rows
        import pandas as pd
        df = pd.read_parquet(RAW_PARQUET)
        manifest["raw_predictions"]["n_samples"] = len(df)
        manifest["raw_predictions"]["columns"] = list(df.columns)
    if EVAL_TEST.exists():
        eval_test = json.load(open(EVAL_TEST))
        manifest["eval_test"]["R@5"] = eval_test.get("R@5")
        manifest["eval_test"]["R@10"] = eval_test.get("R@10")
        manifest["eval_test"]["R@20"] = eval_test.get("R@20")
        manifest["eval_test"]["NDCG@5"] = eval_test.get("NDCG@5")
        manifest["eval_test"]["NDCG@10"] = eval_test.get("NDCG@10")
        manifest["eval_test"]["NDCG@20"] = eval_test.get("NDCG@20")
        manifest["eval_test"]["n_eval"] = eval_test.get("n_eval")
        if "protocol_drift_check" in eval_test:
            manifest["protocol_drift_check"]["this_run_full_R10"] = eval_test["protocol_drift_check"].get("this_run_full_R10")
            manifest["protocol_drift_check"]["abs_diff"] = eval_test["protocol_drift_check"].get("abs_diff")
            manifest["protocol_drift_check"]["blocked"] = eval_test["protocol_drift_check"].get("blocked")

    out_path = OUT_DIR / "stage4_full_oracle_manifest.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  -> {out_path}", flush=True)
    print(f"[manifest] 完成 ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()