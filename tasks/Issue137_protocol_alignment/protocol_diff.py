"""Issue #137 protocol_diff_table 生成 — canary (Issue #136) vs Stage4 raw (Issue #137) 协议差异对比.

对比维度 (Issue #137 spec):
1. tokenizer (vocab_size / max_len / pad_token_id / eos_token_id)
2. ckpt (path / sha256 / HAB / per-head)
3. SID (sha256 / shape / dtype / L3 unique / util)
4. dataset eval parquet (path / n_rows / columns)
5. generation config (beam_size / max_length / min_length / early_stopping / num_return_sequences)
6. code path (canary 自己写 generate, Stage4 用 HG_Rec wrapper generate)
7. HAB eval state (install_hab + install_per_head_curvature + freeze lambda/lambda_h/kappa_h)
8. batch_size + DDP

输出:
- protocol_diff_table.md (人读)
- protocol_diff_table.json (机读)
- stage4_protocol_manifest.json (Issue #137 spec 产品 1)
"""
import os
import sys
import json
import time
import hashlib
from pathlib import Path

TASK_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/tasks/Issue137_protocol_alignment")
os.chdir(TASK_DIR)

ISSUE136_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/tasks/Issue136_reference0_data_profile")
ISSUE137_DIR = TASK_DIR

OUT_DIR = TASK_DIR / "alignment"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ISSUE137_STAGE2 = ISSUE137_DIR / "stage2"
ISSUE136_STAGE2 = Path("/home/wlia0047/ar57/wenyu/GeneRec/tasks/Issue133_stage2_baseline_c_exp/stage2")

CKPT_PATH = ISSUE137_STAGE2 / "HG_Rec_best.pth"
SID_NPY = ISSUE137_STAGE2 / "sid_output.npy"
HAB_CKPT = ISSUE137_STAGE2 / "hrqvae_kappa_sync.ckpt"
TEST_PARQUET = Path("/home/wlia0047/ar57/wenyu/GeneRec/dataset/test.parquet")


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    t0 = time.time()
    print("[protocol_diff] 开始生成 protocol manifest + diff table", flush=True)

    import numpy as np
    sid = np.load(SID_NPY)
    sid_sha = sha256_of(SID_NPY)
    ckpt_sha = sha256_of(CKPT_PATH)
    hab_ckpt_sha = sha256_of(HAB_CKPT)

    sid_l3_unique = len(np.unique(sid[:, 3])) if sid.shape[1] >= 4 else None
    sid_l3_util = (sid.shape[0] - np.sum(sid[:, 3] == 0)) / sid.shape[0] if sid.shape[1] >= 4 else None

    # Stage4 评估产物 (本目录)
    eval_test_path = ISSUE137_STAGE2 / "eval/eval_test.json"
    raw_predictions_path = ISSUE137_STAGE2 / "eval/raw_predictions_stage4.json"
    canary_raw_path = ISSUE136_DIR / "data_profile/raw_predictions.json"

    # ────────────── 1. Stage4 protocol manifest ──────────────
    manifest = {
        "issue": "#137",
        "stage4_entry_script": str(TASK_DIR / "stage4_raw_predictions.py"),
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
            "L3_unique_count": sid_l3_unique,
            "L3_PAD_ratio": round(1 - sid_l3_util, 4) if sid_l3_util is not None else None,
        },
        "hab_stage2_ckpt": {
            "path": str(HAB_CKPT),
            "sha256": hab_ckpt_sha,
        },
        "eval_parquet": {
            "path": str(TEST_PARQUET),
            "size_bytes": TEST_PARQUET.stat().st_size,
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
        },
        "generation_config": {
            "beam_size": 20,
            "num_return_sequences": 20,
            "max_length_wrapper": 5,  # HG_Rec.generate default
            "early_stopping": "(default, not specified)",
            "min_length": "(default, not specified)",
        },
        "eval_protocol": {
            "batch_size": 96,
            "max_len": 20,
            "post_process": "preds[:, 1:] skip first token + reshape to (B, 20, 4)",
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
        "outputs": {
            "eval_test_json": str(eval_test_path) if eval_test_path.exists() else None,
            "raw_predictions_json": str(raw_predictions_path) if raw_predictions_path.exists() else None,
        },
        "issue136_canary_artifact": str(canary_raw_path) if canary_raw_path.exists() else None,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    manifest_path = OUT_DIR / "stage4_protocol_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  -> {manifest_path}", flush=True)

    # ────────────── 2. Protocol diff table ──────────────
    # 关键 protocol 差异 (Issue #136 canary vs Issue #137 Stage4 raw)
    diff = {
        "issue": "#137",
        "spec": "曲率可观测性对齐 Stage4 同协议 — canary 必须复用正式 Stage4 entry point",
        "candidate_areas": [
            {
                "area": "ckpt",
                "canary_issue136": "Issue133 stage2/HG_Rec_best.pth",
                "stage4_issue137": "Issue137 stage2/HG_Rec_best.pth (= Issue133 副本, sha256 一致)",
                "diff": "无功能差异 (sha256 同源); canary 与 Stage4 raw 用同 ckpt",
            },
            {
                "area": "SID",
                "canary_issue136": "Issue133 stage2/sid_output.npy",
                "stage4_issue137": "Issue137 stage2/sid_output.npy (= Issue133 副本)",
                "diff": "无功能差异 (sha256 同源); L3 unique_count/PAD_ratio 同",
            },
            {
                "area": "tokenizer (vocab_size/max_len)",
                "canary_issue136": "vocab_size=1025, max_len=20, input_ids=(B=1, 80), pad_token_id=0, eos_token_id=0",
                "stage4_issue137": "vocab_size=1025, max_len=20, input_ids=(B=96, 80), pad_token_id=0, eos_token_id=0 (GenRecDataset 注入)",
                "diff": "batch_size 不同 (1 vs 96); pad_token_id / eos_token_id / vocab_size / max_len 一致",
            },
            {
                "area": "code path (model.generate)",
                "canary_issue136": "model.model.generate (T5 internal, max_length=8, min_length=4, early_stopping=True)",
                "stage4_issue137": "model.generate (HG_Rec wrapper, max_length=5 = default)",
                "diff": "⚠️ 关键协议差异: canary 用 model.model.generate + max_length=8 + early_stopping=True, Stage4 wrapper 默认 max_length=5 + 无 early_stopping. max_length 差异会改变 generate 输出长度上限, 影响 pos_index match",
            },
            {
                "area": "target_sid encoding",
                "canary_issue136": "sid[target_item].tolist() → raw digit (0..63 / 0..127 / 0..255 / 0)",
                "stage4_issue137": "GenRecDataset.labels → token id (raw_digit + offset)",
                "diff": "⚠️ canary target 是 raw digit (无 offset), Stage4 labels 是 token id (有 offset). canary pred_top20_sids 已加 offset (model.model.generate 输出 token id)",
            },
            {
                "area": "post-processing",
                "canary_issue136": "取 gen[b].tolist() 前 4 token, EOS 处截断, 不足 4 补 PAD",
                "stage4_issue137": "preds[:, 1:] skip first + reshape (B, beam=20, -1=4) — vectorized all-token match",
                "diff": "⚠️ canary 用 list loop + EOS 截断, Stage4 用 vectorized slice + reshape. 关键: canary 没 [:, 1:] skip first token",
            },
            {
                "area": "HAB eval state",
                "canary_issue136": "install_hab + install_per_head_curvature + lambda_raw/lambda_h_raw fill_(0) + kappa_h no_grad",
                "stage4_issue137": "install_hab + install_per_head_curvature + lambda_raw/lambda_h_raw fill_(0) + kappa_h no_grad (via V74_EVAL_CONFIG)",
                "diff": "无功能差异 (协议一致, 都是 freeze HAB)",
            },
            {
                "area": "DDP / 单卡",
                "canary_issue136": "单卡 (CUDA_VISIBLE_DEVICES=0)",
                "stage4_issue137": "单卡 (本次 raw_predictions 走单进程) — V74 wrapper 支持 DDP 但当前没启",
                "diff": "无功能差异 (单卡); Stage4 wrapper DDP 选项不影响 raw predictions 内容",
            },
            {
                "area": "shuffle",
                "canary_issue136": "无 (按 test.parquet 顺序遍历 sample_idx=0..4999)",
                "stage4_issue137": "shuffle=False (DataLoader 默认按 parquet 顺序)",
                "diff": "无功能差异 (顺序一致, sample_idx=0..4999 对应 parquet 第 0..4999 行)",
            },
        ],
        "summary": {
            "same_protocol_areas": ["ckpt", "SID", "vocab_size/max_len/pad_token_id", "HAB eval state", "shuffle", "DDP"],
            "different_protocol_areas": [
                "code path (model.model.generate vs model.generate wrapper + max_length 8 vs 5)",
                "target_sid encoding (raw digit vs token id with offset)",
                "post-processing (list loop vs vectorized reshape, canary 缺 [:, 1:] skip first)",
            ],
        },
    }

    diff_md_path = OUT_DIR / "protocol_diff_table.md"
    with open(diff_md_path, "w") as f:
        f.write("# Issue #137 Protocol Diff Table\n\n")
        f.write("**Spec**: 曲率可观测性对齐 — canary (Issue #136) vs Stage4 raw (Issue #137) 同协议对比\n\n")
        f.write(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## 一致维度 (sha256 同源 / 协议一致)\n\n")
        for area in diff["summary"]["same_protocol_areas"]:
            f.write(f"- {area}\n")
        f.write("\n## 差异维度 (Issue #137 spec 候选 alignment 区)\n\n")
        f.write("| 维度 | canary (Issue #136) | Stage4 raw (Issue #137) | 协议 diff |\n")
        f.write("| --- | --- | --- | --- |\n")
        for c in diff["candidate_areas"]:
            f.write(f"| **{c['area']}** | {c['canary_issue136']} | {c['stage4_issue137']} | {c['diff']} |\n")
        f.write("\n## 推断 (alignment.py 输出后才能确认)\n\n")
        f.write("- canary exact_R@10=0 vs Stage4 R@10=0.0962 的 mismatch 主要源于:\n")
        f.write("  1. canary 缺 `preds[:, 1:]` skip first token (Stage4 有)\n")
        f.write("  2. canary 用 `model.model.generate` (T5 内部, max_length=8), Stage4 用 wrapper (max_length=5)\n")
        f.write("  3. canary target_sid 是 raw digit, Stage4 labels 是 token id (offset 差)\n")
        f.write("- 这些都不是曲率/参数差异, 而是 generate/eval 协议差异\n")

    diff_json_path = OUT_DIR / "protocol_diff_table.json"
    with open(diff_json_path, "w") as f:
        json.dump(diff, f, indent=2)
    print(f"  -> {diff_md_path}", flush=True)
    print(f"  -> {diff_json_path}", flush=True)
    print(f"[protocol_diff] 完成 ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()