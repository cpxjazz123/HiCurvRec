"""Issue #137 alignment — canary (Issue #136) vs Stage4 raw predictions 同协议对比.

目的: 验证 Issue #136 reference-0 canary (exact_R@10=0) 与正式 Stage4 (test_R@10=0.0962)
的 mismatch 是否因协议差异导致. 本脚本:
1. 读 Issue #136 canary_forward.py 输出的 raw_predictions.json (5000 samples)
2. 读 Issue #137 stage4_raw_predictions.py 输出的 raw_predictions_stage4.json (5000 samples)
3. 计算每 sample 的 exact match / prefix match / first-3-token match / last-token match / PAD ratio
4. 输出 alignment_table.csv + alignment_summary.json
5. 不修改任何 Stage2/Stage3/Stage4 代码, 仅对齐观察
"""
import os
import sys
import json
import time
from pathlib import Path

TASK_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/tasks/Issue137_protocol_alignment")
os.chdir(TASK_DIR)

ISSUE136_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/tasks/Issue136_reference0_data_profile")
ISSUE137_DIR = TASK_DIR

CANARY_PATH = ISSUE136_DIR / "data_profile/raw_predictions.json"
STAGE4_PATH = ISSUE137_DIR / "stage2/eval/raw_predictions_stage4.json"
OUT_DIR = ISSUE137_DIR / "alignment"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PAD_TOKEN = 0


def load_json(path):
    with open(path) as f:
        return json.load(f)


def compute_metrics(samples):
    """samples: list of dicts (sample_idx, pred_top20_sids, target_sid[, pos_index_top20]).

    Returns dict of per-sample metrics.
    """
    rows = []
    for s in samples:
        idx = s.get("sample_idx", s.get("sample_id"))
        target = s["target_sid"]
        preds = s["pred_top20_sids"]
        # exact: any beam 完全等于 target
        exact = any(p == target for p in preds)
        # prefix: any beam 前 3 token match
        prefix3 = any(p[:3] == target[:3] for p in preds)
        # first token match
        first1 = any(p[0] == target[0] for p in preds)
        # last token (4th) match — Issue #136 发现 L3 全 0
        last1 = any(p[3] == target[3] for p in preds)
        # rank of exact match (1-indexed, None if not found)
        exact_rank = None
        for r, p in enumerate(preds, 1):
            if p == target:
                exact_rank = r
                break
        # top-1 prediction
        top1 = preds[0] if preds else [0, 0, 0, 0]
        rows.append({
            "sample_idx": idx,
            "exact_hit_top20": exact,
            "prefix3_hit_top20": prefix3,
            "first1_hit_top20": first1,
            "last1_hit_top20": last1,
            "exact_rank": exact_rank if exact_rank is not None else 21,
            "top1_eq_target": top1 == target,
        })
    return rows


def summarize(rows, label):
    n = len(rows)
    if n == 0:
        return {"label": label, "n": 0}
    exact_n = sum(r["exact_hit_top20"] for r in rows)
    prefix_n = sum(r["prefix3_hit_top20"] for r in rows)
    first_n = sum(r["first1_hit_top20"] for r in rows)
    last_n = sum(r["last1_hit_top20"] for r in rows)
    top1_eq = sum(r["top1_eq_target"] for r in rows)
    ranks = [r["exact_rank"] for r in rows if r["exact_hit_top20"]]
    mean_rank = sum(ranks) / len(ranks) if ranks else None
    return {
        "label": label,
        "n": n,
        "exact_R@10": round(exact_n / n, 4),  # Issue #136 用 exact_R@10
        "exact_R@20": round(exact_n / n, 4),  # canary 只看 top20
        "prefix3_R@10": round(prefix_n / n, 4),
        "prefix3_R@20": round(prefix_n / n, 4),
        "first1_R@10": round(first_n / n, 4),
        "last1_R@10": round(last_n / n, 4),
        "top1_eq_target_rate": round(top1_eq / n, 4),
        "n_exact_hits": exact_n,
        "mean_exact_rank": round(mean_rank, 2) if mean_rank else None,
    }


def main():
    t0 = time.time()
    print("[alignment] 开始对比 Issue #136 canary vs Issue #137 Stage4 raw", flush=True)

    if not CANARY_PATH.exists():
        raise FileNotFoundError(f"Issue #136 canary 不存在: {CANARY_PATH}")
    if not STAGE4_PATH.exists():
        raise FileNotFoundError(f"Issue #137 Stage4 raw 不存在: {STAGE4_PATH}")

    canary = load_json(CANARY_PATH)
    stage4 = load_json(STAGE4_PATH)

    # Issue #136 canary 是 list[dict] (samples), Issue #137 Stage4 raw 是 dict{samples, n_raw, ...}
    if isinstance(canary, dict) and "samples" in canary:
        canary_samples = canary["samples"]
    else:
        canary_samples = canary
    if isinstance(stage4, dict) and "samples" in stage4:
        stage4_samples = stage4["samples"]
    else:
        stage4_samples = stage4

    print(f"  canary samples={len(canary_samples)}", flush=True)
    print(f"  Stage4 raw samples={len(stage4_samples)}", flush=True)

    # 取交集 sample_idx (R11: 不允许 fallback, 交集为空则 raise)
    canary_idx = {s.get("sample_id", s.get("sample_idx")): s for s in canary_samples}
    stage4_idx = {s.get("sample_idx"): s for s in stage4_samples}
    common = sorted(set(canary_idx.keys()) & set(stage4_idx.keys()))
    if not common:
        raise RuntimeError(f"canary 与 Stage4 raw 无交集 sample_idx (canary={list(canary_idx)[:3]}, "
                           f"stage4={list(stage4_idx)[:3]})")
    print(f"  common sample_idx={len(common)}", flush=True)

    canary_aligned = [{**canary_idx[i], "sample_idx": i} for i in common]
    stage4_aligned = [{**stage4_idx[i], "sample_idx": i} for i in common]

    # 计算 metrics
    canary_rows = compute_metrics(canary_aligned)
    stage4_rows = compute_metrics(stage4_aligned)

    canary_summary = summarize(canary_rows, "canary_issue136")
    stage4_summary = summarize(stage4_rows, "stage4_raw_issue137")

    # Diff metrics (R20+R36 强制: 至少 3-5 行/Gate 维度回答)
    diff = {}
    for k in canary_summary:
        if k in ("label", "n") or k in stage4_summary and isinstance(canary_summary[k], (int, float)):
            if isinstance(canary_summary[k], (int, float)) and k != "n_exact_hits":
                diff[k] = round(stage4_summary[k] - canary_summary[k], 4)

    # Per-sample 一致性 (canary exact hit 但 Stage4 not hit 等)
    exact_canary_only = 0
    exact_stage4_only = 0
    exact_both = 0
    exact_neither = 0
    pred_diff_n = 0
    for cr, sr in zip(canary_rows, stage4_rows):
        c_hit = cr["exact_hit_top20"]
        s_hit = sr["exact_hit_top20"]
        if c_hit and s_hit:
            exact_both += 1
        elif c_hit and not s_hit:
            exact_canary_only += 1
        elif s_hit and not c_hit:
            exact_stage4_only += 1
        else:
            exact_neither += 1
        # top-1 prediction same?
        c_top1 = canary_idx[cr["sample_idx"]]["pred_top20_sids"][0]
        s_top1 = stage4_idx[sr["sample_idx"]]["pred_top20_sids"][0]
        if c_top1 != s_top1:
            pred_diff_n += 1

    # 保存 alignment_table.csv
    csv_path = OUT_DIR / "alignment_table.csv"
    with open(csv_path, "w") as f:
        f.write("sample_idx,canary_exact,stage4_exact,canary_prefix3,stage4_prefix3,"
                "canary_first1,stage4_first1,canary_last1,stage4_last1,"
                "canary_top1_eq,stage4_top1_eq,canary_rank,stage4_rank\n")
        for cr, sr in zip(canary_rows, stage4_rows):
            f.write(f"{cr['sample_idx']},{int(cr['exact_hit_top20'])},{int(sr['exact_hit_top20'])},"
                    f"{int(cr['prefix3_hit_top20'])},{int(sr['prefix3_hit_top20'])},"
                    f"{int(cr['first1_hit_top20'])},{int(sr['first1_hit_top20'])},"
                    f"{int(cr['last1_hit_top20'])},{int(sr['last1_hit_top20'])},"
                    f"{int(cr['top1_eq_target'])},{int(sr['top1_eq_target'])},"
                    f"{cr['exact_rank']},{sr['exact_rank']}\n")

    # 保存 alignment_summary.json
    summary = {
        "issue": "#137",
        "spec": "曲率可观测性对齐: canary (Issue #136) vs Stage4 raw 同协议对比 (5000 sample 交集)",
        "canary": canary_summary,
        "stage4_raw": stage4_summary,
        "diff": diff,
        "sample_intersection": {
            "canary_n": len(canary_samples),
            "stage4_n": len(stage4_samples),
            "common_n": len(common),
            "pred_top1_diff_n": pred_diff_n,
        },
        "exact_match_consistency": {
            "both": exact_both,
            "canary_only": exact_canary_only,
            "stage4_only": exact_stage4_only,
            "neither": exact_neither,
        },
        "products": {
            "alignment_table_csv": str(csv_path),
        },
    }
    summary_path = OUT_DIR / "alignment_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # 输出
    print(f"  canary summary: {canary_summary}", flush=True)
    print(f"  Stage4 raw summary: {stage4_summary}", flush=True)
    print(f"  diff: {diff}", flush=True)
    print(f"  exact consistency: both={exact_both} canary_only={exact_canary_only} "
          f"stage4_only={exact_stage4_only} neither={exact_neither}", flush=True)
    print(f"  pred_top1_diff_n={pred_diff_n}/{len(common)}", flush=True)
    print(f"  -> {csv_path}", flush=True)
    print(f"  -> {summary_path}", flush=True)
    print(f"[alignment] 完成 ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()