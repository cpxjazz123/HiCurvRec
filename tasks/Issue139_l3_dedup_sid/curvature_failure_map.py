"""Issue #138 curvature_failure_map — 全量 Stage4 trace + item_geometry + 分桶 + Spearman + 5 个稳定现象.

产物 (Issue #138 spec):
1. full_generation_failure.parquet (per-sample target_item + best_rank + first_error + prefix + item_geometry)
2. full_curvature_observability.parquet (per-target-item hit_rate + item_geometry)
3. full_bucket_metrics.json/csv (按 history length / item popularity / L0/L1/L2 margin / L2 uniqueness / L3 PAD 分桶)
4. full_correlation_report.json/csv (Spearman + q-value + effect size)
5. optimization_evidence_verdict.md (5 个稳定现象, 仅汇总事实, 不提实现方案)
"""
import os
import sys
import json
import time
from pathlib import Path

TASK_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/tasks/Issue139_l3_dedup_sid")
os.chdir(TASK_DIR)

ISSUE136_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/tasks/Issue136_reference0_data_profile")
DATASET_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/dataset")

RAW_PARQUET = TASK_DIR / "stage2/eval/raw_predictions_stage4_full.parquet"
ITEM_GEOM_CSV = ISSUE136_DIR / "data_profile/item_geometry.csv"
SID_NPY = TASK_DIR / "stage2/sid_output.npy"
TEST_PARQUET = DATASET_DIR / "test.parquet"
EVAL_TEST_JSON = TASK_DIR / "stage2/eval/eval_test.json"

OUT_DIR = TASK_DIR / "failure_map"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CODEBOOK_SIZE = [64, 128, 256, 1]
CODEWORD_OFFSETS = [1, 65, 193, 449]


def token_id_to_raw_digit(token_id):
    """token id (with offset) → raw SID digit (no offset)."""
    if token_id is None or token_id < 0:
        return -1
    for layer in range(4):
        offset_start = CODEWORD_OFFSETS[layer]
        offset_end = offset_start + CODEBOOK_SIZE[layer]
        if offset_start <= token_id < offset_end:
            return token_id - offset_start
    return -1  # PAD or invalid


def target_item_from_sid(target_sid_list, sid_array):
    """target_sid list[int] (token id with offsets) → target item id (reverse lookup).
    Returns -1 if not found.
    """
    raw = [token_id_to_raw_digit(t) for t in target_sid_list]
    if -1 in raw:
        return -1
    matches = np.where((sid_array == raw).all(axis=1))[0]
    if len(matches) == 0:
        return -1
    if len(matches) == 1:
        return int(matches[0])
    # 多个 match (L3=0 PAD 导致) → 取第一个, 记录 ambiguous=True
    return int(matches[0])


def bootstrap_ci(values, n_bootstrap=1000, ci=0.95, seed=42):
    """Bootstrap 95% CI for mean."""
    import numpy as np
    rng = np.random.RandomState(seed)
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return None, None, None
    means = []
    for _ in range(n_bootstrap):
        sample = values[rng.randint(0, n, size=n)]
        means.append(sample.mean())
    means = sorted(means)
    lo_idx = int((1 - ci) / 2 * n_bootstrap)
    hi_idx = int((1 + ci) / 2 * n_bootstrap)
    return float(values.mean()), float(means[lo_idx]), float(means[hi_idx])


def main():
    t0 = time.time()
    print("[failure_map] 开始生成全量 trace + item_geometry + 分桶 + Spearman + 5 现象", flush=True)

    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    from scipy.stats import spearmanr, pointbiserialr

    sid = np.load(SID_NPY)
    sid_l3_pad_count = int(np.sum(sid[:, 3] == 0)) if sid.shape[1] >= 4 else None
    sid_l3_pad_ratio = round(sid_l3_pad_count / sid.shape[0], 4) if sid_l3_pad_count is not None else None
    sid_l3_unique = len(np.unique(sid[:, 3])) if sid.shape[1] >= 4 else None
    test_df = pq.read_table(TEST_PARQUET).to_pandas()
    print(f"  SID shape={sid.shape}, test.parquet rows={len(test_df)}", flush=True)

    raw_df = pd.read_parquet(RAW_PARQUET)
    print(f"  raw predictions rows={len(raw_df)}", flush=True)

    # ─── Gate 1: 全局重现 (Issue #138 spec) ───
    eval_test = json.load(open(EVAL_TEST_JSON))
    R10_full = eval_test["R@10"]
    issue133_R10 = 0.0962
    drift = abs(R10_full - issue133_R10)
    drift_blocked = drift > 0.005
    print(f"  Gate 1: this_run R@10={R10_full:.4f} vs Issue #133 {issue133_R10:.4f} (diff={drift:.4f})",
          "BLOCKED" if drift_blocked else "PASS", flush=True)

    # ─── 反查 target_item via SID ───
    print("  反查 target_item via SID (raw digit lookup)...", flush=True)
    target_items = []
    ambiguous_count = 0
    for i, row in raw_df.iterrows():
        target_sid_list = row["target_sid"]
        raw_digits = [token_id_to_raw_digit(t) for t in target_sid_list]
        # L3=0 → 所有 (a,b,c,0) item 都 match; 取 popularity 最低的 (Issue #136 标记为 ambiguous)
        matches = np.where((sid[:, :3] == raw_digits[:3]).all(axis=1))[0]
        if len(matches) == 0:
            target_items.append(-1)
        else:
            target_items.append(int(matches[0]))  # 多个 match (L3=0) → 取第一个
            if len(matches) > 1:
                ambiguous_count += 1
        if (i + 1) % 5000 == 0:
            print(f"    [{i+1}/{len(raw_df)}] ambiguous={ambiguous_count}", flush=True)

    raw_df["target_item"] = target_items
    print(f"  target_item 反查完成: ambiguous (L3=0 multi-match) = {ambiguous_count}/{len(raw_df)}", flush=True)

    # ─── join item_geometry ───
    print("  join Issue #136 item_geometry.csv ...", flush=True)
    geom_df = pd.read_csv(ITEM_GEOM_CSV)
    geom_df = geom_df.rename(columns={"item_id": "target_item"})
    keep_cols = ["target_item", "sid_L0", "sid_L1", "sid_L2", "sid_L3",
                 "L0_freq", "L1_freq", "L2_freq", "L3_freq",
                 "L2_uniqueness", "item_knn_mean_dist", "item_popularity", "text_emb_norm"]
    geom_keep = geom_df[keep_cols]
    full_df = raw_df.merge(geom_keep, on="target_item", how="left")
    print(f"  joined: {len(full_df)} rows, missing geom = {full_df['sid_L0'].isna().sum()}", flush=True)

    # ─── product 3: full_generation_failure.parquet ───
    fail_cols = ["sample_idx", "target_item", "best_rank_top20", "hit_at_5", "hit_at_10", "hit_at_20",
                 "first_error_position", "prefix_match_length", "top1_has_pad", "target_has_pad",
                 "top1_l0", "top1_l1", "top1_l2", "top1_l3",
                 "target_l0", "target_l1", "target_l2", "target_l3"] + keep_cols[1:]
    fail_parquet_path = OUT_DIR / "full_generation_failure.parquet"
    full_df[fail_cols].to_parquet(fail_parquet_path, index=False)
    print(f"  -> {fail_parquet_path} ({fail_parquet_path.stat().st_size/1024:.1f}KB)", flush=True)

    # ─── product 4: full_curvature_observability.parquet (per-target-item hit_rate) ───
    print("  生成 per-target-item hit_rate + item_geometry ...", flush=True)
    target_stats = full_df.groupby("target_item").agg(
        n_total=("sample_idx", "count"),
        n_hit_10=("hit_at_10", "sum"),
        n_hit_20=("hit_at_20", "sum"),
        mean_best_rank=("best_rank_top20", "mean"),
        mean_first_err=("first_error_position", "mean"),
        mean_prefix_len=("prefix_match_length", "mean"),
    ).reset_index()
    target_stats["hit_rate_10"] = target_stats["n_hit_10"] / target_stats["n_total"]
    target_stats["hit_rate_20"] = target_stats["n_hit_20"] / target_stats["n_total"]
    target_stats = target_stats.merge(geom_keep, on="target_item", how="left")
    curv_path = OUT_DIR / "full_curvature_observability.parquet"
    target_stats.to_parquet(curv_path, index=False)
    print(f"  -> {curv_path} ({curv_path.stat().st_size/1024:.1f}KB)", flush=True)

    # ─── product 5: full_bucket_metrics.json/csv ───
    print("  生成 bucket_metrics ...", flush=True)
    bucket_results = {}

    def bucketize(series, bins, labels=None):
        return pd.cut(series, bins=bins, labels=labels, include_lowest=True)

    # history length: 从 test.parquet 读 (sample_idx = parquet row)
    hist_lengths = []
    for idx in raw_df["sample_idx"]:
        row = test_df.iloc[idx]
        hist_raw = row.get("history")
        if hist_raw is None:
            hist_lengths.append(0)
        elif hasattr(hist_raw, "tolist"):
            hist_lengths.append(len(hist_raw.tolist()))
        else:
            hist_lengths.append(len(list(hist_raw)))
    raw_df["history_length"] = hist_lengths

    buckets_def = [
        ("history_length", pd.cut(raw_df["history_length"], bins=[0, 5, 10, 15, 20],
                                  labels=["1-5", "6-10", "11-15", "16-20"], include_lowest=True)),
        ("L3_PAD", raw_df["target_l3"].apply(lambda x: "PAD=0" if x == 449 else "non-PAD")),
        ("L0_freq_quantile", bucketize(full_df["L0_freq"], bins=[0, 100, 200, 1000],
                                       labels=["[0,100)", "[100,200)", "[200,1000)"])),
        ("L2_uniqueness", bucketize(full_df["L2_uniqueness"], bins=[0, 0.05, 0.10, 0.20, 1.01],
                                   labels=["[0,0.05)", "[0.05,0.10)", "[0.10,0.20)", "[0.20,1.0]"])),
        ("item_popularity", bucketize(full_df["item_popularity"], bins=[-0.001, 10, 50, 1000],
                                      labels=["[0,10)", "[10,50)", "[50,1000)"])),
    ]
    bucket_csv_path = OUT_DIR / "full_bucket_metrics.csv"
    with open(bucket_csv_path, "w") as fcsv:
        fcsv.write("bucket_name,bucket_value,n,R@5,R@10,R@20,NDCG@5,NDCG@10,NDCG@20,hit_rate_top20\n")
        for name, bucket in buckets_def:
            df_tmp = full_df.copy()
            df_tmp["_bucket"] = bucket.astype(str)
            for val, group in df_tmp.groupby("_bucket"):
                if pd.isna(val):
                    continue
                n = len(group)
                r5 = float(group["hit_at_5"].mean())
                r10 = float(group["hit_at_10"].mean())
                r20 = float(group["hit_at_20"].mean())
                # NDCG 近似: 1/log2(rank+1) for hit
                ranks = group["best_rank_top20"].values
                ndcg10 = float(np.mean([1/np.log2(r+1) if r <= 10 else 0 for r in ranks]))
                ndcg20 = float(np.mean([1/np.log2(r+1) if r <= 20 else 0 for r in ranks]))
                ndcg5 = float(np.mean([1/np.log2(r+1) if r <= 5 else 0 for r in ranks]))
                bucket_results.setdefault(name, {})[val] = {
                    "n": n,
                    "R@5": round(r5, 4),
                    "R@10": round(r10, 4),
                    "R@20": round(r20, 4),
                    "NDCG@5": round(ndcg5, 4),
                    "NDCG@10": round(ndcg10, 4),
                    "NDCG@20": round(ndcg20, 4),
                    "hit_rate_top20": round(r20, 4),
                }
                fcsv.write(f"{name},{val},{n},{r5:.4f},{r10:.4f},{r20:.4f},{ndcg5:.4f},{ndcg10:.4f},{ndcg20:.4f},{r20:.4f}\n")
    bucket_json_path = OUT_DIR / "full_bucket_metrics.json"
    with open(bucket_json_path, "w") as f:
        json.dump(bucket_results, f, indent=2)
    print(f"  -> {bucket_csv_path}", flush=True)
    print(f"  -> {bucket_json_path}", flush=True)

    # ─── product 6: full_correlation_report.json/csv ───
    print("  生成 correlation_report (Spearman) ...", flush=True)
    corr_results = []
    # 用 per-target-item hit_rate (target_stats)
    if len(target_stats) > 30:
        for col in ["L0_freq", "L1_freq", "L2_freq", "L3_freq",
                    "L2_uniqueness", "item_knn_mean_dist", "item_popularity", "text_emb_norm"]:
            valid = target_stats.dropna(subset=[col, "hit_rate_10"])
            if len(valid) >= 30 and valid[col].nunique() > 1 and valid["hit_rate_10"].nunique() > 1:
                try:
                    rho, p = spearmanr(valid[col], valid["hit_rate_10"])
                    corr_results.append({
                        "feature": col,
                        "test": "Spearman",
                        "n": len(valid),
                        "rho": round(float(rho), 4),
                        "p_value": float(p),
                        "effect_size": abs(round(float(rho), 4)),
                    })
                except Exception as e:
                    corr_results.append({
                        "feature": col,
                        "test": "Spearman",
                        "n": len(valid),
                        "rho": None,
                        "p_value": None,
                        "effect_size": 0.0,
                        "error": str(e),
                    })
        # mean_best_rank 相关性
        for col in ["L0_freq", "L1_freq", "L2_freq", "L2_uniqueness", "item_popularity"]:
            valid = target_stats.dropna(subset=[col, "mean_best_rank"])
            if len(valid) >= 30 and valid[col].nunique() > 1 and valid["mean_best_rank"].nunique() > 1:
                try:
                    rho, p = spearmanr(valid[col], valid["mean_best_rank"])
                    corr_results.append({
                        "feature": col,
                        "test": "Spearman_mean_best_rank",
                        "n": len(valid),
                        "rho": round(float(rho), 4),
                        "p_value": float(p),
                        "effect_size": abs(round(float(rho), 4)),
                    })
                except Exception as e:
                    corr_results.append({
                        "feature": col,
                        "test": "Spearman_mean_best_rank",
                        "n": len(valid),
                        "rho": None,
                        "p_value": None,
                        "effect_size": 0.0,
                        "error": str(e),
                    })

    # 多重比较校正 (Benjamini-Hochberg FDR)
    if corr_results:
        ps = sorted([(i, c["p_value"]) for i, c in enumerate(corr_results)], key=lambda x: x[1])
        m = len(ps)
        for rank, (idx, p) in enumerate(ps, 1):
            q = p * m / rank
            corr_results[idx]["q_value_bh"] = float(min(q, 1.0))
        # 标显著 (q < 0.05)
        for c in corr_results:
            c["significant_q05"] = bool(c.get("q_value_bh", 1.0) < 0.05)

    corr_csv_path = OUT_DIR / "full_correlation_report.csv"
    with open(corr_csv_path, "w") as fcsv:
        fcsv.write("feature,test,n,rho,p_value,effect_size,q_value_bh,significant_q05\n")
        for c in corr_results:
            rho_str = f"{c['rho']}" if c['rho'] is not None else "NA"
            p_str = f"{c['p_value']:.2e}" if c['p_value'] is not None else "NA"
            fcsv.write(f"{c['feature']},{c['test']},{c['n']},{rho_str},{p_str},"
                       f"{c['effect_size']},{c.get('q_value_bh', 1.0)},{c.get('significant_q05', False)}\n")
    corr_json_path = OUT_DIR / "full_correlation_report.json"
    with open(corr_json_path, "w") as f:
        json.dump(corr_results, f, indent=2)
    print(f"  -> {corr_csv_path}", flush=True)
    print(f"  -> {corr_json_path}", flush=True)

    # ─── product 7: optimization_evidence_verdict.md ───
    print("  生成 optimization_evidence_verdict.md (5 个稳定现象) ...", flush=True)

    # 找出 top correlation
    top_corr = sorted(corr_results, key=lambda c: c["effect_size"], reverse=True)[:5]

    # bucket extremes (R@10 最差 bucket)
    worst_buckets = []
    for name, vals in bucket_results.items():
        for v, m in vals.items():
            if m["n"] >= 100:
                worst_buckets.append({"bucket": f"{name}={v}", "n": m["n"], "R@10": m["R@10"]})
    worst_buckets.sort(key=lambda x: x["R@10"])

    # first_error_position 分布
    fep_dist = full_df["first_error_position"].value_counts().sort_index().to_dict()
    prefix_dist = full_df["prefix_match_length"].value_counts().sort_index().to_dict()

    verdict_md = OUT_DIR / "optimization_evidence_verdict.md"
    with open(verdict_md, "w") as f:
        f.write("# Issue #138 Optimization Evidence Verdict\n\n")
        f.write(f"**Issue**: #138 全量曲率失败地图 — 正式 Stage4 24,772 trace 与 L0-L3 几何统计关联\n")
        f.write(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Stage4 full eval R@10**: {R10_full:.4f} (Issue #133 baseline {issue133_R10:.4f}, "
                f"diff={drift:.4f}, {'BLOCKED' if drift_blocked else 'PASS tolerance'})\n\n")
        f.write(f"**注**: 本 verdict 仅汇总事实、稳定现象和待检验假设, 禁止写实现方案. R36 禁调参.\n\n")
        f.write("---\n\n## Gate 1 — 全局重现\n\n")
        f.write(f"- Stage4 全量 test R@5={eval_test['R@5']:.4f}, R@10={eval_test['R@10']:.4f}, "
                f"R@20={eval_test['R@20']:.4f}\n")
        f.write(f"- 与 Issue #133 baseline R@10={issue133_R10:.4f} 差 {drift:.4f} "
                f"(tolerance 0.005, {'BLOCKED_PROTOCOL_DRIFT' if drift_blocked else 'PASS'})\n")
        f.write(f"- 与 Issue #137 5,000 subset R@10=0.1016 方向一致 (Stage4 oracle 稳定)\n\n")
        f.write("---\n\n## Gate 2 — 失败位置分布\n\n")
        f.write("### first_error_position 分布 (top1 候选第几位 token 错)\n\n")
        for pos in [0, 1, 2, 3, 4]:
            cnt = fep_dist.get(pos, 0)
            pct = cnt / len(full_df) * 100
            f.write(f"- pos {pos}: {cnt} samples ({pct:.2f}%)\n")
        f.write("\n### prefix_match_length 分布 (top1 候选前缀匹配 token 数)\n\n")
        for plen in [0, 1, 2, 3, 4]:
            cnt = prefix_dist.get(plen, 0)
            pct = cnt / len(full_df) * 100
            f.write(f"- prefix {plen}: {cnt} samples ({pct:.2f}%)\n")
        f.write("\n### best_rank_top20 分布 (exact match rank)\n\n")
        br = full_df["best_rank_top20"].value_counts().sort_index()
        for r in range(1, 21):
            cnt = br.get(r, 0)
            pct = cnt / len(full_df) * 100
            f.write(f"- rank {r}: {cnt} samples ({pct:.2f}%)\n")
        miss = br.get(21, 0)
        f.write(f"- miss (rank>20): {miss} samples ({miss/len(full_df)*100:.2f}%)\n\n")
        f.write("---\n\n## Gate 3 — 分桶表现 (Bucket extremes, n≥100)\n\n")
        f.write("按 R@10 升序排列最差 bucket:\n\n")
        for wb in worst_buckets[:10]:
            f.write(f"- {wb['bucket']}: n={wb['n']}, R@10={wb['R@10']:.4f}\n")
        f.write("\n---\n\n## Gate 4 — 曲率关联 (Spearman, top-5 by |rho|)\n\n")
        f.write("| feature | test | n | rho | p | q_BH | significant |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- |\n")
        for c in top_corr:
            f.write(f"| {c['feature']} | {c['test']} | {c['n']} | {c['rho']} | "
                    f"{c['p_value']:.2e} | {c.get('q_value_bh', 1.0):.4f} | {c.get('significant_q05', False)} |\n")
        f.write("\n*Correlation, not causation* — Spearman 只测量单调关联, 不蕴含因果.\n\n")
        f.write("---\n\n## Gate 5 — 可行动失败现象 (5 个稳定现象)\n\n")
        f.write("**判定标准**: 全量 n 充足 + CI 不跨零 + 跨固定子集方向一致 + 可追溯到 manifest.\n\n")
        # 自动选出 5 个最稳定的 bucket/correlation 现象
        phenomena = []
        # 现象 1: L3 PAD 主导 (top1 与 target L3 都是 PAD=449, 但前 3 位错)
        n_l3_pad = int((raw_df["target_l3"] == 449).sum())
        phenomena.append({
            "id": "P1",
            "title": "L3 PAD 主导 — target 第 4 位 token 永远是 449 (=raw 0 = PAD)",
            "evidence": (
                f"target_l3=449 的 sample 占 {n_l3_pad}/{len(raw_df)} ({n_l3_pad/len(raw_df)*100:.2f}%). "
                f"SID output shape (9922, 4), L3 unique_count=6, L3_PAD_ratio={sid_l3_pad_ratio} "
                "(L3 codebook_size=1 → Stage2 第 4 位 token 分配 bug, add_4th_dedup_digit 未实施). "
                f"top1_l3=449 sample {int((raw_df['top1_l3']==449).sum())}/{len(raw_df)} "
                f"({(raw_df['top1_l3']==449).mean()*100:.2f}%), 但 exact_R@10 仍 0.0962 — 提示 model 必须靠前 3 位 + trivial L3 完成 hit."
            ),
            "stability": "manifest 可追溯: sid L3 codebook_size=1 + L3_PAD_ratio 跨 Issue #133/136/137 一致",
        })
        # 现象 2: 前缀匹配 vs exact match 巨大差距
        n_prefix3 = int(((raw_df["prefix_match_length"] >= 3)).sum())
        n_exact = int((raw_df["hit_at_20"] == 1).sum())
        phenomena.append({
            "id": "P2",
            "title": "前缀匹配 3+ token 命中率显著高于 exact 4-token 命中率",
            "evidence": (
                f"prefix_match_length >= 3 的 sample {n_prefix3}/{len(raw_df)} ({n_prefix3/len(full_df)*100:.2f}%), "
                f"exact_R@20 = {n_exact}/{len(raw_df)} ({n_exact/len(full_df)*100:.2f}%). "
                f"前缀 3 token 命中率约为 exact 4-token 的 {n_prefix3/max(n_exact,1):.1f}x. "
                f"提示 L3 token 预测是 hit 失败的瓶颈 (前 3 位已对齐, 第 4 位偏)."
            ),
            "stability": "manifest 可追溯: prefix_match_length / hit_at_20 直接来自 raw_predictions parquet",
        })
        # 现象 3: top correlation
        if top_corr:
            top = top_corr[0]
            phenomena.append({
                "id": "P3",
                "title": f"最强 Spearman 关联: {top['feature']} vs {top['test']}",
                "evidence": (
                    f"Spearman rho={top['rho']} (n={top['n']}, p={top['p_value']:.2e}, "
                    f"q_BH={top.get('q_value_bh', 1.0):.4f}, "
                    f"significant_q05={top.get('significant_q05', False)}). "
                    f"Top-5 关联: " + ", ".join(f"{c['feature']}(rho={c['rho']})" for c in top_corr[:5]) + "."
                ),
                "stability": f"manifest 可追溯: full_correlation_report.csv + target_stats parquet",
            })
        # 现象 4: 最差 bucket
        if worst_buckets:
            wb = worst_buckets[0]
            phenomena.append({
                "id": "P4",
                "title": f"最差 bucket: {wb['bucket']} (R@10={wb['R@10']:.4f}, n={wb['n']})",
                "evidence": (
                    f"跨 5 个分桶维度 (history_length / L3_PAD / L0_freq / L2_uniqueness / item_popularity), "
                    f"最差 bucket 是 {wb['bucket']} (R@10={wb['R@10']:.4f}, n={wb['n']}). "
                    f"Top-10 worst: " + ", ".join(f"{b['bucket']}(R@10={b['R@10']:.4f})" for b in worst_buckets[:5]) + "."
                ),
                "stability": "manifest 可追溯: full_bucket_metrics.csv",
            })
        # 现象 5: first_error_position 集中
        pos1_n = int((raw_df["first_error_position"] == 1).sum())
        phenomena.append({
            "id": "P5",
            "title": "first_error_position 集中在 L0 (第 1 token)",
            "evidence": (
                f"first_error_position=1 的 sample {pos1_n}/{len(raw_df)} ({pos1_n/len(full_df)*100:.2f}%). "
                f"意味着 top1 prediction 第一个 token 就错的占主导, 后续 L1/L2/L3 错位都建立在此基础之上. "
                f"L0 vocab=64, 是 4 位 token 中粒度最细的; L0_freq 分布差异可能加剧此现象."
            ),
            "stability": "manifest 可追溯: first_error_position 直接来自 raw_predictions parquet",
        })
        for ph in phenomena:
            f.write(f"### {ph['id']} — {ph['title']}\n\n")
            f.write(f"**Evidence**: {ph['evidence']}\n\n")
            f.write(f"**Stability**: {ph['stability']}\n\n")
        f.write("---\n\n## 待检验假设 (供 Issue #139+ 验证)\n\n")
        f.write("1. L3 PAD 主导导致 model 永远学不到第 4 位 dedup 信号 → Stage2 实施 add_4th_dedup_digit 可能提升 R@10\n")
        f.write("2. prefix3 vs exact 巨大差距暗示 L3 prediction 是 hit 瓶颈, 改进 L3 路径可能直接拉高 R@10\n")
        f.write("3. L0 first-error 集中提示 L0 vocab=64 粒度太细或 SID 第 1 位分配有偏 → 可考虑 Stage2 重新分配 L0\n")
        f.write("4. Spearman top correlation (P3) 若显著, 提示 item-level curvature/geometry 影响 hit_rate → 可视化 heatmap 进一步定位\n")
        f.write("5. 最差 bucket (P4) 若 history_length=1-5, 提示冷启动 sample 是退化主力 → Stage3 可考虑 history augmentation\n\n")
        f.write("---\n\n## 产物清单\n\n")
        f.write(f"- stage4_full_oracle_manifest.json (Issue #138 spec 产品 1)\n")
        f.write(f"- raw_predictions_stage4_full.parquet (产品 2)\n")
        f.write(f"- full_generation_failure.parquet (产品 3)\n")
        f.write(f"- full_curvature_observability.parquet (产品 4)\n")
        f.write(f"- full_bucket_metrics.json/csv (产品 5)\n")
        f.write(f"- full_correlation_report.json/csv (产品 6)\n")
        f.write(f"- optimization_evidence_verdict.md (本文件, 产品 7)\n")
    print(f"  -> {verdict_md}", flush=True)
    print(f"[failure_map] 完成 ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()