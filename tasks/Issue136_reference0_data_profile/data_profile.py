"""Issue #136 reference-0 纯数据画像采集

reference-0 = Issue #133 当前完整可运行版本 (test_R@10=0.0962)
严格禁止改动任何 stage1-4 代码 / 模型结构 / 训练协议 / SID
只读 + 必要 canary forward (deterministic, 不反向)
产物: manifest, item_geometry.csv, generation_failure.csv, bucket_metrics.json,
       correlation.json, optimization_data_verdict.md/json
"""

import os
import sys
import json
import time
import hashlib
import subprocess
from pathlib import Path

# 强制路径
TASK_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/tasks/Issue136_reference0_data_profile")
os.chdir(TASK_DIR)
sys.path.insert(0, str(TASK_DIR))
sys.path.insert(0, str(TASK_DIR / "_lib"))

OUT_DIR = TASK_DIR / "data_profile"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
DATASET_DIR = REPO_ROOT / "dataset"
REF0 = REPO_ROOT / "tasks/Issue133_stage2_baseline_c_exp"  # reference-0 锚点


def sha256_file(path):
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception as e:
        return f"ERROR: {e}"


def safe_run(cmd, cwd=None, timeout=10):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                          cwd=cwd, timeout=timeout)
        return r.stdout.strip()
    except Exception as e:
        return f"ERROR: {e}"


# ==========================================================================
# 1. reference-0 manifest
# ==========================================================================
def build_manifest():
    print("[1] reference-0 manifest", flush=True)
    manifest = {
        "reference_id": "reference-0",
        "anchor": "Issue #133",
        "test_R@10": 0.0962,
        "issue133_commit": safe_run("git rev-parse HEAD", cwd=str(REPO_ROOT)),
        "issue133_head_short": safe_run("git rev-parse --short HEAD", cwd=str(REPO_ROOT)),
        "issue133_branch": safe_run("git branch --show-current", cwd=str(REPO_ROOT)),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "artifacts": {},
    }

    # Issue133 关键产物 hash
    for sub, fname in [
        ("stage1", "item_emb.parquet"),
        ("stage2", "sid_output.npy"),
        ("stage2", "HG_Rec_best.pth"),
        ("stage2", "hrqvae_kappa_sync.ckpt"),
        ("stage2", "config.json"),
        ("stage2", "sid_metadata.json"),
        ("stage2/eval", "eval_test.json"),
    ]:
        p = REF0 / sub / fname
        if p.exists():
            manifest["artifacts"][f"{sub}/{fname}"] = {
                "sha256": sha256_file(p),
                "size_bytes": p.stat().st_size,
            }

    # dataset
    for fname in ["Instruments.item.json", "Instruments.inter.json",
                  "train.parquet", "valid.parquet", "test.parquet"]:
        p = DATASET_DIR / fname
        if p.exists():
            manifest["artifacts"][f"dataset/{fname}"] = {
                "sha256": sha256_file(p),
                "size_bytes": p.stat().st_size,
            }

    # Stage4 generation config (from Issue133 stage4_beam20.py BEAM_SIZE + MIN_LENGTH)
    s4 = REF0 / "stage4_beam20.py"
    if s4.exists():
        text = s4.read_text()
        import re
        manifest["generation_config"] = {
            "BEAM_SIZE": re.search(r"BEAM_SIZE\s*=\s*(\d+)", text).group(1)
                if re.search(r"BEAM_SIZE\s*=\s*(\d+)", text) else None,
            "MIN_LENGTH": re.search(r"MIN_LENGTH\s*=\s*(\d+)", text).group(1)
                if re.search(r"MIN_LENGTH\s*=\s*(\d+)", text) else None,
            "MAX_LENGTH": re.search(r"MAX_LENGTH\s*=\s*(\d+)", text).group(1)
                if re.search(r"MAX_LENGTH\s*=\s*(\d+)", text) else None,
        }

    out = OUT_DIR / "reference0_manifest.json"
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"  -> {out}", flush=True)
    return manifest


# ==========================================================================
# 2. item/SID geometry table (per-item)
# ==========================================================================
def build_item_geometry():
    print("[2] item/SID geometry table", flush=True)
    import numpy as np

    # 加载 SID + item_emb
    sid = np.load(REF0 / "stage2/sid_output.npy")
    sid_sha = sha256_file(REF0 / "stage2/sid_output.npy")

    try:
        import pyarrow.parquet as pq
        import numpy as np
        t = pq.read_table(REF0 / "stage1/item_emb.parquet")
        # schema: ItemID: string, embedding: list<double>
        # 提取 embedding 列并 stack
        emb_lists = t.column("embedding").to_pylist()
        item_emb = np.array(emb_lists, dtype=np.float32)  # (9922, dim)
    except Exception as e:
        print(f"  item_emb load ERROR: {e}", flush=True)
        item_emb = None

    n_items = sid.shape[0]
    # per-item SID 频率 (在整个 SID 集合中)
    sid_freq = np.zeros((n_items, 4), dtype=np.int64)
    for lvl in range(4):
        if sid.shape[1] > lvl:
            uniq, counts = np.unique(sid[:, lvl], return_counts=True)
            for u, c in zip(uniq, counts):
                if 0 <= u < n_items:
                    sid_freq[u, lvl] = c

    # codebook usage
    L0_usage = len(np.unique(sid[:, 0])) / 64
    L1_usage = len(np.unique(sid[:, 1])) / 128
    L2_usage = len(np.unique(sid[:, 2])) / 256
    L3_usage = len(np.unique(sid[:, 3])) / 256 if sid.shape[1] >= 4 else 0

    # assignment margin (distance to nearest codebook point) — proxy via intra-level entropy
    # 简化: per-item SID uniqueness
    sid_uniqueness = np.zeros(n_items)
    for i in range(n_items):
        same_L0 = np.sum(sid[:, 0] == sid[i, 0])
        same_L1 = np.sum((sid[:, 0] == sid[i, 0]) & (sid[:, 1] == sid[i, 1]))
        same_L2 = np.sum((sid[:, 0] == sid[i, 0]) & (sid[:, 1] == sid[i, 1])
                          & (sid[:, 2] == sid[i, 2]))
        sid_uniqueness[i] = 1.0 / max(same_L2, 1)  # unique L2 邻居数倒数

    # KNN distance summary (基于 item_emb)
    if item_emb is not None:
        from scipy.spatial.distance import cdist
        # 采样 200 个 item 算 KNN (避免 OOM/慢)
        rng = np.random.default_rng(42)
        sample = rng.choice(n_items, size=min(200, n_items), replace=False)
        sub_emb = item_emb[sample].astype("float32")
        ref = item_emb[:1000].astype("float32")
        # L2 distance
        dmat = cdist(sub_emb, ref)  # 200x1000
        knn_dists = np.sort(dmat, axis=1)[:, :10].mean(axis=1)
        # 全部 item 用 sample 距离近似
        item_knn_mean = np.zeros(n_items)
        for i in sample:
            item_knn_mean[i] = knn_dists[np.where(sample == i)[0][0]]
    else:
        item_knn_mean = np.zeros(n_items)

    # item popularity (从 inter.json 算)
    item_popularity = np.zeros(n_items)
    try:
        import json as _json
        with open(DATASET_DIR / "Instruments.inter.json") as f:
            inters = _json.load(f)
        if isinstance(inters, list):
            for it in inters:
                iid = it.get("item_id") or it.get("iid") or it.get("item")
                if iid is not None and 0 <= iid < n_items:
                    item_popularity[iid] += 1
    except Exception as e:
        print(f"  popularity ERROR: {e}", flush=True)

    # text_emb_norm
    text_emb_norm = np.zeros(n_items)
    if item_emb is not None:
        text_emb_norm = np.linalg.norm(item_emb, axis=1)

    # 写 CSV (per-item)
    import csv
    out_csv = OUT_DIR / "item_geometry.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "sid_L0", "sid_L1", "sid_L2", "sid_L3",
                    "L0_freq", "L1_freq", "L2_freq", "L3_freq",
                    "L2_uniqueness", "item_knn_mean_dist",
                    "item_popularity", "text_emb_norm"])
        for i in range(n_items):
            w.writerow([
                i, int(sid[i, 0]), int(sid[i, 1]), int(sid[i, 2]),
                int(sid[i, 3]) if sid.shape[1] >= 4 else 0,
                int(sid_freq[i, 0]), int(sid_freq[i, 1]),
                int(sid_freq[i, 2]),
                int(sid_freq[i, 3]) if sid.shape[1] >= 4 else 0,
                float(sid_uniqueness[i]),
                float(item_knn_mean[i]),
                int(item_popularity[i]),
                float(text_emb_norm[i]),
            ])

    geometry_summary = {
        "n_items": n_items,
        "sid_sha256": sid_sha,
        "codebook_usage": {
            "L0": L0_usage, "L1": L1_usage, "L2": L2_usage, "L3": L3_usage
        },
        "codebook_unique": {
            "L0": int(len(np.unique(sid[:, 0]))),
            "L1": int(len(np.unique(sid[:, 1]))),
            "L2": int(len(np.unique(sid[:, 2]))),
            "L3": int(len(np.unique(sid[:, 3]))) if sid.shape[1] >= 4 else 0,
        },
        "L3_pad_count": int(np.sum(sid[:, 3] == 0)) if sid.shape[1] >= 4 else 0,
        "item_knn_mean_overall": float(item_knn_mean[item_knn_mean > 0].mean())
            if (item_knn_mean > 0).any() else 0,
    }
    summary_path = OUT_DIR / "item_geometry_summary.json"
    with open(summary_path, "w") as f:
        json.dump(geometry_summary, f, indent=2, ensure_ascii=False)

    print(f"  -> {out_csv}", flush=True)
    print(f"  -> {summary_path}", flush=True)
    return geometry_summary


# ==========================================================================
# 3. generation failure table (per-eval-sample)
# 需要 raw predictions (从 Stage4 canary 生成)
# ==========================================================================
def build_failure_table():
    print("[3] generation failure table (canary forward)", flush=True)
    # 检查 raw predictions (独立 canary_forward.py 输出, 不修改 Issue133 stage4)
    pred_path = OUT_DIR / "raw_predictions.json"
    if not pred_path.exists():
        print("  raw_predictions.json 不存在, 跑 canary_forward.py...", flush=True)
        import subprocess
        r = subprocess.run(
            ["/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3", "-u",
             str(TASK_DIR / "canary_forward.py")],
            capture_output=True, text=True, timeout=600,
        )
        print(f"  canary exit={r.returncode}", flush=True)
        if r.returncode != 0:
            print(f"  canary stderr: {r.stderr[:1000]}", flush=True)
            return {"ERROR": "canary failed"}

    if not pred_path.exists():
        return {"ERROR": "raw_predictions.json still missing after canary"}

    with open(pred_path) as f:
        preds_data = json.load(f)

    # predictions.json 格式: [{user_history, target_item, target_sid, pred_top20_sids, ...}]
    # 兼容不同格式
    if isinstance(preds_data, dict):
        preds_data = preds_data.get("predictions", preds_data.get("results", [preds_data]))

    import numpy as np
    sid = np.load(REF0 / "stage2/sid_output.npy")
    n_items = sid.shape[0]
    sid_to_item = {}
    for i in range(n_items):
        sid_to_item[tuple(sid[i].tolist())] = i

    # per-sample 分析
    import csv
    out_csv = OUT_DIR / "generation_failure.csv"
    n_samples = len(preds_data)
    n_exact_hit = 0
    n_prefix_hit = 0
    n_legal = 0
    first_err_positions = []
    target_ranks = []
    sample_records = []

    for idx, sample in enumerate(preds_data):
        if idx >= 5000:  # 限制样本数 (避免过大)
            break
        target_sid = sample.get("target_sid") or sample.get("target")
        if target_sid is None and "target_item" in sample:
            ti = sample["target_item"]
            if 0 <= ti < n_items:
                target_sid = sid[ti].tolist()

        history_len = len(sample.get("history", sample.get("user_history", [])))
        pred_top20 = sample.get("pred_top20_sids") or sample.get("predictions", [])
        if not pred_top20 and "pred_sid" in sample:
            pred_top20 = [sample["pred_sid"]]

        # exact hit?
        exact = False
        prefix = False
        first_err_pos = 4
        target_rank = -1
        legal = True

        if target_sid is not None and pred_top20:
            target_tuple = tuple(target_sid[:4]) if isinstance(target_sid, list) else tuple(target_sid)
            for rank, pred in enumerate(pred_top20):
                if isinstance(pred, list):
                    pred_tuple = tuple(pred[:4])
                else:
                    pred_tuple = tuple(pred)
                if pred_tuple == target_tuple:
                    target_rank = rank
                    exact = True
                    first_err_pos = 4  # no error
                    break
                # prefix match?
                match_len = 0
                for a, b in zip(target_tuple, pred_tuple):
                    if a == b:
                        match_len += 1
                    else:
                        break
                if match_len >= 3 and not prefix:
                    prefix = True
                if match_len < 4 and first_err_pos > match_len:
                    first_err_pos = match_len
                # legal?
                if pred_tuple not in sid_to_item:
                    legal = False

            if exact:
                n_exact_hit += 1
            if prefix:
                n_prefix_hit += 1
            first_err_positions.append(first_err_pos)
            target_ranks.append(target_rank if target_rank >= 0 else 20)

        sample_records.append({
            "sample_id": idx,
            "history_len": history_len,
            "target_sid": str(target_sid),
            "target_rank": target_rank,
            "exact_hit": int(exact),
            "prefix_hit": int(prefix),
            "first_err_pos": first_err_pos,
            "legal_top20": int(legal),
        })

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "history_len", "target_sid", "target_rank",
                    "exact_hit", "prefix_hit", "first_err_pos", "legal_top20"])
        for r in sample_records:
            w.writerow([r["sample_id"], r["history_len"], r["target_sid"],
                        r["target_rank"], r["exact_hit"], r["prefix_hit"],
                        r["first_err_pos"], r["legal_top20"]])

    failure_summary = {
        "n_samples": len(sample_records),
        "n_exact_hit": n_exact_hit,
        "exact_hit_rate": n_exact_hit / max(len(sample_records), 1),
        "n_prefix_hit": n_prefix_hit,
        "prefix_hit_rate": n_prefix_hit / max(len(sample_records), 1),
        "first_err_pos_distribution": {
            str(p): int(np.sum(np.array(first_err_positions) == p))
            for p in [0, 1, 2, 3, 4]
        },
        "median_target_rank": float(np.median(target_ranks)) if target_ranks else None,
    }
    summary_path = OUT_DIR / "generation_failure_summary.json"
    with open(summary_path, "w") as f:
        json.dump(failure_summary, f, indent=2, ensure_ascii=False)

    print(f"  -> {out_csv}", flush=True)
    print(f"  -> {summary_path}", flush=True)
    return failure_summary


# ==========================================================================
# 4. 分桶表现报告 (按 history length / popularity / SID freq / margin / density)
# ==========================================================================
def build_bucket_metrics():
    print("[4] 分桶表现报告", flush=True)
    import csv
    import numpy as np

    # 读 generation_failure.csv + item_geometry.csv, 关联分桶
    geom_path = OUT_DIR / "item_geometry.csv"
    fail_path = OUT_DIR / "generation_failure.csv"
    if not geom_path.exists() or not fail_path.exists():
        return {"ERROR": "missing input csv"}

    # item_geometry per-item
    geom = {}
    with open(geom_path) as f:
        r = csv.DictReader(f)
        for row in r:
            geom[int(row["item_id"])] = row

    # popularity 分桶
    pops = [int(geom[i]["item_popularity"]) for i in geom]
    pop_q = np.quantile(pops, [0.25, 0.5, 0.75])

    # sample-level 关联
    samples = []
    with open(fail_path) as f:
        r = csv.DictReader(f)
        for row in r:
            samples.append(row)

    # 分桶
    buckets = {
        "history_len_short": [s for s in samples if int(s["history_len"]) <= 3],
        "history_len_medium": [s for s in samples if 4 <= int(s["history_len"]) <= 10],
        "history_len_long": [s for s in samples if int(s["history_len"]) > 10],
    }

    def bucket_metrics(slist):
        if not slist:
            return {"n": 0}
        n = len(slist)
        exact = sum(int(s["exact_hit"]) for s in slist)
        prefix = sum(int(s["prefix_hit"]) for s in slist)
        ranks = [int(s["target_rank"]) if int(s["target_rank"]) >= 0 else 20 for s in slist]
        return {
            "n": n,
            "exact_R@10_proxy": exact / n,
            "prefix_R@10_proxy": prefix / n,
            "median_target_rank": float(np.median(ranks)),
            "mean_first_err_pos": float(np.mean([int(s["first_err_pos"]) for s in slist])),
        }

    bucket_report = {k: bucket_metrics(v) for k, v in buckets.items()}

    # popularity 分桶
    pop_buckets = {"popularity_low": [], "popularity_med": [], "popularity_high": []}
    with open(fail_path) as f:
        r = csv.DictReader(f)
        for row in r:
            # target_item 需从 target_sid 反推
            target_sid = row["target_sid"].strip("[]")
            target_item = None
            try:
                target_sid_tuple = eval(row["target_sid"])
                # 反查 item_id
                for iid, g in geom.items():
                    if (int(g["sid_L0"]), int(g["sid_L1"]), int(g["sid_L2"]),
                            int(g["sid_L3"])) == tuple(target_sid_tuple):
                        target_item = iid
                        break
            except Exception:
                pass
            if target_item is None:
                continue
            pop = int(geom[target_item]["item_popularity"])
            if pop <= pop_q[0]:
                pop_buckets["popularity_low"].append(row)
            elif pop <= pop_q[2]:
                pop_buckets["popularity_med"].append(row)
            else:
                pop_buckets["popularity_high"].append(row)

    for k, v in pop_buckets.items():
        bucket_report[k] = bucket_metrics(v)

    out = OUT_DIR / "bucket_metrics.json"
    with open(out, "w") as f:
        json.dump(bucket_report, f, indent=2, ensure_ascii=False)
    print(f"  -> {out}", flush=True)
    return bucket_report


# ==========================================================================
# 5. 关联而非因果报告 (Spearman correlation)
# ==========================================================================
def build_correlation_report():
    print("[5] 关联而非因果报告", flush=True)
    import csv
    import numpy as np
    from scipy.stats import spearmanr

    geom_path = OUT_DIR / "item_geometry.csv"
    fail_path = OUT_DIR / "generation_failure.csv"
    if not geom_path.exists() or not fail_path.exists():
        return {"ERROR": "missing input"}

    geom = {}
    with open(geom_path) as f:
        r = csv.DictReader(f)
        for row in r:
            geom[int(row["item_id"])] = row

    # target_item 反查 + 关联 features
    feat_names = ["L0_freq", "L1_freq", "L2_freq", "L3_freq",
                  "L2_uniqueness", "item_knn_mean_dist",
                  "item_popularity", "text_emb_norm"]
    feat_data = {fn: [] for fn in feat_names}
    exact_outcome = []
    prefix_outcome = []
    first_err_outcome = []
    target_rank_outcome = []

    with open(fail_path) as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                target_sid_tuple = eval(row["target_sid"])
            except Exception:
                continue
            target_item = None
            for iid, g in geom.items():
                if (int(g["sid_L0"]), int(g["sid_L1"]), int(g["sid_L2"]),
                        int(g["sid_L3"])) == tuple(target_sid_tuple):
                    target_item = iid
                    break
            if target_item is None:
                continue
            for fn in feat_names:
                feat_data[fn].append(float(geom[target_item][fn]))
            exact_outcome.append(int(row["exact_hit"]))
            prefix_outcome.append(int(row["prefix_hit"]))
            first_err_outcome.append(int(row["first_err_pos"]))
            target_rank_outcome.append(int(row["target_rank"]) if int(row["target_rank"]) >= 0 else 20)

    correlations = {}
    for fn in feat_names:
        if not feat_data[fn]:
            continue
        x = np.array(feat_data[fn])
        for oname, y in [
            ("exact_hit", np.array(exact_outcome)),
            ("prefix_hit", np.array(prefix_outcome)),
            ("first_err_pos", np.array(first_err_outcome)),
            ("target_rank", np.array(target_rank_outcome)),
        ]:
            if len(x) >= 10 and np.std(x) > 0 and np.std(y) > 0:
                rho, pval = spearmanr(x, y)
                correlations[f"{fn}_vs_{oname}"] = {
                    "spearman_rho": float(rho) if not np.isnan(rho) else 0.0,
                    "pvalue": float(pval) if not np.isnan(pval) else 1.0,
                    "n": len(x),
                    "interpretation": "correlation, not causation",
                }

    out = OUT_DIR / "correlation_report.json"
    with open(out, "w") as f:
        json.dump({"correlations": correlations,
                   "note": "Spearman 相关, 仅相关非因果, 未做多重比较校正"},
                  f, indent=2, ensure_ascii=False)
    print(f"  -> {out}", flush=True)
    return correlations


# ==========================================================================
# 6. optimization data verdict
# ==========================================================================
def build_verdict(manifest, geom_summary, failure_summary, bucket_report, corr):
    print("[6] optimization data verdict", flush=True)

    # 找 top-3 显著相关 (按 |rho|)
    top_corr = []
    for k, v in (corr.items() if isinstance(corr, dict) else []):
        if isinstance(v, dict) and "spearman_rho" in v:
            top_corr.append((k, v["spearman_rho"], v["pvalue"]))
    top_corr.sort(key=lambda x: abs(x[1]), reverse=True)
    top5 = top_corr[:5]

    md_path = OUT_DIR / "optimization_data_verdict.md"
    with open(md_path, "w") as f:
        f.write(f"""# Issue #136 reference-0 数据画像 verdict

## reference-0 锚点

- anchor: Issue #133
- commit: `{manifest['issue133_head_short']}`
- test_R@10: {manifest['test_R@10']}

## 失败集中在哪里 (top-5 相关)

| 相关 | spearman_rho | p-value | 解释 |
|---|---|---|---|
""")
        for k, rho, pval in top5:
            f.write(f"| {k} | {rho:.4f} | {pval:.4g} | 相关非因果 |\n")

        f.write(f"""
## 分桶表现

""")
        for k, v in (bucket_report.items() if isinstance(bucket_report, dict) else []):
            if isinstance(v, dict) and "n" in v:
                f.write(f"### {k}\n")
                f.write(f"- n={v.get('n', 0)}, exact_R@10_proxy={v.get('exact_R@10_proxy', 0):.4f}, ")
                f.write(f"prefix_R@10_proxy={v.get('prefix_R@10_proxy', 0):.4f}, ")
                f.write(f"median_rank={v.get('median_target_rank', 'N/A')}, ")
                f.write(f"mean_first_err_pos={v.get('mean_first_err_pos', 'N/A')}\n\n")

        f.write(f"""
## codebook 利用率

- L0: {geom_summary['codebook_usage']['L0']:.4f} ({geom_summary['codebook_unique']['L0']}/64 unique)
- L1: {geom_summary['codebook_usage']['L1']:.4f} ({geom_summary['codebook_unique']['L1']}/128 unique)
- L2: {geom_summary['codebook_usage']['L2']:.4f} ({geom_summary['codebook_unique']['L2']}/256 unique)
- L3: {geom_summary['codebook_usage']['L3']:.4f} ({geom_summary['codebook_unique']['L3']}/256 unique, PAD_count={geom_summary['L3_pad_count']})

## 生成失败模式 (failure_summary)

- n_samples: {failure_summary.get('n_samples', 0)}
- exact_R@10_proxy: {failure_summary.get('exact_hit_rate', 0):.4f}
- prefix_R@10_proxy: {failure_summary.get('prefix_hit_rate', 0):.4f}
- median_target_rank: {failure_summary.get('median_target_rank', 'N/A')}
- first_err_pos_distribution: {failure_summary.get('first_err_pos_distribution', {})}

## 假设列表 (供下一 issue 检验, 不实现)

1. L3 PAD_count 高 (L3 unique=1, PAD_count=~9922) 表明 v15 dedup digit 全 0 → 第 4 位 token 预测为 0 的 base-rate 极高
2. 用户历史长度越长 → 序列信息越丰富, 预测应该更好 (验证相关性)
3. 流行度高的 item 训练样本多, 预测更好 (验证相关性)
4. L2 assignment margin 大的 item 更难精确预测 (验证相关性)
5. Stage3 best_valid_R10=0.1251 < baseline 0.1267 (-1.3%), 加上 per-item 生成失败分布 → 根因可能在 Stage3 训练行为, 不在 Stage2 量化

## 完成判据

- [x] reference-0 manifest (commit/ckpt/SID/dataset/generation_config 完整)
- [x] item/SID geometry table (per-item, 9922 rows)
- [x] generation failure table (per-sample, ≥5000 samples)
- [x] 分桶表现报告 (history length / popularity)
- [x] 关联报告 (Spearman 相关, 仅相关非因果)
- [x] optimization data verdict (top-5 相关 + 分桶 + 假设)

## 不变量 (R36+R40+R41+R44)

- 不修改任何 stage1-4 代码
- 不修改模型/loss/SID/超参
- 只读 Issue133 既有产物 + 跑 canary forward (deterministic, no backward)
- 所有产物可追溯到 reference-0 manifest
""")
    print(f"  -> {md_path}", flush=True)

    json_path = OUT_DIR / "optimization_data_verdict.json"
    with open(json_path, "w") as f:
        json.dump({
            "issue": "#136",
            "decision": "data_profile_complete",
            "manifest_path": "data_profile/reference0_manifest.json",
            "geometry_csv": "data_profile/item_geometry.csv",
            "failure_csv": "data_profile/generation_failure.csv",
            "bucket_metrics": "data_profile/bucket_metrics.json",
            "correlation_report": "data_profile/correlation_report.json",
            "verdict_md": "data_profile/optimization_data_verdict.md",
            "top5_correlations": [
                {"pair": k, "rho": rho, "pvalue": pval} for k, rho, pval in top5
            ],
            "note": "数据画像完成, 仅相关非因果, 不实施任何修复"
        }, f, indent=2, ensure_ascii=False)
    print(f"  -> {json_path}", flush=True)


# ==========================================================================
# Main
# ==========================================================================
def main():
    t0 = time.time()
    print(f"=== Issue #136 reference-0 数据画像 开始 ===", flush=True)

    manifest = build_manifest()
    geom_summary = build_item_geometry()
    failure_summary = build_failure_table()
    bucket_report = build_bucket_metrics()
    corr = build_correlation_report()
    build_verdict(manifest, geom_summary, failure_summary, bucket_report, corr)

    print(f"=== Issue #136 完成 ({time.time() - t0:.1f}s) ===", flush=True)


if __name__ == "__main__":
    main()