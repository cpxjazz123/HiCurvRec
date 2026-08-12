"""Issue #137 curvature_observability_table.csv — Stage4 raw predictions 命中率 × Issue #136 item geometry.

把 Stage4 raw (本 issue 唯一 oracle) 的 5000 sample 命中率与 Issue #136 item_geometry.csv 关联,
输出 curvature_observability_table.csv (per-sample: sample_idx, target_item, hit, item_geometry cols).

本表用作 Issue #138 (per-item 曲率可观测性扩展) 的种子数据.
"""
import os
import sys
import json
import csv
import time
from pathlib import Path

TASK_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/tasks/Issue137_protocol_alignment")
os.chdir(TASK_DIR)

ISSUE136_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/tasks/Issue136_reference0_data_profile")
ISSUE137_DIR = TASK_DIR

RAW_PATH = ISSUE137_DIR / "stage2/eval/raw_predictions_stage4.json"
ITEM_GEOM_CSV = ISSUE136_DIR / "data_profile/item_geometry.csv"
TEST_PARQUET = Path("/home/wlia0047/ar57/wenyu/GeneRec/dataset/test.parquet")
OUT_DIR = ISSUE137_DIR / "alignment"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "curvature_observability_table.csv"


def main():
    t0 = time.time()
    print("[curvature_observability] 开始生成 Stage4 raw × item_geometry 关联表", flush=True)

    import numpy as np
    import pyarrow.parquet as pq

    raw = json.load(open(RAW_PATH))
    samples = raw["samples"]
    print(f"  raw samples: {len(samples)}", flush=True)

    # 读 test.parquet 拿 target_item (sample_idx 是 test.parquet 行索引 0..4999)
    test_df = pq.read_table(TEST_PARQUET).to_pandas()
    print(f"  test.parquet rows: {len(test_df)}", flush=True)

    # 读 item_geometry.csv
    geom_rows = {}
    with open(ITEM_GEOM_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            geom_rows[int(row["item_id"])] = row
    print(f"  item_geometry rows: {len(geom_rows)}", flush=True)

    # 输出 csv
    out_fields = [
        "sample_idx", "target_item", "hit_top20",
        "sid_L0", "sid_L1", "sid_L2", "sid_L3",
        "L0_freq", "L1_freq", "L2_freq", "L3_freq",
        "L2_uniqueness", "item_knn_mean_dist", "item_popularity", "text_emb_norm",
    ]
    n_with_geom = 0
    n_hit = 0
    with open(OUT_CSV, "w") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        for s in samples:
            idx = s["sample_idx"]
            target_item = int(test_df.iloc[idx].get("target", -1))
            if target_item < 0 or target_item not in geom_rows:
                # no geometry, 跳过 (R8: 不允许 silent fallback, 必须显式记 missing)
                continue
            geom = geom_rows[target_item]
            # 实际 hit 计算: pos_index_top20 任意 True (Stage4 vectorized all-token match)
            hit = 1 if any(s["pos_index_top20"]) else 0
            n_hit += hit
            n_with_geom += 1
            writer.writerow({
                "sample_idx": idx,
                "target_item": target_item,
                "hit_top20": hit,
                "sid_L0": geom["sid_L0"],
                "sid_L1": geom["sid_L1"],
                "sid_L2": geom["sid_L2"],
                "sid_L3": geom["sid_L3"],
                "L0_freq": geom["L0_freq"],
                "L1_freq": geom["L1_freq"],
                "L2_freq": geom["L2_freq"],
                "L3_freq": geom["L3_freq"],
                "L2_uniqueness": geom["L2_uniqueness"],
                "item_knn_mean_dist": geom["item_knn_mean_dist"],
                "item_popularity": geom["item_popularity"],
                "text_emb_norm": geom["text_emb_norm"],
            })

    print(f"  -> {OUT_CSV} ({n_with_geom} rows, {n_hit} hits)", flush=True)
    print(f"[curvature_observability] 完成 ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()