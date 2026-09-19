"""验证 first-token 可预测性的因果分析 (TIGER / iter5 / iter10).

计算每个 SID 表的:
- H(T0)              边际熵
- H(T0|H[k])         条件熵 (history 前 k 个 item 作为条件变量, k=1,2,3,last)
- I(T0;H[k]) = H(T0) - H(T0|H[k])   互信息 (history 把不确定性消掉多少)
- oracle_top_k(H[k], T0)  给定 history 前 k 个 item, top-1/5/20 codes 命中率
- prefix_collision     同 first-token T0 下挂多少不同 item (mean, max)
- T5 实测 first-token CE / acc (来自 token-level eval, SOS position)

输出对比表 + 写入 /home/wlia0047/.claude/jobs/acff3725/tmp/first_token_verify.log
"""
from __future__ import annotations
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
STAGE3 = ROOT / "stage3_T5Train"
DATASET = STAGE3 / "dataset/Amazon_2023_Instruments"
OUT_LOG = Path("/home/wlia0047/.claude/jobs/acff3725/tmp/first_token_verify.log")

SID_FILES = {
    "TIGER":  ROOT / "results/stage2_RQ-VAE/TIGER_RQ-VAE/item_sids_recbole.json",
    "iter5":  DATASET / "item_sids_iter5.json",
    "iter10": DATASET / "item_sids_iter10.json",
}
# T5 实测 first-token CE/acc (SOS position from token-level eval)
T5_OBSERVED = {
    "TIGER":  {"CE": 4.3979, "acc@1": 0.0996, "acc@5": 0.2782, "acc@20": 0.5485},
    "iter5":  {"CE": 4.7806, "acc@1": 0.0543, "acc@5": 0.1803, "acc@20": 0.4178},
    "iter10": {"CE": 4.8028, "acc@1": 0.0596, "acc@5": 0.1898, "acc@20": 0.4236},
}

# iter11..15 是 L0-only sk_eps sweep; SID 文件可能在 sweep 跑完后才落地,
# 缺失时跳过 (不允许 fallback, 直接 raise KeyError 给主流程).
for _iter in range(11, 16):
    _sid_path = DATASET / f"item_sids_iter{_iter}.json"
    if _sid_path.is_file():
        SID_FILES[f"iter{_iter}"] = _sid_path


def entropy_bits(counts: np.ndarray) -> float:
    if counts.sum() == 0:
        return 0.0
    p = counts[counts > 0] / counts.sum()
    return float(-np.sum(p * np.log2(p)))


def gini(counts: np.ndarray) -> float:
    a = np.sort(np.asarray(counts, dtype=np.float64))
    n = len(a)
    if n == 0 or a.sum() == 0:
        return 0.0
    cum = np.cumsum(a)
    return float((n + 1 - 2 * cum.sum() / cum[-1]) / n)


def load_sid(path: Path) -> dict[int, list[int]]:
    with open(path) as f:
        d = json.load(f)
    # dict[item_id (str)] -> [t0, t1, t2, t3]
    return {int(k): list(v) for k, v in d.items()}


def cond_entropy(counts_per_bin: dict[tuple, Counter]) -> float:
    """counts_per_bin[history_bin] -> Counter of T0; compute H(T0|bin) weighted by bin size."""
    total = 0
    H = 0.0
    for bin_key, ctr in counts_per_bin.items():
        if len(ctr) == 0:
            continue
        arr = np.array(list(ctr.values()), dtype=np.int64)
        n = arr.sum()
        H += entropy_bits(arr) * n
        total += n
    return H / total if total else 0.0


def oracle_topk(counts_per_bin: dict[tuple, Counter], k_list=(1, 5, 20)) -> dict[str, float]:
    """Given each bin, top-k codes; report the fraction of (bin, target) pairs
    where ground-truth T0 is in the top-k."""
    total = 0
    hit = {k: 0 for k in k_list}
    for ctr in counts_per_bin.values():
        if not ctr:
            continue
        items = list(ctr.items())  # [(t0, count), ...]
        items.sort(key=lambda x: -x[1])
        ranked = [t0 for t0, _ in items]
        n_in_bin = sum(ctr.values())
        for tgt_t0, c in ctr.items():
            total += c
            for k in k_list:
                if tgt_t0 in ranked[:k]:
                    hit[k] += c
    return {f"top{k}": (hit[k] / total if total else 0.0) for k in k_list}


def prefix_collision(sid_table: dict[int, list[int]]) -> dict[str, float]:
    """统计每个 first-token T0 下挂多少不同 item."""
    pre2item = defaultdict(set)
    for item_id, code in sid_table.items():
        pre2item[code[0]].add(item_id)
    counts = np.array([len(s) for s in pre2item.values()], dtype=np.int64)
    return {
        "n_unique_T0":     int(len(pre2item)),
        "mean_items_per_T0": float(counts.mean()),
        "max_items_per_T0":  int(counts.max()),
        "std_items_per_T0":  float(counts.std()),
    }


def main() -> None:
    df = pd.read_parquet(DATASET / "valid_recbole.parquet")
    n = len(df)

    print(f"=== valid set n={n} ===\n")

    rows = []
    for name, sid_path in SID_FILES.items():
        print(f"=== {name}  (SID={sid_path.name}) ===")
        sid_table = load_sid(sid_path)
        assert len(sid_table) == 24587, f"got {len(sid_table)}"

        # target_first_token: 每行 target item_id 对应的 code[0]
        tgt_t0 = np.array([sid_table[int(t)][0] for t in df["target"]], dtype=np.int64)

        # history 是 ndarray of item_ids (按时间顺序)
        history_lists = df["history"].tolist()  # list of np.ndarray

        # ---- 边际 H(T0) ----
        counts_marg = np.bincount(tgt_t0)
        H_marg = entropy_bits(counts_marg)
        G = gini(counts_marg)
        eff_vocab = 2 ** H_marg
        print(f"  H(T0) = {H_marg:.4f} bits, Gini = {G:.4f}, effective vocab = {eff_vocab:.1f}, "
              f"unique used = {(counts_marg > 0).sum()}/{counts_marg.size}")

        # ---- prefix collision ----
        pc = prefix_collision(sid_table)
        print(f"  prefix_collision: n_unique_T0={pc['n_unique_T0']} "
              f"mean_items/T0={pc['mean_items_per_T0']:.2f} max={pc['max_items_per_T0']}")

        # ---- conditioning on history ----
        # 构建多个 bin 方案:
        #   bin by history[0]
        #   bin by (history[0], history[1])
        #   bin by (history[0], history[1], history[2])
        #   bin by history[-1]
        #   bin by history mean semantic embedding?  -> 不在本次范围内
        schemes = {
            "H[0]":         lambda hist: (int(hist[0]),) if len(hist) >= 1 else None,
            "H[0,1]":       lambda hist: (int(hist[0]), int(hist[1])) if len(hist) >= 2 else None,
            "H[0,1,2]":     lambda hist: (int(hist[0]), int(hist[1]), int(hist[2])) if len(hist) >= 3 else None,
            "H[last]":      lambda hist: (int(hist[-1]),) if len(hist) >= 1 else None,
        }

        cond = {}
        for sname, keyfn in schemes.items():
            bin2ctr = defaultdict(Counter)
            for hist, tgt in zip(history_lists, tgt_t0):
                key = keyfn(hist)
                if key is None:
                    continue
                bin2ctr[key][int(tgt)] += 1
            H_c = cond_entropy(bin2ctr)
            I = H_marg - H_c
            topk = oracle_topk(bin2ctr)
            print(f"  cond({sname}): H(T0|cond)={H_c:.4f}  I(T0;{sname})={I:+.4f} bits  "
                  f"oracle_top1={topk['top1']:.4f}  top5={topk['top5']:.4f}  top20={topk['top20']:.4f}")
            cond[sname] = (H_c, I, topk)

        # ---- T5 实际表现 ----
        t5 = T5_OBSERVED.get(name)
        if t5 is None:
            print(f"  T5 实测 first-token (SOS): (尚未评估)")
        else:
            print(f"  T5 实测 first-token (SOS): CE={t5['CE']:.4f} acc@1={t5['acc@1']:.4f} "
                  f"acc@5={t5['acc@5']:.4f} acc@20={t5['acc@20']:.4f}")

        rows.append({
            "name": name,
            "H_marg": H_marg, "Gini": G, "eff_vocab": eff_vocab,
            "pc": pc, "cond": cond, "t5": t5,
        })

    # ===== 动态表头: 跟 SID_FILES 顺序一致 =====
    header_names = [r["name"] for r in rows]
    n_rows = len(rows)
    label_w = 22
    col_w = 14
    sep_w = "-" * (label_w + 1 + col_w * n_rows)

    print("\n=== 对比表 ===\n")
    header_line = f"{'metric':<{label_w}} " + " ".join(f"{n:>{col_w}}" for n in header_names)
    print(header_line)
    print(sep_w)
    for key_label, fn in [
        ("H(T0) bits",            lambda r: f"{r['H_marg']:.4f}"),
        ("Gini L0",               lambda r: f"{r['Gini']:.4f}"),
        ("eff vocab 2^H",         lambda r: f"{r['eff_vocab']:.1f}"),
        ("prefix max items/T0",   lambda r: f"{r['pc']['max_items_per_T0']}"),
        ("prefix mean items/T0",  lambda r: f"{r['pc']['mean_items_per_T0']:.2f}"),
    ]:
        cells = " ".join(f"{fn(r):>{col_w}}" for r in rows)
        print(f"{key_label:<{label_w}} {cells}")

    print("\n=== 条件熵 vs 互信息 (H unit = bits) ===")
    scheme_w = 14
    metric_w = 14
    val_w = 12
    val_sep = " ".join(f"{'':>{val_w}}" for _ in header_names)
    print(f"{'scheme':<{scheme_w}} {'metric':<{metric_w}} " + " ".join(f"{n:>{val_w}}" for n in header_names))
    print("-" * (scheme_w + 1 + metric_w + 1 + val_w * n_rows))
    for sname in ["H[0]", "H[0,1]", "H[0,1,2]", "H[last]"]:
        for label, idx in [("H(T0|cond)", 0), ("I(T0;cond)", 1), ("oracle_top1", 2), ("oracle_top5", 2)]:
            vals = []
            for r in rows:
                v = r["cond"][sname][idx]
                if isinstance(v, dict):
                    v = v[label.split("_")[1]]
                vals.append(v)
            cells = " ".join(f"{vals[i]:>{val_w}.4f}" for i in range(n_rows))
            print(f"{sname:<{scheme_w}} {label:<{metric_w}} {cells}")

    print("\n=== T5 实测 first-token (SOS 位置) ===")
    print(f"{'metric':<{scheme_w}} " + " ".join(f"{n:>{val_w}}" for n in header_names))
    print("-" * (scheme_w + 1 + val_w * n_rows))
    for m in ["CE", "acc@1", "acc@5", "acc@20"]:
        cells = []
        for r in rows:
            obs = T5_OBSERVED.get(r["name"])
            if obs is None:
                cells.append(f"{'(no T5 obs)':>{val_w}}")
            else:
                cells.append(f"{obs[m]:>{val_w}.4f}")
        print(f"{m:<{scheme_w}} " + " ".join(cells))

    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()