"""task421: 三 Poincaré Ball 变体码本利用率诊断

计算 per-level unique codes / Gini / Perplexity / marginal distribution，
与 E-E-E-E baseline 对照，验证"码字坍缩"假说。

复算自:
  /fs04/ar57/wenyu/GeneRec/GRID/logs/inference/runs/{task13_group_a_s22, task388v4_hee_s22, task388v4_hhee_s22, task388v4_hhhh_s22}/pickle/merged_predictions_tensor.pt

输出: result/task421_codebook_utilization/{summary.json, per_level_*.csv, verdict.md}
"""

import os
import json
import numpy as np
import torch

ROOT = "/home/wlia0047/ar57/wenyu/GeneRec/GRID"
SID_PATHS = {
    "E_E_E_E": f"{ROOT}/logs/inference/runs/task13_group_a_s22/pickle/merged_predictions_tensor.pt",
    "H_E_E_E": f"{ROOT}/logs/inference/runs/task388v4_hee_s22/pickle/merged_predictions_tensor.pt",
    "H_H_E_E": f"{ROOT}/logs/inference/runs/task388v4_hhee_s22/pickle/merged_predictions_tensor.pt",
    "H_H_H_H": f"{ROOT}/logs/inference/runs/task388v4_hhhh_s22/pickle/merged_predictions_tensor.pt",
}
CODEBOOK_SIZE = 256  # max+1, all tokens in [0, 255]
NUM_LAYERS = 4

OUT_DIR = f"{ROOT}/task_artifacts/results/exp388v5/task421_codebook_utilization"
os.makedirs(OUT_DIR, exist_ok=True)


def gini_coefficient(counts: np.ndarray) -> float:
    """Gini = 1 - 2 * area under Lorenz curve, range [0,1].
    Gini=0: perfect equality (all codes used equally); Gini→1: extreme concentration."""
    counts = counts.astype(np.float64)
    counts = counts[counts > 0]
    if counts.size == 0:
        return 0.0
    counts = np.sort(counts)
    n = counts.size
    cum = np.cumsum(counts)
    # Lorenz: cumulative fraction of total
    lorenz = cum / cum[-1]
    # Trapezoidal integral of lorenz
    area = np.sum(lorenz) / n - 0.5 / n  # approx
    return float(1.0 - 2.0 * area)


def entropy_perplexity(counts: np.ndarray) -> tuple:
    """Shannon entropy (natural log) and perplexity = exp(H).
    For uniform over K: H=log(K), perplexity=K.
    Lower entropy → more concentrated → lower effective perplexity."""
    counts = counts.astype(np.float64)
    total = counts.sum()
    if total == 0:
        return 0.0, 1.0
    p = counts / total
    p_nonzero = p[p > 0]
    H = -np.sum(p_nonzero * np.log(p_nonzero))
    return float(H), float(np.exp(H))


def per_level_stats(level_codes: np.ndarray, codebook_size: int) -> dict:
    """level_codes: 1D array of integer codes for one layer."""
    counts = np.bincount(level_codes, minlength=codebook_size)
    n_items = counts.sum()
    n_active = int((counts > 0).sum())
    utilization = n_active / codebook_size

    # Gini over the **active** codes (zero entries would inflate Gini artificially,
    # so measure concentration of usage among *used* codes)
    gini_active = gini_coefficient(counts[counts > 0])

    # Also compute "augmented Gini" including inactive slots as zeros
    gini_full = gini_coefficient(counts)

    H, perp = entropy_perplexity(counts)

    # Top-1 code share
    top1_share = float(counts.max() / n_items)
    # Effective K: sum_{i} p_i^2 = collision prob; effective_K = 1 / sum_p2
    p = counts / n_items
    p = p[p > 0]
    sum_p2 = float((p ** 2).sum())
    eff_K = 1.0 / sum_p2 if sum_p2 > 0 else 0.0

    # fraction singleton (codes used exactly 1 time)
    n_singleton = int((counts == 1).sum())
    frac_singleton = n_singleton / max(n_active, 1)

    return {
        "n_items": int(n_items),
        "n_codes_total": codebook_size,
        "n_codes_used": n_active,
        "utilization": utilization,
        "gini_active": gini_active,
        "gini_full": gini_full,
        "entropy_nats": H,
        "entropy_max_nats": float(np.log(codebook_size)),
        "perplexity": perp,
        "perplexity_max": float(codebook_size),
        "top1_share": top1_share,
        "effective_K": eff_K,
        "n_singleton_codes": n_singleton,
        "frac_singleton": frac_singleton,
        "counts": counts.tolist(),  # full marginal
    }


def joint_uniqueness(sid: np.ndarray) -> dict:
    """Compute joint-unique codes: number of unique 4-tuples vs total items."""
    n = sid.shape[1]
    sid_t = sid.T  # (n_items, 4)
    # Pack as single 64-bit int: code0*2^48 + code1*2^32 + code2*2^16 + code3 (arbitrary order)
    packed = sid_t[:, 0].astype(np.int64)
    packed = packed * (CODEBOOK_SIZE ** 3) + sid_t[:, 1] * (CODEBOOK_SIZE ** 2) + sid_t[:, 2] * CODEBOOK_SIZE + sid_t[:, 3]
    unique = np.unique(packed).size
    return {
        "n_items": n,
        "n_joint_unique_codes": unique,
        "joint_utilization": unique / n,
        "n_collisions": n - unique,
        "frac_collision": float((n - unique) / n),
    }


def main():
    summary = {}
    for variant, path in SID_PATHS.items():
        print(f"\n=== {variant}: {path}")
        t = torch.load(path, map_location="cpu", weights_only=False)
        sid = t.numpy().astype(np.int64)  # (4, n_items)
        assert sid.shape[0] == NUM_LAYERS, f"unexpected layers: {sid.shape}"

        per_level = [per_level_stats(sid[l], CODEBOOK_SIZE) for l in range(NUM_LAYERS)]
        joint = joint_uniqueness(sid)

        summary[variant] = {"per_level": per_level, "joint": joint}

        print(f"  joint unique {joint['n_joint_unique_codes']}/{joint['n_items']} ({joint['joint_utilization']*100:.2f}%), collisions {joint['n_collisions']} ({joint['frac_collision']*100:.2f}%)")
        for l in range(NUM_LAYERS):
            ps = per_level[l]
            print(f"  L{l}: used {ps['n_codes_used']}/{ps['n_codes_total']} ({ps['utilization']*100:.2f}%)  "
                  f"Gini(active)={ps['gini_active']:.3f}  perp={ps['perplexity']:.1f}/{CODEBOOK_SIZE}  "
                  f"top1={ps['top1_share']*100:.2f}%  effK={ps['effective_K']:.1f}")

    # Save summary
    out_json = f"{OUT_DIR}/summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_json}")

    # Save per-variant CSV (one row per layer)
    import csv
    csv_path = f"{OUT_DIR}/per_level_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "layer", "n_codes_used", "utilization", "gini_active", "gini_full",
                    "entropy", "entropy_max", "perplexity", "top1_share", "effective_K",
                    "n_singleton", "frac_singleton"])
        for variant, data in summary.items():
            for l, ps in enumerate(data["per_level"]):
                w.writerow([variant, l,
                            ps["n_codes_used"], f"{ps['utilization']:.6f}",
                            f"{ps['gini_active']:.6f}", f"{ps['gini_full']:.6f}",
                            f"{ps['entropy_nats']:.6f}", f"{ps['entropy_max_nats']:.6f}",
                            f"{ps['perplexity']:.4f}", f"{ps['top1_share']:.6f}",
                            f"{ps['effective_K']:.4f}", ps["n_singleton_codes"],
                            f"{ps['frac_singleton']:.6f}"])
    print(f"Saved: {csv_path}")

    # Save joint CSV
    joint_csv = f"{OUT_DIR}/joint_summary.csv"
    with open(joint_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "n_items", "n_joint_unique", "joint_utilization", "n_collisions", "frac_collision"])
        for variant, data in summary.items():
            j = data["joint"]
            w.writerow([variant, j["n_items"], j["n_joint_unique_codes"],
                        f"{j['joint_utilization']:.6f}", j["n_collisions"], f"{j['frac_collision']:.6f}"])
    print(f"Saved: {joint_csv}")


if __name__ == "__main__":
    main()
