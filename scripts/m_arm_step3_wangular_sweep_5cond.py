"""Sweep w_angular ∈ {1, 2, 3, 5} best_collision ckpts to extract cos_std.
Same logic as m_arm_step3_sweep_5cond.py but targets the 4 sweep directories.
"""
import os, sys
import torch

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec')
from m_arm_step3_sweep_5cond import evaluate_ckpt  # reuse proven function


CKPTS = [
    ("w_angular=1", "/home/wlia0047/ar57/wenyu/GeneRec/products/m_arm/m_radius_spread_step3_50ep_wdiv100_wangular_sweep_w1/Jul-27-2026_15-30-44_beta_0.500_codebook_[64,128,256]_sk_0.000/best_collision_model.pth"),
    ("w_angular=2", "/home/wlia0047/ar57/wenyu/GeneRec/products/m_arm/m_radius_spread_step3_50ep_wdiv100_wangular_sweep_w2/Jul-27-2026_15-30-44_beta_0.500_codebook_[64,128,256]_sk_0.000/best_collision_model.pth"),
    ("w_angular=3", "/home/wlia0047/ar57/wenyu/GeneRec/products/m_arm/m_radius_spread_step3_50ep_wdiv100_wangular_sweep_w3/Jul-27-2026_15-33-09_beta_0.500_codebook_[64,128,256]_sk_0.000/best_collision_model.pth"),
    ("w_angular=5", "/home/wlia0047/ar57/wenyu/GeneRec/products/m_arm/m_radius_spread_step3_50ep_wdiv100_wangular_sweep_w5/Jul-27-2026_15-33-09_beta_0.500_codebook_[64,128,256]_sk_0.000/best_collision_model.pth"),
]


def main():
    print(f"{'variant':<14} {'ckpt':<55} {'collision':<10} {'agree':<6} {'util%':<6} {'cos_std':<8} {'r-only':<7} {'maxc2':<6} {'5cond':<7} {'coll':<7}")
    print("-" * 130)
    rows = []
    for name, path in CKPTS:
        if not os.path.exists(path):
            print(f"{name:<14} MISSING: {path}")
            continue
        r = evaluate_ckpt(path)
        epoch_name = os.path.basename(path).replace("_model.pth", "")
        agree = f"{min(r['agreement']):.3f}"
        util = f"{min(r['util'])*100:.1f}"
        cos_std = f"{max(r['cos_std']):.3f}"
        cos_std_L0 = f"{r['cos_std'][0]:.3f}"
        cos_std_L1 = f"{r['cos_std'][1]:.3f}"
        cos_std_L2 = f"{r['cos_std'][2]:.3f}"
        r_only = f"{min(r['radius_only'])*100:.1f}"
        maxc2 = f"{max(r['max_c_norm_sq']):.3f}"
        cond5 = "PASS" if r['all_5cond_pass'] else "FAIL"
        coll = "PASS" if r['collision_pass'] else "FAIL"
        print(f"{name:<14} {epoch_name:<55} {r['tuple_collision']*100:<10.2f} {agree:<6} {util:<6} {cos_std:<8} {r_only:<7} {maxc2:<6} {cond5:<7} {coll:<7}")
        print(f"  {'(per-layer)':<14} {'cos_std [L0, L1, L2]':<55} [{cos_std_L0}, {cos_std_L1}, {cos_std_L2}]")
        rows.append({
            "variant": name,
            "collision": r['tuple_collision'],
            "agreement": min(r['agreement']),
            "util_min": min(r['util']),
            "cos_std_max": max(r['cos_std']),
            "cos_std_L0": r['cos_std'][0],
            "cos_std_L1": r['cos_std'][1],
            "cos_std_L2": r['cos_std'][2],
            "radius_only_max": max(r['radius_only']),
            "maxc2": max(r['max_c_norm_sq']),
            "cond5_pass": r['all_5cond_pass'],
            "coll_pass": r['collision_pass'],
        })

    print("\n=== Summary (w_angular sweep) ===")
    print(f"{'variant':<14} {'cos_std':<10} {'collision':<12} {'5cond':<6} {'coll':<6} {'Goldilocks?':<12}")
    for r in rows:
        goldilocks = "YES" if (0.30 <= r['cos_std_max'] <= 0.40 and r['collision'] <= 0.12) else "no"
        print(f"{r['variant']:<14} {r['cos_std_max']:<10.3f} {r['collision']*100:<12.2f} {'PASS' if r['cond5_pass'] else 'FAIL':<6} {'PASS' if r['coll_pass'] else 'FAIL':<6} {goldilocks:<12}")


if __name__ == "__main__":
    main()
