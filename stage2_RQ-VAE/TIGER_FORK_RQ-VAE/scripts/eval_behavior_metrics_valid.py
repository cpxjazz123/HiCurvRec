"""Independent-validation behaviour metrics for the accepted mechanism arms.

Read-only. Uses the valid.parquet-derived pair set so the scored relations never
entered training, and the same metric definitions the programme already uses for
the train-based canonical set.

Run: /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 stage2_RQ-VAE/TIGER_FORK_RQ-VAE/scripts/eval_behavior_metrics_valid.py
"""
import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS = "/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_FORK_RQ-VAE/scripts"
sys.path.insert(0, SCRIPTS)
import analyze_prefix_behavior as AP  # noqa: E402
import eval_behavior_metrics as EB  # noqa: E402
import monotonicity_metrics as MM  # noqa: E402

MONO = Path("/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/monotonicity")
STAGE3 = Path("/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train")
STAGE2 = Path("/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE")

ARMS = [
    ("Exp1a_resample", STAGE2 / "TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample/item_sids.json"),
    ("plain_nce_euclid_group", STAGE2 / "TIGER_FORK_RQ-VAE/nce_euclid_group/item_sids.json"),
    ("hyper_l1_s42", STAGE2 / "TIGER_FORK_RQ-VAE/hyper_l1_s42/item_sids.json"),
    ("hyper_l1_br_euclid_s42", STAGE2 / "TIGER_FORK_RQ-VAE/hyper_l1_br_euclid_s42/item_sids.json"),
    ("hyper_l1_br_s42_geodesic", STAGE2 / "TIGER_FORK_RQ-VAE/hyper_l1_br_s42/item_sids.json"),
    ("hyper_l1_ang_s42", STAGE2 / "TIGER_FORK_RQ-VAE/hyper_l1_ang_s42/item_sids.json"),
]
STAGE3_DIRS = {
    "Exp1a_resample": None,
    "plain_nce_euclid_group": STAGE3 / "tiger_fork_nce_euclid_group",
    "hyper_l1_s42": STAGE3 / "tiger_fork_hyper_l1_s42",
    "hyper_l1_br_euclid_s42": STAGE3 / "tiger_fork_hyper_l1_br_euclid_s42",
    "hyper_l1_br_s42_geodesic": STAGE3 / "tiger_fork_hyper_l1_br_s42",
    "hyper_l1_ang_s42": STAGE3 / "tiger_fork_hyper_l1_ang_s42",
}


def recall_of(label):
    d = STAGE3_DIRS.get(label)
    if d is None:
        return None
    f = sorted(d.rglob("test_final.json"))
    if not f:
        return None
    return json.loads(f[-1].read_text())


def main():
    pairs = EB.load_pairs(MONO / "behavior_pairs_valid_s42.npz")
    print(f"valid pairs: anchors={pairs['anchors'].shape} "
          f"positives={pairs['positives'].shape} negatives={pairs['negatives'].shape}")
    print()
    hdr = f"{'arm':26s} {'lvl':10s} {'pos_share':>10s} {'neg_share':>10s} {'gap':>10s} {'ER':>9s}"
    print(hdr)
    for label, sid in ARMS:
        if not Path(sid).is_file():
            print(f"{label:26s} MISSING {sid}")
            continue
        levels, info = AP.load_levels(Path(sid))
        if len(levels) != MM.N_ITEMS:
            print(f"{label:26s} n_items={len(levels)} expected {MM.N_ITEMS}")
            continue
        m = EB.metrics_for(levels, pairs)
        for lvl in ("L1", "L1_L2", "L1_L2_L3"):
            e = m[lvl]
            er = f"{e['er']:.4f}" if e["er"] is not None else "undefined"
            print(f"{label:26s} {lvl:10s} {e['p_share_pos']:>10.5f} {e['p_share_neg']:>10.5f} "
                  f"{e['gap']:>+10.5f} {er:>9s}")
        cb = m["codebook"]
        r = recall_of(label)
        rtxt = (f"R@10={r['test_recall@10']:.6f} NDCG@10={r['test_ndcg@10']:.6f} "
                f"n_eval={r['n_eval']}") if r else "stage3=none"
        print(f"{label:26s} {'codebook':10s} collision={cb['full_sid']['collision_rate']:.4f} "
              f"L1={cb['L1']['distinct_codes']} L2={cb['L2']['distinct_codes']} "
              f"L3={cb['L3']['distinct_codes']}  {rtxt}")
        print()


if __name__ == "__main__":
    main()
