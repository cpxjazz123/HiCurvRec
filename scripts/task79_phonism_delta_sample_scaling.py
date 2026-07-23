#!/usr/bin/env python3
"""Task #79 sample-size sensitivity analysis.

User feedback (2026-07-23): δ_max from 5000 4-tuples has statistical noise.
Re-run with 5000 / 20000 / 50000 samples to see if rankings/values stabilize.

Goal: estimate sampling-induced variance of δ_max.
Output: verdicts/task79_phonism_delta_sample_scaling.json + .md
"""
import argparse
import json
import os
import subprocess
import sys
import time

# Three sample sizes, focus on layer 3 (final residual) which is the headline
# number for "δ/diameter = 7.8% vs 9.2%" claim.
SAMPLE_SIZES = [5000, 20000, 50000]
SEEDS = [42]  # match Task #79 verdict seed
DEVICE = "cuda:0"  # CUDA_VISIBLE_DEVICES=2 in shell maps physical GPU 2 to cuda:0 inside subprocess


def run_one(ckpt: str, emb: str, num_samples: int, seed: int, output: str) -> dict:
    """Invoke task79_phonism_delta_hyperbolicity.py once."""
    cmd = [
        "python3",
        "/home/wlia0047/ar57/wenyu/GeneRec/scripts/task79_phonism_delta_hyperbolicity.py",
        "--ckpt", ckpt,
        "--emb", emb,
        "--num_samples", str(num_samples),
        "--seed", str(seed),
        "--device", DEVICE,
        "--output", output,
    ]
    t0 = time.time()
    print(f"\n=== Running num_samples={num_samples}, seed={seed} ===")
    print(" ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"num_samples={num_samples} failed: {proc.stderr[-2000:]}")
    with open(output) as f:
        result = json.load(f)
    print(f"  done in {elapsed:.1f}s")
    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--emb", required=True)
    p.add_argument("--out_dir", default="/home/wlia0047/ar57/wenyu/GeneRec/verdicts")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    runs = []
    for ns in SAMPLE_SIZES:
        out_json = os.path.join(args.out_dir, f"task79_delta_n{ns}_seed42.json")
        r = run_one(args.ckpt, args.emb, ns, SEEDS[0], out_json)
        # Extract per-layer deltas
        layers = []
        for layer in r["layers"]:
            layers.append({
                "layer": layer["layer"],
                "label": layer["label"],
                "delta_max": layer["delta_max"],
                "delta_95": layer["delta_95"],
                "delta_median": layer["delta_median"],
                "diameter_approx": layer["diameter_approx"],
                "delta_over_diameter": layer["delta_max"] / layer["diameter_approx"]
                                       if layer["diameter_approx"] > 0 else None,
                "residual_norm_mean": layer["residual_norm_mean"],
                "codebook_utilization": layer["codebook_utilization"],
            })
        runs.append({"num_samples": ns, "seed": SEEDS[0], "layers": layers})

    # Stability analysis: how much does δ_max change with sample size?
    # We focus on layer 3 (final residual).
    print("\n=== Sample-size sensitivity (Layer 3, final residual) ===")
    l3 = [(r["num_samples"], r["layers"][3]["delta_max"]) for r in runs]
    print(f"  num_samples=5000 : δ_max={l3[0][1]:.4f}")
    print(f"  num_samples=20000: δ_max={l3[1][1]:.4f} (Δ vs 5000 = {(l3[1][1]-l3[0][1])/l3[0][1]*100:+.2f}%)")
    print(f"  num_samples=50000: δ_max={l3[2][1]:.4f} (Δ vs 5000 = {(l3[2][1]-l3[0][1])/l3[0][1]*100:+.2f}%)")

    print("\n=== Sample-size sensitivity (Layer 3, δ/diameter) ===")
    l3_n = [(r["num_samples"], r["layers"][3]["delta_over_diameter"]) for r in runs]
    for ns, d in l3_n:
        print(f"  num_samples={ns}: δ/diameter = {d:.4f} ({d*100:.2f}%)")

    # Save final summary
    summary = {
        "task": "task79_phonism_delta_sample_scaling",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "user_feedback": "5000 4-tuples may have statistical noise; rerun with 20000 / 50000 to test stability",
        "ckpt": args.ckpt,
        "emb": args.emb,
        "device": DEVICE,
        "seed": SEEDS[0],
        "runs": runs,
    }
    out_path = os.path.join(args.out_dir, "task79_phonism_delta_sample_scaling.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {out_path}")


if __name__ == "__main__":
    main()