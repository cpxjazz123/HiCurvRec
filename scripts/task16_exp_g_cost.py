"""
Task #71 — Exp G: 三流形残差计算成本对比

测量 r_E/r_H/r_S 三种残差在 RQ-VAE 训练中的：
  - wall-clock 时间 (per step)
  - GPU 显存占用
  - Forward + backward 总耗时

启动:
  python scripts/task6_exp_g_cost.py [--device cuda:0]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans


def project_to_poincare(x):
    norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    return torch.tanh(norm) * (x / norm)


def project_to_sphere(x):
    norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    return x / norm


def poincare_log_map(q, r):
    q_sqnorm = (q ** 2).sum(dim=-1, keepdim=True).clamp(min=1e-7, max=1 - 1e-5)
    r_sqnorm = (r ** 2).sum(dim=-1, keepdim=True).clamp(min=1e-7, max=1 - 1e-5)
    inner = (q * r).sum(dim=-1, keepdim=True)
    num = (1 + 2 * inner + r_sqnorm) * q + (1 - q_sqnorm) * r
    denom = (1 + 2 * inner + q_sqnorm * r_sqnorm).clamp(min=1e-7)
    u = num / denom
    u_norm = u.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    diff_sq = ((r - q) ** 2).sum(dim=-1, keepdim=True)
    arg = (1 + 2 * diff_sq / ((1 - q_sqnorm) * (1 - r_sqnorm))).clamp(min=1 + 1e-7)
    d = torch.acosh(arg)
    lambda_q = 2 / (1 - q_sqnorm)
    return (d / (lambda_q * u_norm)) * u


def spherical_log_map(q, r):
    q_norm = q.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    r_norm = r.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    cos_sim = (q * r).sum(dim=-1, keepdim=True) / (q_norm * r_norm)
    cos_sim = cos_sim.clamp(-1 + 1e-7, 1 - 1e-7)
    d = torch.arccos(cos_sim)
    inner = (q * r).sum(dim=-1, keepdim=True)
    proj = r - (inner / (q_norm ** 2)) * q
    proj_norm = proj.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    return (d / proj_norm) * proj


class SimpleRQ(nn.Module):
    def __init__(self, n_features, n_clusters):
        super().__init__()
        self.codebook = nn.Parameter(
            torch.randn(n_clusters, n_features) * (1.0 / n_features ** 0.5)
        )

    def forward(self, x, residual_type="E"):
        d = torch.cdist(x, self.codebook, p=2)
        idx = d.argmin(dim=1)
        q = self.codebook[idx]
        if residual_type == "E":
            r = x - q
        elif residual_type == "H":
            r = poincare_log_map(project_to_poincare(q), project_to_poincare(x))
        elif residual_type == "S":
            r = spherical_log_map(project_to_sphere(q), project_to_sphere(x))
        loss = F.mse_loss(x, q.detach()) + F.mse_loss(x.detach(), q)
        return loss, r, idx


def benchmark_one(residual_type, embeddings_np, n_steps=50, batch_size=512,
                  n_clusters=64, lr=1e-3, device="cuda:0", seed=42):
    """Benchmark wall-clock + memory for one residual type"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_items, n_features = embeddings_np.shape

    # Reset GPU memory stats
    if "cuda" in device:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated() / 1024 ** 2

    model = SimpleRQ(n_features, n_clusters).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    embeddings_t = torch.tensor(embeddings_np, dtype=torch.float32, device=device)

    # Warm-up
    for _ in range(3):
        idx = np.random.randint(0, n_items, size=batch_size)
        batch = embeddings_t[idx]
        optimizer.zero_grad()
        loss, r, q_idx = model(batch, residual_type=residual_type)
        loss.backward()
        optimizer.step()

    if "cuda" in device:
        torch.cuda.synchronize()
    t_start = time.time()
    forward_times = []
    backward_times = []

    for step in range(n_steps):
        idx = np.random.randint(0, n_items, size=batch_size)
        batch = embeddings_t[idx]

        optimizer.zero_grad()

        # Forward timing
        if "cuda" in device:
            torch.cuda.synchronize()
        t0 = time.time()
        loss, r, q_idx = model(batch, residual_type=residual_type)
        if "cuda" in device:
            torch.cuda.synchronize()
        forward_times.append(time.time() - t0)

        # Backward timing
        if "cuda" in device:
            torch.cuda.synchronize()
        t0 = time.time()
        loss.backward()
        if "cuda" in device:
            torch.cuda.synchronize()
        backward_times.append(time.time() - t0)

        optimizer.step()

    total_time = time.time() - t_start

    if "cuda" in device:
        peak_mem = torch.cuda.max_memory_allocated() / 1024 ** 2
        current_mem = torch.cuda.memory_allocated() / 1024 ** 2
    else:
        peak_mem = 0
        current_mem = 0

    return {
        "residual_type": residual_type,
        "total_time_sec": total_time,
        "time_per_step_ms": total_time / n_steps * 1000,
        "forward_mean_ms": float(np.mean(forward_times)) * 1000,
        "backward_mean_ms": float(np.mean(backward_times)) * 1000,
        "peak_mem_mb": peak_mem,
        "current_mem_mb": current_mem,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_pt", default="products/task16/from_task62/item_embeddings.pt")
    parser.add_argument("--out_json", default="products/task16/from_task71/exp_g_cost.json")
    parser.add_argument("--n_steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--n_clusters", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("Task #71 — Exp G: 三流形残差计算成本对比")
    print("=" * 60)

    embeddings = torch.load(args.embeddings_pt, map_location="cpu", weights_only=False)
    emb_np = embeddings.numpy()
    print(f"\nEmbeddings shape: {emb_np.shape}")
    print(f"Device: {args.device}")

    if "cuda" in args.device and not torch.cuda.is_available():
        args.device = "cpu"
        print(f"  CUDA not available, using CPU")

    # Benchmark each residual type
    results = {}
    for rt in ["E", "H", "S"]:
        print(f"\nBenchmarking residual_type={rt}...")
        results[rt] = benchmark_one(
            residual_type=rt, embeddings_np=emb_np,
            n_steps=args.n_steps, batch_size=args.batch_size,
            n_clusters=args.n_clusters, device=args.device, seed=args.seed,
        )
        r = results[rt]
        print(f"  Total: {r['total_time_sec']:.2f}s")
        print(f"  Per step: {r['time_per_step_ms']:.2f}ms")
        print(f"    Forward: {r['forward_mean_ms']:.2f}ms")
        print(f"    Backward: {r['backward_mean_ms']:.2f}ms")
        if "cuda" in args.device:
            print(f"  Peak GPU mem: {r['peak_mem_mb']:.1f} MiB")

    # 决策
    print(f"\n{'='*60}")
    print("Decision (R7: 计算成本是否可行?)")
    print(f"{'='*60}")
    time_E = results["E"]["time_per_step_ms"]
    time_H = results["H"]["time_per_step_ms"]
    time_S = results["S"]["time_per_step_ms"]
    print(f"  Per-step time:")
    print(f"    E: {time_E:.2f}ms (baseline)")
    print(f"    H: {time_H:.2f}ms (H/E ratio: {time_H/time_E:.2f}×)")
    print(f"    S: {time_S:.2f}ms (S/E ratio: {time_S/time_E:.2f}×)")

    max_ratio = max(time_H, time_S) / time_E
    if max_ratio > 10:
        decision = "R7_REJECTED (wall-clock > 10× baseline → 不可行 → 关闭 E)"
    elif max_ratio > 3:
        decision = "R7_MARGINAL (3× < ratio < 10× → 需优化)"
    else:
        decision = "R7_CONFIRMED (ratio < 3× → 实际部署可行 → 可启动 E)"

    print(f"\n  Decision: {decision}")

    # 保存
    out = {
        "task": "#71 Exp G — computational cost",
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_clusters": args.n_clusters,
        "device": args.device,
        "results": results,
        "ratios": {
            "H_over_E": time_H / time_E,
            "S_over_E": time_S / time_E,
            "max_ratio": max_ratio,
        },
        "decision": decision,
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {args.out_json}")


if __name__ == "__main__":
    main()