"""
Task #71 — Exp E Stage 2.3: 三流形 RQ-VAE 推断生成 SID

输入: products/task16/from_task71/exp_e_s22/codebooks_{E,H,S}.pt
输出: products/task16/from_task71/exp_e_s22/sid_{E,H,S}.pt  shape=(11924, 3) int64

最近码字选择：欧氏距离（与 Task #70 D1 一致：任何距离排序相同 → 用欧氏即可）
残差几何：H/S 仍然 cascade 输入（但不影响 SID 选择本身）

启动:
  python scripts/task6_exp_e_s23_sid_gen.py --residual_type {E,H,S} --device cuda:0
"""

import argparse
import os

import numpy as np
import torch


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


def compute_residual(x, q, residual_type):
    if residual_type == "E":
        return x - q
    elif residual_type == "H":
        return poincare_log_map(project_to_poincare(q), project_to_poincare(x))
    elif residual_type == "S":
        return spherical_log_map(project_to_sphere(q), project_to_sphere(x))
    else:
        raise ValueError(residual_type)


def generate_sid(residual_type, embeddings_pt, codebooks_pt, device="cuda:0",
                 batch_size=1024):
    """生成 L=3 SID"""
    embeddings = torch.load(embeddings_pt, map_location="cpu", weights_only=False)
    ck = torch.load(codebooks_pt, map_location="cpu", weights_only=False)
    codebooks = [cb.to(device) for cb in ck["codebooks"]]
    n_layers = ck["n_layers"]
    n_items = embeddings.shape[0]

    print(f"  residual={residual_type}, n_items={n_items}, n_layers={n_layers}")

    sid = torch.zeros(n_items, n_layers, dtype=torch.long)
    embeddings_gpu = embeddings.to(device)

    with torch.no_grad():
        for start in range(0, n_items, batch_size):
            end = min(start + batch_size, n_items)
            batch = embeddings_gpu[start:end]
            cur = batch
            for l in range(n_layers):
                d = torch.cdist(cur, codebooks[l], p=2)
                idx = d.argmin(dim=1)
                sid[start:end, l] = idx.cpu()
                q = codebooks[l][idx]
                cur = compute_residual(cur, q, residual_type)

    return sid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--residual_type", choices=["E", "H", "S"], required=True)
    parser.add_argument("--embeddings_pt", default="products/task16/from_task62/item_embeddings.pt")
    parser.add_argument("--codebooks_pt",
                        default="products/task16/from_task71/exp_e_s22/codebooks_{}.pt")
    parser.add_argument("--out_pt", default="products/task16/from_task71/exp_e_s22/sid_{}.pt")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Task #71 Exp E Stage 2.3 — SID Generation")
    print(f"  residual_type: {args.residual_type}")
    print(f"  device: {args.device}")
    print("=" * 60)

    codebooks_pt = args.codebooks_pt.format(args.residual_type)
    out_pt = args.out_pt.format(args.residual_type)

    sid = generate_sid(args.residual_type, args.embeddings_pt, codebooks_pt, args.device)

    # 统计每层 unique codeword
    print(f"\n  SID shape: {sid.shape}")
    print(f"  Layer unique counts:")
    for l in range(sid.shape[1]):
        u = sid[:, l].unique().numel()
        print(f"    L{l+1}: {u}/{256} unique")

    # 检查冲突：每层是否有 sample 选到同一个 codeword 在所有层
    print(f"\n  Total unique (sid_1, sid_2, sid_3) tuples: {len(sid.unique(dim=0))}/{sid.shape[0]}")

    os.makedirs(os.path.dirname(out_pt), exist_ok=True)
    torch.save({
        "sid": sid,
        "residual_type": args.residual_type,
        "source": codebooks_pt,
    }, out_pt)
    print(f"\n  Saved: {out_pt}")


if __name__ == "__main__":
    main()
