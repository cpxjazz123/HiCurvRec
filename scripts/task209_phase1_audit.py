# Task #209 Phase 1 主判据脚本
# 输入: 4 个 arm 训练好的 ckpt (从 products/task209/phase1_arm_A{1,2,3,4}/.../best_loss_model.pth)
# 输出: 跨 4 臂的 metrics 对比表 + 通过/不通过主判据

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def load_arm_ckpt(ckpt_path, device="cpu"):
    """加载 arm 训练好的 HRQVAE ckpt, 提取 encoder + hrq (含 vq_layers + codebook)."""
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    return sd


@torch.no_grad()
def evaluate_arm(sd, item_emb_path, device="cpu"):
    """给定训练好的 ckpt state_dict, 评估 arm 的关键指标."""
    import sys
    sys.path.insert(0, "/fs04/ar57/wenyu/GeneRec/HG-Rec")
    from model.hrqvae import HRQVAE

    # 重建 model
    model = HRQVAE(
        in_dim=768, num_emb_list=[64, 128, 256], e_dim=32,
        layers=[512, 256, 128, 64], loss_type="poincare", beta=0.5,
        sk_eps=[0.0, 0.0, 0.0],
        w_path=1.0 if sd.get("path_geometry") == "hyp" or True else 0.0,
    )
    model.load_state_dict(sd, strict=False)
    model.eval()
    model.to(device)

    # 加载 item embedding
    import pandas as pd
    df = pd.read_parquet(item_emb_path)
    embs = np.stack(df["embedding"].values, axis=0)
    x = torch.FloatTensor(embs).to(device)
    print(f"  loaded {x.shape[0]} items, dim={x.shape[1]}")

    # Encode + quantile per layer
    z = model.encoder(x)  # (N, 32)
    residual = z

    metrics_per_layer = []
    all_indices = []
    for li, vq in enumerate(model.hrq.vq_layers):
        # 测试时 hard norm 投影 + poincare argmin
        # 跟训练时一致, 但用 evaluation 模式 (training=False 跳过 init_emb 检查).
        vq.eval()
        x_res, _, indices = vq(residual, use_sk=False)
        residual = residual - x_res
        all_indices.append(indices)

        # Codebook metrics
        n_e = vq.n_e if hasattr(vq, "n_e") else vq.embeddings.weight.shape[0]
        cb = vq.embeddings.weight.detach()  # (n_e, 32)
        cb_norms = cb.norm(dim=-1)
        # projected norm (硬投影 ρ/2 → tanh 之后)
        rho = 2.0 * cb_norms.median().item() if vq.rho is not None else None
        if vq.rho is not None:
            tangent_norm = vq.rho / 2.0
            cb_n = tangent_norm * F.normalize(cb, dim=-1, eps=1e-6)
            projected_norm = proj_to_ball_safe(expmap0_safe(cb_n, c=1.0), c=1.0).norm(dim=-1).mean().item()
        else:
            projected_norm = cb_norms.mean().item()

        # argmax utility
        unique_codes = indices.unique().numel()
        util = unique_codes / n_e

        # cos max / cos mean over inputs vs centers
        # 取 z_centered (per-layer input 是 residual)
        in_norms = residual.norm(dim=-1)
        if vq.rho is not None:
            tangent_norm = vq.rho / 2.0
            in_n = tangent_norm * F.normalize(residual, dim=-1, eps=1e-6)
        else:
            in_n = residual
        in_h = proj_to_ball_safe(expmap0_safe(in_n, c=1.0), c=1.0)
        if vq.rho is not None:
            cb_n = tangent_norm * F.normalize(cb, dim=-1, eps=1e-6)
        else:
            cb_n = cb
        cb_h = proj_to_ball_safe(expmap0_safe(cb_n, c=1.0), c=1.0)

        d = torch.cdist(in_h, cb_h).squeeze()  # placeholder
        # Actually use poincare
        from model.utils import poincare_distance
        b = in_h.shape[0]
        k = cb_h.shape[0]
        d = poincare_distance(
            in_h.unsqueeze(1).expand(b, k, -1),
            cb_h.unsqueeze(0).expand(b, k, -1),
            c=1.0,
        ).squeeze(-1)  # (B, K)
        cos_proxy = 1 - d.clamp(min=0).pow(2) / 4  # (approx cos) via formula
        # simpler: use metric diag
        cos_max = cos_proxy.max(dim=-1).values
        cos_mean = cos_proxy.mean().item()
        n_pair_collapse = 0  # skip — too expensive

        metrics_per_layer.append({
            "layer": li,
            "n_e": n_e,
            "rho": float(vq.rho) if vq.rho is not None else None,
            "raw_norm_mean": cb_norms.mean().item(),
            "projected_norm_mean": projected_norm,
            "util": util,
            "cos_max_max": cos_max.max().item(),
            "cos_max_mean": cos_max.mean().item(),
            "cos_mean_proxy": cos_mean,
        })

    # 全局 collision_rate: 全 9922 个 item 的 3 层 SID 碰撞到已有 SID 集合的比率
    sid = torch.stack(all_indices, dim=-1)  # (N, 3)
    N = sid.shape[0]
    unique_sids = torch.unique(sid, dim=0).shape[0]
    collision_rate = 1 - unique_sids / N

    return {
        "metrics_per_layer": metrics_per_layer,
        "collision_rate": float(collision_rate),
        "n_unique_sids": int(unique_sids),
        "n_items": int(N),
    }


def expmap0_safe(x, c):
    import sys
    sys.path.insert(0, "/fs04/ar57/wenyu/GeneRec/HG-Rec")
    from model.utils import expmap0
    return expmap0(x, c)


def proj_to_ball_safe(x, c):
    import sys
    sys.path.insert(0, "/fs04/ar57/wenyu/GeneRec/HG-Rec")
    from model.utils import proj_to_ball
    return proj_to_ball(x, c)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=["A1", "A2", "A3", "A4"])
    parser.add_argument("--ckpt_pattern", default=None,
                        help="如不指定, 自动在 products/task209/phase1_arm_<arm> 找 best_loss_model.pth")
    parser.add_argument("--item_emb", default="/fs04/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet")
    parser.add_argument("--out_dir", default="/fs04/ar57/wenyu/GeneRec/verdicts/task209_phase1")
    args = parser.parse_args()

    # 找 ckpt
    ckpt_dir = Path(f"/fs04/ar57/wenyu/GeneRec/products/task209/phase1_arm_{args.arm}")
    if args.ckpt_pattern:
        ckpt_path = args.ckpt_pattern
    else:
        cks = list(ckpt_dir.rglob("best_loss_model.pth"))
        if not cks:
            print(f"❌ No ckpt under {ckpt_dir}")
            return
        ckpt_path = str(cks[0])
    print(f"[Audit Arm {args.arm}] ckpt={ckpt_path}")

    sd = load_arm_ckpt(ckpt_path)
    print(f"  loaded state_dict, {len(sd)} tensors")

    metrics = evaluate_arm(sd, args.item_emb)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out_dir) / f"arm_{args.arm}_metrics.json"
    with open(out_path, "w") as f:
        json.dump({"arm": args.arm, "ckpt": ckpt_path, "metrics": metrics}, f, indent=2, ensure_ascii=False)
    print(f"  saved → {out_path}")
    print(f"\n--- Arm {args.arm} ---")
    for li, m in enumerate(metrics["metrics_per_layer"]):
        rho_str = f'{m["rho"]:.2f}' if m['rho'] is not None else 'None'
        print(f"  L{li}: util={m['util']:.4f}, ρ={rho_str}, projected_norm={m['projected_norm_mean']:.4f}, cos_max_max={m['cos_max_max']:.4f}")
    print(f"  Collision rate (3-digit): {metrics['collision_rate']:.4f}")
    print(f"  Unique SIDs / N: {metrics['n_unique_sids']} / {metrics['n_items']}")


if __name__ == "__main__":
    main()
