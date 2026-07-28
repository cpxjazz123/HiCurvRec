"""Standard downstream collision rate on v6 ckpt.

Usage: python3 scripts/m_arm_collision_rate.py <ckpt_path>

Collision rate definition (下游 SID 评估):
  - L*_collision = 1 - (unique c_* 分配 / N_items)
  - tuple_collision = 1 - (unique (c1,c2,c3) tuples / N_items)

HG-Rec reproduction range ≤ 12% 指的是 overall tuple collision rate.
"""
import sys, os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO = "/home/wlia0047/ar57/wenyu/GeneRec"
HYP_DIM, RADIAL_DIM, E_DIM = 2, 32, 34
R_TARGET_NORMS = [1.0, 1.35, 1.70]
R_SPREAD = 0.3


def expmap0(u, c=1):
    nu = u.norm(dim=-1, keepdim=True).clamp_min(1e-15)
    return torch.tanh(c**0.5 * nu) / (c**0.5 * nu) * u


def proj_to_ball(x, c=1, eps=1e-6):
    n = x.norm(dim=-1, keepdim=True).clamp_min(eps)
    m = (1 - eps) / (c**0.5)
    s = torch.where(n > m, m/n, torch.ones_like(n))
    return x * s


def poincare_distance(x, y, c=1):
    d = (x - y).pow(2).sum(-1)
    xn, yn = x.pow(2).sum(-1), y.pow(2).sum(-1)
    num = 2 * c * d
    den = (1 - c*xn) * (1 - c*yn)
    arg = (1 + num/den.clamp_min(1e-15)).clamp_min(1+1e-15)
    return torch.acosh(arg) / c**0.5


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(REPO, "HG-Rec"))
    from model.utils import MLP

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    df = pd.read_parquet(f"{REPO}/HG-Rec/dataset/Instruments/item_emb.parquet")
    x = torch.from_numpy(np.stack(df["embedding"].values, axis=0)).float()
    N = x.shape[0]
    print(f"[input] N={N}, dim={x.shape[1]}")

    enc = MLP(layers=[768, 512, 256, 128, 64, E_DIM])
    enc_w = {k.replace("encoder.", ""): v for k, v in state.items() if k.startswith("encoder.")}
    enc.load_state_dict(enc_w, strict=False)
    enc.eval()

    with torch.no_grad():
        z = enc(x)
        ep = z[:, HYP_DIM:]
        scale = torch.clamp(0.3 / ep.norm(dim=-1, keepdim=True).clamp_min(1e-12), max=1.0)
        z = torch.cat([z[:, :HYP_DIM], ep * scale], dim=-1)
    zh, ze = z[:, :HYP_DIM], z[:, HYP_DIM:]

    print(f"\n=== Standard collision rate (downstream SID assignment) ===\n")
    print(f"  N items = {N}")
    print(f"\n{'层':<4} {'K':<5} {'unique':<8} {'util%':<8} {'collision%':<12}")
    print("-" * 50)

    all_indices = []
    for li in [0, 1, 2]:
        K = [64, 128, 256][li]
        r_tgt = R_TARGET_NORMS[li]
        w = state[f"hrq.vq_layers.{li}.embeddings.weight"]
        cb_h = w[:, :HYP_DIM]
        cb_e = w[:, HYP_DIM:]
        # 必须跟 _product_manifold_distance 实际公式一致:
        # z 钉到 scalar r_tgt, cb 用 per-codeword r_per_k (= r_target + spread*(2σ(log_r)-1))
        zhn = r_tgt * F.normalize(zh, dim=-1, eps=1e-6)
        log_r = state[f"hrq.vq_layers.{li}.log_r"]
        r_per_k = r_tgt + R_SPREAD * (2 * torch.sigmoid(log_r) - 1)
        cbn = r_per_k.unsqueeze(-1) * F.normalize(cb_h, dim=-1, eps=1e-6)
        zhs = proj_to_ball(expmap0(zhn, c=1), c=1)
        cbs = proj_to_ball(expmap0(cbn, c=1), c=1)

        # d_hyp (argmin)
        zhe = zhs.unsqueeze(1).expand(N, K, -1)
        cbe = cbs.unsqueeze(0).expand(N, K, -1)
        d_hyp = poincare_distance(zhe, cbe, c=1)
        # d_euc (32D MSE-style)
        zee = ze.unsqueeze(1).expand(N, K, -1)
        cee = cb_e.unsqueeze(0).expand(N, K, -1)
        d_euc = ((zee - cee) ** 2).mean(-1)
        combined = d_hyp + d_euc  # α=β=1
        idx_l = combined.argmin(-1)
        all_indices.append(idx_l.cpu().numpy())
        unique_l = np.unique(idx_l.numpy()).size
        util = unique_l / K
        collision = 1 - unique_l / N
        print(f"L{li:<3} {K:<5} {unique_l:<8} {util*100:<8.2f} {collision*100:<12.2f}")

    arr = np.stack(all_indices, axis=1)  # (N, 3)
    tuples = [tuple(r) for r in arr]
    unique_tuple = len(set(tuples))
    tuple_collision = 1 - unique_tuple / N
    print(f"\n--- 整 (c1, c2, c3) tuple ---")
    print(f"  unique tuples = {unique_tuple} / {N}")
    print(f"  tuple collision_rate = {tuple_collision*100:.2f}%")
    print(f"\n  HG-Rec 复现范围目标: ≤ 12%")
    print(f"  实际: {tuple_collision*100:.2f}%")
    if tuple_collision <= 0.12:
        print(f"  ✅ PASS")
    else:
        print(f"  ❌ FAIL (差 {(tuple_collision-0.12)*100:.2f}%)")


if __name__ == "__main__":
    main()
