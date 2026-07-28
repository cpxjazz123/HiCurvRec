"""Corrected Step 3 评估 — 真正测 "曲率在同子空间是否改变分配".

旧版错误: 把 32D euc argmin 跟 2D hyp argmin 比 → 测的是分支差异.
正确版: 都在 z_h (2D) / cb_h (2D) 这同一批点上:
  - 曲线 argmin = argmin_k d_poincare(z_h, h_k)
  - 直线 argmin = argmin_k ‖z_h - h_k‖  (普通欧氏, 跟 z/h 在同一 2D 平面)
对比这两条 — 这才是 "曲率是否改变码字分配".

顺带: 还做一个反向 32D-only, 看 d_hyp(2D) 跟 32D 的 d_euc 比 — 这是 "分支差", 跟旧版一样, 作为 sanity ref.
"""
import sys, os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO = "/home/wlia0047/ar57/wenyu/GeneRec"
ANGULAR_DIM = 2
RADIAL_DIM = 32
E_DIM = 34
HYP_DIM = ANGULAR_DIM


def expmap0(u, c):
    norm_u = u.norm(dim=-1, keepdim=True).clamp_min(1e-15)
    factor = torch.tanh(c ** 0.5 * norm_u) / (c ** 0.5 * norm_u)
    return factor * u


def proj_to_ball(x, c=1.0, eps=1e-6):
    norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
    max_norm = (1 - eps) / (c ** 0.5)
    scale = torch.where(norm > max_norm, max_norm / norm, torch.ones_like(norm))
    return x * scale


def poincare_distance(x, y, c=1.0):
    diff_sq = (x - y).pow(2).sum(dim=-1)
    x_norm_sq = x.pow(2).sum(dim=-1)
    y_norm_sq = y.pow(2).sum(dim=-1)
    num = 2.0 * c * diff_sq
    den = (1.0 - c * x_norm_sq) * (1.0 - c * y_norm_sq)
    arg = 1.0 + num / den.clamp_min(1e-15)
    arg = arg.clamp_min(1.0 + 1e-15)
    return torch.acosh(arg) / (c ** 0.5)


class MLP(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        mods = []
        for idx, (in_d, out_d) in enumerate(zip(layers[:-1], layers[1:])):
            if idx > 0:
                mods.append(torch.nn.ReLU())
            mods.append(torch.nn.Linear(in_d, out_d))
        self.mlp = torch.nn.Sequential(*mods)
    def forward(self, x):
        return self.mlp(x)


def load_model(ckpt_path, device="cpu"):
    ckpt_raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt_raw["state_dict"] if isinstance(ckpt_raw, dict) and "state_dict" in ckpt_raw else ckpt_raw
    sys.path.insert(0, os.path.join(REPO, "HG-Rec"))
    from model.utils import MLP as RealMLP
    enc = RealMLP(layers=[768, 512, 256, 128, 64, E_DIM])
    enc_w = {k.replace("encoder.", ""): v
             for k, v in state.items() if k.startswith("encoder.")}
    enc.load_state_dict(enc_w, strict=False)
    enc.eval()
    return enc, state


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    args = ap.parse_args()

    df = pd.read_parquet(os.path.join(REPO, "HG-Rec/dataset/Instruments/item_emb.parquet"))
    x = torch.from_numpy(np.stack(df["embedding"].values, axis=0)).float()
    N = x.shape[0]
    enc, state = load_model(args.ckpt)
    with torch.no_grad():
        z_lat = enc(x)
        euc_part = z_lat[:, HYP_DIM:]
        euc_norm = euc_part.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        scale_euc = torch.clamp(0.3 / euc_norm, max=1.0)
        z_lat = torch.cat([z_lat[:, :HYP_DIM], euc_part * scale_euc], dim=-1)
    z_hyp = z_lat[:, :HYP_DIM]
    z_euc = z_lat[:, HYP_DIM:]

    R_TARGET_NORMS = [1.0, 1.35, 1.70]
    R_SPREAD = 0.3  # 跟 launcher --r_spread_list 0.3 一致
    print(f"=== 修正 Step 3 v2: cb 侧 per-codeword r spread, z 侧 scalar r (跟 _product_manifold_distance 一致) ===\n")
    print(f"{'层':<4} {'agreement_2D_correct':<22} {'agreement_branch(旧错)':<22} "
          f"{'2D hyp util':<12} {'2D L2 util':<12}")
    print(f"{'-'*4} {'-'*22} {'-'*22} {'-'*12} {'-'*12}")

    for li in [0, 1, 2]:
        K = int([64, 128, 256][li])
        r_tgt = R_TARGET_NORMS[li]
        w = state[f"hrq.vq_layers.{li}.embeddings.weight"]
        cb_h = w[:, :HYP_DIM]
        cb_e = w[:, HYP_DIM:]

        # ===== 跟 _product_manifold_distance 实际公式一致 =====
        # z 侧: scalar tangent_norm = r_tgt
        z_h_n = r_tgt * F.normalize(z_hyp, dim=-1, eps=1e-6)
        # cb 侧: per-codeword r_per_k = r_tgt + r_spread*(2σ(log_r)-1)
        # log_r 必须从 ckpt 读 (state 里有 'hrq.vq_layers.{li}.log_r')
        try:
            log_r = state[f"hrq.vq_layers.{li}.log_r"]
            r_per_k = r_tgt + R_SPREAD * (2 * torch.sigmoid(log_r) - 1)  # (K,)
            # 用 EMA 中心化 (跟 _hyp_layernorm_normalize 一致)
            # 简单版: 直接 F.normalize 然后乘 r_per_k
            cb_h_normed = F.normalize(cb_h, dim=-1, eps=1e-6)
            cb_h_n = r_per_k.unsqueeze(-1) * cb_h_normed  # (K, 2)
        except KeyError:
            # 备选: 全用 scalar r (退化情况)
            cb_h_n = r_tgt * F.normalize(cb_h, dim=-1, eps=1e-6)

        # 投影到 Poincaré 球
        z_h_sphere = proj_to_ball(expmap0(z_h_n, c=1.0), c=1.0)
        cb_h_sphere = proj_to_ball(expmap0(cb_h_n, c=1.0), c=1.0)
        # 同 2D 子空间内的 d_poincare
        z_h_exp = z_h_sphere.unsqueeze(1).expand(N, K, -1)
        cb_h_exp = cb_h_sphere.unsqueeze(0).expand(N, K, -1)
        d_curve = poincare_distance(z_h_exp, cb_h_exp, c=1.0)  # (N, K)
        idx_curve = d_curve.argmin(dim=-1)

        # 同 2D 子空间内的欧氏 L2 (用同一批 z_h_sphere, cb_h_sphere)
        d_euc_2d = (z_h_exp - cb_h_exp).pow(2).sum(dim=-1).sqrt()  # (N, K)
        idx_euc_2d = d_euc_2d.argmin(dim=-1)

        # (b) 同 2D 子空间的欧氏 L2 距离
        # 用的就是 2D 球面坐标 (跟 d_curve 完全同一批点)
        d_euc_2d = (z_h_exp - cb_h_exp).pow(2).sum(dim=-1).sqrt()  # (N, K)
        idx_euc_2d = d_euc_2d.argmin(dim=-1)

        # ====== 修正版的核心指标 ======
        agreement_2d_corr = (idx_curve == idx_euc_2d).float().mean().item()

        # ====== 旧版 (错) 的核心指标, 作为对照 ======
        # 32D euc 分支 + 2D hyp 分支
        z_e_n = F.normalize(z_euc, dim=-1)
        cb_e_n = F.normalize(cb_e, dim=-1)
        ze_exp = z_e_n.unsqueeze(1).expand(N, K, -1)
        ce_exp = cb_e_n.unsqueeze(0).expand(N, K, -1)
        d_euc_32d = ((ze_exp - ce_exp) ** 2).sum(dim=-1).sqrt()
        idx_euc_32d = d_euc_32d.argmin(dim=-1)
        agreement_branch = (idx_curve == idx_euc_32d).float().mean().item()

        # 利用率 (跟分支无关)
        util_2d_curve = idx_curve.unique().numel() / K
        util_2d_euc = idx_euc_2d.unique().numel() / K

        print(f"L{li}   agreement_2D={agreement_2d_corr:<18.4f}   "
              f"agreement_branch={agreement_branch:<14.4f}   "
              f"{util_2d_curve:<12.4f} {util_2d_euc:<12.4f}")

    print(f"\n=== 解读 ===")
    print(f"  - 修正版 agreement (同 2D 子空间内 d_poincare vs L2): 测的是曲率是否改变码字分配.")
    print(f"  - 旧版 agreement (32D 分支 vs 2D 分支): 测的是分支差, 跟曲率无关.")
    print(f"  - 如果两者数字相近 → 旧版碰巧没伪测, 仍可能 PASS.")
    print(f"  - 如果修正版数字接近 1.0 → 曲率真没改变 2D 子空间内分配, 旧版 PASS 是 false positive.")


if __name__ == "__main__":
    main()
