"""Phase 0 判据: κ-Stereographic (Chami 2019 / Ganea 2018) 距离 vs Poincaré.

不重训, 只换 distance formula 在 v11 ep4 ckpt 上重算 assignment.
看 collision / 5-cond 是否有改善 — 这就是攻前提 d (distance formula) 的 Phase 0 验证.

公式 (Chami et al. 2019 "Hyperbolic Graph Convolutional Neural Networks", Sec 3.1):
  Ball: B_c = {x ∈ R^n : c·||x||² < 1}
  exp_0^c(v) = tanh(√c·||v||/2) / (√c·||v||/2) · v
  log_0^c(y) = 2/√c · arctanh(√c·||y||) / (√c·||y||) · y
  d_c(x, y) = (2/√c) · arctan(√c·||x-y|| / |1 - c·⟨x,y⟩|)

vs Poincaré (Berman-Metzler 2020):
  d(x, y) = (1/√c) · arccosh(1 + 2c·||x-y||² / ((1-c||x||²)(1-c||y||²)))

关键差别:
- arctan bounded → κ-Stereo 不会 blow up 近 boundary
- denominator 1 - c·⟨x,y⟩ 对反向码字 (= -1) 有更强 angular repulsion
"""
import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from inspect import signature

REPO = "/home/wlia0047/ar57/wenyu/GeneRec"


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


# ────────────────────────────────────────────────────────────────────
# κ-Stereographic (Chami 2019)
# ────────────────────────────────────────────────────────────────────
def expmap0_stereo(u, c=1):
    """exp_0^c(v) at origin for κ-Stereo (Chami 2019, eq. 5)."""
    n = u.shape[-1]
    sqnorm = u.pow(2).sum(-1, keepdim=True).clamp_min(1e-15)
    return u * (torch.tanh(c**0.5 * sqnorm.sqrt() / 2) / (c**0.5 * sqnorm.sqrt() / 2)) / 1


def logmap0_stereo(y, c=1):
    """log_0^c(y) at origin for κ-Stereo (Chami 2019, eq. 6)."""
    sqnorm = y.pow(2).sum(-1, keepdim=True).clamp_min(1e-15)
    n_sqrt = sqnorm.sqrt()
    return y * (2 / c**0.5 / n_sqrt) * torch.arctanh(c**0.5 * n_sqrt.clamp(max=1.0 - 1e-5 / c**0.5))


def proj_to_stereo_ball(x, c=1, eps=1e-6):
    """Project to κ-Stereo ball: c·||x||² < 1."""
    n = x.norm(dim=-1, keepdim=True).clamp_min(eps)
    m = (1 - eps) / (c**0.5)
    s = torch.where(n > m, m/n, torch.ones_like(n))
    return x * s


def stereo_distance(x, y, c=1):
    """d_c(x, y) = (2/√c) · arctan(√c·||x-y|| / |1 - c·⟨x,y⟩|)  (Chami 2019, eq. 7).

    Note: arctan is bounded by π/2, so distance never blows up.
    """
    diff = x - y
    inner = (x * y).sum(-1)
    num = c**0.5 * diff.norm(dim=-1).clamp_min(1e-15)
    den = (1 - c * inner).abs().clamp_min(1e-15)
    return (2.0 / c**0.5) * torch.atan(num / den)


def load_item_embeddings():
    df = pd.read_parquet(f"{REPO}/HG-Rec/dataset/Instruments/item_emb.parquet")
    x = np.stack(df["embedding"].values, axis=0).astype(np.float32)
    return torch.from_numpy(x)


def build_model(ckpt):
    sys.path.insert(0, os.path.join(REPO, "HG-Rec"))
    from model.hrqvae import HRQVAE
    sig = signature(HRQVAE.__init__)
    args_d = vars(ckpt["args"])
    mkw = {"in_dim": 768, "num_emb_list": [64, 128, 256], "e_dim": 34}
    for k, v in args_d.items():
        nk = k.replace("-", "_")
        if nk in sig.parameters:
            mkw[nk] = v
    if "norm_target" in args_d and "r_target_norm_list" in sig.parameters:
        mkw["r_target_norm_list"] = list(args_d["norm_target"])
    for k in ["r_target_list"]:
        v = mkw.get(k)
        if isinstance(v, str):
            mkw[k] = [float(x) for x in v.split(",")]
    return HRQVAE(**mkw)


def check_stereo_distance_properties():
    """Sanity check on κ-Stereo distance: bounded? symmetric? triangle? near boundary?"""
    print("=" * 70)
    print("Phase 0.1 — κ-Stereo distance sanity check")
    print("=" * 70)
    torch.manual_seed(42)
    c = 1.0
    # Random points in ball
    for trial in range(3):
        x = torch.randn(10, 2) * 0.3
        y = torch.randn(10, 2) * 0.3
        x = proj_to_stereo_ball(x, c=c)
        y = proj_to_stereo_ball(y, c=c)
        d_stereo = stereo_distance(x, y, c=c)
        d_poinc = poincare_distance(x, y, c=c)
        print(f"  trial {trial}: stereo dist min/max/mean = {d_stereo.min():.4f} / {d_stereo.max():.4f} / {d_stereo.mean():.4f}")
        print(f"           poinc  dist min/max/mean = {d_poinc.min():.4f} / {d_poinc.max():.4f} / {d_poinc.mean():.4f}")
        # Symmetry
        assert torch.allclose(d_stereo, stereo_distance(y, x, c=c), atol=1e-5), "Stereo not symmetric!"
        # d(x,x) = 0
        assert torch.allclose(stereo_distance(x, x, c=c), torch.zeros_like(d_stereo), atol=1e-5), "Stereo d(x,x) != 0"
    # Near boundary stability
    print("\nNear-boundary stability (2D, norm=0.95):")
    x = torch.tensor([[0.95, 0.0]])
    for r in [0.5, 0.95, 0.999]:
        y = torch.tensor([[r * 0.95, r * 0.3]]).clamp_max(0.99)
        y = y / y.norm() * 0.95
        try:
            d_s = stereo_distance(x, y, c=c).item()
            d_p = poincare_distance(x, y, c=c).item()
        except Exception as e:
            d_s, d_p = float('nan'), float('nan')
        print(f"  r={r}: stereo={d_s:.4f}, poincare={d_p:.4f}")
    print()


def evaluate_ckpt_with_distance(ckpt_path, distance_mode="poincare"):
    """Re-evaluate ckpt with swapped distance formula (residual-aware).

    distance_mode ∈ {'poincare', 'stereo', 'stereo_ball'}.
    'stereo' uses ball coords (no expmap). 'stereo_ball' uses expmap0_stereo.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    m = build_model(ckpt)
    m.load_state_dict(state, strict=False)
    m.eval()

    x = load_item_embeddings()
    N = x.shape[0]
    with torch.no_grad():
        z = m.encoder(x)
    if m.use_normcap and m.normcap_euc_only and m.product_manifold and m.hyp_dim > 0:
        ep = z[:, m.hyp_dim:]
        scale = torch.clamp(m.normcap_target / ep.norm(dim=-1, keepdim=True).clamp_min(1e-12), max=1.0)
        z = torch.cat([z[:, :m.hyp_dim], ep * scale], dim=-1)

    if distance_mode == "poincare":
        use_stereo = False
        use_expmap = True
    elif distance_mode == "stereo":
        use_stereo = True
        use_expmap = False
    elif distance_mode == "stereo_ball":
        use_stereo = True
        use_expmap = True
    else:
        raise ValueError(distance_mode)

    # Residual-aware loop: layer 1 input = encoder - L0_q, layer 2 input = encoder - L0_q - L1_q
    residual = z
    idx_per_layer = []
    for li, vq in enumerate(m.hrq.vq_layers):
        K = [64, 128, 256][li]
        r_tgt = vq.r_target_norm
        r_spread = vq.r_spread
        w = vq.embeddings.weight.detach()
        cb_h = w[:, :m.hyp_dim]
        cb_e = w[:, m.hyp_dim:]
        if r_spread > 0 and hasattr(vq, "log_r"):
            r_per_k = r_tgt + r_spread * (2 * torch.sigmoid(vq.log_r.detach()) - 1)
        else:
            r_per_k = None
        hyp_mean = vq.hyp_mean.detach()
        hyp_scale = vq.hyp_scale.detach()
        # Use residual (not encoder) for hyp part input
        zh_r, ze_r = residual[:, :m.hyp_dim], residual[:, m.hyp_dim:]
        z_c = (zh_r - hyp_mean) / (hyp_scale + 1e-6)
        cb_c = (cb_h - hyp_mean) / (hyp_scale + 1e-6)
        z_n = r_tgt * F.normalize(z_c, dim=-1, eps=1e-6)
        if r_per_k is not None:
            c_n = r_per_k.unsqueeze(-1) * F.normalize(cb_c, dim=-1, eps=1e-6)
        else:
            c_n = r_tgt * F.normalize(cb_c, dim=-1, eps=1e-6)

        z_ball = z_n
        c_ball = c_n
        if use_expmap:
            if use_stereo:
                z_ball = proj_to_stereo_ball(expmap0_stereo(z_n, c=1), c=1)
                c_ball = proj_to_stereo_ball(expmap0_stereo(c_n, c=1), c=1)
            else:
                z_ball = proj_to_ball(expmap0(z_n, c=1), c=1)
                c_ball = proj_to_ball(expmap0(c_n, c=1), c=1)

        z_exp = z_ball.unsqueeze(1).expand(N, K, -1)
        c_exp = c_ball.unsqueeze(0).expand(N, K, -1)
        if use_stereo:
            hyp_d = stereo_distance(z_exp, c_exp, c=1)
        else:
            hyp_d = poincare_distance(z_exp, c_exp, c=1).squeeze(-1) ** 2
        euc_exp = ze_r.unsqueeze(1).expand(N, K, -1)
        euc_cb = cb_e.unsqueeze(0).expand(N, K, -1)
        euc_d = ((euc_exp - euc_cb) ** 2).mean(-1)

        combined = m.alpha * hyp_d + m.beta_radial * euc_d
        idx_combined = torch.argmin(combined, dim=-1)
        idx_per_layer.append(idx_combined)
        # Subtract quantized for next layer's residual
        x_q = w.index_select(0, idx_combined)
        residual = residual - x_q

    arr = torch.stack(idx_per_layer, dim=1).cpu().numpy()
    tuples = [tuple(r) for r in arr]
    tuple_collision = 1 - len(set(tuples)) / N
    return tuple_collision


def main():
    check_stereo_distance_properties()

    print("=" * 70)
    print("Phase 0.2 — κ-Stereo on v11 ep4 (collision 5.45% Poincaré baseline)")
    print("=" * 70)
    ckpt_path = (f"{REPO}/products/m_arm/m_radius_spread_step3_v11_angdim8/"
                 "Jul-27-2026_14-21-44_beta_0.500_codebook_[64,128,256]_sk_0.000/"
                 "epoch_4_collision_0.0548_model.pth")
    print(f"ckpt: {ckpt_path}\n")

    for mode in ["poincare", "stereo", "stereo_ball"]:
        c = evaluate_ckpt_with_distance(ckpt_path, distance_mode=mode)
        print(f"  distance_mode={mode:14s}  collision = {c*100:6.2f}%")

    print("\n" + "=" * 70)
    print("Phase 0.3 — Same on v12 ep4 (collision 8.35% Poincaré baseline)")
    print("=" * 70)
    ckpt_path = (f"{REPO}/products/m_arm/m_radius_spread_step3_v12_angdim16/"
                 "Jul-27-2026_14-21-44_beta_0.500_codebook_[64,128,256]_sk_0.000/"
                 "epoch_4_collision_0.0844_model.pth")
    print(f"ckpt: {ckpt_path}\n")
    for mode in ["poincare", "stereo", "stereo_ball"]:
        c = evaluate_ckpt_with_distance(ckpt_path, distance_mode=mode)
        print(f"  distance_mode={mode:14s}  collision = {c*100:6.2f}%")

    print("\n" + "=" * 70)
    print("Phase 0.4 — Same on v6 ep50 (Task #226 PASS 5-cond, collision 95.68%)")
    print("=" * 70)
    v6_dir = f"{REPO}/products/m_arm/m_radius_spread_step3_50ep_wdiv100_wangular"
    # Find an epoch ckpt
    import glob as _g
    ckpts = sorted(_g.glob(f"{v6_dir}/*/epoch_*_collision_*.pth"))
    if not ckpts:
        ckpts = sorted(_g.glob(f"{v6_dir}/**/epoch_*_collision_*.pth"))
    # Pick ep 50 or last
    if ckpts:
        ckpt_path = ckpts[-1]
        print(f"ckpt: {ckpt_path}\n")
        for mode in ["poincare", "stereo", "stereo_ball"]:
            c = evaluate_ckpt_with_distance(ckpt_path, distance_mode=mode)
            print(f"  distance_mode={mode:14s}  collision = {c*100:6.2f}%")


if __name__ == "__main__":
    main()