"""Sweep v11/v12 epoch ckpts via directly-loaded HRQVAE (verified collision match).

Trainer-style collision = get_indices(x) over layerwise argmin; matches within 0.05%.
5-cond extracted from trained model's buffers (hyp_mean/hyp_scale/codebooks/log_r).
"""
import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from inspect import signature

REPO = "/home/wlia0047/ar57/wenyu/GeneRec"
HYP_DIMS = {"v11": 8, "v12": 16}


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
    # Remap: norm_target → r_target_norm_list
    if "norm_target" in args_d and "r_target_norm_list" in sig.parameters:
        mkw["r_target_norm_list"] = list(args_d["norm_target"])
    # Parse comma-separated strings
    for k in ["r_target_list"]:
        v = mkw.get(k)
        if isinstance(v, str):
            mkw[k] = [float(x) for x in v.split(",")]
    return HRQVAE(**mkw)


def load_item_embeddings():
    df = pd.read_parquet(f"{REPO}/HG-Rec/dataset/Instruments/item_emb.parquet")
    x = np.stack(df["embedding"].values, axis=0).astype(np.float32)
    return torch.from_numpy(x)


def evaluate_ckpt(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    m = build_model(ckpt)
    m.load_state_dict(state, strict=False)
    m.eval()

    x = load_item_embeddings()
    N = x.shape[0]

    # 1) Collision via get_indices (matches trainer: sk_eps=0 → argmin)
    with torch.no_grad():
        indices = m.get_indices(x, use_sk=True)
    indices_np = indices.cpu().numpy()
    tuples = [tuple(r) for r in indices_np]
    tuple_collision = 1 - len(set(tuples)) / N

    # 2) 5 conditions via saved buffers
    # Get encoder output (post-normcap)
    with torch.no_grad():
        z = m.encoder(x)
    if m.use_normcap and m.normcap_euc_only and m.product_manifold and m.hyp_dim > 0:
        ep = z[:, m.hyp_dim:]
        scale = torch.clamp(m.normcap_target / ep.norm(dim=-1, keepdim=True).clamp_min(1e-12), max=1.0)
        z = torch.cat([z[:, :m.hyp_dim], ep * scale], dim=-1)
    zh, ze = z[:, :m.hyp_dim], z[:, m.hyp_dim:]

    agreements, utils_, cos_stds, radius_onlys, maxc2s = [], [], [], [], []
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
        # Centering + normalize
        z_c = (zh - hyp_mean) / (hyp_scale + 1e-6)
        cb_c = (cb_h - hyp_mean) / (hyp_scale + 1e-6)
        z_n = r_tgt * F.normalize(z_c, dim=-1, eps=1e-6)
        if r_per_k is not None:
            c_n = r_per_k.unsqueeze(-1) * F.normalize(cb_c, dim=-1, eps=1e-6)
        else:
            c_n = r_tgt * F.normalize(cb_c, dim=-1, eps=1e-6)
        # Ball
        z_ball = proj_to_ball(expmap0(z_n, c=1), c=1)
        c_ball = proj_to_ball(expmap0(c_n, c=1), c=1)
        z_exp = z_ball.unsqueeze(1).expand(N, K, -1)
        c_exp = c_ball.unsqueeze(0).expand(N, K, -1)
        hyp_d = poincare_distance(z_exp, c_exp, c=1).squeeze(-1) ** 2
        # Euc MSE
        euc_exp = ze.unsqueeze(1).expand(N, K, -1)
        euc_cb = cb_e.unsqueeze(0).expand(N, K, -1)
        euc_d = ((euc_exp - euc_cb) ** 2).mean(-1)

        combined = m.alpha * hyp_d + m.beta_radial * euc_d
        idx_combined = torch.argmin(combined, dim=-1)
        idx_hyp = torch.argmin(hyp_d, dim=-1)
        idx_euc = torch.argmin(euc_d, dim=-1)
        agreement = (idx_hyp == idx_euc).float().mean().item()
        unique = torch.unique(idx_combined).numel()
        util = unique / K
        # cos_std of latent hyp raw
        z_h_raw_norm = F.normalize(zh, dim=-1, eps=1e-6)
        mean_dir = F.normalize(z_h_raw_norm.mean(0, keepdim=True), dim=-1, eps=1e-6)
        cos_to_mean = (z_h_raw_norm * mean_dir).sum(-1).clamp(-1, 1)
        cos_std = cos_to_mean.std().item()
        # radius-only
        z_r = z_n.norm(dim=-1)
        c_r = c_n.norm(dim=-1)
        d_radius = torch.abs(z_r.unsqueeze(1) - c_r.unsqueeze(0))
        idx_radius = d_radius.argmin(-1)
        radius_only = (idx_radius == idx_hyp).float().mean().item()
        max_c_norm_sq = (cb_e ** 2).sum(-1).max().item()

        agreements.append(agreement)
        utils_.append(util)
        cos_stds.append(cos_std)
        radius_onlys.append(radius_only)
        maxc2s.append(max_c_norm_sq)

    cond1 = all(a < 0.90 for a in agreements)
    cond2 = all(u >= 0.80 for u in utils_)
    cond3 = max(cos_stds) > 0.3
    cond4 = all(r < 0.40 for r in radius_onlys)
    cond5 = max(maxc2s) < 0.5
    return {
        'tuple_collision': tuple_collision,
        'agreement': agreements, 'util': utils_,
        'cos_std': cos_stds, 'radius_only': radius_onlys, 'max_c_norm_sq': maxc2s,
        'all_5cond_pass': cond1 and cond2 and cond3 and cond4 and cond5,
        'collision_pass': tuple_collision <= 0.12,
    }


def main():
    print(f"{'variant':<8} {'ckpt':<55} {'collision':<10} {'agree':<6} {'util%':<6} {'cos_std':<8} {'r-only':<7} {'maxc2':<6} {'5cond':<7} {'coll':<7}")
    print("-" * 130)

    for variant, ang in HYP_DIMS.items():
        sub = "angdim8" if "11" in variant else "angdim16"
        base = f"{REPO}/products/m_arm/m_radius_spread_step3_{variant}_{sub}"
        all_ckpts = []
        if os.path.isdir(base):
            for root, _, files in os.walk(base):
                for f in files:
                    if f.startswith("epoch_") and "_collision_" in f and f.endswith("_model.pth"):
                        all_ckpts.append(os.path.join(root, f))
        all_ckpts = sorted(all_ckpts)
        for ckpt in all_ckpts:
            try:
                r = evaluate_ckpt(ckpt)
                epoch_name = os.path.basename(ckpt).replace("_model.pth", "")
                agree = f"{min(r['agreement']):.3f}"
                util = f"{min(r['util'])*100:.1f}"
                cos_std = f"{max(r['cos_std']):.3f}"
                r_only = f"{min(r['radius_only'])*100:.1f}"
                maxc2 = f"{max(r['max_c_norm_sq']):.3f}"
                cond5 = "PASS" if r['all_5cond_pass'] else "FAIL"
                coll = "PASS" if r['collision_pass'] else "FAIL"
                print(f"{variant:<8} {epoch_name:<55} {r['tuple_collision']*100:<10.2f} {agree:<6} {util:<6} {cos_std:<8} {r_only:<7} {maxc2:<6} {cond5:<7} {coll:<7}")
            except Exception as e:
                import traceback
                print(f"{variant:<8} {os.path.basename(ckpt)}: ERROR {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
