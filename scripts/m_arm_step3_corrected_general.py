"""Generalized 5-cond verification on M-arm ckpt with variable angular_dim.

Usage: python3 scripts/m_arm_step3_corrected_general.py <ckpt_path> [ang_dim]
"""
import sys, os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO = "/home/wlia0047/ar57/wenyu/GeneRec"
HYP_DIM_DEFAULT = 2
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


def get_ckpt_angular_dim(state):
    for k in state.keys():
        if k.startswith("hrq.vq_layers.0.embeddings.weight"):
            return state[k].shape[1] - (state["hrq.vq_layers.0.embeddings.weight"].shape[1] - 2)
    raise RuntimeError("Can't determine angular_dim from state")


def main():
    ckpt_path = sys.argv[1]
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    # Detect angular_dim from state
    L0_weight = state["hrq.vq_layers.0.embeddings.weight"]
    E_DIM = L0_weight.shape[1]  # 34 for our configs
    # Identify hyp_dim by structure: normcap splits hyp (first) + euc (rest)
    # We have `radial_dim` from normcap_target config — but ckpt doesn't store.
    # Inspect embeddings: cb_h = weight[:, :hyp_dim], cb_e = weight[:, hyp_dim:]
    # For product_manifold, hyp_dim = angular_dim (config).
    # We must parse from a saved arg like '--angular_dim' — let's look for any hint.
    # Default fallback: ang_dim from sys.argv[2] if provided.
    if len(sys.argv) > 2:
        HYP_DIM = int(sys.argv[2])
    else:
        # Heuristic: normcap_target=0.3 means euc part kept at 0.3 — weight[:, hyp_dim:].norm(dim=-1) ~0.3
        euc_norms = L0_weight.norm(dim=-1)
        # euc_norms should be small (~0.3) if normcap active; > 1 if hyp dominant
        # Just try default 2 and warn if bad
        HYP_DIM = 2
        print(f"⚠️ ang_dim not specified, assuming 2")

    RADIAL_DIM = E_DIM - HYP_DIM
    print(f"[input] ckpt={ckpt_path}")
    print(f"        E_DIM={E_DIM}, HYP_DIM={HYP_DIM}, RADIAL_DIM={RADIAL_DIM}")

    df = pd.read_parquet(f"{REPO}/HG-Rec/dataset/Instruments/item_emb.parquet")
    x = torch.from_numpy(np.stack(df["embedding"].values, axis=0)).float()
    N = x.shape[0]

    sys.path.insert(0, os.path.join(REPO, "HG-Rec"))
    from model.utils import MLP

    enc = MLP(layers=[768, 512, 256, 128, 64, E_DIM])
    enc_w = {k.replace("encoder.", ""): v for k, v in state.items() if k.startswith("encoder.")}
    enc.load_state_dict(enc_w, strict=False)
    enc.eval()

    with torch.no_grad():
        z = enc(x)
        # If normcap_euc_only, scale euc part to normcap_target=0.3
        ep = z[:, HYP_DIM:]
        scale = torch.clamp(0.3 / ep.norm(dim=-1, keepdim=True).clamp_min(1e-12), max=1.0)
        z = torch.cat([z[:, :HYP_DIM], ep * scale], dim=-1)
    zh, ze = z[:, :HYP_DIM], z[:, HYP_DIM:]

    R_TARGET_NORMS = [1.0, 1.35, 1.70]
    print(f"\n=== 5-condition verification (ang_dim={HYP_DIM}) ===\n")
    print(f"{'层':<4} {'agreement':<10} {'util%':<8} {'cos_std':<8} {'radius-only%':<14} {'max(c·‖e_k‖²)':<14}")

    cos_stds = []
    all_agreements = []
    all_radius_only = []
    all_util = []
    all_max_cnorm = []

    for li in [0, 1, 2]:
        K = [64, 128, 256][li]
        r_tgt = R_TARGET_NORMS[li]
        w = state[f"hrq.vq_layers.{li}.embeddings.weight"]
        cb_h = w[:, :HYP_DIM]
        cb_e = w[:, HYP_DIM:]
        log_r = state[f"hrq.vq_layers.{li}.log_r"]
        r_per_k = r_tgt + R_SPREAD * (2 * torch.sigmoid(log_r) - 1)

        # Project to ball with per-codeword radius
        zhn = r_tgt * F.normalize(zh, dim=-1, eps=1e-6)
        cbn = r_per_k.unsqueeze(-1) * F.normalize(cb_h, dim=-1, eps=1e-6)
        zhs = proj_to_ball(expmap0(zhn, c=1), c=1)
        cbs = proj_to_ball(expmap0(cbn, c=1), c=1)

        # argmin in hyp (poincare)
        zhe = zhs.unsqueeze(1).expand(N, K, -1)
        cbe = cbs.unsqueeze(0).expand(N, K, -1)
        d_hyp = poincare_distance(zhe, cbe, c=1)
        # argmin in euclidean (same subspace, in ball coords)
        d_euc = ((zhe - cbe) ** 2).mean(-1)

        idx_hyp = d_hyp.argmin(-1)
        idx_euc = d_euc.argmin(-1)
        agreement = (idx_hyp == idx_euc).float().mean().item()

        # utilization (euclidean argmin, since that's the natural partition)
        unique = torch.unique(idx_euc).numel()
        util = unique / K

        # cos_std of z_h (encoder hyp output direction)
        z_h_norm = F.normalize(zh, dim=-1, eps=1e-6)
        mean_dir = F.normalize(z_h_norm.mean(0, keepdim=True), dim=-1, eps=1e-6)
        cos_to_mean = (z_h_norm * mean_dir).sum(-1).clamp(-1, 1)
        cos_std = cos_to_mean.std().item()

        # radius-only reproduction rate
        z_r = zhn.norm(dim=-1)
        cb_r = cbn.norm(dim=-1)
        # Sort by radius, find argmin
        z_r_idx = z_r.argsort()
        cb_r_idx = cb_r.argsort()
        # Map: items with same rank share a code
        # radius-only reproduction: items within same radius-bucket share same code
        # Simpler: split items into K bins by radius, items in same bin → same code
        rank_z = z_r.argsort().argsort()  # rank of each item
        bin_size = N // K
        bin_z = rank_z // bin_size
        rank_cb = cb_r.argsort().argsort()
        bin_size_cb = K // K  # each code is its own bin
        bin_cb = rank_cb
        # For each item, find cb with closest radius bin
        radius_code = bin_z  # simple: each item gets code = its radius bin
        # Simpler metric: count how often argmin(rank) coincides with argmin(d_hyp)
        # We use a much simpler test: does argmin by radius alone produce same idx?
        d_radius_only = torch.abs(z_r.unsqueeze(1) - cb_r.unsqueeze(0))  # (N, K)
        idx_radius = d_radius_only.argmin(-1)
        radius_only = (idx_radius == idx_hyp).float().mean().item()

        # max(c·‖e_k‖²)
        max_c_norm_sq = (cb_e ** 2).sum(-1).max().item()

        all_agreements.append(agreement)
        all_util.append(util)
        cos_stds.append(cos_std)
        all_radius_only.append(radius_only)
        all_max_cnorm.append(max_c_norm_sq)

        print(f"L{li:<3} {agreement:<10.4f} {util*100:<8.2f} {cos_std:<8.4f} {radius_only*100:<14.4f} {max_c_norm_sq:<14.4f}")

    print(f"\n=== 条件检查 ===")
    cond1 = all(a < 0.90 for a in all_agreements)
    cond2 = all(u >= 0.80 for u in all_util)
    cond3 = max(cos_stds) > 0.3
    cond4 = all(r < 0.40 for r in all_radius_only)
    cond5 = max(all_max_cnorm) < 0.5
    print(f"1. agreement < 0.90 (3 层): {all_agreements} {'✅' if cond1 else '❌'}")
    print(f"2. utilization ≥ 0.80 (3 层): {all_util} {'✅' if cond2 else '❌'}")
    print(f"3. cos_std > 0.3: max={max(cos_stds):.4f} {'✅' if cond3 else '❌'}")
    print(f"4. radius-only < 40% (3 层): {all_radius_only} {'✅' if cond4 else '❌'}")
    print(f"5. max(c·‖e_k‖²) < 0.5: max={max(all_max_cnorm):.4f} {'✅' if cond5 else '❌'}")

    if all([cond1, cond2, cond3, cond4, cond5]):
        print(f"\n✅ 全部 5 条件 PASS")
    else:
        print(f"\n⚠️ 5 条件有失败, 见上")


if __name__ == "__main__":
    main()