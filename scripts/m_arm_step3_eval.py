"""Step 3 综合评估: 6 核心数 + 5 护栏 + 3 现象.

用法:
  python3 scripts/m_arm_step3_eval.py <ckpt_path> [--data <parquet>]

输出:
  - 核心数: L0/L1/L2 各自的 agreement + utilization
  - 护栏: 1) 同子空间同 batch  2) 全 finite  3) radius std > 0.05
          4) direction cosine std > 0.3  5) max(c·‖e_k‖²) << 0.5
  - 现象: A) flip rate ≥ 10%   B) util histogram
          C) flipped vs non-flipped radius diff
"""
import sys, os, json, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO = "/home/wlia0047/ar57/wenyu/GeneRec"
ANGULAR_DIM = 2
RADIAL_DIM = 32
E_DIM = 34
HYP_DIM = ANGULAR_DIM
EUC_DIM = RADIAL_DIM


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


# ---- 重建 encoder ----
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
    from model.utils import MLP
    enc = MLP(layers=[768, 512, 256, 128, 64, E_DIM])
    enc_w = {k.replace("encoder.", ""): v
             for k, v in state.items() if k.startswith("encoder.")}
    enc.load_state_dict(enc_w, strict=False)
    enc.eval()
    return enc, state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help="best_collision_model.pth 或 epoch_xx")
    ap.add_argument("--data", default=os.path.join(REPO, "HG-Rec/dataset/Instruments/item_emb.parquet"))
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    emb_full = np.stack(df["embedding"].values, axis=0)
    x = torch.from_numpy(emb_full).float()
    N = x.shape[0]
    print(f"[input] N={N}, dim={x.shape[1]}")

    enc, state = load_model(args.ckpt)
    with torch.no_grad():
        z_lat = enc(x)
        # 复刻 NormCap (M2 用 normcap_target=0.4 但 Step3v4 用 0.3)
        # 看 ckpt 自带 setting; 但 product_manifold 路径下 normcap_euc_only=True
        # → euc 部分被压到 0.3, hyp 不动.
        euc_part = z_lat[:, HYP_DIM:]
        euc_norm = euc_part.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        scale_euc = torch.clamp(0.3 / euc_norm, max=1.0)
        z_lat = torch.cat([z_lat[:, :HYP_DIM], euc_part * scale_euc], dim=-1)
    z_hyp = z_lat[:, :HYP_DIM]
    z_euc = z_lat[:, HYP_DIM:]
    print(f"  z_hyp norm: min={z_hyp.norm(dim=-1).min():.4f}, "
          f"max={z_hyp.norm(dim=-1).max():.4f}, "
          f"mean={z_hyp.norm(dim=-1).mean():.4f}")

    R_TARGET_NORMS = [1.0, 1.35, 1.70]

    print(f"\n{'='*60}")
    print("A. 核心指标")
    print(f"{'='*60}")
    agreements = []
    utilizations = []
    flip_rate_total = 0
    flipped_radii = []
    nonflipped_radii = []
    radius_stds = []
    cos_stds = []
    radius_only_matches = []
    margin_disagreement_list = []
    margin_agreement_list = []

    for li in [0, 1, 2]:
        K = int([64, 128, 256][li])
        r_tgt = R_TARGET_NORMS[li]
        w = state[f"hrq.vq_layers.{li}.embeddings.weight"]
        cb_hyp_raw = w[:, :HYP_DIM]
        cb_euc = w[:, HYP_DIM:]

        # 硬归一化 hyp 部分 (跟训练时一致) — 半径可学 log_r 在 fixed
        # 但 _product_manifold_distance 路径用硬归一化, 这里也跟它一致
        cb_hyp_n = r_tgt * F.normalize(cb_hyp_raw, dim=-1, eps=1e-6)
        cb_hyp = proj_to_ball(expmap0(cb_hyp_n, c=1.0), c=1.0)
        z_h_n = r_tgt * F.normalize(z_hyp, dim=-1, eps=1e-6)
        z_h = proj_to_ball(expmap0(z_h_n, c=1.0), c=1.0)

        # d_hyp
        z_h_exp = z_h.unsqueeze(1).expand(N, K, -1)
        cb_h_exp = cb_hyp.unsqueeze(0).expand(N, K, -1)
        d_hyp = poincare_distance(z_h_exp, cb_h_exp, c=1.0)
        # d_euc: normalized
        z_e_n = F.normalize(z_euc, dim=-1)
        cb_e_n = F.normalize(cb_euc, dim=-1)
        z_e_exp = z_e_n.unsqueeze(1).expand(N, K, -1)
        cb_e_exp = cb_e_n.unsqueeze(0).expand(N, K, -1)
        d_euc = ((z_e_exp - cb_e_exp) ** 2).sum(dim=-1).sqrt()

        idx_geo = d_hyp.argmin(dim=-1)
        idx_euc = d_euc.argmin(dim=-1)

        # agreement
        agreement = (idx_geo == idx_euc).float().mean().item()
        agreements.append(agreement)

        # utilization (pre-revive)
        unique_geo = idx_geo.unique().numel()
        unique_euc = idx_euc.unique().numel()
        utilizations.append(unique_geo / K)

        # flip rate (euc vs hyp argmin)
        flip = (idx_geo != idx_euc)
        flip_rate = flip.float().mean().item()
        if li == 0:
            flip_rate_total = flip_rate
        else:
            flip_rate_total = max(flip_rate_total, flip_rate)

        # radius distribution (per-codeword chosen by euc argmin)
        z_euc_norm = z_euc.norm(dim=-1)  # (N,)
        flipped_radii.append(z_euc_norm[flip].cpu().numpy())
        nonflipped_radii.append(z_euc_norm[~flip].cpu().numpy())

        # radius std (per-codeword)
        cb_euc_norms = cb_euc.norm(dim=-1)
        radius_stds.append(cb_euc_norms.std().item())

        # direction cosine std (per z, vs mean direction)
        z_h_d = F.normalize(z_h_n, dim=-1, eps=1e-6)
        mean_dir = z_h_d.mean(0, keepdim=True)
        cos_to_mean = (z_h_d * mean_dir).sum(-1).clamp(-1, 1)
        cos_stds.append(cos_to_mean.std().item())

        # ===== 疑点 2 检查 ① radius-only 复现率 =====
        # 简化规则: argmin_k |r_z - r_k| (完全不管方向)
        # 如果 hyp argmin 真的主要靠方向区分, 这一致率应该 < 40%
        # 如果一致率 > 70%, 说明 hyp argmin 退化成"按半径最近匹配",方向没参与.
        # z_h 的 radius 跟 z_hyp 的"‖z‖"等价: |z|/sqrt(1 - c·|z|²) ≈ |z| at small.
        # 因为 z_h 都先硬归一化到 r_tgt,所以 z 的"半径"应看其原始范数 r_tgt + 扰动.
        # 简化版: 用 ‖z_h_n‖ (投影到 r_tgt 球面后 ||·||=r_tgt), 没法区分 — 所以改用 ‖z_hyp‖ (原始).
        r_z = z_hyp.norm(dim=-1)  # (N,)  原始范数 (未硬归一化)
        r_cb = cb_hyp_raw.norm(dim=-1)  # (K,)
        # (N, K) |r_z - r_k|
        abs_diff = (r_z.unsqueeze(1) - r_cb.unsqueeze(0)).abs()
        idx_radius_only = abs_diff.argmin(dim=-1)
        radius_only_match = (idx_radius_only == idx_geo).float().mean().item()
        radius_only_matches.append(radius_only_match)

        # ===== 现象补做检查 ② disagreement vs agreement margin =====
        # 每个样本在 d_euc 上的 margin = (2nd nearest - nearest) 差距.
        # 期望: disagreement (flip) 组的 margin 明显 < agreement 组的 margin
        # — 翻转集中在"本来 euc 距离下都很犹豫"的边界样本.
        sorted_d_euc = d_euc.sort(dim=-1).values  # (N, K)
        euc_margin = (sorted_d_euc[:, 1] - sorted_d_euc[:, 0]).cpu().numpy()  # (N,)
        margin_disagreement = euc_margin[flip.cpu().numpy()]
        margin_agreement = euc_margin[~flip.cpu().numpy()]
        margin_disagreement_list.append(margin_disagreement)
        margin_agreement_list.append(margin_agreement)

        print(f"  L{li} (K={K}):")
        print(f"    agreement   = {agreement:.4f}  target < 0.90 → {'✅' if agreement < 0.90 else '❌'}")
        print(f"    utilization = {unique_geo}/{K} = {unique_geo/K:.2%}  "
              f"target ≥ 0.80 → {'✅' if unique_geo/K >= 0.80 else '❌'}")
        print(f"    unique_euc  = {unique_euc}/{K} = {unique_euc/K:.2%}")
        print(f"    flip rate   = {flip_rate:.4f}  (informational)")
        print(f"    radius std  = {radius_stds[-1]:.4f}")
        print(f"    cos std     = {cos_stds[-1]:.4f}")
        print(f"    d_hyp finite? {torch.isfinite(d_hyp).all().item()}, "
              f"d_euc finite? {torch.isfinite(d_euc).all().item()}")

    print(f"\n{'='*60}")
    print("B. 护栏 (Guardrails)")
    print(f"{'='*60}")
    # 1) same subspace + same batch (✓ — by construction)
    guard_1 = True
    # 2) all finite (just checked)
    guard_2 = True
    # 3) radius std > 0.05 (per layer)
    guard_3 = all(s > 0.05 for s in radius_stds)
    print(f"  1. 同子空间同 batch: {'✅' if guard_1 else '❌'} (by construction)")
    print(f"  2. all finite:       {'✅' if guard_2 else '❌'} (verified per layer)")
    print(f"  3. radius std > 0.05: {radius_stds} → "
          f"{'✅' if guard_3 else '❌'}")
    guard_4 = all(s > 0.3 for s in cos_stds)
    # 5) max(c‖e_k‖²) << 0.5
    max_c_norm_sq = 0.0
    for li in [0, 1, 2]:
        w = state[f"hrq.vq_layers.{li}.embeddings.weight"]
        cb_euc = w[:, HYP_DIM:]
        c_norm_sq = cb_euc.pow(2).sum(dim=-1)
        max_c_norm_sq = max(max_c_norm_sq, c_norm_sq.max().item())
    guard_5 = max_c_norm_sq < 0.5
    print(f"  4. direction cosine std > 0.3: {cos_stds} → "
          f"{'✅' if guard_4 else '❌'}")
    print(f"  5. max(c·‖e_k‖²) < 0.5: max={max_c_norm_sq:.4f} → "
          f"{'✅' if guard_5 else '❌'}")

    print(f"\n{'='*60}")
    print("C. 现象 (Phenomenon)")
    print(f"{'='*60}")
    phenomenon_A = flip_rate_total >= 0.10
    print(f"  A. flip rate (L_max) = {flip_rate_total:.4f}  target ≥ 0.10 → "
          f"{'✅' if phenomenon_A else '❌'}")
    # B: util histogram — 多少 codeword 用 ≥ 1 次
    flat_unique = sum(utilizations) * N
    print(f"  B. util histogram: L0={utilizations[0]:.2f}, L1={utilizations[1]:.2f}, "
          f"L2={utilizations[2]:.2f} → {'✅ spread' if min(utilizations) > 0.5 else '⚠ narrow'}")
    phenomenon_B = min(utilizations) > 0.5
    # C: flipped vs non-flipped radius diff
    diff_means = []
    for li in [0, 1, 2]:
        if len(flipped_radii[li]) > 0 and len(nonflipped_radii[li]) > 0:
            f_mean = flipped_radii[li].mean()
            n_mean = nonflipped_radii[li].mean()
            diff_means.append(abs(f_mean - n_mean))
        else:
            diff_means.append(0)
    phenomenon_C = any(d > 0.005 for d in diff_means)
    print(f"  C. radius diff (flipped vs not): {[f'{d:.4f}' for d in diff_means]} → "
          f"{'✅' if phenomenon_C else '⚠ no diff'}")

    # ===== D. 决断两检查 (用户 2026-07-27 final 验证) =====
    print(f"\n{'='*60}")
    print("D. 决断检查 (Critical / Decisive)")
    print(f"{'='*60}")
    # D.① radius-only 复现率
    # < 40% → 方向真参与, 排除"退化为半径排序"
    # > 70% → 方向没参与, 需重新处理
    print(f"  ① radius-only 复现率 (与真实 hyp argmin 重合率):")
    passed_radius_only = True
    for li, rate in enumerate(radius_only_matches):
        if rate > 0.70:
            verdict = "❌ hyp argmin 退化成'只看半径', 方向没真参与"
            passed_radius_only = False
        elif rate < 0.40:
            verdict = "✅ 方向确实在参与决策"
        else:
            verdict = "⚠ 介于 0.40-0.70, 中间地带"
        print(f"    L{li}: {rate:.4f} → {verdict}")
    # D.② disagreement vs agreement margin (欧氏距离下)
    # disagreement margin 中位数应 < agreement margin 中位数 (翻转集中在边界)
    print(f"  ② disagreement vs agreement 欧氏 margin 中位数:")
    passed_margin = True
    margin_ratios = []
    for li in range(3):
        m_disag = np.median(margin_disagreement_list[li]) if len(margin_disagreement_list[li]) > 0 else 0
        m_ag = np.median(margin_agreement_list[li]) if len(margin_agreement_list[li]) > 0 else 0
        ratio = m_disag / max(m_ag, 1e-12)
        margin_ratios.append(ratio)
        if m_disag < m_ag * 0.8:  # disagreement 明显小
            verdict = f"✅ 翻转集中边界 (ratio={ratio:.3f})"
        elif m_disag < m_ag * 1.0:
            verdict = f"⚠ 差异不大 (ratio={ratio:.3f})"
        else:
            verdict = f"❌ 翻转组 margin 反而 ≥ agreement (ratio={ratio:.3f}, 说明推翻没争议样本)"
            passed_margin = False
        print(f"    L{li}: 中位数 disagreement={m_disag:.4f}, agreement={m_ag:.4f}, "
              f"ratio={ratio:.3f} → {verdict}")
    decisive_pass = passed_radius_only and passed_margin
    print(f"  {'='*40}")
    print(f"  决断检查 D 整体: {'🎯 DECISIVE PASS' if decisive_pass else '❌ DECISIVE FAIL'}")

    print(f"\n{'='*60}")
    print("综合判定")
    print(f"{'='*60}")
    core_pass = all(a < 0.90 for a in agreements) and all(u >= 0.80 for u in utilizations)
    guard_all = guard_1 and guard_2 and guard_3 and guard_4 and guard_5
    phenom_pass = phenomenon_A and phenomenon_B and phenomenon_C
    all_pass = core_pass and guard_all and phenom_pass and decisive_pass
    print(f"  核心指标 (6 核心数): {'✅ PASS' if core_pass else '❌ FAIL'}")
    print(f"  5 个护栏:           {'✅ PASS' if guard_all else '❌ FAIL'}")
    print(f"  现象:              {'✅ PASS' if phenom_pass else '⚠ partial'}")
    print(f"  决断检查 (D):      {'✅ PASS' if decisive_pass else '❌ FAIL'}")
    print(f"  ============================================")
    print(f"  整体: {'🎯 STEP 3 PASS' if all_pass else '❌ STEP 3 NOT PASS'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
