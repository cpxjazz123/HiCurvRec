"""
Task #339 — Issue #48 Gate 0/1/2 — 码字间隔合理性诊断.

Gate 0: 噪声底线 — 对输入加扰动, 测 SID 分配稳定性 + residual 波动
Gate 1: 健康参照系 — Task #194 K=64 baseline 码字几何 + 势力范围
Gate 2: 坍缩案例对比 — 对比 collapsed checkpoint vs 健康分布

Usage:
  python3 scripts/task339_issue48_diagnose_spacing.py
"""
from __future__ import annotations
import sys, json, math
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
HGREC = REPO / "HG-Rec"
sys.path.insert(0, str(HGREC))

from model.hrqvae import HRQVAE

EPS = 1e-15
device = 'cpu'


# ══════════════════════════════════════════════════════════════════════
# 加载
# ══════════════════════════════════════════════════════════════════════

def load_model(ckpt_path: str) -> HRQVAE:
    ckpt = torch.load(ckpt_path, weights_only=False, map_location='cpu')
    args = ckpt['args']
    sd = ckpt['state_dict']
    model = HRQVAE(
        in_dim=768,
        num_emb_list=args.num_emb_list,
        e_dim=args.e_dim,
        layers=args.layers,
        dropout_prob=args.dropout_prob,
        bn=args.bn,
        loss_type=args.loss_type,
        quant_loss_weight=args.quant_loss_weight,
        beta=args.beta,
        kmeans_init=args.kmeans_init,
        kmeans_iters=args.kmeans_iters,
        sk_eps=args.sk_epsilons,
        sk_iters=args.sk_iters,
    )
    model.load_state_dict(sd)
    model.eval()
    return model


def get_sid_and_residuals(model: HRQVAE, z: torch.Tensor):
    """Get SID codes and per-layer residuals.

    Returns:
        sids: (N, 3) — per-layer codeword indices
        residuals: list of (N, e_dim) — residual before each layer
        final_zq: (N, e_dim) — final quantized output
    """
    z_q = torch.zeros_like(z)
    residual = z
    sids = []
    residuals = [residual.clone()]  # residual before layer 0

    for vq in model.hrq.vq_layers:
        x_res, _, indices = vq(residual, use_sk=False)
        residual = residual - x_res
        z_q = z_q + x_res
        sids.append(indices)
        residuals.append(residual.clone())

    sids = torch.stack(sids, dim=-1)  # (N, 3)
    return sids, residuals, z_q


# ══════════════════════════════════════════════════════════════════════
# Gate 0: 噪声底线
# ══════════════════════════════════════════════════════════════════════

def gate0_noise_floor(model, embeddings, noise_sigmas=[0.01, 0.02, 0.05],
                      n_samples=500, n_trials=20):
    """Gate 0: measure how input noise propagates through VQ layers.

    For each σ, we add Gaussian noise to input and measure:
    - SID flip rate: fraction of codewords that change (per layer)
    - Residual shift: L2 distance between clean and noisy residual (per layer)
    """
    print(f"\n{'='*60}")
    print("Gate 0: 噪声底线测量")
    print(f"{'='*60}")
    rng = np.random.RandomState(42)
    idxs = rng.choice(len(embeddings), min(n_samples, len(embeddings)), replace=False)
    base = torch.tensor(embeddings[idxs], dtype=torch.float32)

    results = {}
    for sigma in noise_sigmas:
        print(f"\n  σ={sigma}:")
        layer_flip_rates = []
        layer_residual_shifts = []

        for trial in range(n_trials):
            noisy = base + torch.randn_like(base) * sigma

            # Encode both
            z_clean = model.encoder(base)
            z_noisy = model.encoder(noisy)

            sid_clean, res_clean, _ = get_sid_and_residuals(model, z_clean)
            sid_noisy, res_noisy, _ = get_sid_and_residuals(model, z_noisy)

            # Per-layer SID flip rate
            flip_rates = []
            for lyr in range(3):
                flip = (sid_clean[:, lyr] != sid_noisy[:, lyr]).float().mean().item()
                flip_rates.append(flip)
            layer_flip_rates.append(flip_rates)

            # Per-layer residual shift (residual BEFORE each layer, i.e. layer input)
            shifts = []
            for lyr in range(3):
                r_c = res_clean[lyr]  # (N, 32) — residual before this layer
                r_n = res_noisy[lyr]
                shift = (r_c - r_n).norm(dim=-1).mean().item()
                shifts.append(shift)
            layer_residual_shifts.append(shifts)

            if trial % 5 == 4:
                print(f"    Trial {trial+1}/{n_trials} — "
                      f"flip L0={flip_rates[0]:.3f} L1={flip_rates[1]:.3f} L2={flip_rates[2]:.3f}")

        # Aggregate
        flip_arr = np.array(layer_flip_rates)  # (n_trials, 3)
        shift_arr = np.array(layer_residual_shifts)  # (n_trials, 3)

        results[f"sigma_{sigma}"] = {
            "sid_flip_rate": {
                "mean": flip_arr.mean(axis=0).tolist(),
                "std": flip_arr.std(axis=0).tolist(),
            },
            "residual_shift": {
                "mean": shift_arr.mean(axis=0).tolist(),
                "std": shift_arr.std(axis=0).tolist(),
                "p50": np.median(shift_arr, axis=0).tolist(),
                "p90": np.percentile(shift_arr, 90, axis=0).tolist(),
            },
        }
        print(f"    SID flip: L0={flip_arr.mean(axis=0)[0]:.3f} "
              f"L1={flip_arr.mean(axis=0)[1]:.3f} L2={flip_arr.mean(axis=0)[2]:.3f}")
        print(f"    Residual shift: L0={shift_arr.mean(axis=0)[0]:.4e} "
              f"L1={shift_arr.mean(axis=0)[1]:.4e} L2={shift_arr.mean(axis=0)[2]:.4e}")

    return results


# ══════════════════════════════════════════════════════════════════════
# Gate 1: 健康参照系码字画像
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def gate1_healthy_profile(model, embeddings):
    """Gate 1: codebook geometry + catchment areas from healthy baseline.

    Per VQ layer:
      1. Codebook pairwise distances (NN dist distribution)
      2. Catchment area (item count per codeword)
      3. Within-catchment item dispersion (avg distance from codeword to its items)
    """
    print(f"\n{'='*60}")
    print("Gate 1: 健康参照系码字几何画像")
    print(f"{'='*60}")

    z = model.encoder(torch.tensor(embeddings, dtype=torch.float32))
    sid, _, _ = get_sid_and_residuals(model, z)
    sid_np = sid.numpy()  # (N, 3)

    profiles = {}
    for lyr_idx, vq_layer in enumerate(model.hrq.vq_layers):
        codebook = vq_layer.embeddings.weight.data  # (K, 32)
        K, dim = codebook.shape

        print(f"\n  Layer {lyr_idx} (K={K}, dim={dim}):")

        # 1. Pairwise NN-distance distribution (batch_size=32 to avoid mask < K)
        batch_sz = min(32, K)
        all_nn = []
        for i in range(0, K, batch_sz):
            j_end = min(i + batch_sz, K)
            batch = codebook[i:j_end, None, :]  # (B, 1, d)
            dists = (batch - codebook[None, :, :]).norm(dim=-1)  # (B, K)
            # Mask self: set diagonal entries to inf
            for b_idx, k_idx in enumerate(range(i, j_end)):
                dists[b_idx, k_idx] = float('inf')
            nn_d = dists.min(dim=-1).values  # (B,)
            all_nn.append(nn_d)
        nn_all = torch.cat(all_nn).numpy()

        nn_stats = {
            "min": float(nn_all.min()), "max": float(nn_all.max()),
            "mean": float(nn_all.mean()), "median": float(np.median(nn_all)),
            "p1": float(np.percentile(nn_all, 1)),
            "p5": float(np.percentile(nn_all, 5)),
            "p10": float(np.percentile(nn_all, 10)),
            "p90": float(np.percentile(nn_all, 90)),
            "p99": float(np.percentile(nn_all, 99)),
            "std": float(nn_all.std()),
        }

        # 2. Catchment area
        sid_l = sid_np[:, lyr_idx]
        unique, counts = np.unique(sid_l, return_counts=True)
        catch = np.zeros(K, dtype=int)
        catch[unique] = counts
        used_pct = (catch > 0).sum() / K * 100

        catchment_stats = {
            "util_pct": float(used_pct),
            "min": int(catch.min()), "max": int(catch.max()),
            "mean": float(catch.mean()), "median": float(np.median(catch)),
            "p10": float(np.percentile(catch, 10)),
            "p90": float(np.percentile(catch, 90)),
            "std": float(catch.std()),
            "zero_catch": int((catch == 0).sum()),
        }

        # 3. Within-catchment dispersion
        z_blocks = torch.split(z, z.size(-1)//3, dim=-1)  # dummy — not per-layer
        # Actually, we need the input to EACH layer's quantizer which is the residual
        # We already computed residuals in get_sid_and_residuals
        # But we don't have the per-layer inputs in this simplified version.
        # Let's compute: for each codeword k, the avg L2 from z to codeword for items that map to k
        # z is the encoder output (input to VQ layer 0)
        # For higher layers, the "input" is the residual

        # Re-run with residuals saved
        _, residuals, _ = get_sid_and_residuals(model, z)
        layer_input = residuals[lyr_idx]  # (N, 32)

        disp_list = []
        for k in range(K):
            mask = (sid_l == k)
            n = mask.sum()
            if n > 1:
                items = layer_input[mask]
                disp = (items - codebook[k]).norm(dim=-1).mean().item()
                disp_list.append(disp)

        disp_stats = {}
        if disp_list:
            disp_stats = {
                "mean": float(np.mean(disp_list)),
                "median": float(np.median(disp_list)),
                "p10": float(np.percentile(disp_list, 10)),
                "p90": float(np.percentile(disp_list, 90)),
                "n_cw_with_items": len(disp_list),
            }

        profiles[f"layer_{lyr_idx}"] = {
            "K": K, "dim": dim,
            "nn_dist": nn_stats,
            "catchment": catchment_stats,
            "within_dispersion": disp_stats,
        }

        # Summary print
        print(f"    NN dist: mean={nn_stats['mean']:.4f} "
              f"p5={nn_stats['p5']:.4f} p50={nn_stats['median']:.4f} p95={nn_stats['p99']:.4f}")
        print(f"    Catchment: util={used_pct:.1f}% "
              f"mean={catch.mean():.1f} zero={catchment_stats['zero_catch']}")
        if disp_stats:
            print(f"    Within-disp: mean={disp_stats['mean']:.4f} "
                  f"p10={disp_stats['p10']:.4f} p90={disp_stats['p90']:.4f}")

    return profiles


# ══════════════════════════════════════════════════════════════════════
# Gate 2: 坍缩案例对比
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def gate2_compare(healthy_profiles: dict, collapsed_profiles: dict,
                  label: str):
    """Compare collapsed vs healthy and produce deviation report."""
    print(f"\n{'='*60}")
    print(f"Gate 2: 坍缩案例对比 — {label}")
    print(f"{'='*60}")

    deviations = {}
    for lyr_idx in range(3):
        hk = f"layer_{lyr_idx}"
        ck = f"layer_{lyr_idx}"
        if hk not in healthy_profiles or ck not in collapsed_profiles:
            continue

        h = healthy_profiles[hk]
        c = collapsed_profiles[ck]
        print(f"\n  Layer {lyr_idx}:")

        # NN dist comparison
        nn_h = h['nn_dist']
        nn_c = c['nn_dist']
        ratio = nn_c['mean'] / (nn_h['mean'] + EPS)
        print(f"    NN-dist mean: healthy={nn_h['mean']:.4f} "
              f"collapsed={nn_c['mean']:.4f} ratio={ratio:.2f}x")
        print(f"    NN-dist p5:   healthy={nn_h['p5']:.4f} "
              f"collapsed={nn_c['p5']:.4f}")
        print(f"    NN-dist p99:  healthy={nn_h['p99']:.4f} "
              f"collapsed={nn_c['p99']:.4f}")

        # Catchment comparison
        ct_h = h['catchment']
        ct_c = c['catchment']
        print(f"    Catchment util: healthy={ct_h['util_pct']:.1f}% "
              f"collapsed={ct_c['util_pct']:.1f}%")
        print(f"    Catchment mean: healthy={ct_h['mean']:.1f} "
              f"collapsed={ct_c['mean']:.1f}")
        print(f"    Zero-catch cw:  healthy={ct_h['zero_catch']} "
              f"collapsed={ct_c['zero_catch']}")

        # Within dispersion
        dh = h.get('within_dispersion', {})
        dc = c.get('within_dispersion', {})
        if dh and dc:
            print(f"    Within-disp mean: healthy={dh['mean']:.4f} "
                  f"collapsed={dc['mean']:.4f}")

        deviations[f"layer_{lyr_idx}"] = {
            "nn_mean_ratio": ratio,
            "util_healthy": ct_h['util_pct'],
            "util_collapsed": ct_c['util_pct'],
            "zero_catch_healthy": ct_h['zero_catch'],
            "zero_catch_collapsed": ct_c['zero_catch'],
        }

    return deviations


# ══════════════════════════════════════════════════════════════════════
# 搜索坍缩 checkpoint
# ══════════════════════════════════════════════════════════════════════

def find_collapsed_checkpoints():
    """Search for collapsed checkpoints with low utilization."""
    candidates = []
    for path in sorted(REPO.glob("products/task*/**/best_collision_model.pth")):
        # Exclude healthy baselines
        if 'protocol_match' in str(path) or 'task84' in str(path):
            continue
        # Quick inspect: load and check utilization
        try:
            ckpt = torch.load(path, weights_only=False, map_location='cpu')
            # We need to load the model to check utilization
            # For now, just return the path
            candidates.append(path)
        except:
            continue
    return candidates


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("Task #339 — Issue #48 Gate 0/1/2 — 码字间隔合理性诊断")
    print("=" * 70)

    # Load embeddings
    print("\n加载 embeddings...")
    df = pd.read_parquet(HGREC / "dataset/Instruments/item_emb.parquet")
    emb_col = "emb" if "emb" in df.columns else df.columns[-1]
    item_emb = np.stack(df[emb_col].values).astype(np.float32)
    print(f"  Shape: {item_emb.shape}")

    all_results = {}

    # ─── Gate 1: 健康参照系 ───
    ckpt_healthy = (REPO / "products/task194/protocol_match_k064"
                    "/Jul-30-2026_12-56-39_beta_1.000_codebook_[64,128,256]_sk_0.000"
                    "/best_collision_model.pth")
    print(f"\n加载健康 baseline: {ckpt_healthy.name}")
    m_healthy = load_model(str(ckpt_healthy))
    all_results["gate1_healthy"] = gate1_healthy_profile(m_healthy, item_emb)

    # ─── Gate 0: 噪声底线 ───
    all_results["gate0_noise_floor"] = gate0_noise_floor(m_healthy, item_emb)

    # ─── Gate 2: 坍缩案例 ───
    # Try multiple known collapsed checkpoints
    collapsed_targets = [
        (REPO / "products/task144/arm_B" / "Jul-25-2026_17-09-05_beta_0.500_codebook_[64,128,256]_sk_0.000" / "best_collision_model.pth",
         "Task #144 Arm B (κ-decouple Phase A)"),
        (REPO / "products/task178" / "Jul-25-2026_11-35-05_beta_0.500_codebook_[64,128,256]_sk_0.000" / "best_collision_model.pth",
         "Task #178 Phase 0 c=1.0+b=0.5"),
        (REPO / "products/task299" / "hrqvae_issue28_gate1" / "Jul-29-2026_23-37-11_beta_0.500_codebook_[64,128,256]_sk_0.000" / "best_collision_model.pth",
         "Task #299 Issue #28 Gumbel-Softmax"),
    ]

    for ckpt_path, label in collapsed_targets:
        if ckpt_path.exists():
            print(f"\n加载坍缩案例: {label}")
            try:
                m_coll = load_model(str(ckpt_path))
                coll_key = f"gate2_{label.split()[0].replace('#','task')}"
                coll_profile = gate1_healthy_profile(m_coll, item_emb)
                all_results[coll_key] = coll_profile

                # Compare with healthy
                dev_key = f"deviation_{label.split()[0].replace('#','task')}"
                all_results[dev_key] = gate2_compare(
                    all_results["gate1_healthy"], coll_profile, label)
            except Exception as e:
                print(f"    ⚠️ 加载/分析失败: {e}")
                import traceback; traceback.print_exc()
        else:
            print(f"\n  ⚠️ 未找到 {label}: {ckpt_path}")

    # ─── 综合报告 ───
    print("\n" + "=" * 70)
    print("综合报告")
    print("=" * 70)

    g1 = all_results["gate1_healthy"]
    g0 = all_results.get("gate0_noise_floor", {})

    print("\nGate 1 — 健康 baseline NN 分布:")
    for lyr_idx in range(3):
        h = g1.get(f"layer_{lyr_idx}", {})
        nn = h.get('nn_dist', {})
        ct = h.get('catchment', {})
        if nn:
            print(f"  L{lyr_idx}: NN={nn['mean']:.4f}±{nn['std']:.4f} "
                  f"[p5={nn['p5']:.4f}, p99={nn['p99']:.4f}] "
                  f"util={ct.get('util_pct', '?'):.1f}% "
                  f"catch={ct.get('mean', '?'):.1f}")

    print("\nGate 0 — 噪声底线 (σ=0.02):")
    g0_02 = g0.get("sigma_0.02", {})
    rs = g0_02.get("residual_shift", {})
    fr = g0_02.get("sid_flip_rate", {})
    if rs:
        for lyr_idx in range(3):
            print(f"  L{lyr_idx}: residual shift={rs['mean'][lyr_idx]:.4e} "
                  f"SID flip={fr['mean'][lyr_idx]:.3f}")

    print("\nNN min dist vs 噪声底线 (检测 '挤在一起' 风险):")
    g0_01 = g0.get("sigma_0.01", {})
    rs_01 = g0_01.get("residual_shift", {})
    for lyr_idx in range(3):
        nn_min = g1.get(f"layer_{lyr_idx}", {}).get('nn_dist', {}).get('p5', float('nan'))
        noise_l = rs.get('mean', [float('nan')]*3)[lyr_idx] if rs else float('nan')
        if noise_l and noise_l > 0:
            print(f"  L{lyr_idx}: NN_p5={nn_min:.4e} vs noise_floor={noise_l:.4e} "
                  f"→ ratio={nn_min/noise_l:.1f}x")
        else:
            print(f"  L{lyr_idx}: NN_p5={nn_min:.4e} (noise floor N/A)")

    # Save
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    out_path = REPO / "verdicts/task339_issue48_diagnose_spacing.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=convert)
    print(f"\n结果写入: {out_path}")


if __name__ == "__main__":
    main()
