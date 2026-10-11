"""Step 1: does the Poincare metric actually move the L1 grouping boundary?

Everything is frozen. Exp1a resample's encoder and its three codebooks are
loaded and never updated, the continuous representation is mapped into the
Poincare ball with the repository's own fixed normalizer, and the question is
whether the geodesic distance ranks L1 codewords differently from the Euclidean
one. If it does not, no training is warranted, because a mechanism built on a
metric that orders candidates identically to the one already in use cannot
change which codeword an item lands on.

Four things are measured, in the order they decide the answer:

  1. Distance geometry. Radial spread of the mapped items and prototypes, and
     how much of the item-prototype distance is radial versus angular. If the
     radial spread is small the ball map is barely being used and the curvature
     term is a near-constant scale factor.
  2. Ordering. Per item, the rank correlation between the Euclidean and geodesic
     distance to all 256 codewords, and the fraction of items whose NEAREST
     codeword changes. A high rank correlation with a near-zero reassignment rate
     means the two metrics agree in practice even if not pointwise.
  3. Grouping. L1 positive and negative sharing rates under the Euclidean and the
     geodesic assignment, on the canonical seed-42 pairs and, separately, on
     relations built from valid.parquet that the model never trained on.
  4. Where the wrongly-shared negatives sit. If they sit at a similar radius to
     their anchor's prototype, curvature cannot separate them, because curvature
     acts on radius.

The valid.parquet relations are constructed here and used ONLY for evaluation.
They never enter training, and a negative is excluded if the item has an observed
relation to the anchor in EITHER train or valid, so a known positive can never be
drawn as a negative. Absent evidence is still absent evidence: these negatives
mean UNOBSERVED, not irrelevant.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
STAGE2 = HERE.parent
# train_rqvae.py and hyperbolic.py live one level up, next to the scripts.
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(STAGE2))
import analyze_prefix_behavior as AP  # noqa: E402
import monotonicity_metrics as MM  # noqa: E402
import train_rqvae as T  # noqa: E402
from hyperbolic import ball_radius, encode_ball, poincare_distance  # noqa: E402

SEED = 42
CKPT = AP.ROOT / "results/stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample/rqvae_best.pth"
PAIRS = MM.OUT_DIR / "behavior_pairs_s42.npz"
OUT = MM.OUT_DIR / "hyper_step1_geometry.json"


@torch.no_grad()
def frozen_state():
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    T.set_seed(SEED)
    emb = T.maybe_apply_pca(T.load_embeddings(T.EMBEDDING_FILE), T.PCA_DIM, T.SEED)
    model = T.RQVAE(T._tokenizer_config(), in_dim=emb.shape[1]).to(dev)
    model.load_state_dict(torch.load(CKPT, map_location=dev)["state_dict"])
    model.eval()
    z = model.encoder(torch.from_numpy(emb).to(dev)).detach()
    e1 = model.rq.vq_layers[0].get_code_embs().detach()
    return model, emb, z, e1, dev


def valid_relations(n_items: int):
    """Behaviour relations from valid.parquet, never used for training."""
    va = pd.read_parquet(AP.DATA / "valid.parquet")
    pairs = AP.transitions(va, n_items)
    a, b = pairs[:, 0], pairs[:, 1]
    keep = (a != b) & (a >= 0) & (a < n_items) & (b >= 0) & (b < n_items)
    return pairs[keep]


def build_eval_sets(pairs, succ_tr, succ_va, pop_bin, freq_bin, tag):
    """Positives from the given relation, negatives matched and fully excluded."""
    anchors, positives = pairs[:, 0].copy(), pairs[:, 1].copy()
    related = []
    for a in anchors:
        a = int(a)
        rel = set(succ_tr.get(a, ())) | set(succ_va.get(a, ()))
        rel.discard(a)
        related.append(rel)
    rng = np.random.default_rng(SEED)
    negs, stats = MM.sample_negatives(
        anchors, positives, related, pop_bin, freq_bin, 10, rng
    )
    return {
        "tag": tag, "anchors": anchors, "positives": positives, "negatives": negs,
        "violations": int(stats["invariant_violations"]), "stats": stats,
    }


def sharing(codes, anchors, others):
    return float((codes[others] == codes[anchors]).mean())


def rates(codes, s):
    k = s["negatives"].shape[1]
    anchor_rep = np.repeat(s["anchors"], k)
    p = sharing(codes, s["anchors"], s["positives"])
    n = sharing(codes, anchor_rep, s["negatives"].reshape(-1))
    return {
        "p_share_pos": p, "p_share_neg": n, "gap": p - n,
        "er": (p / n) if n > 0 else None,
    }


def radii(z_ball, e_ball):
    """Radius of each item and each prototype in the ball.

    The item-prototype angle would be a (N, 256) matrix and is not needed here:
    curvature acts on radius, so radius is the quantity the diagnosis needs, and
    the direction count below is what says whether the angular dimension is
    carrying any structure at all.
    """
    return torch.linalg.vector_norm(z_ball, dim=-1), torch.linalg.vector_norm(e_ball, dim=-1)


def main() -> None:
    model, emb, z, e1, dev = frozen_state()
    n = len(z)
    report = {
        "seed": SEED,
        "frozen": "Exp1a resample encoder and L1 codebook, loaded and never updated",
        "checkpoint": str(CKPT.relative_to(AP.ROOT)),
    }

    # ---- 1. distance geometry -------------------------------------------
    normaliser = T.resolve_normalizer(model, torch.from_numpy(emb).to(dev))
    report["ball_normalizer"] = float(normaliser)
    zb = encode_ball(z, T.CURVATURE, normaliser)
    eb = encode_ball(e1, T.CURVATURE, normaliser)
    br = ball_radius(T.CURVATURE)
    ri, ri_e = radii(zb, eb)
    inside = ((torch.linalg.vector_norm(zb, dim=-1) < br).float().mean().item())
    # Did the map even stay inside the ball, or did tanh saturate it?
    rho = (torch.linalg.vector_norm(zb, dim=-1) / br).clamp(0, 1 - 1e-7)
    report["geometry"] = {
        "pre_map_norm_mean": float(torch.linalg.vector_norm(z, dim=-1).mean()),
        "post_map_radius_mean": float(rho.mean()),
        "post_map_radius_std": float(rho.std()),
        "post_map_radius_p50": float(rho.median()),
        "post_map_radius_p99": float(torch.quantile(rho.float(), 0.99)),
        "frac_radii_above_0.95": float((rho > 0.95).float().mean()),
        "prototype_radius_mean": float((torch.linalg.vector_norm(eb, dim=-1) / br).mean()),
        "frac_items_inside_ball": inside,
    }
    # Angular dispersion of the items: how many directions are actually used.
    ui = (zb / torch.linalg.vector_norm(zb, dim=-1).unsqueeze(1).clamp_min(1e-12)).cpu().numpy()
    _u, _c = np.unique(np.round(ui, 4), axis=0, return_counts=True)
    report["geometry"]["distinct_item_directions"] = int(len(_u))

    # ---- 2. ordering ------------------------------------------------------
    d_eu = torch.cdist(z, e1)                       # (N, 256)
    d_hyp = poincare_distance(zb[:, None, :], eb[None, :, :], T.CURVATURE)
    a_eu = d_eu.argmin(1)
    a_hyp = d_hyp.argmin(1)
    changed = float((a_eu != a_hyp).float().mean())
    # Rank agreement per item, averaged, and the top-k overlap that decides the
    # assignment in practice.
    r_eu = d_eu.argsort(1).argsort(1).float()
    r_hyp = d_hyp.argsort(1).argsort(1).float()
    per_item_rho = []
    for i in range(0, n, 37):                      # strided sample, deterministic
        a = r_eu[i] - r_eu[i].mean()
        b = r_hyp[i] - r_hyp[i].mean()
        den = (a.norm() * b.norm()).clamp_min(1e-12)
        per_item_rho.append(float((a @ b / den).item()))
    topk = {k: float((d_eu.topk(k, 1, largest=False).indices
                      == d_hyp.topk(k, 1, largest=False).indices).float().mean())
            for k in (1, 5, 20)}
    report["ordering"] = {
        "assignment_change_rate": changed,
        "mean_rank_spearman": float(np.mean(per_item_rho)),
        "rank_spearman_p05": float(np.percentile(per_item_rho, 5)),
        "rank_spearman_p50": float(np.percentile(per_item_rho, 50)),
        "topk_identical_rate": topk,
        "spearman_distance_pointwise": float(
            __import__("scipy.stats", fromlist=["stats"]).spearmanr(
                d_eu.flatten().cpu(), d_hyp.flatten().cpu()).statistic
        ),
    }

    # ---- 3. grouping on train and on an independent valid relation --------
    train = pd.read_parquet(AP.DATA / "train.parquet")
    valid = pd.read_parquet(AP.DATA / "valid.parquet")
    succ_tr = AP.successors(train, MM.N_ITEMS)
    succ_va = AP.successors(valid, MM.N_ITEMS)
    pop, freq = AP.popularity_and_frequency(train, MM.N_ITEMS)
    pb, fb = AP.decile_bins(pop), AP.decile_bins(freq)

    zp = np.load(PAIRS)
    sets = {
        "train_pairs": build_eval_sets(
            np.stack([zp["anchors"].astype(np.int64), zp["positives"].astype(np.int64)], 1),
            succ_tr, succ_va, pb, fb, "train canonical pairs"),
        "valid_relations": build_eval_sets(
            valid_relations(MM.N_ITEMS), succ_tr, succ_va, pb, fb,
            "valid.parquet relations, held out from training"),
    }
    codes_eu = a_eu.cpu().numpy().astype(np.int64)
    codes_hyp = a_hyp.cpu().numpy().astype(np.int64)
    report["grouping"] = {}
    for name, s in sets.items():
        report["grouping"][name] = {
            "n_pairs": int(len(s["anchors"])),
            "negative_violations": int(s["violations"]),
            "euclidean": rates(codes_eu, s),
            "geodesic": rates(codes_hyp, s),
        }

    # ---- 4. where the wrongly shared negatives sit ------------------------
    diag = {}
    for name, s in sets.items():
        k = s["negatives"].shape[1]
        anchor_rep = np.repeat(s["anchors"], k)
        flat = s["negatives"].reshape(-1)
        shared = codes_eu[flat] == codes_eu[anchor_rep]
        if shared.sum() == 0:
            diag[name] = {"wrongly_shared_negatives": 0}
            continue
        rho_i = rho.cpu().numpy()
        rho_p = rho[torch.as_tensor(anchor_rep, device=dev)].cpu().numpy()
        rho_n = rho[torch.as_tensor(flat, device=dev)].cpu().numpy()
        idx = np.where(shared)[0]
        diag[name] = {
            "wrongly_shared_negatives": int(shared.sum()),
            "radius_gap_positive_mean": float(np.mean(
                rho_p[idx] - rho_n[idx])),
            "radius_gap_abs_mean": float(np.mean(np.abs(rho_p[idx] - rho_n[idx]))),
            "radial_overlap": float(
                np.mean(np.minimum(rho_p[idx], rho_n[idx]) /
                         np.maximum(np.maximum(rho_p[idx], rho_n[idx]), 1e-12))),
        }
    report["wrongly_shared_negatives"] = diag

    OUT.write_text(json.dumps(report, indent=2, default=str))
    g, o = report["geometry"], report["ordering"]
    print("=== 1. 几何 ===")
    print(f"  映射后半径 mean={g['post_map_radius_mean']:.4f} std={g['post_map_radius_std']:.4f} "
          f"p50={g['post_map_radius_p50']:.4f} >0.95 占比={g['frac_radii_above_0.95']:.4f}")
    print(f"  商品方向数 {g['distinct_item_directions']}  (24,587 个商品)")
    print("=== 2. 排序 ===")
    print(f"  分配变化率 {o['assignment_change_rate']:.4f}")
    print(f"  逐项秩相关 mean={o['mean_rank_spearman']:.4f} p05={o['rank_spearman_p05']:.4f} "
          f"p50={o['rank_spearman_p50']:.4f}")
    print(f"  top-k 完全一致率 {o['topk_identical_rate']}")
    for name, v in report["grouping"].items():
        print(f"=== 3. 分组 ({name}, {v['n_pairs']} 对, 违规 {v['negative_violations']}) ===")
        for m in ("euclidean", "geodesic"):
            g2 = v[m]
            print(f"  {m:10s} pos={g2['p_share_pos']:.5f} neg={g2['p_share_neg']:.5f} "
                  f"ER={g2['er']}")
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()