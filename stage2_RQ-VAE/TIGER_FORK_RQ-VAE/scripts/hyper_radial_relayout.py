"""Decisive test: can a radial re-layout of the L1 prototypes restore the metric?

Step 1 established that the geodesic and Euclidean metrics order the 256 L1
codewords almost identically (per-item rank correlation 0.9602, 90.7% of items
keeping the same nearest codeword), and step 1b located why: only 2.28% of the
spread of the 256 item-to-codeword distances comes from radius, the rest from
angle. The curvature of the Poincare metric buys its discriminating power
through radius, so with 2.28% radial there is nothing for it to act on. The
normaliser is not the culprit, the sweep shows the change rate rising
monotonically with map depth while no item leaves the ball.

So step 2, swapping only the distance, cannot work on the layout as it stands.
This script tests the prerequisite instead: re-place the L1 prototypes at
DIFFERENT radii, keeping every prototype's direction and the item
representation completely untouched, and ask whether the metric then separates.

The layout rule is the natural one for a hyperbolic ball. Volume grows
exponentially with radius, so a cluster that has to hold many items is placed
further out, and the target radius is scaled with the square root of the cluster
size. One factor changes: where the prototypes sit radially. Directions are
fixed, the encoder is frozen, the codebook size is unchanged, and no training
happens here.

Reported for the frozen layout and for the re-laid one, on both the training
pairs and the independent valid relations, with the negatives excluded against
train UNION valid:
  radial share of the distance spread, which is what was 2.28%
  ordering divergence between the two metrics
  positive and negative sharing and the enrichment ratio
  whether the items whose assignment moved carried behaviour signal

If re-laying the prototypes raises the radial share but the enrichment does not
follow, then the metric is not the lever and the whole hyperbolic direction is
dead on this architecture, which is worth knowing before any training.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import analyze_prefix_behavior as AP  # noqa: E402
import monotonicity_metrics as MM  # noqa: E402
import train_rqvae as T  # noqa: E402
from hyperbolic import encode_ball, poincare_distance  # noqa: E402

SEED = 42
CKPT = AP.ROOT / "results/stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample/rqvae_best.pth"
OUT = MM.OUT_DIR / "hyper_radial_relayout.json"


@torch.no_grad()
def frozen():
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    T.set_seed(SEED)
    emb = T.maybe_apply_pca(T.load_embeddings(T.EMBEDDING_FILE), T.PCA_DIM, T.SEED)
    model = T.RQVAE(T._tokenizer_config(), in_dim=emb.shape[1]).to(dev)
    model.load_state_dict(torch.load(CKPT, map_location=dev)["state_dict"])
    model.eval()
    z = model.encoder(torch.from_numpy(emb).to(dev)).detach()
    e1 = model.rq.vq_layers[0].get_code_embs().detach()
    return z, e1, dev


def relayout_codes(z_ball, e_ball, assign, r_min, r_max):
    """Move prototypes radially, keeping direction. Size scales with sqrt(count)."""
    e_dir = e_ball / torch.linalg.vector_norm(e_ball, dim=-1, keepdim=True).clamp_min(1e-12)
    counts = torch.bincount(assign, minlength=e_ball.shape[0]).float()
    key = counts.clamp_min(1.0).sqrt()
    # Equal-count quantiles give a stable spread even when the size distribution
    # is heavily skewed, which it is here.
    qs = torch.argsort(torch.argsort(key)).float() / max(len(key) - 1, 1)
    target = r_min + (r_max - r_min) * qs
    return e_dir * target.unsqueeze(1)


def evaluate(e_ball, z_ball, sets, tag, d_eu):
    d_h = poincare_distance(z_ball[:, None, :], e_ball[None, :, :], 1.0)
    a_h = d_h.argmin(1).cpu().numpy().astype(np.int64)
    res = {"tag": tag}
    for name, s in sets.items():
        k = s["negatives"].shape[1]
        anchor_rep = np.repeat(s["anchors"], k)
        p = float((a_h[s["positives"]] == a_h[s["anchors"]]).mean())
        n = float((a_h[s["negatives"].reshape(-1)] == a_h[anchor_rep]).mean())
        res[name] = {"p_share_pos": p, "p_share_neg": n, "gap": p - n,
                     "er": (p / n) if n > 0 else None}
    # Did anything move relative to the Euclidean assignment?
    res["change_vs_euclidean"] = float((a_h != d_eu).mean())
    return res


def main() -> None:
    z, e1, dev = frozen()
    n, K = z.shape[0], e1.shape[0]
    normaliser = float(np.arctanh(0.7) / float(torch.linalg.vector_norm(z, dim=-1).median().item()))

    zb = encode_ball(z, 1.0, normaliser)
    eb = encode_ball(e1, 1.0, normaliser)
    d_eu = torch.cdist(z, e1).argmin(1).cpu().numpy().astype(np.int64)

    train = pd.read_parquet(AP.DATA / "train.parquet")
    valid = pd.read_parquet(AP.DATA / "valid.parquet")
    succ_tr = AP.successors(train, MM.N_ITEMS)
    succ_va = AP.successors(valid, MM.N_ITEMS)
    pop, freq = AP.popularity_and_frequency(train, MM.N_ITEMS)
    pb, fb = AP.decile_bins(pop), AP.decile_bins(freq)

    def mk(pairs, tag):
        a_, p_ = pairs[:, 0].copy(), pairs[:, 1].copy()
        rel = []
        for x in a_:
            x = int(x)
            r = set(succ_tr.get(x, ())) | set(succ_va.get(x, ()))
            r.discard(x)
            rel.append(r)
        negs, st = MM.sample_negatives(a_, p_, rel, pb, fb, 10,
                                       np.random.default_rng(SEED))
        return {"anchors": a_, "positives": p_, "negatives": negs,
                "violations": int(st["invariant_violations"])}

    zp = np.load(MM.OUT_DIR / "behavior_pairs_s42.npz")
    va_pairs = AP.transitions(valid, MM.N_ITEMS)
    a_, b_ = va_pairs[:, 0], va_pairs[:, 1]
    va_pairs = va_pairs[(a_ != b_) & (a_ >= 0) & (a_ < MM.N_ITEMS) & (b_ >= 0) & (b_ < MM.N_ITEMS)]
    sets = {
        "train_pairs": mk(np.stack([zp["anchors"].astype(np.int64),
                                    zp["positives"].astype(np.int64)], 1), "train"),
        "valid_relations": mk(va_pairs, "valid"),
    }
    print("eval sets:", {k: (len(v["anchors"]), v["violations"]) for k, v in sets.items()})

    base_assign = torch.as_tensor(d_eu, device=dev)
    counts = torch.bincount(base_assign, minlength=K).float()
    item_r = torch.linalg.vector_norm(zb, dim=-1)
    proto_r = torch.linalg.vector_norm(eb, dim=-1)

    variants = {}
    variants["as_trained"] = evaluate(eb, zb, sets, "as trained (frozen codebook)", d_eu)
    for lo, hi, tag in ((0.30, 0.92, "relayout_r030_092"),
                        (0.15, 0.95, "relayout_r015_095"),
                        (0.45, 0.90, "relayout_r045_090")):
        eb2 = relayout_codes(zb, eb, base_assign, lo, hi)
        variants[tag] = evaluate(eb2, zb, sets, f"prototypes re-laid to radius [{lo},{hi}]", d_eu)

    # Radial share of the distance spread, which is the quantity step 1b found
    # to be 2.28%.
    def radial_share(e_ball):
        ui = zb / item_r.unsqueeze(1).clamp_min(1e-12)
        ue = e_ball / torch.linalg.vector_norm(e_ball, dim=-1, keepdim=True).clamp_min(1e-12)
        cos = (ui @ ue.T).clamp(-1, 1)
        dr = item_r.unsqueeze(1) - torch.linalg.vector_norm(e_ball, dim=-1).unsqueeze(0)
        chord = (2.0 * (1.0 - cos)).clamp_min(0)
        rv = float(dr.var(dim=1).mean())
        av = float((chord ** 2).var(dim=1).mean())
        return rv / max(rv + av, 1e-12)

    report = {
        "seed": SEED, "normalizer": normaliser,
        "frozen": "Exp1a resample encoder and item representations never updated",
        "layout_rule": "target radius from sqrt(cluster size), direction of each prototype fixed",
        "as_trained": {
            "prototype_radius_mean": float(proto_r.mean()),
            "prototype_radius_std": float(proto_r.std()),
            "prototype_radius_cv": float(proto_r.std() / proto_r.mean()),
            "radial_share_of_distance_spread": radial_share(eb),
            "l1_cluster_size": {"min": int(counts.min()), "max": int(counts.max()),
                                "median": float(counts.median())},
        },
        "variants": {},
    }
    for tag, v in variants.items():
        if tag == "as_trained":
            report["variants"][tag] = v
            continue
        lo, hi = (0.30, 0.92) if tag == "relayout_r030_092" else (
            (0.15, 0.95) if tag == "relayout_r015_095" else (0.45, 0.90))
        eb2 = relayout_codes(zb, eb, base_assign, lo, hi)
        r2 = torch.linalg.vector_norm(eb2, dim=-1)
        report["variants"][tag] = {
            **v,
            "prototype_radius_mean": float(r2.mean()),
            "prototype_radius_std": float(r2.std()),
            "prototype_radius_cv": float(r2.std() / r2.mean()),
            "radial_share_of_distance_spread": radial_share(eb2),
        }

    OUT.write_text(json.dumps(report, indent=2))
    print("\n=== 原型径向布局 ===")
    print(f"  训练态: 半径 mean={report['as_trained']['prototype_radius_mean']:.4f} "
          f"CV={report['as_trained']['prototype_radius_cv']:.4f} "
          f"径向占比={report['as_trained']['radial_share_of_distance_spread']:.4f}")
    print(f"  L1 簇大小 min={report['as_trained']['l1_cluster_size']['min']} "
          f"median={report['as_trained']['l1_cluster_size']['median']:.0f} "
          f"max={report['as_trained']['l1_cluster_size']['max']}")
    print("\n=== 变体对比 (valid 独立关系) ===")
    print(f"  {'variant':22s} {'径向占比':>9} {'vs欧氏变化率':>12} {'pos':>8} {'neg':>9} {'ER':>7}")
    for tag, v in report["variants"].items():
        rs = v.get("radial_share_of_distance_spread", report["as_trained"]["radial_share_of_distance_spread"])
        vv = v["valid_relations"]
        print(f"  {tag:22s} {rs:>9.4f} {v['change_vs_euclidean']:>12.4f} "
              f"{vv['p_share_pos']:>8.5f} {vv['p_share_neg']:>9.5f} {vv['er']:>7.3f}")
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()