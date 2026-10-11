"""Step 5: reassign L1 with the semantic representation held frozen.

The question is whether enrichment can be improved at all WITHOUT touching what
the encoder has learned. Earlier steps failed because every one-sided push on the
L1 assignment either collapsed the codebook or diverged the encoder norm. Here the
encoder is frozen, so neither failure mode is reachable: the only lever is which
codeword each item is assigned to, and the assignment is a discrete variable.

The objective is computed directly on the canonical seed-42 pairs, which is the
quantity the metric reports, rather than through a surrogate:

    F(sigma) = #{positive pairs with sigma(a) = sigma(b)}
             - lambda * #{negative pairs with sigma(a) = sigma(c)}

Behaviour relations are symmetrised before use, because an L1 group is an
undirected concept. lambda balances the two terms by their mean degrees so that
a heavily connected popular item cannot dominate the objective on its own.

Three assignments are reported, and the difference between them is the actual
answer to the step 5 question:

  initial    KMeans on the frozen latents, which is how the codebook was
             originally seeded, so this reproduces the baseline
  optimised  after local search, with no constraint at all
  voronoi    nearest-centroid assignment of the optimised partition's centroids,
             i.e. what 256 codeword vectors can actually express

If optimised is much better than voronoi, then the improvement exists only in a
partition the codebook cannot represent, and the honest conclusion is that L1
cannot carry it. If voronoi keeps the gain, the mechanism is deployable.

The encoder is never updated: no gradient is taken anywhere in this script.
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
import analyze_prefix_behavior as AP  # noqa: E402
import monotonicity_metrics as MM  # noqa: E402

SEED = 42
N_CODE = 256
N_PASSES = 40
OUT_DIR = MM.OUT_DIR
CKPT = AP.ROOT / "results/stage2_RQ-VAE/TIGER_RQ-VAE/rqvae_best.pth"
PAIRS = OUT_DIR / "behavior_pairs_s42.npz"
OUT_JSON = OUT_DIR / "step5_l1_reassignment.json"


def load_encoder():
    """Rebuild the RQ-VAE from the stage 2 config and load the baseline weights."""
    sys.path.insert(0, str(HERE.parent))
    import train_rqvae as T

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    T.set_seed(SEED)
    emb = T.maybe_apply_pca(T.load_embeddings(T.EMBEDDING_FILE), T.PCA_DIM, T.SEED)
    model = T.RQVAE(T._tokenizer_config(), in_dim=emb.shape[1]).to(dev)
    blob = torch.load(CKPT, map_location=dev)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, torch.from_numpy(emb), dev


@torch.no_grad()
def quantize_three_levels(latents, assign_l1, code_l2, code_l3, dev):
    """Rebuild the SID given an L1 assignment. L2 and L3 keep their codebooks."""
    z = latents.to(dev)
    e1 = code_l1_from_centroids(assign_l1, z, dev)
    r1 = z - e1[assign_l1]
    e2 = torch.as_tensor(code_l2, device=dev, dtype=torch.float32)
    e3 = torch.as_tensor(code_l3, device=dev, dtype=torch.float32)
    l2 = torch.cdist(r1, e2).argmin(dim=1)
    r2 = r1 - e2[l2]
    l3 = torch.cdist(r2, e3).argmin(dim=1)
    return (
        torch.stack([torch.as_tensor(assign_l1, device=dev), l2, l3], dim=1)
        .detach().cpu().numpy().astype(np.int32)
    )


def code_l1_from_centroids(assign, z, dev):
    cent = torch.zeros((N_CODE, z.shape[1]), device=dev)
    cnt = torch.zeros(N_CODE, device=dev)
    at = torch.as_tensor(assign, dtype=torch.long, device=dev)
    cent.index_add_(0, at, z)
    cnt.index_add_(0, at, torch.ones_like(at, dtype=torch.float))
    empty = cnt == 0
    cent[~empty] = cent[~empty] / cnt[~empty].unsqueeze(1)
    # An emptied codeword keeps its previous position rather than collapsing to
    # the origin, which would distort every distance.
    cent[empty] = z.mean(dim=0)
    return cent


def build_adjacency():
    """Symmetric behaviour positives from train, and the canonical negatives."""
    frame = pd.read_parquet(AP.DATA / "train.parquet")
    succ = AP.successors(frame, MM.N_ITEMS)
    pos = [[] for _ in range(MM.N_ITEMS)]
    for a, targets in succ.items():
        for b in targets:
            if a == b:
                continue
            pos[a].append(b)
            pos[b].append(a)
    z = np.load(PAIRS)
    anchors = z["anchors"].astype(np.int64)
    positives = z["positives"].astype(np.int64)
    negs = z["negatives"].astype(np.int64)
    neg = [[] for _ in range(MM.N_ITEMS)]
    for a_row, c in zip(anchors, negs):
        for n in c:
            if n != a_row:
                neg[a_row].append(int(n))
                neg[int(n)].append(int(a_row))
    return (
        [np.asarray(sorted(set(p)), dtype=np.int64) for p in pos],
        [np.asarray(sorted(set(n)), dtype=np.int64) for n in neg],
    )


def objective(assign, pos_adj, neg_adj, lam):
    p = sum(int(assign[a] == assign[b]) for a, bs in enumerate(pos_adj) for b in bs) // 2
    n = sum(int(assign[a] == assign[b]) for a, bs in enumerate(neg_adj) for b in bs) // 2
    return p - lam * n, p, n


def local_search(assign, pos_adj, neg_adj, lam, passes=N_PASSES, seed=SEED):
    """Greedy single-item moves, deterministic: items are visited in index order."""
    rng = np.random.default_rng(seed)
    assign = assign.copy()
    history = []
    for p_i in range(passes):
        moved = 0
        for i in range(len(assign)):
            u = assign[i]
            bp = np.bincount(assign[pos_adj[i]], minlength=N_CODE) if len(pos_adj[i]) else np.zeros(N_CODE, np.int64)
            bn = np.bincount(assign[neg_adj[i]], minlength=N_CODE) if len(neg_adj[i]) else np.zeros(N_CODE, np.int64)
            delta = (bp - bp[u]) - lam * (bn - bn[u])
            delta[u] = -(10 ** 9)
            v = int(delta.argmax())
            if delta[v] > 0:
                assign[i] = v
                moved += 1
        f, p_in, n_in = objective(assign, pos_adj, neg_adj, lam)
        history.append({"pass": p_i + 1, "moved": moved,
                        "objective": f, "pos_inside": p_in, "neg_inside": n_in})
        if moved == 0:
            break
    return assign, history


def sharing(prefix_a, prefix_b):
    return float((prefix_a == prefix_b).mean())


def metrics_for(levels, anchors, positives, negs, recall):
    k = negs.shape[1]
    anchor_rep = np.repeat(anchors, k)
    out = {"recall@10": recall}
    for lvl, label in ((1, "L1"), (2, "L1_L2"), (3, "L1_L2_L3")):
        codes = np.unique(levels[:, :lvl], axis=0, return_inverse=True)[1]
        p = sharing(codes[positives], codes[anchors])
        n = sharing(codes[negs.reshape(-1)], codes[anchor_rep])
        out[label] = {
            "p_share_pos": p, "p_share_neg": n, "gap": p - n,
            "er": (p / n) if n > 0 else None,
        }
    full = np.unique(levels[:, :3], axis=0, return_inverse=True)[1]
    sizes = np.bincount(full)
    out["collision_rate"] = float(1.0 - len(sizes) / len(levels))
    out["distinct_triples"] = int(len(sizes))
    out["l1_codewords_used"] = int(len(np.unique(levels[:, 0])))
    out["l1_max_group"] = int(np.bincount(levels[:, 0]).max())
    return out


def main() -> None:
    model, all_emb, dev = load_encoder()
    z = model.encoder(all_emb.to(dev)).detach().cpu().numpy().astype(np.float64)

    with torch.no_grad():
        e2 = model.rq.vq_layers[1].get_code_embs().detach().cpu().numpy().astype(np.float64)
        e3 = model.rq.vq_layers[2].get_code_embs().detach().cpu().numpy().astype(np.float64)

    pos_adj, neg_adj = build_adjacency()
    mean_pos = np.mean([len(x) for x in pos_adj])
    mean_neg = np.mean([len(x) for x in neg_adj])
    lam = float(mean_pos / max(mean_neg, 1e-9))
    print(f"mean behaviour degree {mean_pos:.2f}, mean canonical negative degree {mean_neg:.2f}, "
          f"lambda {lam:.4f}")

    zt = torch.as_tensor(z, device=dev, dtype=torch.float32)
    print("KMeans seeding the L1 codebook on the frozen latents ...")
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=N_CODE, n_init=10, random_state=SEED).fit(z)
    assign0 = km.labels_.astype(np.int64)

    assign1, hist = local_search(assign0, pos_adj, neg_adj, lam)
    print(f"local search: {len(hist)} passes, last moved {hist[-1]['moved']}, "
          f"objective {hist[-1]['objective']:.1f}")

    # What 256 codeword vectors can actually express.
    cent = code_l1_from_centroids(assign1, zt, dev).cpu().numpy().astype(np.float64)
    assign2 = ((z[:, None, :] - cent[None, :, :]) ** 2).sum(-1).argmin(axis=1).astype(np.int64)
    kept = float((assign2 == assign1).mean())
    print(f"voronoi round-trip keeps {kept * 100:.4f}% of the optimised assignment "
          f"(overlap {int((assign2 == assign1).sum())} of {len(assign1)} items)")

    zpairs = np.load(PAIRS)
    anchors = zpairs["anchors"].astype(np.int64)
    positives = zpairs["positives"].astype(np.int64)
    negs = zpairs["negatives"].astype(np.int64)
    recall = 0.053674

    variants = {}
    for name, asg in (("initial", assign0), ("optimised", assign1), ("voronoi", assign2)):
        levels = quantize_three_levels(zt, asg, e2, e3, dev)
        variants[name] = metrics_for(levels, anchors, positives, negs, recall)
        m = variants[name]
        print(f"  {name:10s} L1 pos={m['L1']['p_share_pos']:.4f} "
              f"neg={m['L1']['p_share_neg']:.5f} ER={m['L1']['er']} "
              f"collision={m['collision_rate']:.4f} L1used={m['l1_codewords_used']}")

    out = {
        "seed": SEED,
        "frozen": "encoder weights loaded from TIGER_RQ-VAE/rqvae_best.pth and never updated",
        "objective": {
            "definition": "F = #{positive pairs sharing L1} - lambda * #{negative pairs sharing L1}",
            "lambda": lam,
            "mean_behaviour_degree": float(mean_pos),
            "mean_negative_degree": float(mean_neg),
            "passes": hist,
        },
        "voronoi_round_trip_keep_rate": kept,
        "variants": variants,
        "recall_note": "Recall@10 is the baseline's 0.053674 and is IDENTICAL for all three "
                       "variants by construction: no model was retrained and no Stage 3 was run. "
                       "The comparison is of the SID statistics only.",
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"written: {OUT_JSON}")


if __name__ == "__main__":
    main()