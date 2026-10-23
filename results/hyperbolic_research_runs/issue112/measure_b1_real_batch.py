"""The two measurable tests the revised proposal demands, on the real batch (issue #112)."""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np, torch
REPO = Path("/fs04/ar57/wenyu/GeneRec")
sys.path.insert(0, str(REPO / "tools" / "cgv")); sys.path.insert(0, str(REPO / "tools" / "hyperbolic_verifier"))
import busemann as B, engine
TREE = REPO / "stage2_RQ-VAE/TIGER_hyp_busemann_boundary"
CONTROL = REPO / "results/stage2_RQ-VAE/r8_control/rqvae_best.pth"
OUT = Path(__file__).resolve().parent / "b1_real_batch_tests.json"
SAMPLE = 4000

def rank(v):
    o = torch.argsort(v); r = torch.empty(len(v), dtype=torch.float64); r[o] = torch.arange(len(v), dtype=torch.float64); return r
def spearman(a, b):
    ra, rb = rank(a), rank(b); ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm()))

trainer, saved, mods = engine.load_trainer(TREE, module_name="b1_probe")
try:
    emb = trainer.maybe_apply_pca(trainer.load_embeddings(trainer.EMBEDDING_FILE), trainer.PCA_DIM, trainer.SEED)
    model = trainer.RQVAE(trainer._tokenizer_config(), in_dim=int(emb.shape[1]))
    state = torch.load(CONTROL, map_location="cpu"); model.load_state_dict(state.get("state_dict") or state, strict=False); model.eval()
    import pandas as pd, scripts.hyp_losses as HL
    frame = pd.read_parquet(trainer.TRAIN_FILE)
    pairs = [(int(h[-1]), int(t)) for h, t in zip(frame["seen_history"].to_numpy(), frame["target"].to_numpy(dtype=np.int64)) if h is not None and len(h)]
    pairs = np.asarray(pairs, dtype=np.int64)
    rng = np.random.default_rng(20261023)
    if pairs.shape[0] > SAMPLE: pairs = pairs[rng.choice(pairs.shape[0], size=SAMPLE, replace=False)]
    with torch.no_grad():
        src = model.encoder(torch.from_numpy(emb[pairs[:, 0]])).double()
        tgt = model.encoder(torch.from_numpy(emb[pairs[:, 1]])).double()
    c = 1.0
    sb = HL.encode_points(src.float(), c).double(); tb = HL.encode_points(tgt.float(), c).double()
    omega = tb / torch.linalg.vector_norm(tb, dim=-1, keepdim=True).clamp_min(1e-12) / math.sqrt(c)
    levels = B.busemann_boundary(sb, omega)
    # an interior point derived from the data (the centroid, scaled inside the ball)
    xi = sb.mean(0, keepdim=True); xi = xi / torch.linalg.vector_norm(xi).clamp_min(1e-12) * 0.3
    d_xi = B.poincare_distance(sb, xi.expand_as(sb), c)
    rho_ours = spearman(levels, d_xi)
    # the R10 operator on the same points, for the contrast
    r10 = HL.busemann_poincare(sb, xi.expand_as(sb), c)
    rho_r10 = spearman(r10, d_xi)
    # does the gap separate the host's own label from the other codewords?
    codebook = model.rq.vq_layers[0].get_code_embs().double()
    codes = HL.encode_points(codebook.float(), c).double()
    cl = B.busemann_boundary(codes[:, None, :], omega[None, :, :]).T   # (n, 256)
    gaps = (cl - levels[:, None]).abs()
    label = torch.cdist(tgt, codebook).argmin(-1)
    label_gap = gaps.gather(1, label[:, None]).squeeze(1)
    others = gaps.clone(); others.scatter_(1, label[:, None], float("inf"))
    best_other = others.min(dim=1).values
    separation = float((best_other - label_gap).mean())
    frac_best = float((label_gap < best_other).double().mean())
    payload = {
        "schema_version": 1, "issue": 112, "pairs": int(pairs.shape[0]),
        "checkpoint": str(CONTROL), "checkpoint_epoch": state.get("epoch"),
        "b1_level_is_not_radial": {"spearman_level_vs_distance_to_interior_point": rho_ours,
                                   "r10_same_measurement": rho_r10,
                                   "passes": bool(-0.999 < rho_ours < -0.95)},
        "b1_gap_separates_the_codewords": {"mean_gap_to_the_next_best_codeword": separation,
                                           "fraction_where_the_label_is_the_closest": frac_best,
                                           "passes": bool(frac_best > 0.5)},
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=1))
finally:
    engine.unload_trainer(saved, mods)
