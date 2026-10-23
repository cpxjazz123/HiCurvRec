"""Pre-check the LEVEL-WISE factorisation (a different family from the approved one).

The approved candidate (hyp_product_l1_quantiser) splits ONE level into two factors. This
candidate splits the LEVELS: level 0 lives in a ball of curvature c_0, level 1 in one of curvature
c_1, level 2 in one of curvature c_2, and the residual that enters level l is mapped into that
level's ball before the assignment. The three curvatures are RELATIVE to each other, which is the
only way a curvature can matter in the Poincare ball (F1).

Why this family and not another: the cohort's L2->L3 branching is about 1.47, i.e. the third token
is nearly determined by the second -- the tree below the first level carries almost no
information. A mechanism that changes the geometry of the DEEPER levels attacks that, and it
touches a different parameter group from the L1 assignment, so it cannot coarsen the content
partition the way the behaviour terms did.

Measured on the frozen data, before any implementation: the per-level assignment change rate
(against the host's Euclidean chain) and the branching of the resulting tree.

    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        results/hyperbolic_research_runs/issue112/probe_levelwise_factors.py
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np, torch
REPO = Path("/fs04/ar57/wenyu/GeneRec")
sys.path.insert(0, str(REPO / "tools/cgv")); sys.path.insert(0, str(REPO / "tools/hyperbolic_verifier"))
import busemann as B, engine, transport as VT
TREE = REPO / "stage2_RQ-VAE/TIGER_hyp_product_l1_quantiser"
CONTROL = REPO / "results/stage2_RQ-VAE/r8_control/rqvae_best.pth"
OUT = Path(__file__).resolve().parent / "levelwise_factors_probe.json"
SAMPLE = 4000
# the three levels' relative curvatures (a ratio, never a sweep) and ONE common tangent scale
CURVATURES = (1.0, 2.0, 4.0)
SCALE = 1.4937731472770945

trainer, saved, mods = engine.load_trainer(TREE, module_name="levelwise_probe")
try:
    emb = trainer.maybe_apply_pca(trainer.load_embeddings(trainer.EMBEDDING_FILE), trainer.PCA_DIM, trainer.SEED)
    model = trainer.RQVAE(trainer._tokenizer_config(), in_dim=int(emb.shape[1]))
    state = torch.load(CONTROL, map_location="cpu"); model.load_state_dict(state.get("state_dict") or state, strict=False); model.eval()
    rng = np.random.default_rng(20261023)
    rows = rng.choice(emb.shape[0], size=min(SAMPLE, emb.shape[0]), replace=False)
    with torch.no_grad():
        z = model.encoder(torch.from_numpy(emb[rows])).double()
    codes = [layer.get_code_embs().double() for layer in model.rq.vq_layers]
    def assign(residual, codebook, c):
        """The level's assignment in its own ball: map the residual in, compare geodesically."""
        x = VT.expmap0(SCALE * residual, c)
        points = VT.expmap0(SCALE * codebook, c)
        out = torch.empty(x.shape[0], dtype=torch.long)
        for j in range(points.shape[0]):
            d = B.poincare_distance(x, points[j].expand_as(x), c)
            out = torch.where(out == out, out, j) if j == 0 else torch.where(d < out.float(), torch.full_like(out, j), out)
        return out
    def assign_fast(residual, codebook, c):
        x = VT.expmap0(SCALE * residual, c)
        points = VT.expmap0(SCALE * codebook, c)
        best = torch.full((x.shape[0],), float("inf"), dtype=torch.float64)
        out = torch.zeros(x.shape[0], dtype=torch.long)
        for j in range(points.shape[0]):
            d = B.poincare_distance(x, points[j].expand_as(x), c)
            better = d < best
            best = torch.where(better, d, best)
            out = torch.where(better, torch.full_like(out, j), out)
        return out
    # the host's chain
    residual = z.clone()
    host = []
    for level, codebook in enumerate(codes):
        quant = codebook[torch.cdist(residual, codebook).argmin(-1)]
        host.append(torch.cdist(residual, codebook).argmin(-1))
        residual = residual - quant
    # the candidate's chain: each level's residual is mapped into that level's own ball
    residual = z.clone()
    ours = []
    for level, codebook in enumerate(codes):
        c = CURVATURES[level]
        idx = assign_fast(residual, codebook, c)
        ours.append(idx)
        residual = residual - codebook[idx]
    def branching(levels):
        l1, l2 = levels[0], levels[1]
        branches = len(set(zip(l1.tolist(), l2.tolist())))
        deeper = len(set(zip(l2.tolist(), levels[2].tolist())))
        return branches, deeper
    host_branches, host_deeper = branching(host)
    our_branches, our_deeper = branching(ours)
    payload = {
        "schema_version": 1, "issue": 112,
        "candidate_family": "level_wise_factorisation (one factor per level, relative curvatures)",
        "declaration": {"curvatures": list(CURVATURES), "common_tangent_scale": SCALE,
                        "scale_source": "results/hyperbolic_research_runs/subspace_111/latent_scale.json"},
        "items": int(len(rows)), "checkpoint": str(CONTROL), "checkpoint_epoch": state.get("epoch"),
        "not_a_no_op": {
            "level0_assignments_changed": float((host[0] != ours[0]).double().mean()),
            "level1_assignments_changed": float((host[1] != ours[1]).double().mean()),
            "level2_assignments_changed": float((host[2] != ours[2]).double().mean()),
        },
        "tree": {
            "host_l1_to_l2_branches": host_branches,
            "candidate_l1_to_l2_branches": our_branches,
            "host_l2_to_l3_branches": host_deeper,
            "candidate_l2_to_l3_branches": our_deeper,
            "note": "the cohort's control sits near 1.47 at L2->L3, so a candidate that makes the "
                    "deeper tree richer is attacking a real weakness",
        },
        "distinct_codes": {
            "host": [int(len(set(level.tolist()))) for level in host],
            "candidate": [int(len(set(level.tolist()))) for level in ours],
        },
    }
    OUT.write_text(json.dumps(payload, indent=2) + chr(10))
    print(json.dumps(payload, indent=1))
finally:
    engine.unload_trainer(saved, mods)
