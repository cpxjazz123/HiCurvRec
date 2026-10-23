"""Why is the third token redundant, and can a HYPERBOLIC residual chain change that?

The measured baseline: 92.08% of the (L1, L2) cells carry exactly one L3 code, and every
level's codebook is essentially fully used, so the codebook is not the bottleneck. This asks the
next question: is the redundancy because the items inside a cell have residuals that are much
CLOSER to each other than the level-3 codewords are to each other -- and does replacing the
residual's Euclidean difference by a MÖBIUS difference (a hyperbolic displacement) change that
ratio?

The comparison is inside one fixture (the frozen host, the matched control checkpoint):
the Euclidean chain and the hyperbolic chain, the same encoder, the same codebooks, the same
tangent scale, and the same measurement of the within-cell spread against the codebook spacing.

    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        results/hyperbolic_research_runs/issue112/probe_hyperbolic_residual_chain.py
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
OUT = Path(__file__).resolve().parent / "hyperbolic_residual_chain_probe.json"
SAMPLE = 4000
C = 1.0
SCALE = 1.4937731472770945

def mobius_diff(x, a, c):
    """The hyperbolic displacement x (-) a, mapped back to the tangent at the origin."""
    return VT.logmap0(VT.mobius_add(x, -a, c), c)

trainer, saved, mods = engine.load_trainer(TREE, module_name="hyperbolic_chain_probe")
try:
    emb = trainer.maybe_apply_pca(trainer.load_embeddings(trainer.EMBEDDING_FILE), trainer.PCA_DIM, trainer.SEED)
    model = trainer.RQVAE(trainer._tokenizer_config(), in_dim=int(emb.shape[1]))
    state = torch.load(CONTROL, map_location="cpu"); model.load_state_dict(state.get("state_dict") or state, strict=False); model.eval()
    rng = np.random.default_rng(20261023)
    rows = rng.choice(emb.shape[0], size=min(SAMPLE, emb.shape[0]), replace=False)
    with torch.no_grad():
        z = model.encoder(torch.from_numpy(emb[rows])).double()
    codes = [layer.get_code_embs().double() for layer in model.rq.vq_layers]
    points = [VT.expmap0(SCALE * codebook, C) for codebook in codes]
    x = VT.expmap0(SCALE * z, C)

    def chain(hyperbolic):
        """Run the three levels and return the assignments and the level-3 input residuals."""
        residual = z.clone() if not hyperbolic else x.clone()
        assignments = []
        residuals = []
        for level, codebook in enumerate(codes):
            if hyperbolic:
                d = torch.stack([B.poincare_distance(residual, points[level][j].expand_as(residual), C)
                                 for j in range(points[level].shape[0])], dim=1)
            else:
                d = torch.cdist(residual, codebook)
            idx = d.argmin(-1)
            assignments.append(idx)
            if hyperbolic:
                residual = mobius_diff(residual, points[level][idx], C)
            else:
                residual = residual - codebook[idx]
            residuals.append(residual.clone())
        return assignments, residuals

    def redundancy(assignments, residuals):
        """The within-cell spread against the level-3 codebook's spacing."""
        cells = {}
        for a, b, r in zip(assignments[0].tolist(), assignments[1].tolist(), residuals[2].tolist()):
            cells.setdefault((a, b), []).append(r)
        spreads = []
        for vectors in cells.values():
            if len(vectors) < 2:
                continue
            centre = torch.stack([torch.tensor(v, dtype=torch.float64) for v in vectors]).mean(0)
            stacked = torch.stack([torch.tensor(v, dtype=torch.float64) for v in vectors])
            spreads.append(float((stacked - centre).norm(dim=-1).mean()))
        distances = torch.cdist(codes[2], codes[2])
        distances = distances + torch.eye(codes[2].shape[0], dtype=distances.dtype) * 1e9
        spacing = float(distances.min(dim=1).values.mean())
        mean_spread = float(np.mean(spreads)) if spreads else None
        return {"cells": len(cells), "mean_within_cell_spread": mean_spread,
                "level3_codeword_nearest_neighbour_spacing": spacing,
                "spread_over_spacing": (mean_spread / spacing) if mean_spread else None}

    def cell_redundancy(assignments):
        cells = {}
        for a, b, c3 in zip(assignments[0].tolist(), assignments[1].tolist(), assignments[2].tolist()):
            cells.setdefault((a, b), set()).add(c3)
        per = [len(v) for v in cells.values()]
        return {"cells": len(cells), "mean_l3_codes_per_cell": float(sum(per) / len(per)),
                "fraction_single_l3_code": sum(1 for v in per if v == 1) / len(per)}

    euclid_assign, euclid_res = chain(False)
    hyper_assign, hyper_res = chain(True)
    payload = {
        "schema_version": 1, "issue": 112,
        "purpose": ("does a MÖBIUS residual (a hyperbolic displacement) change the within-cell "
                    "spread relative to the level-3 codebook's spacing -- the ratio that makes the "
                    "third token redundant?"),
        "items": int(len(rows)), "checkpoint": str(CONTROL), "checkpoint_epoch": state.get("epoch"),
        "declaration": {"curvature": C, "common_tangent_scale": SCALE},
        "euclidean_chain": {"redundancy": cell_redundancy(euclid_assign),
                            "spread": redundancy(euclid_assign, euclid_res)},
        "hyperbolic_chain": {"redundancy": cell_redundancy(hyper_assign),
                             "spread": redundancy(hyper_assign, hyper_res)},
    }
    payload["comparison"] = {
        "single_l3_code_fraction": {
            "euclidean": payload["euclidean_chain"]["redundancy"]["fraction_single_l3_code"],
            "hyperbolic": payload["hyperbolic_chain"]["redundancy"]["fraction_single_l3_code"]},
        "spread_over_spacing": {
            "euclidean": payload["euclidean_chain"]["spread"]["spread_over_spacing"],
            "hyperbolic": payload["hyperbolic_chain"]["spread"]["spread_over_spacing"]},
    }
    OUT.write_text(json.dumps(payload, indent=2) + chr(10))
    print(json.dumps(payload["comparison"], indent=1))
    print("euclidean:", json.dumps(payload["euclidean_chain"]["redundancy"]))
    print("hyperbolic:", json.dumps(payload["hyperbolic_chain"]["redundancy"]))
finally:
    engine.unload_trainer(saved, mods)
