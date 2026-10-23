"""Pre-check the product-geodesic level-0 quantiser on the frozen data (issue #112).

The candidate (hyp_product_l1_quantiser v2): the level-0 assignment minimises a sum of two
factor geodesic distances -- d_{c_s}^2(x_s, A_j) + d_{c_b}^2(x_b, B_j) -- with ONE common tangent
scale and RELATIVE curvatures (c_s = 1, c_b = 2), the scale being the measured 1.4937731.

Before any implementation, three questions on the frozen data (the matched control checkpoint,
epoch 3000, 4000 items):

1. IS IT A NO-OP? How many level-0 assignments does the product metric change against the host's
   Euclidean assignment?
2. IS IT INERT? If the joint assignment agrees with ONE factor alone almost always, the split is a
   rotation and the mechanism does nothing (this is the proposal's own declared falsifier).
3. WHAT HAPPENS TO THE L0 TAIL? The distinct codes and the largest cell of the proposed partition
   against the control's, because the whole point of the candidate is the L0 tail.

    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        results/hyperbolic_research_runs/issue112/probe_product_quantiser.py
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np, torch
REPO = Path("/fs04/ar57/wenyu/GeneRec")
sys.path.insert(0, str(REPO / "tools" / "cgv")); sys.path.insert(0, str(REPO / "tools" / "hyperbolic_verifier"))
import busemann as B, engine
TREE = REPO / "stage2_RQ-VAE/TIGER_hyp_busemann_boundary"
CONTROL = REPO / "results/stage2_RQ-VAE/r8_control/rqvae_best.pth"
OUT = Path(__file__).resolve().parent / "product_quantiser_probe.json"
SAMPLE = 4000
C_S, C_B, SCALE = 1.0, 2.0, 1.4937731472770945   # the v2 declaration

def expmap0(v, c):
    root = math.sqrt(c)
    norm = torch.linalg.vector_norm(v, dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.tanh(root * norm) / (root * norm) * v

trainer, saved, mods = engine.load_trainer(TREE, module_name="product_probe")
try:
    emb = trainer.maybe_apply_pca(trainer.load_embeddings(trainer.EMBEDDING_FILE), trainer.PCA_DIM, trainer.SEED)
    model = trainer.RQVAE(trainer._tokenizer_config(), in_dim=int(emb.shape[1]))
    state = torch.load(CONTROL, map_location="cpu"); model.load_state_dict(state.get("state_dict") or state, strict=False); model.eval()
    rng = np.random.default_rng(20261023)
    rows = rng.choice(emb.shape[0], size=min(SAMPLE, emb.shape[0]), replace=False)
    with torch.no_grad():
        z = model.encoder(torch.from_numpy(emb[rows])).double()
    code = model.rq.vq_layers[0].get_code_embs().double()          # the level-0 codebook (tangent)
    half = code.shape[1] // 2
    a_s, a_b = code[:, :half], code[:, half:]
    x_s = expmap0(SCALE * z[:, :half], C_S)
    x_b = expmap0(SCALE * z[:, half:], C_B)
    A_s, A_b = expmap0(a_s, C_S), expmap0(a_b, C_B)
    def pair_dist(x, codes, c):
        out = torch.empty(x.shape[0], codes.shape[0], dtype=torch.float64)
        for j in range(codes.shape[0]):
            out[:, j] = B.poincare_distance(x, codes[j].expand_as(x), c)
        return out
    d_s = pair_dist(x_s, A_s, C_S)
    d_b = pair_dist(x_b, A_b, C_B)
    joint = (d_s ** 2 + d_b ** 2).argmin(-1)
    only_s = (d_s ** 2).argmin(-1)
    only_b = (d_b ** 2).argmin(-1)
    host = torch.cdist(z, code).argmin(-1)                          # the host's Euclidean assignment
    def tail(labels):
        counts = torch.bincount(labels, minlength=256)
        return int((counts > 0).sum()), int(counts.max()), float(counts.float().mean())
    host_codes, host_max, _ = tail(host)
    joint_codes, joint_max, _ = tail(joint)
    payload = {
        "schema_version": 1, "issue": 112, "items": int(len(rows)),
        "checkpoint": str(CONTROL), "checkpoint_epoch": state.get("epoch"),
        "declaration": {"c_s": C_S, "c_b": C_B, "common_tangent_scale": SCALE,
                        "scale_source": "results/hyperbolic_research_runs/subspace_111/latent_scale.json"},
        "no_op_check": {"level0_assignments_changed_vs_the_host": float((joint != host).double().mean()),
                        "host_distinct_codes": host_codes, "product_distinct_codes": joint_codes},
        "inert_check": {"joint_equals_the_semantic_factor_only": float((joint == only_s).double().mean()),
                        "joint_equals_the_behaviour_factor_only": float((joint == only_b).double().mean()),
                        "the_two_factors_disagree": float((only_s != only_b).double().mean()),
                        "declared_falsifier": ">= 0.99 agreement with one factor alone is inert"},
        "tail_check": {"host_largest_cell": host_max, "product_largest_cell": joint_max,
                       "host_distinct_codes": host_codes, "product_distinct_codes": joint_codes},
    }
    OUT.write_text(json.dumps(payload, indent=2) + chr(10))
    print(json.dumps(payload, indent=1))
finally:
    engine.unload_trainer(saved, mods)
