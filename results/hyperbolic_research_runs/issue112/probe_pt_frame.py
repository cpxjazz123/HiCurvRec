"""Can the codeword-frame residual chain work at all? (issue #112, the path-B candidate)

Two questions, answered on the frozen data before anything is implemented:

1. IS THE PROPOSAL WELL-POSED? The proposal transports the RESIDUAL into the level-0 codeword's
   tangent frame and then quantises it with the SHARED level-1 codebook. But the transported
   frame depends on the item, and the shared codebook has no fixed coordinates in an item's frame.
   The DUAL keeps the codebook shared in its canonical (origin) frame and transports the
   CODEBOOK into the item's frame instead; that is well posed. This script measures the dual.
2. DOES IT DO ANYTHING? The transported frame is a position-dependent isometry of the tangent
   space, so the quantisation error can only change if the local metric does. Measured here: the
   level-1 residual error with and without the transport, against the item's radius.
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np, torch
REPO = Path("/fs04/ar57/wenyu/GeneRec")
sys.path.insert(0, str(REPO / "tools" / "cgv"))
import engine
TREE = REPO / "stage2_RQ-VAE/TIGER_hyp_busemann_boundary"
CONTROL = REPO / "results/stage2_RQ-VAE/r8_control/rqvae_best.pth"
OUT = Path(__file__).resolve().parent / "pt_frame_probe.json"
SAMPLE = 4000
FLOOR = 1e-30

def conformal(x, c):
    return 2.0 / (1.0 - c * (x * x).sum(-1)).clamp_min(FLOOR)

def mobius_add(x, y, c):
    x2 = (x * x).sum(-1, keepdim=True); y2 = (y * y).sum(-1, keepdim=True)
    xy = (x * y).sum(-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    den = 1 + 2 * c * xy + c * c * x2 * y2
    return num / den.clamp_min(FLOOR)

def gyration(a, b, v, c):
    return mobius_add(-mobius_add(a, b, c), mobius_add(a, mobius_add(b, v, c), c), c)

sys.path.insert(0, str(REPO / "tools" / "hyperbolic_verifier"))
import transport as VT          # the verifier's regression-tested transport


def parallel_transport(v, x, y, c):
    return VT.parallel_transport(v, x, y, c)


def transport_from_origin(v, y, c):
    return VT.transport_from_origin(v, y, c)

trainer, saved, mods = engine.load_trainer(TREE, module_name="pt_probe")
try:
    emb = trainer.maybe_apply_pca(trainer.load_embeddings(trainer.EMBEDDING_FILE), trainer.PCA_DIM, trainer.SEED)
    model = trainer.RQVAE(trainer._tokenizer_config(), in_dim=int(emb.shape[1]))
    state = torch.load(CONTROL, map_location="cpu"); model.load_state_dict(state.get("state_dict") or state, strict=False); model.eval()
    rng = np.random.default_rng(20261023)
    rows = rng.choice(emb.shape[0], size=min(SAMPLE, emb.shape[0]), replace=False)
    with torch.no_grad():
        z = model.encoder(torch.from_numpy(emb[rows])).double()
    c = 1.0
    import scripts.hyp_losses as HL
    x = HL.encode_points(z.float(), c).double()                       # the item's ball point
    code0 = model.rq.vq_layers[0].get_code_embs().double()             # the level-0 codebook (tangent)
    code1 = model.rq.vq_layers[1].get_code_embs().double()             # the level-1 codebook (tangent)
    # the host's level-0 assignment: the nearest codeword in the AMBIENT latent space
    label0 = torch.cdist(z, code0).argmin(-1)
    a = x[label0]                                                      # the assigned codeword's ball point
    def log0(p, c):
        root = math.sqrt(c)
        norm = torch.linalg.vector_norm(p, dim=-1, keepdim=True).clamp_min(1e-12)
        return torch.atanh((root * norm).clamp(max=1 - 1e-12)) / (root * norm) * p
    tx = log0(x, c)                                                    # the item's tangent vector
    ta = log0(a, c)                                                    # the codeword's tangent vector
    r_amb = tx - ta                                                    # the ambient residual
    # the DUAL: the shared level-1 codebook stays in its canonical frame and is TRANSPORTED
    # into the item's level-0 frame, so the comparison happens in one frame
    code1_in_a = transport_from_origin(code1[None, :, :], a[:, None, :], c)
    r_tr = transport_from_origin(r_amb, a, c)                           # the residual in a's frame
    # the level-1 assignment and the quantisation error in each frame
    err_amb = torch.linalg.vector_norm(r_amb[:, None, :] - code1[None, :, :], dim=-1)
    j_amb = err_amb.argmin(-1)
    err_tr = torch.linalg.vector_norm(r_tr[:, None, :] - code1_in_a, dim=-1)
    j_tr = err_tr.argmin(-1)
    # the Riemannian norm of the residual that survives quantisation, in each frame
    lam_a = conformal(a, c)
    lam_0 = conformal(torch.zeros_like(a), c)
    norm_amb = torch.linalg.vector_norm(r_amb - code1[j_amb], dim=-1) * lam_0
    norm_tr = torch.linalg.vector_norm(r_tr - code1_in_a[torch.arange(len(j_tr)), j_tr], dim=-1) * lam_a
    radius = torch.linalg.vector_norm(x, dim=-1)
    change = float((j_amb != j_tr).double().mean())
    hi = radius > radius.median()
    payload = {
        "schema_version": 1, "issue": 112, "items": int(len(rows)),
        "checkpoint": str(CONTROL), "checkpoint_epoch": state.get("epoch"),
        "well_posedness": (
            "the proposal as written transports the RESIDUAL into the item's level-0 frame and "
            "then quantises it with the SHARED level-1 codebook; the shared codebook has no fixed "
            "coordinates in an item-dependent frame, so that version is not well posed. The dual -- "
            "the codebook stays canonical and is transported INTO the item's frame -- is well "
            "posed and is what this probe measures."),
        "assignment_change_rate": change,
        "mean_residual_norm_ambient": float(norm_amb.mean()),
        "mean_residual_norm_transported": float(norm_tr.mean()),
        "near_items_median_radius": {
            "ambient": float(norm_amb[~hi].mean()), "transported": float(norm_tr[~hi].mean()),
            "radius_mean": float(radius[~hi].mean())},
        "far_items_above_median_radius": {
            "ambient": float(norm_amb[hi].mean()), "transported": float(norm_tr[hi].mean()),
            "radius_mean": float(radius[hi].mean())},
        "radius_range": [float(radius.min()), float(radius.max())],
        "transport_is_an_isometry_check": float(
            (torch.linalg.vector_norm(transport_from_origin(r_amb, a, c), dim=-1) * lam_a
             - torch.linalg.vector_norm(r_amb, dim=-1) * lam_0).abs().max()),
        "transport_convention": ("the residual lives in the ORIGIN's tangent space, so it is "
                                 "transported FROM THE ORIGIN into the codeword's frame; the "
                                 "earlier probe transported it 'from x', which rotates it by the "
                                 "wrong gyration"),
    }
    OUT.write_text(json.dumps(payload, indent=2) + chr(10))
    print(json.dumps(payload, indent=1))
finally:
    engine.unload_trainer(saved, mods)
