"""Calibrate the three geometry branches onto one scale, on the TRAIN split only.

The curved half of the joint codebook is scored under three geometries: Euclidean in the tangent
block, the baseline's hyperbolic branch, and the spherical branch.  Their natural magnitudes are
not the same -- a squared geodesic on the unit sphere lives in ``[0, pi^2]`` while the hyperbolic
squared distance grows with the radius -- so a mixture of them would be dominated by whichever
branch happens to be largest.

The factors are calibrated once, on the training items only, with the frozen F0 model and its
frozen checkpoint, and then written into the trainer as constants.  The hyperbolic branch keeps the
factor one, so a row that routes entirely to it is the baseline's number, element for element.

    python3 results/research/issue215/calibrate_scales.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = REPO_ROOT / "analysis_sid_recall" / "r2" / "19" / "TIGER_mixed_rq_F0"
CHECKPOINT = REPO_ROOT / "analysis_sid_recall" / "r2" / "19" / "rqvae_best.pth"
TRAIN = REPO_ROOT / "results" / "stage0_build_parquet" / "train.parquet"
SEED = 42
SAMPLE = 4096

sys.path.insert(0, str(REPO_ROOT / "stage2_RQ-VAE" / "TIGER_role_geom_rq"))


def load_archive_model():
    """The archived F0 model package, imported under its own name."""
    spec = importlib.util.spec_from_file_location(
        "f0_model_pkg", ARCHIVE / "model" / "__init__.py",
        submodule_search_locations=[str(ARCHIVE / "model")])
    module = importlib.util.module_from_spec(spec)
    sys.modules["f0_model_pkg"] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    from types import SimpleNamespace

    package = load_archive_model()
    from f0_model_pkg.model import MixedRQVAE
    from f0_model_pkg.poincare_contract import expmap0, logmap0, squared_distance
    # the spherical branch is new in this issue, so it comes from the candidate tree while the
    # baseline arithmetic comes from the archived F0 source
    sphere_spec = importlib.util.spec_from_file_location(
        "candidate_spherical", REPO_ROOT / "stage2_RQ-VAE" / "TIGER_role_geom_rq" / "model" / "spherical_contract.py")
    sphere = importlib.util.module_from_spec(sphere_spec)
    sphere_spec.loader.exec_module(sphere)
    sphere_expmap0 = sphere.sphere_expmap0
    sphere_squared_distance = sphere.sphere_squared_distance

    config = SimpleNamespace(hidden_sizes=(512, 256, 128), codebook_num=3,
                             codebook_size=(256, 256, 256), codebook_dim=32, dropout=0.0,
                             beta=0.25, vq_type="vq", ema_decay=0.99, fix_code_embs=False,
                             sk_epsilon=0.003, sk_iters=50)
    model = MixedRQVAE(config, in_dim=768, curvature=1.0)
    state = torch.load(CHECKPOINT, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"checkpoint: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    model.eval()

    frame = pd.read_parquet(TRAIN, columns=["target"])
    train_ids = np.unique(frame["target"].to_numpy(dtype=np.int64))
    generator = np.random.default_rng(SEED)
    sample = np.sort(generator.choice(train_ids, size=min(SAMPLE, len(train_ids)), replace=False))

    embeddings = torch.from_numpy(np.load(REPO_ROOT / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"))
    with torch.no_grad():
        encoded = model.encoder(embeddings[torch.from_numpy(sample)])
    z_e, z_t = encoded[..., :16], encoded[..., 16:]
    residual_h = expmap0(z_t, 1.0)
    tangent_residual = logmap0(residual_h, 1.0)
    level = model.rq.vq_layers[0]
    codes = level.get_code_embs()
    codes_e, codes_h = codes[:, :16], codes[:, 16:]

    d_euclidean = (tangent_residual.unsqueeze(1) - codes_h.unsqueeze(0)).pow(2).sum(-1)
    d_hyperbolic = squared_distance(residual_h.unsqueeze(1), expmap0(codes_h, 1.0).unsqueeze(0), 1.0)
    d_spherical = sphere_squared_distance(sphere_expmap0(tangent_residual).unsqueeze(1),
                                           sphere_expmap0(codes_h).unsqueeze(0))
    means = {"euclidean": float(d_euclidean.mean()), "hyperbolic": float(d_hyperbolic.mean()),
             "spherical": float(d_spherical.mean())}
    scales = {"euclidean": means["hyperbolic"] / max(means["euclidean"], 1e-12),
              "hyperbolic": 1.0,
              "spherical": means["hyperbolic"] / max(means["spherical"], 1e-12)}
    print("mean branch distances:", json.dumps(means, indent=1))
    print("scale factors:", json.dumps(scales, indent=1))
    print(f"GEOMETRY_SCALES = ({scales['euclidean']:.6f}, 1.0, {scales['spherical']:.6f})")
    (REPO_ROOT / "results" / "research" / "issue215" / "geometry_scales.json").write_text(
        json.dumps({"means": means, "scales": scales, "sample": int(len(sample)),
                    "checkpoint": str(CHECKPOINT.relative_to(REPO_ROOT)), "seed": SEED}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
