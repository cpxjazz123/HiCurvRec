"""The pre-specification measurement the v2 subspace proposals need (issue #112, P1).

The geometry review (F1) established that a Poincare-ball curvature is a pure rescaling, so a
tangent scale that is not tied to the data makes the curvature cosmetic. This script measures the
HOST's latent-norm distribution on the frozen stage-1 embeddings with a matched control
checkpoint, so the v2 proposals can declare a scale rule with real numbers in it:

    s = atanh(rho) / (sqrt(c) * ||z||_p95)        =>  the p95 latent lands at normalised
                                                      radius sqrt(c)||x|| = rho

and can state the boundary margin the largest latent would have.

Run under the project interpreter:

    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        results/hyperbolic_research_runs/subspace_111/measure_latent_scale.py

Writes: .../latent_scale.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path("/fs04/ar57/wenyu/GeneRec")
sys.path.insert(0, str(REPO_ROOT / "tools" / "cgv"))

import engine  # noqa: E402  (loads the arm's trainer in isolation)

TREE = REPO_ROOT / "stage2_RQ-VAE/TIGER_r10_hyp_busemann_horosphere"
CONTROL = REPO_ROOT / "results/stage2_RQ-VAE/r8_control/rqvae_best.pth"
OUT = Path(__file__).resolve().parent / "latent_scale.json"
RHO = 0.6          # declared: the p95 latent sits at 60% of the ball radius
CURVATURES = (1.0, 4.0)


def main() -> int:
    trainer, saved_path, saved_mods = engine.load_trainer(TREE, module_name="scale_probe")
    try:
        embeddings = trainer.maybe_apply_pca(
            trainer.load_embeddings(trainer.EMBEDDING_FILE), trainer.PCA_DIM, trainer.SEED)
        in_dim = int(embeddings.shape[1])
        model = trainer.RQVAE(trainer._tokenizer_config(), in_dim=in_dim)
        state = torch.load(CONTROL, map_location="cpu")
        state_dict = state.get("state_dict") or state
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        model.eval()

        sample = torch.from_numpy(np.asarray(embeddings[:20000], dtype="float32"))
        with torch.no_grad():
            latent = model.encoder(sample)
        norms = latent.norm(dim=-1)
        quantiles = {f"p{int(q * 100)}": float(torch.quantile(norms, q))
                     for q in (0.5, 0.9, 0.95, 0.99)}
        maximum = float(norms.max())

        # the two halves of the (whitened) PCA space have the same scale by construction
        half = latent.shape[1] // 2
        halves = {"first": latent[:, :half].norm(dim=-1), "second": latent[:, half:].norm(dim=-1)}

        rules = {}
        for c in CURVATURES:
            scale = math.atanh(RHO) / (math.sqrt(c) * quantiles["p95"])
            normalised = {name: math.tanh(math.sqrt(c) * scale * value) ** 2
                          for name, value in (("p50", quantiles["p50"]), ("p95", quantiles["p95"]),
                                              ("p99", quantiles["p99"]), ("max", maximum))}
            rules[f"c{c:g}"] = {
                "tangent_scale": scale,
                "normalised_radius_squared": normalised,
                "boundary_margin_at_max": 1.0 - normalised["max"],
            }

        payload = {
            "schema_version": 1,
            "issue": 112,
            "measured_utc": "2026-10-23T07:30:00Z",
            "purpose": ("the pre-specification latent measurement for the v2 subspace proposals: "
                        "the tangent scale must be tied to the data (geometry review F1)"),
            "tree": str(TREE),
            "checkpoint": str(CONTROL),
            "checkpoint_epoch": state.get("epoch"),
            "checkpoint_selection": state.get("selection"),
            "embeddings": {"file": str(trainer.EMBEDDING_FILE),
                           "rows_used": int(sample.shape[0]),
                           "in_dim_after_pca": in_dim,
                           "pca_dim": trainer.PCA_DIM, "pca_seed": trainer.SEED},
            "load_state_dict": {"missing": list(missing), "unexpected": list(unexpected)},
            "latent_norm": {"n": int(norms.numel()), "mean": float(norms.mean()),
                            "std": float(norms.std()), **quantiles, "max": maximum},
            "halves": {name: {"p50": float(torch.quantile(value, 0.5)),
                              "p95": float(torch.quantile(value, 0.95))}
                       for name, value in halves.items()},
            "scale_rule": "s = atanh(rho) / (sqrt(c) * ||z||_p95); rho declared at specification time",
            "rho": RHO,
            "rules": rules,
            "interpretation": (
                "With ONE common tangent scale and c_b = 4 * c_s the two factors' normalised "
                "radii differ by construction (0.60 vs tanh(2*atanh(0.60))^2 = 0.766): that "
                "relative difference is the only curvature content the Poincare ball can carry "
                "(F1), and it is what the v2 declarations must use instead of a curvature sweep."
            ),
        }
        OUT.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload["latent_norm"], indent=1))
        print(json.dumps(payload["rules"], indent=1))
        print("wrote", OUT)
        return 0
    finally:
        engine.unload_trainer(saved_path, saved_mods)


if __name__ == "__main__":
    raise SystemExit(main())
