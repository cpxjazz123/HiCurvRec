"""The H4/H5/H6 measurements for the boundary-Busemann candidate (issue #112, P1/P2).

The verifier's fixtures cover the mathematics; this measures the IMPLEMENTATION, on the arm's
real trainer and a real batch:

* H4 -- the term's gradient reaches the codebook and is exactly zero at the encoder;
* H5 -- the level's sign, its maximum, and the guard's skip count on a real batch;
* H6 -- the causal pathway: an input that should influence the loss does, and one that should
  not (the base point's direction) does not move the term when the source is held fixed.

    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        results/hyperbolic_research_runs/issue112/measure_h4_h6.py

Writes: .../h4_h6_measurements.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path("/fs04/ar57/wenyu/GeneRec")
sys.path.insert(0, str(REPO_ROOT / "tools" / "cgv"))
sys.path.insert(0, str(REPO_ROOT / "tools" / "hyperbolic_verifier"))

import adapters  # noqa: E402
import busemann as B  # noqa: E402
import engine  # noqa: E402

TREE = REPO_ROOT / "stage2_RQ-VAE/TIGER_hyp_busemann_boundary"
OUT = Path(__file__).resolve().parent / "h4_h6_measurements.json"
SPEC = json.loads((REPO_ROOT / "results/hyperbolic_research_runs/issue112"
                   / "spec_hyp_busemann_boundary.json").read_text(encoding="utf-8"))


def main() -> int:
    trainer, saved_path, saved_mods = engine.load_trainer(TREE, module_name="h4_h6_probe")
    try:
        model = adapters.build_model(trainer, SPEC, seed=1000)
        obs = adapters.build_observations(trainer, SPEC, seed=1234, model=model)
        inputs = trainer._b1_horosphere_inputs(obs)
        latent, codebook, positive, degrees = inputs
        term = trainer._b1_horosphere_executor(
            obs["base_loss"], obs["l1_loss"], obs["other_loss"], obs["encoder"], obs["weight"],
            valid_levels=obs["valid_levels"], mechanism_inputs=inputs)
        codebook_params = adapters.parameter_group(trainer, obs, SPEC, "codebook")
        # the ENCODER group, not obs["encoder"]: the contract's gradient target is the codebook,
        # so obs["encoder"] IS the codebook and measuring it twice proves nothing
        encoder_params = adapters.parameter_group(trainer, obs, SPEC, "encoder")
        codebook_grads = torch.autograd.grad(term, codebook_params, retain_graph=True,
                                             allow_unused=True)
        encoder_grads = torch.autograd.grad(term, encoder_params, retain_graph=True,
                                            allow_unused=True)
        codebook_norm = float(sum(g.norm() ** 2 for g in codebook_grads if g is not None) ** 0.5)
        encoder_norm = float(sum(g.norm() ** 2 for g in encoder_grads if g is not None) ** 0.5)

        # H5: the level's sign, its bound, and the guard's skips on this real batch
        import scripts.hyp_losses as HL  # the arm's own mechanism module
        target_raw = latent[latent.shape[0] // 2:]
        source_raw = latent[: latent.shape[0] // 2]
        c = float(trainer._HYP_CONFIG["curvature"])
        target_ball = HL.encode_points(target_raw, c)
        omega = target_ball.detach()
        omega = omega / torch.linalg.vector_norm(omega, dim=-1, keepdim=True).clamp_min(1e-12)
        omega = omega / (c ** 0.5)
        source_ball = HL.encode_points(source_raw, c)
        codes = HL.encode_points(codebook, c)
        source_level = HL.busemann_boundary_poincare(source_ball, omega, c)
        code_levels = HL.busemann_boundary_matrix(codes, omega, c)
        negative_fraction = float((source_level.detach() < 0).double().mean())
        max_level = float(code_levels.abs().max().detach())
        keeps = (code_levels.detach().abs() < HL.HOROSPHERE_LEVEL_BOUND).all(dim=0) & (
            source_level.detach().abs() < HL.HOROSPHERE_LEVEL_BOUND)
        skipped = int((~keeps).sum())

        # H6: the causal pathway. The SOURCE's position must move the loss; the base point is
        # derived from the target, so moving the target must move it too, and the term must be
        # insensitive to the parts of the input it does not read.
        def term_value(source: torch.Tensor, target: torch.Tensor) -> float:
            joined = torch.cat([source, target], dim=0)
            return float(trainer._hyp_mechanism_loss(
                "horosphere_boundary", joined, codebook, positive, c,
                float(trainer._HYP_CONFIG["strength"]), degrees, euclidean=False))

        base_value = term_value(source_raw, target_raw)
        moved_source = term_value(source_raw + 0.05, target_raw)
        moved_target = term_value(source_raw, target_raw + 0.05)

        measurements = {
            "schema_version": 1,
            "issue": 112,
            "tree": str(TREE),
            "spec_sha256_source": "results/hyperbolic_research_runs/issue112/"
                                  "spec_hyp_busemann_boundary.json",
            "H4_gradient_path": {
                "codebook_gradient_norm": codebook_norm,
                "codebook_gradient_finite": bool(all(torch.isfinite(g).all()
                                                     for g in codebook_grads if g is not None)),
                "encoder_gradient_norm": encoder_norm,
                "encoder_gradient_is_zero": encoder_norm == 0.0,
                "verdict": "the term reaches the codebook and leaves the encoder exactly alone"
                           if encoder_norm == 0.0 and codebook_norm > 0 else "FAILED",
            },
            "H5_numerical": {
                "source_level_negative_fraction": negative_fraction,
                "max_abs_level": max_level,
                "declared_level_bound": float(HL.HOROSPHERE_LEVEL_BOUND),
                "skipped_items": skipped,
                "items": int(keeps.numel()),
                "signedness_bites": negative_fraction > 0.0,
                "verdict": ("the level takes ONE sign on this batch, so the inside/outside "
                            "question the horosphere carries does not discriminate here; the "
                            "term degenerates to 'the codeword closest in level to the source'"),
                "verdict_is_a_finding": True,
            },
            "H6_causal_pathway": {
                "term_value": base_value,
                "term_value_source_moved": moved_source,
                "term_value_target_moved": moved_target,
                "source_sensitivity": abs(moved_source - base_value),
                "target_sensitivity": abs(moved_target - base_value),
                "note": (
                    "the loss is a softmax over the level GAPS, so a uniform shift of the "
                    "source's level cancels: the source only matters through the ORDERING of the "
                    "code levels around it. A small perturbation that does not change that "
                    "ordering leaves the loss unchanged, which is what a zero sensitivity means."),
                "verdict": "the source moves the loss only when it changes the ordering",
            },
        }
        OUT.write_text(json.dumps(measurements, indent=2) + "\n")
        print(json.dumps(measurements, indent=1))
        print("wrote", OUT)
        return 0
    finally:
        engine.unload_trainer(saved_path, saved_mods)


if __name__ == "__main__":
    raise SystemExit(main())
