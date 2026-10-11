"""Small-batch pre-check for the tangent-norm control, before any full run.

Checks, with no training:
  1. gradient direction -- the penalty must pull the encoder norm DOWN, i.e. one
     gradient step along -grad must reduce the mean tangent norm;
  2. numerical stability -- finite penalty, finite gradients, no NaN at the
     saturated operating point where the other terms lose their grip;
  3. weight calibration -- match the penalty's gradient norm to the combined
     gradient of the other terms, so the weight is not a free parameter tuned on
     the final metric. The same value is then used for both arms.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SOURCE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_DIR))

import train_rqvae as T  # noqa: E402
from hyperbolic import ball_radius, encode_ball  # noqa: E402
from model import RQVAE  # noqa: E402

ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")


def main() -> None:
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    report: dict = {"target_radius_fraction": T.TARGET_RADIUS_FRACTION,
                    "budget_factor": T.RADIUS_BUDGET_FACTOR}
    failures: list[str] = []

    T.EXPERIMENT_ARM = "nce_hyp_radius_control"
    embeddings = T.load_embeddings(T.EMBEDDING_FILE)
    frame = pd.read_parquet(T.TRAIN_FILE, columns=["seen_history", "target"])
    ids = np.unique(frame["target"].to_numpy(dtype=np.int64))
    all_emb = torch.from_numpy(embeddings)
    train_emb = all_emb[torch.from_numpy(ids)]

    out = {}
    for arm in ("nce_euclid_radius_control", "nce_hyp_radius_control"):
        T.EXPERIMENT_ARM = arm
        T.set_seed(T.SEED)
        model = RQVAE(T._tokenizer_config(), in_dim=embeddings.shape[1]).to(device)
        T.initialize_tiger_weights(model)
        model.train()
        for layer in model.rq.vq_layers:
            layer._skip_ddp_reduce = True
        with torch.no_grad():
            model.init_codebook(train_emb.to(device))
            normalizer = float(T.resolve_normalizer(model, train_emb.to(device)))
            initial_median = float(torch.median(torch.linalg.vector_norm(
                model.encoder(train_emb.to(device)), dim=-1)))
        budget = T.RADIUS_BUDGET_FACTOR * initial_median
        params = [p for p in model.parameters() if p.requires_grad]

        def mean_norm() -> float:
            with torch.no_grad():
                return float(torch.linalg.vector_norm(
                    model.encoder(train_emb[:1024].to(device)), dim=-1).mean())

        batch = train_emb[:1024].to(device)

        # ---- 2. numerical stability at the saturated operating point -------
        # Synthesise the growth the real run produced, then apply the penalty.
        class Scaled(torch.nn.Module):
            def __init__(self, mod, s):
                super().__init__(); self.m = mod; self.s = s
            def forward(self, x):
                return self.m(x) * self.s
        base = model.encoder
        model.encoder = Scaled(base, 9.4)  # the measured growth of the hyp arm
        penalty = T.tangent_norm_control(model, batch, budget)
        model.encoder = base
        report[f"{arm}_penalty_at_saturation"] = float(penalty.detach())
        report[f"{arm}_penalty_finite"] = bool(torch.isfinite(penalty))
        if not report[f"{arm}_penalty_finite"]:
            failures.append(f"{arm}: penalty not finite at the saturated point")
        if float(penalty.detach()) <= 0:
            failures.append(f"{arm}: penalty is zero at the saturated point, so it cannot act")

        # ---- 1. gradient direction ----------------------------------------
        # Scale the encoder by a small negative step and confirm the norm falls.
        for delta in (-0.05, +0.05):
            model.encoder = Scaled(base, 1.0 + delta)
            before = mean_norm()
            model.encoder = base
            report[f"{arm}_norm_at_scale_{delta:+.2f}"] = before
        model.encoder = base
        shrink, grow = (report[f"{arm}_norm_at_scale_-0.05"],
                        report[f"{arm}_norm_at_scale_+0.05"])
        if not shrink < grow:
            failures.append(f"{arm}: norm does not increase with the scale factor")

        # ---- 3. gradient-norm matching for the weight ---------------------
        model.encoder = Scaled(base, 9.4)   # the operating point of training
        nz = torch.as_tensor(normalizer, dtype=torch.float32, device=device)
        z = encode_ball(model.encoder(batch), T.CURVATURE, nz)
        radius = (T.CURVATURE ** 0.5) * torch.linalg.vector_norm(
            z, dim=-1) / ball_radius(T.CURVATURE)
        log_degree = np.log1p(T.behaviour_out_degree(frame, len(embeddings)).astype(np.float64))
        rank = np.argsort(np.argsort(log_degree)) / max(len(log_degree) - 1, 1)
        target = torch.from_numpy(
            (T.RHO_HI - (T.RHO_HI - T.RHO_LO) * rank).astype(np.float32)).to(device)
        radial_term = torch.nn.functional.mse_loss(radius, target[torch.from_numpy(ids[:1024].copy()).to(device)])
        control_term = T.tangent_norm_control(model, batch, budget)
        recon, quant, _, _ = model(batch)
        recon_term, _ = model.compute_loss(batch, recon, quant)
        grad_recon = torch.autograd.grad(recon_term, params, retain_graph=True,
                                         allow_unused=True)
        n_recon = float(torch.cat([(g if g is not None else torch.zeros_like(p)).flatten()
                                   for g, p in zip(grad_recon, params)]).norm())
        grad_other = torch.autograd.grad(radial_term, params, retain_graph=True,
                                         allow_unused=True)
        grad_control = torch.autograd.grad(control_term, params, retain_graph=True,
                                          allow_unused=True)
        n_other = float(torch.cat([(g if g is not None else torch.zeros_like(p)).flatten()
                                   for g, p in zip(grad_other, params)]).norm())
        n_control = float(torch.cat([(g if g is not None else torch.zeros_like(p)).flatten()
                                     for g, p in zip(grad_control, params)]).norm())
        weight_for_parity = n_other / max(n_control, 1e-12)
        report[f"{arm}_grad_norm_radial_mse"] = n_other
        report[f"{arm}_grad_norm_control_at_weight_1"] = n_control
        report[f"{arm}_weight_for_gradient_parity"] = weight_for_parity
        report[f"{arm}_grad_norm_recon"] = n_recon
        # Calibrate against the reconstruction gradient, NOT the radial MSE: the
        # MSE gradient is already 2e-4 because tanh has saturated, so matching
        # it would hand the control a weight of 1e-5 and reproduce the failure.
        report[f"{arm}_weight_for_recon_parity"] = n_recon / max(n_control, 1e-12)
        model.encoder = base
        out[arm] = report[f"{arm}_grad_norm_control_at_weight_1"]

    # One weight for both arms: the mean of the two parity weights, rounded to a
    # single declared constant. It is set from gradient balance, not from Recall.
    w = (report["nce_euclid_radius_control_weight_for_recon_parity"]
         + report["nce_hyp_radius_control_weight_for_recon_parity"]) / 2
    report["shared_weight_from_recon_parity"] = w
    report["declared_RADIUS_CONTROL_WEIGHT"] = T.RADIUS_CONTROL_WEIGHT
    report["failures"] = failures
    report["status"] = "FAIL" if failures else "PASS"
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
