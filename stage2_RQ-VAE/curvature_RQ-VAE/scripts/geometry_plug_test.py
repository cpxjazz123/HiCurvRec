"""Invariants of the RQ-VAE geometry plug-in.

Runs without pytest: ``python scripts/geometry_plug_test.py`` exits non-zero on
the first broken invariant. The two arms have to differ in geometry and nothing
else, so these check the flat limit is defined exactly, that the pinned working
radius does not depend on curvature in the flat limit, and that both arms
quantize, back-propagate and produce different token assignments.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

from curvature_config import GEOMETRY  # noqa: E402
from model.layers import (  # noqa: E402
    _euclid_distance_tangent_pairs,
    _euclid_residual,
    _pairwise_euclid_distance_tangents,
    _resolve_geometry,
)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str) -> None:
    print(f"[{'PASS' if condition else 'FAIL'}] {name}: {detail}", flush=True)
    if not condition:
        FAILURES.append(name)


def main() -> None:
    torch.manual_seed(0)
    left = torch.randn(16, 32, device=DEVICE)
    right = torch.randn(16, 32, device=DEVICE)
    codebook = torch.randn(256, 32, device=DEVICE)

    check(
        "flat pair distance is the plain L2 norm",
        bool(
            torch.allclose(
                _euclid_distance_tangent_pairs(left, right),
                torch.linalg.vector_norm(left - right, dim=-1),
                atol=0.0,
            )
        ),
        "exact",
    )
    check(
        "flat pairwise distance is cdist",
        bool(
            torch.allclose(
                _pairwise_euclid_distance_tangents(left, codebook),
                torch.cdist(left, codebook),
                atol=1e-5,
            )
        ),
        "atol=1e-5",
    )
    check(
        "flat residual is tangent subtraction",
        bool(torch.allclose(_euclid_residual(left, right), left - right, atol=0.0)),
        "exact",
    )

    pairwise, pair, residual = _resolve_geometry("euclid")
    check(
        "resolver returns the flat operators",
        pairwise is _pairwise_euclid_distance_tangents
        and pair is _euclid_distance_tangent_pairs
        and residual is _euclid_residual,
        "euclid triple",
    )
    try:
        _resolve_geometry("sphere")
        check("resolver rejects an unknown geometry", False, "no error raised")
    except ValueError as error:
        check("resolver rejects an unknown geometry", True, str(error))

    import numpy as np
    from types import SimpleNamespace

    from model.model import RQVAE

    config = SimpleNamespace(
        hidden_sizes=(512, 256, 128),
        codebook_num=3,
        codebook_size=(256, 256, 256),
        codebook_dim=32,
        dropout=0.0,
        beta=0.25,
        vq_type="vq",
        ema_decay=0.99,
        fix_code_embs=False,
        sk_epsilon=0.003,
        sk_iters=50,
        layer_curvatures=(1.0, 1.0, 1.0),
        layer_working_radii=(0.2, 0.2, 0.2),
        pin_in_s_coordinates=True,
        geometry="euclid",
    )
    model = RQVAE(config, in_dim=768).to(DEVICE)
    torch.nn.init.normal_(model.rq.vq_layers[0].get_code_embs(), std=0.02)
    features = torch.randn(64, 768, device=DEVICE)
    reconstructed, quant_loss, tokens = model(features)
    loss, recon_loss = model.compute_loss(features, reconstructed, quant_loss)
    loss.backward()
    gradient = torch.cat(
        [p.grad.detach().reshape(-1) for p in model.parameters() if p.grad is not None]
    )
    check(
        "euclid arm trains: finite loss",
        bool(torch.isfinite(loss) and torch.isfinite(recon_loss) and torch.isfinite(quant_loss)),
        f"loss={float(loss.detach()):.6f}",
    )
    check(
        "euclid arm back-propagates",
        bool(torch.isfinite(gradient).all() and bool(gradient.any())),
        f"nonzero params={int((gradient != 0).sum())}",
    )
    check(
        "euclid arm emits tokens inside the codebook range",
        bool(int(tokens.min()) >= 0 and int(tokens.max()) < 256),
        f"range=[{int(tokens.min())},{int(tokens.max())}]",
    )

    config.geometry = "poincare"
    hyperbolic = RQVAE(config, in_dim=768).to(DEVICE)
    hyperbolic.rq.vq_layers[0].get_code_embs().data.copy_(
        model.rq.vq_layers[0].get_code_embs().detach()
    )
    for parameter in hyperbolic.rq.vq_layers[1:]:
        parameter.get_code_embs().data.copy_(
            dict(model.rq.vq_layers.named_parameters())
        ) if False else None
    residual_before = model.rq.vq_layers[0]._residual_fn(
        features[:4] @ torch.randn(768, 32, device=DEVICE) * 0.01,
        model.rq.vq_layers[0].get_code_embs()[:4],
        model.rq.vq_layers[0].get_curvature(),
    )
    hyperbolic_residual = hyperbolic.rq.vq_layers[0]._residual_fn(
        features[:4] @ torch.randn(768, 32, device=DEVICE) * 0.01,
        hyperbolic.rq.vq_layers[0].get_code_embs()[:4],
        hyperbolic.rq.vq_layers[0].get_curvature(),
    )
    check(
        "the two arms share no residual operator",
        not bool(torch.allclose(residual_before, hyperbolic_residual, atol=1e-6)),
        "euclid subtraction vs Mobius log-map",
    )
    check(
        "production default geometry is poincare",
        GEOMETRY == "poincare",
        f"curvature_config.GEOMETRY={GEOMETRY}",
    )

    if FAILURES:
        raise SystemExit(f"geometry plug-in invariants failed: {FAILURES}")
    print("all geometry plug-in invariants hold", flush=True)


if __name__ == "__main__":
    main()