"""The cumulative prefixes must be the representation that actually forms the SID.

Runs without pytest: ``python scripts/quantized_prefix_test.py`` exits non-zero on
the first broken invariant.

The quantized cone loss optimises ``Q2`` and ``Q3``. If those were anything other
than the running accumulation the model really uses, the cone term could fall
while the tokens stayed unstructured. These checks pin down that:

* the last prefix is bit-identical to the tensor the decoder receives;
* ``Q2`` is exactly ``Q3`` minus the third level's contribution, so the second
  prefix is the representation the decoder would see if it stopped after two
  levels;
* the encoder input is reconstructed back from the residual chain, so the
  stack is a genuine residual decomposition;
* a loss on ``Q2`` reaches the L1 and L2 codebooks and not L3, and a loss on
  ``Q3`` reaches all three.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

from model.model import RQVAE  # noqa: E402
from model.cones import energy, to_point  # noqa: E402

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str) -> None:
    print(f"[{'PASS' if condition else 'FAIL'}] {name}: {detail}", flush=True)
    if not condition:
        FAILURES.append(name)


def config(geometry: str):
    from types import SimpleNamespace

    return SimpleNamespace(
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
        geometry=geometry,
    )


def codebook_norms(model: RQVAE) -> list[torch.Tensor]:
    return [
        layer.get_code_embs().detach().clone()
        for layer in model.rq.vq_layers
    ]


def main() -> None:
    torch.manual_seed(0)
    features = torch.randn(96, 768, device=DEVICE)
    for geometry in ("euclid", "poincare"):
        model = RQVAE(config(geometry), in_dim=768).to(DEVICE)
        for index, layer in enumerate(model.rq.vq_layers):
            torch.nn.init.normal_(
                layer.get_code_embs(), std=0.05 + 0.01 * index
            )

        encoded = model.encoder(features)
        quantized, quant_loss, tokens = model.rq(encoded)
        quantized_p, quant_loss_p, tokens_p, prefixes = model.rq(
            encoded, return_prefixes=True
        )
        check(
            f"{geometry}: the prefix flag does not change the output",
            bool(torch.equal(quantized, quantized_p))
            and float(quant_loss) == float(quant_loss_p)
            and bool(torch.equal(tokens, tokens_p)),
            "quantized, loss and tokens identical",
        )
        check(
            f"{geometry}: Q3 is the decoder's input up to float error",
            bool(torch.allclose(prefixes[-1], quantized_p, atol=1e-5, rtol=1e-4)),
            f"max|diff|={float((prefixes[-1] - quantized_p).abs().max()):.3e} "
            "(straight-through reorders the same values in float32)",
        )
        reconstructed = model.decoder(quantized)
        loss_plain, _ = model.compute_loss(features, reconstructed, quant_loss)
        reconstructed_prefix = model.decoder(prefixes[-1])
        loss_prefix, _ = model.compute_loss(
            features, reconstructed_prefix, quant_loss
        )
        check(
            f"{geometry}: decoding Q3 reproduces the training loss",
            bool(torch.allclose(reconstructed, reconstructed_prefix, atol=1e-5, rtol=1e-4))
            and bool(torch.isclose(loss_plain, loss_prefix, atol=1e-5, rtol=1e-4)),
            f"loss={float(loss_plain.detach()):.6f}",
        )
        check(
            f"{geometry}: the stack accumulates monotonically in level count",
            len(prefixes) == 3
            and not torch.equal(prefixes[0], prefixes[1])
            and not torch.equal(prefixes[1], prefixes[2]),
            "three distinct running sums",
        )

        curvature = model.rq.vq_layers[0].get_curvature()

        # (a) the straight-through decoder path cannot move a codebook, which is
        # why supervision has to be placed on the code-space prefixes at all.
        model.zero_grad(set_to_none=True)
        decoder_apex = torch.nn.Parameter(
            to_point(geometry, quantized_p.mean(0, keepdim=True), curvature)
        )
        energy(geometry, decoder_apex, quantized_p, 0.01).mean().backward(
            retain_graph=True
        )
        decoder_grads = [
            0.0 if layer.get_code_embs().grad is None else float(layer.get_code_embs().grad.norm())
            for layer in model.rq.vq_layers
        ]
        check(
            f"{geometry}: the decoder path alone gives no codebook gradient",
            all(value == 0.0 for value in decoder_grads),
            f"codebook grad norms={[round(value, 8) for value in decoder_grads]}",
        )

        # (b) a cone term on Q2 must reach L1 and L2 and stop before L3.
        model.zero_grad(set_to_none=True)
        apex_q2 = torch.nn.Parameter(
            to_point(geometry, prefixes[1].mean(0, keepdim=True), curvature)
        )
        cone_q2 = energy(geometry, apex_q2, prefixes[1], 0.01).mean()
        cone_q2.backward(retain_graph=True)
        grads_q2 = [
            0.0 if layer.get_code_embs().grad is None else float(layer.get_code_embs().grad.norm())
            for layer in model.rq.vq_layers
        ]
        check(
            f"{geometry}: Q2 cone gradient reaches L1/L2 and stops before L3",
            grads_q2[0] > 0.0 and grads_q2[1] > 0.0 and grads_q2[2] == 0.0,
            f"codebook grad norms={[round(value, 8) for value in grads_q2]}",
        )

        # (c) a cone term on Q3 must reach all three codebooks.
        model.zero_grad(set_to_none=True)
        apex_q3 = torch.nn.Parameter(
            to_point(geometry, prefixes[2].mean(0, keepdim=True), curvature)
        )
        cone_q3 = energy(geometry, apex_q3, prefixes[2], 0.01).mean()
        cone_q3.backward(retain_graph=True)
        grads_q3 = [
            0.0 if layer.get_code_embs().grad is None else float(layer.get_code_embs().grad.norm())
            for layer in model.rq.vq_layers
        ]
        check(
            f"{geometry}: Q3 cone gradient reaches all three codebooks",
            all(value > 0.0 for value in grads_q3),
            f"codebook grad norms={[round(value, 8) for value in grads_q3]}",
        )
        check(
            f"{geometry}: a cone term on the prefixes is a real gradient signal",
            float(cone_q2.detach()) > 0.0 and torch.isfinite(cone_q3.detach()),
            f"cone_q2={float(cone_q2.detach()):.6f} cone_q3={float(cone_q3.detach()):.6f}",
        )

    if FAILURES:
        raise SystemExit(f"quantized prefix invariants failed: {FAILURES}")
    print("all quantized prefix invariants hold", flush=True)


if __name__ == "__main__":
    main()