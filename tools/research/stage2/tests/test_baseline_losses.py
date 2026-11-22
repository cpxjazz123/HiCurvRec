"""Each loss is its own module, and the trainer only decides where it is applied.

The baseline used to compute both terms inside the model: `VQLayer.forward` built
`commitment + beta * codebook` and `RQVAE.compute_loss` added the reconstruction MSE.
The training loop therefore held the arithmetic of a loss it never named, and any
mechanism that wanted a different term had to edit the quantiser or the model.

Now `losses/` holds one module per term:
    losses/reconstruction.py   the MSE between the reconstruction and its input
    losses/commitment.py       ||q - sg[x]||^2
    losses/codebook.py         ||sg[q] - x||^2
    losses/quantization.py     commitment + beta * codebook (EMA: beta * codebook)
    losses/__init__.py         the registry the trainer builds
The trainer calls the registry and adds nothing of its own.

These tests pin the split -- shape, arithmetic, and the trainer's neutrality -- and
the seeded snapshot run proves the numbers did not move.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
BASELINE = REPO / "stage2_RQ-VAE" / "TIGER_RQ-VAE"
sys.path.insert(0, str(BASELINE))

import torch  # noqa: E402

import losses  # noqa: E402


def test_each_term_has_its_own_module() -> str:
    expected = {"reconstruction": "reconstruction.py",
                "commitment": "commitment.py",
                "codebook": "codebook.py",
                "quantization": "quantization.py"}
    missing = []
    for name, filename in expected.items():
        if not (BASELINE / "losses" / filename).is_file():
            missing.append(filename)
        if name not in losses.LOSSES:
            missing.append(name + " (not registered)")
    if missing:
        raise AssertionError("the loss split is incomplete: " + ", ".join(missing))
    for name, entry in losses.LOSSES.items():
        if not entry.get("description"):
            raise AssertionError("loss %r has no definition to read" % name)
    return "four terms, one module each, each with its own definition"


def test_quantization_is_codebook_plus_beta_commitment() -> str:
    torch.manual_seed(3)
    latent = torch.randn(16, 8)
    quantized = torch.randn(16, 8)
    beta = 0.25
    # The baseline's own arithmetic, written out: the entry term, plus beta times the
    # encoder term. The names say which side each MSE moves.
    expected = (torch.nn.functional.mse_loss(quantized, latent.detach())
                + beta * torch.nn.functional.mse_loss(latent, quantized.detach()))
    got = losses.quantization.quantization_loss(latent, quantized, beta)
    if not torch.allclose(got, expected, rtol=0, atol=0):
        raise AssertionError("the quantisation term changed: %r vs %r" % (got, expected))
    ema = losses.quantization.quantization_loss(latent, quantized, beta, codebook_term=False)
    want = beta * torch.nn.functional.mse_loss(latent, quantized.detach())
    if not torch.allclose(ema, want, rtol=0, atol=0):
        raise AssertionError("the EMA variant changed: %r vs %r" % (ema, want))
    return "the quantisation arithmetic is the one the baseline trained with"


def test_the_total_is_the_sum_of_its_parts() -> str:
    torch.manual_seed(5)
    target = torch.randn(12, 6)
    reconstructed = torch.randn(12, 6)
    parts = [(torch.tensor(0.4), torch.tensor(0.8)), (torch.tensor(0.2), torch.tensor(0.1))]
    criterion = losses.build_loss({"beta": 0.25, "codebook_num": 2})
    total, components = criterion(reconstructed, target, parts)
    # parts are (commitment, codebook): the entry term plus beta times the encoder term.
    expected_quant = ((0.8 + 0.25 * 0.4) + (0.1 + 0.25 * 0.2)) / 2
    expected_recon = torch.nn.functional.mse_loss(reconstructed, target)
    if set(components) != {"reconstruction", "quantization"}:
        raise AssertionError("the components are not named: %r" % (sorted(components),))
    if abs(float(components["quantization"]) - float(expected_quant)) > 1e-6:
        raise AssertionError("the quantisation total is wrong: %r vs %r"
                             % (float(components["quantization"]), float(expected_quant)))
    if abs(float(components["reconstruction"]) - float(expected_recon)) > 1e-6:
        raise AssertionError("the reconstruction total is wrong")
    if abs(float(total) - float(expected_recon + expected_quant)) > 1e-6:
        raise AssertionError("the total is not the sum of the parts")
    return "the total is reconstruction + quantization, and each part is reported"


def test_an_unknown_loss_name_is_refused() -> str:
    try:
        losses.build_loss({"beta": 0.25, "losses": ["reconstruction", "smoothness"]})
    except losses.LossError:
        return "a name the registry does not know is refused"
    raise AssertionError("an unknown loss name was accepted")


def test_the_trainer_does_no_loss_arithmetic() -> str:
    text = (BASELINE / "train_rqvae.py").read_text(encoding="utf-8")
    offenders = []
    for pattern, label in ((r"mse_loss", "an MSE call"), (r"\bbeta\s*\*", "a beta scaling"),
                           (r"quant_loss\s*\+", "a hand-built sum")):
        if re.search(pattern, text):
            offenders.append(label)
    if offenders:
        raise AssertionError("the trainer still does loss arithmetic: " + ", ".join(offenders))
    if "build_loss" not in text:
        raise AssertionError("the trainer never builds the loss from the registry")
    return "the trainer builds the loss and adds nothing of its own"


def test_both_config_spellings_are_accepted() -> str:
    from types import SimpleNamespace
    upper = SimpleNamespace(BETA=0.25, LOSSES=("reconstruction", "quantization"))
    lower = SimpleNamespace(beta=0.25, losses=("reconstruction", "quantization"))
    for label, config in (("upper", upper), ("lower", lower)):
        criterion = losses.build_loss(config)
        if criterion.beta != 0.25 or criterion.terms != ("reconstruction", "quantization"):
            raise AssertionError("%s-case config was not read: %r" % (label, criterion.__dict__))
    return "the trainer's BETA and the tokeniser's beta both build the same criterion"


def _tiny_model():
    from model import RQVAE
    import train_rqvae as trainer
    torch.manual_seed(31)
    model = RQVAE(trainer._tokenizer_config(), in_dim=8)
    generator = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for layer in model.rq.vq_layers:
            weight = layer.get_code_embs()
            weight.copy_(torch.randn(weight.shape, generator=generator))
    model.train()
    return model


def test_compute_loss_accepts_the_parts_the_forward_returns() -> str:
    """The framework's checkpoint probe passes forward's own output back into compute_loss."""
    model = _tiny_model()
    batch = torch.randn(4, 8, generator=torch.Generator().manual_seed(7))
    reconstructed, quant_parts, _, _ = model(batch)
    total, recon = model.compute_loss(batch, reconstructed, quant_parts)
    want_total, components = losses.build_loss(model.config)(reconstructed, batch, quant_parts)
    if not torch.equal(total, want_total):
        raise AssertionError("compute_loss does not score the parts with the registry")
    if not torch.equal(recon, components["reconstruction"]):
        raise AssertionError("compute_loss returned a different reconstruction term")
    return "compute_loss scores the parts list the forward hands out"


def test_compute_loss_still_takes_a_precomputed_scalar() -> str:
    """Trees outside the trainer (older probes) may pass the scalar they computed."""
    model = _tiny_model()
    batch = torch.randn(4, 8, generator=torch.Generator().manual_seed(9))
    reconstructed, _, _, _ = model(batch)
    scalar = torch.tensor(0.5)
    total, recon = model.compute_loss(batch, reconstructed, scalar)
    if not torch.equal(total, recon + scalar):
        raise AssertionError("a precomputed scalar is no longer accepted")
    return "a precomputed quantisation scalar still adds to the reconstruction term"


def main() -> int:
    failures = 0
    for test in (test_each_term_has_its_own_module,
                 test_quantization_is_codebook_plus_beta_commitment,
                 test_the_total_is_the_sum_of_its_parts,
                 test_an_unknown_loss_name_is_refused,
                 test_the_trainer_does_no_loss_arithmetic,
                 test_both_config_spellings_are_accepted,
                 test_compute_loss_accepts_the_parts_the_forward_returns,
                 test_compute_loss_still_takes_a_precomputed_scalar):
        try:
            print("PASS %s: %s" % (test.__name__, test()), flush=True)
        except AssertionError as exc:
            failures += 1
            print("FAIL %s: %s" % (test.__name__, exc), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
