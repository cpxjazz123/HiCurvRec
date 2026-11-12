"""The code/vector distance is its own module; the trainer only passes it in.

`VQLayer._distances` used to hold the Euclidean squared distance inline:

    latent.pow(2).sum(1, keepdim=True) - 2 * latent @ code_embs.t()
        + code_embs.pow(2).sum(1, keepdim=True).t()

so the choice of codeword was decided by a formula living inside the layer that also
owned the codebook, the Sinkhorn assignment and the straight-through trick. It now
lives in `distances/euclidean.py` behind a registry, and the training entry point
imports the built object and hands it to the model -- it names no formula itself.

These tests pin three things: the module exists, the arithmetic is unchanged, and the
injected distance really decides which codeword is chosen.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
BASELINE = REPO / "stage2_RQ-VAE" / "TIGER_RQ-VAE"
sys.path.insert(0, str(BASELINE))

import torch  # noqa: E402

import distances  # noqa: E402
from model.layers import VQLayer  # noqa: E402


def test_the_distance_has_its_own_module() -> str:
    if not (BASELINE / "distances" / "euclidean.py").is_file():
        raise AssertionError("distances/euclidean.py is missing")
    if "euclidean" not in distances.DISTANCES:
        raise AssertionError("the registry does not name euclidean: %r"
                             % sorted(distances.DISTANCES))
    if not distances.DISTANCES["euclidean"].get("description"):
        raise AssertionError("the entry has no definition to read")
    built = distances.build_distance()
    if not callable(built):
        raise AssertionError("build_distance did not return a callable")
    return "the Euclidean distance is one module with a definition and a registry entry"


def test_the_arithmetic_is_unchanged() -> str:
    torch.manual_seed(11)
    latent = torch.randn(32, 8)
    codebook = torch.randn(64, 8)
    expected = (latent.pow(2).sum(1, keepdim=True)
                - 2 * latent @ codebook.t()
                + codebook.pow(2).sum(1, keepdim=True).t())
    got = distances.euclidean.distances(latent, codebook)
    if got.shape != expected.shape or not torch.equal(got, expected):
        raise AssertionError("the distance matrix changed:\n%r\n%r" % (got[:2], expected[:2]))
    return "the squared-distance matrix is bit-identical to the inline expression"


def test_the_injected_distance_decides_the_codeword() -> str:
    torch.manual_seed(13)
    x = torch.randn(16, 8)
    default = VQLayer(64, 8, 0.25, -1.0, -1, distance=distances.build_distance())
    flipped = VQLayer(64, 8, 0.25, -1.0, -1, distance=lambda latent, codebook:
                      -distances.euclidean.distances(latent, codebook))
    flipped.load_state_dict(default.state_dict())
    for layer in (default, flipped):
        layer.eval()
    with torch.no_grad():
        _, _, _, tokens_default = default(x)
        _, _, _, tokens_flipped = flipped(x)
        manual = torch.argmin(distances.euclidean.distances(
            x.view(-1, 8), default.get_code_embs()), dim=-1).view(x.shape[:-1])
    if not torch.equal(tokens_default, manual):
        raise AssertionError("the default layer no longer picks the nearest codeword")
    if torch.equal(tokens_flipped, tokens_default):
        raise AssertionError("the layer ignored the distance it was handed")
    return "the layer follows whatever distance object it is given"


def test_the_trainer_names_no_distance_formula() -> str:
    text = (BASELINE / "train_rqvae.py").read_text(encoding="utf-8")
    offenders = []
    for pattern, label in ((r"\.pow\(2\)\.sum", "an inline distance"), (r"@\s*\w+\.t\(\)", "a matmul"),
                           (r"mse_loss", "an inline MSE")):
        if re.search(pattern, text):
            offenders.append(label)
    if offenders:
        raise AssertionError("the trainer still holds a distance formula: " + ", ".join(offenders))
    if "build_distance" not in text:
        raise AssertionError("the trainer never imports the distance registry")
    return "the trainer imports the built distance and computes none of it"


def test_the_layer_holds_no_distance_formula() -> str:
    text = (BASELINE / "model" / "layers.py").read_text(encoding="utf-8")
    if re.search(r"\.pow\(2\)\.sum", text):
        raise AssertionError("layers.py still computes the distance inline")
    if "self.distance(" not in text:
        raise AssertionError("layers.py never calls the injected distance")
    return "the quantiser calls the injected distance and holds no formula"


def main() -> int:
    failures = 0
    for test in (test_the_distance_has_its_own_module,
                 test_the_arithmetic_is_unchanged,
                 test_the_injected_distance_decides_the_codeword,
                 test_the_trainer_names_no_distance_formula,
                 test_the_layer_holds_no_distance_formula):
        try:
            print("PASS %s: %s" % (test.__name__, test()), flush=True)
        except AssertionError as exc:
            failures += 1
            print("FAIL %s: %s" % (test.__name__, exc), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
