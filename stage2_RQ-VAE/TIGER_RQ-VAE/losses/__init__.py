"""The loss registry: the trainer builds from here and adds nothing of its own.

Every term is a module beside this one with its own `NAME`, `DESCRIPTION` and
function. The training loop picks the terms it wants by name and calls the built
object; it never computes an MSE, scales a beta or sums two losses itself.

The two conjugate terms are named after the side their gradient reaches, which is
the VQ-VAE convention: `commitment` moves the encoder (`MSE(x, sg[q])`) and
`codebook` moves the entry (`MSE(q, sg[x])`). They were swapped in this baseline
until 2026-11-12 -- the formulas were the right ones, the labels were not -- and the
shared suite now fails if a name and its gradient side disagree.

    from losses import build_loss
    criterion = build_loss(CONFIG)
    total, components = criterion(reconstructed, target, quant_parts)
"""

from __future__ import annotations

from .codebook import codebook_loss
from . import codebook as _codebook
from . import commitment as _commitment
from . import quantization as _quantization
from . import reconstruction as _reconstruction
from .commitment import commitment_loss
from .quantization import quantization_loss, quantization_total
from .reconstruction import reconstruction_loss

__all__ = [
    "LOSSES", "LossError", "TotalLoss", "build_loss",
    "reconstruction_loss", "commitment_loss", "codebook_loss",
    "quantization_loss", "quantization_total",
]

#: name -> what the term is, and the callable that computes it.
LOSSES = {
    "reconstruction": {
        "description": "MSE(reconstruction, embedding): keeps the SID decodable",
        "function": reconstruction_loss,
        "spec": _reconstruction.SPEC,
    },
    "commitment": {
        "description": "MSE(latent, sg[quantized]): the gradient reaches the ENCODER only",
        "function": commitment_loss,
        "spec": _commitment.SPEC,
    },
    "codebook": {
        "description": "MSE(quantized, sg[latent]): the gradient reaches the CODEBOOK only",
        "function": codebook_loss,
        "spec": _codebook.SPEC,
    },
    "quantization": {
        "description": "codebook + beta * commitment, averaged over the levels",
        "function": quantization_total,
        "spec": _quantization.SPEC,
    },
}

#: The pair the baseline trains with, in the order it adds them.
DEFAULT_TERMS = ("reconstruction", "quantization")


class LossError(RuntimeError):
    """The requested loss cannot be built; training must not start."""


class TotalLoss:
    """The sum of the enabled terms, with each term reported separately."""

    def __init__(self, beta: float, terms=DEFAULT_TERMS) -> None:
        self.beta = float(beta)
        self.terms = tuple(terms)

    def __call__(self, reconstructed, target, quant_parts):
        components = {}
        for name in self.terms:
            if name == "reconstruction":
                components[name] = reconstruction_loss(reconstructed, target)
            elif name == "quantization":
                components[name] = quantization_total(quant_parts, self.beta)
            elif name == "commitment" or name == "codebook":
                raise LossError("%r is computed inside 'quantization'; enable that instead"
                                % name)
            else:
                raise LossError("unknown loss %r" % (name,))
        total = None
        for value in components.values():
            total = value if total is None else total + value
        if total is None:
            raise LossError("no losses were enabled")
        return total, components


def build_loss(config) -> TotalLoss:
    """Build the criterion from a config object or a plain dict.

    Both spellings are accepted: the trainer's own config uses the upper-case names
    it exposes as constants (`BETA`), while the tokeniser config handed to the model
    uses `beta`.
    """
    def read(name, default=None):
        for candidate in (name, name.lower(), name.upper()):
            if isinstance(config, dict):
                if candidate in config:
                    return config[candidate]
            elif hasattr(config, candidate):
                return getattr(config, candidate)
        return default

    beta = read("beta")
    if beta is None:
        raise LossError("the config does not name a beta for the codebook term")
    terms = read("losses") or DEFAULT_TERMS
    if isinstance(terms, str):
        terms = (terms,)
    unknown = [name for name in terms if name not in LOSSES]
    if unknown:
        raise LossError("unknown loss %s; the registry holds %s"
                        % (", ".join(map(repr, unknown)), ", ".join(sorted(LOSSES))))
    # commitment and codebook are pieces of the quantization term, not terms of their
    # own: accepting them here would build a criterion that fails only when called.
    nested = [name for name in terms if name in ("commitment", "codebook")]
    if nested:
        raise LossError("%s is computed inside 'quantization'; enable that instead"
                        % ", ".join(map(repr, nested)))
    return TotalLoss(float(beta), tuple(terms))
