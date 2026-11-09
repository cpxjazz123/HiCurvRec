"""RecBole3.0-compatible residual quantization used by HG-Rec."""

from .layers import MLP
from .model import RQVAE

__all__ = ["MLP", "RQVAE"]
