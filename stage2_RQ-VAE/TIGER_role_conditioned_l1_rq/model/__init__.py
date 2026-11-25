"""RecBole3.0-compatible residual quantization used by HG-Rec."""

from .layers import EMAVQLayer, MLP, RQLayer, SimVQLayer, VQLayer
from .mechanism import RQVAE

__all__ = ["EMAVQLayer", "MLP", "RQLayer", "RQVAE", "SimVQLayer", "VQLayer"]
