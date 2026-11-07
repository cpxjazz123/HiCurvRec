"""RecBole3.0-compatible residual quantization used by HG-Rec."""

from .layers import EMAVQLayer, MLP, RQLayer, SimVQLayer, VQLayer
from .model import RQVAE as EuclideanRQVAE
from .model import MixedRQVAE

# The framework's model/gradient/checkpoint probes must instantiate this
# candidate's mixed model, not the retained Euclidean reference class.
RQVAE = MixedRQVAE

__all__ = ["EMAVQLayer", "MLP", "RQLayer", "RQVAE", "SimVQLayer", "VQLayer"]

