"""RecBole3.0-compatible residual quantization used by HG-Rec."""

from .layers import EMAVQLayer, MLP, RQLayer, SimVQLayer, VQLayer
from .model import RQVAE as EuclideanRQVAE
from .model import MixedRQVAE

#: The model class of this tree is the mixed one. The framework looks the
#: class up by this name -- the one-epoch smoke probes its loss terms and the
#: checkpoint probe rebuilds it -- so the tree has to answer to it.
RQVAE = MixedRQVAE


__all__ = ["EMAVQLayer", "MLP", "RQLayer", "RQVAE", "SimVQLayer", "VQLayer"]
