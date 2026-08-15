import gin
import torch

from distributions.gumbel import gumbel_softmax_sample
from einops import rearrange
from enum import Enum
from init.kmeans import kmeans_init_
from modules.loss import QuantizeLoss
from modules.normalize import L2NormalizationLayer
from typing import NamedTuple
from torch import nn
from torch import Tensor
from torch.nn import functional as F


@gin.constants_from_enum
class QuantizeForwardMode(Enum):
    GUMBEL_SOFTMAX = 1
    STE = 2
    ROTATION_TRICK = 3


class QuantizeDistance(Enum):
    L2 = 1
    COSINE = 2
    POINCARE = 3  # 双曲机制 M2/M3: Poincare ball 距离 (要求 hyperbolic_mechanism != "none")


class QuantizeOutput(NamedTuple):
    embeddings: Tensor
    ids: Tensor
    loss: Tensor


def efficient_rotation_trick_transform(u, q, e):
    """
    4.2 in https://arxiv.org/abs/2410.06424
    """
    e = rearrange(e, "b d -> b 1 d")
    w = F.normalize(u + q, p=2, dim=1, eps=1e-6).detach()

    return (
        e
        - 2 * (e @ rearrange(w, "b d -> b d 1") @ rearrange(w, "b d -> b 1 d"))
        + 2
        * (
            e
            @ rearrange(u, "b d -> b d 1").detach()
            @ rearrange(q, "b d -> b 1 d").detach()
        )
    ).squeeze()


class Quantize(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_embed: int,
        do_kmeans_init: bool = True,
        codebook_normalize: bool = False,
        sim_vq: bool = False,  # https://arxiv.org/pdf/2411.02038
        commitment_weight: float = 0.25,
        forward_mode: QuantizeForwardMode = QuantizeForwardMode.GUMBEL_SOFTMAX,
        distance_mode: QuantizeDistance = QuantizeDistance.L2,
        poincare_distance_module=None,  # 双曲机制 M2/M3: 由 RqVae 传入共享模块 (RqVae 已注册为 sub-module)
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.embedding = nn.Embedding(n_embed, embed_dim)
        self.forward_mode = forward_mode
        self.distance_mode = distance_mode
        self.do_kmeans_init = do_kmeans_init
        self.kmeans_initted = False

        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False) if sim_vq else nn.Identity(),
            L2NormalizationLayer(dim=-1) if codebook_normalize else nn.Identity(),
        )

        self.quantize_loss = QuantizeLoss(commitment_weight)
        # 用 object.__setattr__ 存为普通 Python 属性 (而非 nn.Module sub-module),
        # 避免 curvature Parameter/buffer 在每层 Quantize.state_dict 中重复 3 次;
        # 真实 ownership 在 RqVae.poincare_distance_module, 各层共享同一实例引用.
        object.__setattr__(self, "poincare_distance_module", poincare_distance_module)
        self._init_weights()

    @property
    def weight(self) -> Tensor:
        return self.embedding.weight

    @property
    def device(self) -> torch.device:
        return self.embedding.weight.device

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight)

    @torch.no_grad
    def _kmeans_init(self, x) -> None:
        kmeans_init_(self.embedding.weight, x=x)
        self.kmeans_initted = True

    def get_item_embeddings(self, item_ids) -> Tensor:
        return self.out_proj(self.embedding(item_ids))

    def forward(self, x, temperature) -> QuantizeOutput:
        assert x.shape[-1] == self.embed_dim

        if self.do_kmeans_init and not self.kmeans_initted:
            self._kmeans_init(x=x)

        codebook = self.out_proj(self.embedding.weight)

        if self.distance_mode == QuantizeDistance.L2:
            dist = (
                (x**2).sum(axis=1, keepdim=True)
                + (codebook.T**2).sum(axis=0, keepdim=True)
                - 2 * x @ codebook.T
            )
        elif self.distance_mode == QuantizeDistance.COSINE:
            dist = -(
                x
                / x.norm(dim=1, keepdim=True)
                @ (codebook.T)
                / codebook.T.norm(dim=0, keepdim=True)
            )
        elif self.distance_mode == QuantizeDistance.POINCARE:
            if self.poincare_distance_module is None:
                raise ValueError(
                    "Quantize.distance_mode=POINCARE requires poincare_distance_module "
                    "to be provided by RqVae (set hyperbolic_mechanism in [poincare_dist, learnable_kappa])"
                )
            # PoincareDistance.forward(x, y) 自动 project x 和 y 到 Poincare ball 内部再算距离
            dist = self.poincare_distance_module(x, codebook)
        else:
            raise Exception("Unsupported Quantize distance mode.")

        _, ids = (dist.detach()).min(axis=1)

        if self.training:
            if self.forward_mode == QuantizeForwardMode.GUMBEL_SOFTMAX:
                weights = gumbel_softmax_sample(
                    -dist, temperature=temperature, device=self.device
                )
                emb = weights @ codebook
                emb_out = emb
            elif self.forward_mode == QuantizeForwardMode.STE:
                emb = self.get_item_embeddings(ids)
                emb_out = x + (emb - x).detach()
            elif self.forward_mode == QuantizeForwardMode.ROTATION_TRICK:
                emb = self.get_item_embeddings(ids)
                emb_out = efficient_rotation_trick_transform(
                    x / (x.norm(dim=-1, keepdim=True) + 1e-8),
                    emb / (emb.norm(dim=-1, keepdim=True) + 1e-8),
                    x,
                )
                emb_out = (
                    emb_out
                    * (
                        torch.norm(emb, dim=1, keepdim=True)
                        / (torch.norm(x, dim=1, keepdim=True) + 1e-6)
                    ).detach()
                )
            else:
                raise Exception("Unsupported Quantize forward mode.")

            loss = self.quantize_loss(query=x, value=emb)

        else:
            emb_out = self.get_item_embeddings(ids)
            loss = self.quantize_loss(query=x, value=emb_out)

        return QuantizeOutput(embeddings=emb_out, ids=ids, loss=loss)
