from typing import NamedTuple

import torch
from torch import nn, Tensor

from init.kmeans import kmeans_init_
from modules.hyperbolic import (
    _expmap0_t,
    _poincare_distance_t,
    _sinkhorn_algorithm,
)
from modules.loss import QuantizeLoss


class QuantizeOutput(NamedTuple):
    embeddings: Tensor
    ids: Tensor
    loss: Tensor


class Quantize(nn.Module):
    """当前 RQ-VAE 的固定量化步骤：Poincaré 距离、Sinkhorn 和 STE。"""

    def __init__(
        self,
        embed_dim: int,
        n_embed: int,
        do_kmeans_init: bool,
        commitment_weight: float,
        sk_eps: float,
        sk_iters: int,
        c_cyclic_min: float,
        c_cyclic_max: float,
        c_cyclic_period: int,
    ) -> None:
        super().__init__()
        if sk_eps <= 0:
            raise ValueError("当前曲率 RQ-VAE 必须启用 Sinkhorn，sk_eps 应为正数")
        if sk_iters <= 0:
            raise ValueError("sk_iters 应为正数")
        if not 0 < c_cyclic_min <= c_cyclic_max:
            raise ValueError("cyclic curvature 范围无效")
        if c_cyclic_period <= 0:
            raise ValueError("cyclic curvature period 必须为正数")

        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.embedding = nn.Embedding(n_embed, embed_dim)
        self.sk_eps = float(sk_eps)
        self.sk_iters = int(sk_iters)
        self.do_kmeans_init = do_kmeans_init
        self.kmeans_initted = False
        self.c_cyclic_min = float(c_cyclic_min)
        self.c_cyclic_max = float(c_cyclic_max)
        self.c_cyclic_period = int(c_cyclic_period)
        self._curriculum_step = 0
        self.quantize_loss = QuantizeLoss(commitment_weight)
        self._init_weights()

    @property
    def weight(self) -> Tensor:
        return self.embedding.weight

    @property
    def device(self) -> torch.device:
        return self.embedding.weight.device

    def get_c(self) -> Tensor:
        """返回当前训练步的 cyclic curvature。"""
        phase = torch.tensor(
            torch.pi * self._curriculum_step / self.c_cyclic_period,
            device=self.device,
            dtype=self.embedding.weight.dtype,
        )
        return self.c_cyclic_min + (
            self.c_cyclic_max - self.c_cyclic_min
        ) * torch.sin(phase).abs()

    def set_curriculum_step(self, step: int) -> None:
        if step < 0:
            raise ValueError("curriculum step 必须非负")
        self._curriculum_step = int(step)

    @staticmethod
    def _center_distance_for_constraint(distances: Tensor) -> Tensor:
        max_distance = distances.max()
        min_distance = distances.min()
        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle
        if amplitude <= 0:
            raise ValueError("量化距离的 Sinkhorn 归一化幅度必须为正数")
        return (distances - middle) / amplitude

    def _init_weights(self) -> None:
        nn.init.uniform_(self.embedding.weight)

    @torch.no_grad()
    def _kmeans_init(self, x: Tensor) -> None:
        kmeans_init_(self.embedding.weight, x=x)
        self.kmeans_initted = True

    def get_item_embeddings(self, item_ids: Tensor) -> Tensor:
        return self.embedding(item_ids)

    def forward(self, x: Tensor) -> QuantizeOutput:
        if x.ndim != 2 or x.shape[-1] != self.embed_dim:
            raise ValueError(
                f"量化输入必须为 [batch, {self.embed_dim}]，实际为 {tuple(x.shape)}"
            )
        if not torch.isfinite(x).all().item():
            raise ValueError("量化输入含 NaN 或 Inf")
        if self.do_kmeans_init and not self.kmeans_initted:
            self._kmeans_init(x)

        codebook = self.embedding.weight
        curvature = self.get_c()
        curvature_3d = curvature.view(1, 1, 1)
        curvature_2d = curvature.view(1, 1)
        batch_size = x.shape[0]
        codebook_size = codebook.shape[0]
        latent_h = _expmap0_t(x.unsqueeze(1), curvature_3d)
        codebook_h = _expmap0_t(
            codebook.unsqueeze(0).expand(batch_size, codebook_size, -1), curvature_3d
        )
        distances = _poincare_distance_t(
            latent_h.expand(batch_size, codebook_size, -1), codebook_h, curvature_3d
        ).squeeze(-1)
        if not torch.isfinite(distances).all().item():
            raise RuntimeError("Poincaré 量化距离含 NaN 或 Inf")

        centered = self._center_distance_for_constraint(distances.detach())
        assignments = _sinkhorn_algorithm(
            centered.double(), self.sk_eps, self.sk_iters
        )
        if not torch.isfinite(assignments).all().item():
            raise RuntimeError("Sinkhorn assignment 含 NaN 或 Inf")
        ids = torch.argmax(assignments, dim=-1)
        if not bool(((ids >= 0) & (ids < codebook_size)).all().item()):
            raise RuntimeError("Sinkhorn 生成了超出 codebook 范围的 id")

        embeddings = self.get_item_embeddings(ids)
        loss = self.quantize_loss(
            query=x,
            value=embeddings,
            c=curvature_2d,
        )
        if not torch.isfinite(loss).all().item():
            raise RuntimeError("Quantize loss 含 NaN 或 Inf")
        embeddings_out = (
            x + (embeddings - x).detach() if self.training else embeddings
        )
        return QuantizeOutput(embeddings=embeddings_out, ids=ids, loss=loss)
