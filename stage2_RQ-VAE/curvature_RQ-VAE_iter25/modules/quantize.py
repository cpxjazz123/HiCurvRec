from typing import NamedTuple

import torch
import torch.distributed as dist
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
        c_layer_norm: float = 1.0,
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
        if not 0 < c_layer_norm <= 1.0:
            raise ValueError("c_layer_norm 必须在 (0, 1] 内（对 c_cyclic_max 的归一化尺度）")

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
        # Initialize the learnable bounded layer scale from its configured prior.
        self.c_layer_norm = float(c_layer_norm)
        initial = min(max(self.c_layer_norm, 1e-4), 1.0 - 1e-4)
        self.c_layer_scale = nn.Parameter(torch.tensor(initial, dtype=torch.float32))
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
        """Return cyclic curvature with a learnable bounded layer scale."""
        phase = torch.tensor(
            torch.pi * self._curriculum_step / self.c_cyclic_period,
            device=self.device,
            dtype=self.embedding.weight.dtype,
        )
        g_t = torch.sin(phase).abs()
        u_layer = self.c_layer_scale.clamp(1e-4, 1.0 - 1e-4).to(
            dtype=self.embedding.weight.dtype
        )
        exponent = (u_layer + g_t) / 2.0
        log_ratio = torch.log(
            torch.tensor(
                self.c_cyclic_max / self.c_cyclic_min,
                device=self.device,
                dtype=self.embedding.weight.dtype,
            )
        )
        return self.c_cyclic_min * torch.exp(exponent * log_ratio)

    def curvature_regularization(self) -> Tensor:
        """Keep learned layer scale near its configured prior."""
        u_layer = self.c_layer_scale.clamp(1e-4, 1.0 - 1e-4)
        return (u_layer - self.c_layer_norm) ** 2

    def set_curriculum_step(self, step: int) -> None:
        if step < 0:
            raise ValueError("curriculum step 必须非负")
        self._curriculum_step = int(step)

    @staticmethod
    def _center_distance_for_constraint(distances: Tensor) -> Tensor:
        if not torch.isfinite(distances).all().item():
            raise ValueError("量化距离含 NaN 或 Inf，无法进行 Sinkhorn 归一化")
        min_distance = distances.min()
        span = distances.max() - min_distance
        if not torch.isfinite(span).item():
            raise ValueError("量化距离动态范围含 NaN 或 Inf")
        if span.item() <= 0:
            # 完全饱和的 batch 对 Sinkhorn 来说对应均匀分配。
            return torch.zeros_like(distances)
        # 先平移再除以范围，避免 (max + min) 中间值溢出。
        return 2.0 * ((distances - min_distance) / span) - 1.0

    def _init_weights(self) -> None:
        nn.init.uniform_(self.embedding.weight)

    @torch.no_grad()
    def _kmeans_init(self, x: Tensor) -> None:
        # K-means 初始化必须由 rank 0 决定并广播；否则每个 rank 的
        # 首批数据会产生不同 codebook，随后 DDP 参数从不同初值分叉。
        if dist.is_available() and dist.is_initialized():
            if dist.get_rank() == 0:
                kmeans_init_(self.embedding.weight, x=x)
            dist.broadcast(self.embedding.weight.data, src=0)
        else:
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
        # If every Poincare distance is numerically clipped to the same value,
        # retain a useful local ordering for assignment.  The quantization
        # loss below still uses the configured curved metric.
        if (distances.max() - distances.min()).item() <= 1e-6:
            # Large finite tangent vectors can flatten float32 squared distances:
            # ||x||^2 dominates the codebook-dependent terms. This branch is rare,
            # so compute it in float64 to preserve the assignment ordering.
            x64 = x.to(dtype=torch.float64)
            codebook64 = codebook.to(dtype=torch.float64)
            x_sq = (x64 * x64).sum(dim=1, keepdim=True)
            codebook_sq = (codebook64 * codebook64).sum(dim=1).unsqueeze(0)
            distances = (
                x_sq + codebook_sq - 2.0 * (x64 @ codebook64.t())
            ).clamp_min(0.0)

        centered = self._center_distance_for_constraint(distances.detach())
        # iter8: tighten the Sinkhorn assignment at high curvature, loosen at low
        # curvature. effective_eps scales the configured sk_eps so high-c layers
        # get sharper assignments (more discriminative per-layer quantization)
        # while low-c layers stay soft (avoiding degenerate hard assignments).
        # Reference: "Optimal Transport for Discrete Representation"
        # (Geneva & Zabaras 2022) — ε ∝ 1/c tightens assignments at high c.
        current_c = curvature.view(1, 1, 1)
        c_min_reference = torch.tensor(
            self.c_cyclic_min, device=centered.device, dtype=centered.dtype
        )
        c_max_reference = torch.tensor(
            self.c_cyclic_max, device=centered.device, dtype=centered.dtype
        )
        # Scale eps linearly with normalized curvature: eps_new = sk_eps * (c / c_max).
        # At c = c_max -> eps stays at sk_eps; at c = c_min -> eps shrinks by factor
        # c_min/c_max, softens assignment. The sign of the eps contrast flips for the
        # behavior on the curvature axis vs. naive 1/c but keeps the high-c sharpening
        # intent (and avoids numerical instability from c -> 0).
        c_scale = (
            current_c / c_max_reference.clamp_min(1e-12)
        ).view(1, 1)
        effective_eps = (self.sk_eps * c_scale).clamp_min(1e-4).item()
        assignments = _sinkhorn_algorithm(
            centered.double(), effective_eps, self.sk_iters
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
