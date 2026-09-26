from typing import NamedTuple, Optional

import torch
import torch.distributed as dist
from torch import nn, Tensor

from init.kmeans import kmeans_init_
from modules.hyperbolic import (
    _expmap0_t,
    _logmap0_t,
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
        # Initialize the learnable bounded layer scale from residual calibration.
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
        """Keep learned layer scale near the residual-calibrated initialization."""
        u_layer = self.c_layer_scale.clamp(1e-4, 1.0 - 1e-4)
        return (u_layer - self.c_layer_norm) ** 2

    def set_curriculum_step(self, step: int) -> None:
        if step < 0:
            raise ValueError("curriculum step 必须非负")
        self._curriculum_step = int(step)

    def apply_hyperbolic_trust_region(
        self,
        prev_codebook: Tensor,
        trust_radius_fraction: float = 0.5,
        trust_radius_min: float = 1e-3,
        eps: float = 1e-9,
    ) -> dict:
        """iter27: clamp codebook AdamW step by actual hyperbolic displacement.

        Computes the post-optimizer.codebook displacement `δ_k = d_{c_l}(p_old, p_new)`
        for every codeword, then rescales each update along the geodesic from
        `p_old` to `p_new` so that the resulting displacement does not exceed
        `τ_l = trust_radius_fraction · D_l`, where `D_l` is the layer-local
        median nearest-neighbor geodesic spacing of the *new* codebook. The
        rescaled tangent parameter is written back into `self.embedding.weight`.
        Returns a per-layer summary dict with δ median, τ, clip fraction,
        and the ratio δ / τ.
        """
        if trust_radius_fraction <= 0 or trust_radius_fraction >= 1:
            raise ValueError("trust_radius_fraction must be in (0, 1)")
        if trust_radius_min <= 0:
            raise ValueError("trust_radius_min must be positive")
        new_codebook = self.embedding.weight
        if prev_codebook.shape != new_codebook.shape:
            raise ValueError(
                f"codebook shape mismatch: prev={tuple(prev_codebook.shape)} "
                f"new={tuple(new_codebook.shape)}"
            )
        if not torch.isfinite(prev_codebook).all().item() or not torch.isfinite(new_codebook).all().item():
            raise RuntimeError("iter27 trust-region: non-finite codebook tensor")

        curvature = self.get_c().to(dtype=new_codebook.dtype)
        prev_ball = _expmap0_t(prev_codebook.unsqueeze(0), curvature).squeeze(0)
        new_ball = _expmap0_t(new_codebook.unsqueeze(0), curvature).squeeze(0)
        displacement = _poincare_distance_t(
            prev_ball.unsqueeze(0), new_ball.unsqueeze(0), curvature
        ).squeeze(0).squeeze(-1)
        pairwise = _poincare_distance_t(
            new_ball.unsqueeze(0), new_ball.unsqueeze(1), curvature
        ).squeeze(-1)
        pairwise.fill_diagonal_(float("inf"))
        nearest = pairwise.min(dim=1).values
        finite_nearest = nearest[torch.isfinite(nearest)]
        if finite_nearest.numel() == 0:
            D_l = float(trust_radius_min)
        else:
            D_l = max(float(trust_radius_min), float(finite_nearest.median().item()))
        tau_l = float(trust_radius_fraction) * D_l

        finite_mask = torch.isfinite(displacement)
        if not finite_mask.all().item():
            displacement = torch.where(finite_mask, displacement, torch.zeros_like(displacement))
        ratio = displacement / (tau_l + eps)
        s_k = torch.clamp(1.0 / (ratio + eps), max=1.0)

        # Origin-space interpolation in tangent space is equivalent (to first
        # order) to a geodesic walk on the Poincaré ball and stays inside the
        # ball because exp/log are mutual inverses. We compute the direction
        # in tangent space, scale by s_k, re-anchor at the previous tangent,
        # then exp back to the ball.
        prev_tangent = _logmap0_t(prev_ball, curvature)
        new_tangent = _logmap0_t(new_ball, curvature)
        scaled_tangent = prev_tangent + s_k.unsqueeze(-1) * (new_tangent - prev_tangent)
        corrected_ball = _expmap0_t(scaled_tangent.unsqueeze(0), curvature).squeeze(0)
        corrected = _logmap0_t(corrected_ball, curvature)
        if not torch.isfinite(corrected).all().item():
            raise RuntimeError("iter27 trust-region: non-finite corrected tangent")
        with torch.no_grad():
            self.embedding.weight.copy_(corrected.to(dtype=new_codebook.dtype))

        clipped_mask = displacement > tau_l
        n_clipped = int(clipped_mask.sum().item())
        clip_frac = n_clipped / max(1, displacement.numel())
        delta_median = float(displacement.median().item()) if displacement.numel() > 0 else 0.0
        delta_over_tau = (displacement / (tau_l + eps)).median().item() if displacement.numel() > 0 else 0.0
        return {
            "layer_index": -1,  # filled by caller
            "delta_median": delta_median,
            "tau_l": tau_l,
            "D_l": D_l,
            "clip_frac": float(clip_frac),
            "n_clipped": n_clipped,
            "delta_over_tau_median": float(delta_over_tau),
            "displacement_max": float(displacement.max().item()) if displacement.numel() > 0 else 0.0,
        }

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
