"""RQ-VAE layers migrated from RecBole3.0.

The implementation intentionally keeps RecBole3.0's contracts: Euclidean
residual quantization, optional Sinkhorn assignment on the last level, and
the same straight-through quantization losses.  The HG-Rec runner consumes
the generated integer SIDs; it does not alter the T5 recommender.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as distributed
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans


class MLP(nn.Module):
    """RecBole3.0's dropout/linear/ReLU MLP helper."""

    def __init__(self, hidden_sizes: list[int], dropout: float = 0.0):
        super().__init__()
        modules: list[nn.Module] = []
        for input_size, output_size in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            modules.extend((nn.Dropout(p=dropout), nn.Linear(input_size, output_size), nn.ReLU()))
        if modules:
            modules.pop()  # no activation after the output projection
        self.mlp = nn.Sequential(*modules)

    def init_tiger_weights(self) -> None:
        """Use the Xavier initialization used by the upstream TIGER RQ-VAE."""
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class VQLayer(nn.Module):
    """Trainable vector quantizer from RecBole3.0."""

    def __init__(
        self,
        codebook_size: int,
        codebook_dim: int,
        beta: float = 0.25,
        sk_epsilon: float = -1.0,
        sk_iters: int = -1,
        geodesic: bool = False,
        geodesic_normalizer: float = 1.0,
        geodesic_curvature: float = 1.0,
        geodesic_codebook_loss: bool = False,
        geodesic_angular_only: bool = False,
    ):
        super().__init__()
        self.dim = int(codebook_dim)
        self.n_embed = int(codebook_size)
        self.beta = float(beta)
        self.use_sk = sk_epsilon > 0 and sk_iters > 0
        self.sk_epsilon = float(sk_epsilon)
        self.sk_iters = int(sk_iters)
        self.embed = nn.Embedding(self.n_embed, self.dim)
        # When set, the CODEWORD SELECTION distance is the Poincare geodesic
        # instead of the squared Euclidean one. This is the only thing that
        # changes: the quantiser still returns the same four-tuple, the residual
        # chain is untouched, and the codebook capacity is identical. It is a
        # property of the layer, not a global switch, so only the L1 layer is
        # given it and L2 and L3 keep the Euclidean distance they always used.
        self.geodesic = bool(geodesic)
        self.geodesic_normalizer = float(geodesic_normalizer)
        self.geodesic_curvature = float(geodesic_curvature)
        # When set together with `geodesic`, the quantisation OBJECTIVE moves into
        # the same ball the selection already measures in: the codeword is pulled
        # toward the mapped item and the item is committed toward the mapped
        # codeword, both as squared distance between the two MAPPED points. The
        # squared chordal distance is used rather than the raw arc because the arc
        # has a hard floor at acosh(1 + 1e-7) = 4.47e-4 with exactly zero gradient
        # below it, and that floor is precisely the region a codebook has to
        # tighten into. Nothing else changes: the residual chain, the capacity,
        # the straight-through value the decoder receives, L2 and L3.
        self.geodesic_codebook_loss = bool(geodesic_codebook_loss)
        # Radius-free (angular) selection. The full geodesic distance between two
        # points grows with BOTH radii, so with items pinned near the boundary the
        # nearest codeword is the one with the smallest norm whatever its angle and
        # the rule stops discriminating by angle at all -- measured on a healthy
        # codebook it keeps 51 of 256 codewords and puts 79.67% of items on ten of
        # them. This branch projects both sides onto a common radius first, so the
        # distance is the angular part of the ball alone and the radial term cannot
        # dominate. It is still the ball's geometry, not the Euclidean rule: on the
        # same codebook the Euclidean rule keeps 236 codewords but disagrees with
        # this one on 51% of items.
        self.geodesic_angular_only = bool(geodesic_angular_only)

    def get_code_embs(self) -> nn.Parameter:
        return self.embed.weight

    def _copy_init_embed(self, init_embed: torch.Tensor) -> None:
        self.embed.weight.data.copy_(init_embed)

    @staticmethod
    def center_distance(distances: torch.Tensor) -> torch.Tensor:
        max_distance = distances.max()
        min_distance = distances.min()
        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        if not bool(amplitude > 0):
            raise ValueError("Cannot center a constant distance matrix.")
        return (distances - middle) / amplitude

    @torch.no_grad()
    def sinkhorn(
        self,
        distances: torch.Tensor,
        epsilon: float = 0.003,
        iterations: int = 5,
    ) -> torch.Tensor:
        q = torch.exp(-distances / epsilon)
        batch_size, codebook_size = q.shape
        q /= q.sum(-1, keepdim=True).sum(-2, keepdim=True)
        for _ in range(iterations):
            q /= q.sum(0, keepdim=True)
            q /= codebook_size
            q /= q.sum(1, keepdim=True)
            q /= batch_size
        q *= batch_size
        return q

    def _distances(self, latent: torch.Tensor) -> torch.Tensor:
        code_embs = self.get_code_embs()
        if self.geodesic:
            # Map BOTH sides with one fixed, batch-independent scale, then take
            # the geodesic distance. The normaliser is a constant of the run, not
            # a statistic of the current batch, so an item's distance to a
            # codeword does not depend on what else is in the batch.
            from hyperbolic import encode_ball, pair_poincare_distance

            nz = self.geodesic_normalizer
            if self.geodesic_angular_only:
                # Radius-free: normalise both sides to the same radius, then take the
                # chart distance. Monotone in the angle between the two directions,
                # so it is the ball's angular channel and nothing else.
                mx = encode_ball(latent, self.geodesic_curvature, nz)
                mc = encode_ball(code_embs, self.geodesic_curvature, nz)
                mx = mx / mx.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                mc = mc / mc.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                return (mx[:, None, :] - mc[None, :, :]).pow(2).sum(-1).clamp_min(1e-6)
            mapped_x = encode_ball(latent, self.geodesic_curvature, nz)
            mapped_c = encode_ball(code_embs, self.geodesic_curvature, nz)
            # All-pairs (N, K) matrix. poincare_distance alone only broadcasts
            # when the two sides already share their leading dims, so it cannot
            # turn (N, D) against (K, D) into a distance matrix.
            return pair_poincare_distance(
                mapped_x, mapped_c, self.geodesic_curvature
            ).clamp_min(1e-6)
        return (
            latent.pow(2).sum(1, keepdim=True)
            - 2 * latent @ code_embs.t()
            + code_embs.pow(2).sum(1, keepdim=True).t()
        )

    def _indices(self, distances: torch.Tensor, infer_use_sk: bool) -> torch.Tensor:
        if (self.training and self.use_sk) or (self.use_sk and infer_use_sk):
            centered = self.center_distance(distances).double()
            q = self.sinkhorn(centered, self.sk_epsilon, self.sk_iters)
            if torch.isnan(q).any() or torch.isinf(q).any():
                raise RuntimeError("Sinkhorn assignment returned NaN or infinity.")
            return torch.argmax(q, dim=-1)
        return torch.argmin(distances, dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        infer_use_sk: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        latent = x.view(-1, self.dim)
        embed_ind = self._indices(self._distances(latent), infer_use_sk)
        onehot = F.one_hot(embed_ind, self.n_embed)
        used = onehot.sum(0)
        # Skip the DDP all_reduce when running inference on rank 0 only:
        # the unused-code accounting is a per-rank stat, and a single-rank
        # all_reduce would deadlock waiting for the other 3 ranks to join.
        if distributed.is_initialized() and infer_use_sk is False and not getattr(self, "_skip_ddp_reduce", False):
            distributed.all_reduce(used, op=distributed.ReduceOp.SUM)
        unused_codes = int((used == 0).sum().item())

        x_q = F.embedding(embed_ind, self.get_code_embs()).view(x.shape)
        if self.geodesic and self.geodesic_codebook_loss:
            # Geometry-consistent objective: the codebook is moved in the same ball
            # the selection measures, not in the pre-map Euclidean space. `x_q` and
            # the straight-through value the decoder receives are unchanged, so the
            # single changed variable is the space the codebook is trained in.
            from hyperbolic import encode_ball

            curv = self.geodesic_curvature
            nz = self.geodesic_normalizer
            z_b = encode_ball(latent, curv, nz)
            e_sel = self.get_code_embs()[embed_ind]
            e_b = encode_ball(e_sel, curv, nz)
            quant_loss = F.mse_loss(e_b, z_b.detach()) + self.beta * F.mse_loss(z_b, e_b.detach())
        else:
            quant_loss = F.mse_loss(x_q, x.detach()) + self.beta * F.mse_loss(x, x_q.detach())
        x_q = x + (x_q - x).detach()
        return x_q, quant_loss, unused_codes, embed_ind.view(*x.shape[:-1])

    def embed_code(self, embed_id: torch.Tensor) -> torch.Tensor:
        return F.embedding(embed_id, self.get_code_embs())

    def init_codebook(self, x: torch.Tensor, device: torch.device) -> torch.Tensor:
        kmeans = KMeans(n_clusters=self.n_embed, n_init="auto").fit(x.detach().cpu().numpy())
        centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device)
        if distributed.is_initialized():
            distributed.broadcast(centers, 0)
        self._copy_init_embed(centers.clone())
        embed_ind = torch.argmin(self._distances(x), dim=-1).view(*x.shape[:-1])
        return x - self.embed_code(embed_ind)


class EMAVQLayer(VQLayer):
    """EMA variant from RecBole3.0."""

    def __init__(
        self,
        codebook_size: int,
        codebook_dim: int,
        beta: float = 0.25,
        sk_epsilon: float = -1.0,
        sk_iters: int = -1,
        decay: float = 0.99,
        eps: float = 1e-5,
    ):
        super().__init__(codebook_size, codebook_dim, beta, sk_epsilon, sk_iters)
        self.decay = float(decay)
        self.eps = float(eps)
        embed = torch.zeros(self.n_embed, self.dim)
        self.embed = nn.Parameter(embed, requires_grad=False)
        nn.init.xavier_normal_(self.embed)
        self.register_buffer("embed_avg", embed.clone())
        self.register_buffer("cluster_size", torch.ones(self.n_embed))

    def _copy_init_embed(self, init_embed: torch.Tensor) -> None:
        self.embed.data.copy_(init_embed)
        self.embed_avg.data.copy_(init_embed)
        self.cluster_size.data.copy_(torch.ones(self.n_embed, device=init_embed.device))

    def get_code_embs(self) -> nn.Parameter:
        return self.embed

    def forward(
        self,
        x: torch.Tensor,
        infer_use_sk: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        latent = x.view(-1, self.dim)
        embed_ind = self._indices(self._distances(latent), infer_use_sk)
        x_q = F.embedding(embed_ind, self.get_code_embs()).view(x.shape)

        if self.training:
            onehot = F.one_hot(embed_ind, self.n_embed).type(latent.dtype)
            used = onehot.sum(0)
            embed_sum = onehot.t() @ latent
            if distributed.is_initialized():
                distributed.all_reduce(used, op=distributed.ReduceOp.SUM)
                distributed.all_reduce(embed_sum, op=distributed.ReduceOp.SUM)
            unused_codes = int((used == 0).sum().item())
            self.cluster_size.data.mul_(self.decay).add_(used, alpha=1 - self.decay)
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            norm_w = n * (self.cluster_size + self.eps) / (n + self.n_embed * self.eps)
            self.embed.data.copy_(self.embed_avg / norm_w.unsqueeze(1))
        else:
            used = F.one_hot(embed_ind, self.n_embed).sum(0)
            if distributed.is_initialized():
                distributed.all_reduce(used, op=distributed.ReduceOp.SUM)
            unused_codes = int((used == 0).sum().item())

        quant_loss = self.beta * F.mse_loss(x, x_q.detach())
        x_q = x + (x_q - x).detach()
        return x_q, quant_loss, unused_codes, embed_ind.view(*x.shape[:-1])


class SimVQLayer(VQLayer):
    """SimVQ variant retained for source-level compatibility."""

    def __init__(
        self,
        codebook_size: int,
        codebook_dim: int,
        beta: float = 0.25,
        sk_epsilon: float = -1.0,
        sk_iters: int = -1,
        fix_code_embs: bool = False,
    ):
        super().__init__(codebook_size, codebook_dim, beta, sk_epsilon, sk_iters)
        nn.init.xavier_normal_(self.embed.weight)
        self.embed_proj = nn.Linear(self.dim, self.dim, bias=False)
        if fix_code_embs:
            for parameter in self.embed.parameters():
                parameter.requires_grad = False

    def get_code_embs(self) -> Any:
        return self.embed_proj(self.embed.weight)

    def _copy_init_embed(self, init_embed: torch.Tensor) -> None:
        del init_embed  # SimVQ initializes through its projection.


class RQLayer(nn.Module):
    """Residual stack matching RecBole3.0's RQLayer."""

    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self.codebook_num = int(config.codebook_num)
        self.codebook_dim = int(config.codebook_dim)
        if isinstance(config.codebook_size, int):
            sizes = [int(config.codebook_size)] * self.codebook_num
        else:
            sizes = [int(size) for size in config.codebook_size]
            if len(sizes) != self.codebook_num:
                raise ValueError("codebook_size must have one entry per quantization level")
        self.codebook_sizes = sizes
        self.vq_type = str(config.vq_type)
        self.vq_beta = float(config.beta)
        self.sk_epsilon = float(config.sk_epsilon)
        self.sk_iters = int(config.sk_iters)

        layers: list[nn.Module] = []
        # Set by the caller when the FIRST level's codeword selection should use
        # the Poincare geodesic. Only level 0 receives it: the residual chain and
        # the deeper levels must keep the Euclidean distance they always used, or
        # the change would not be "only the L1 selection distance".
        geodesic_cfg = getattr(config, "l1_geodesic", None)
        for level, size in enumerate(self.codebook_sizes):
            epsilon = self.sk_epsilon if level == self.codebook_num - 1 else -1.0
            iterations = self.sk_iters if level == self.codebook_num - 1 else -1
            extra = (
                dict(
                    geodesic=True,
                    geodesic_normalizer=float(geodesic_cfg["normalizer"]),
                    geodesic_curvature=float(geodesic_cfg["curvature"]),
                    geodesic_codebook_loss=(str(geodesic_cfg.get("codebook_loss", "euclid"))
                                            == "ball_mse"),
                    geodesic_angular_only=(str(geodesic_cfg.get("selection", "geodesic"))
                                           == "angular"),
                )
                if (geodesic_cfg and level == 0 and self.vq_type == "vq")
                else {}
            )
            if self.vq_type == "vq":
                layer = VQLayer(size, self.codebook_dim, self.vq_beta, epsilon, iterations, **extra)
            elif self.vq_type == "ema":
                layer = EMAVQLayer(size, self.codebook_dim, self.vq_beta, epsilon, iterations, config.ema_decay)
            elif self.vq_type == "simvq":
                layer = SimVQLayer(size, self.codebook_dim, self.vq_beta, epsilon, iterations, config.fix_code_embs)
            else:
                raise ValueError(f"Unsupported vq_type: {self.vq_type}")
            layers.append(layer)
        self.vq_layers = nn.ModuleList(layers)

    def forward(
        self,
        x: torch.Tensor,
        infer_use_sk: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
        quantized_x = torch.zeros(x.shape[0], self.codebook_dim, device=x.device)
        sum_quant_loss: torch.Tensor | float = 0.0
        num_unused_codes = 0.0
        output = torch.empty(x.shape[0], self.codebook_num, dtype=torch.long, device=x.device)
        residual = x
        for level, vq_layer in enumerate(self.vq_layers):
            quant, quant_loss, unused, indices = vq_layer(residual, infer_use_sk)
            residual = residual - quant
            quantized_x = quantized_x + quant
            sum_quant_loss = sum_quant_loss + quant_loss
            num_unused_codes += unused
            output[:, level] = indices
        return quantized_x, sum_quant_loss / self.codebook_num, num_unused_codes, output

    def init_codebook(self, x: torch.Tensor, device: torch.device) -> torch.Tensor:
        residual = x
        for vq_layer in self.vq_layers:
            residual = vq_layer.init_codebook(residual, device)
        return residual
