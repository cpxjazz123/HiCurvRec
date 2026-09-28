"""RQ-VAE layers migrated from RecBole3.0.

The implementation retains TIGER's model, codebook initialization, Sinkhorn
assignment, and straight-through losses. Only VQ geometry is Poincaré with
fixed curvature c=1; code vectors and residuals use origin-tangent coordinates.
The HG-Rec runner consumes the generated integer SIDs without changing T5.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as distributed
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans


CURVATURE = 1.0
_BALL_EPS = 1e-6


def _expmap0_tangent(x: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    scale = torch.tanh(norm) / norm.clamp_min(torch.finfo(x.dtype).eps)
    point = scale * x
    point_norm = torch.linalg.vector_norm(point, dim=-1, keepdim=True)
    max_norm = 1.0 - _BALL_EPS
    return point * torch.clamp(max_norm / point_norm.clamp_min(torch.finfo(x.dtype).eps), max=1.0)


def _logmap0_point(x: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    norm_safe = norm.clamp_min(torch.finfo(x.dtype).eps)
    scaled = norm.clamp(max=1.0 - _BALL_EPS)
    return (torch.atanh(scaled) / norm_safe) * x


def _mobius_add_unit_ball(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    numerator = (1.0 + 2.0 * xy + y2) * x + (1.0 - x2) * y
    denominator = 1.0 + 2.0 * xy + x2 * y2
    return numerator / denominator.clamp_min(torch.finfo(x.dtype).tiny)


def _poincare_distance_tangent_pairs(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_point = _expmap0_tangent(x)
    y_point = _expmap0_tangent(y)
    difference = _mobius_add_unit_ball(-x_point, y_point)
    norm = torch.linalg.vector_norm(difference, dim=-1)
    return 2.0 * torch.atanh(norm.clamp(max=1.0 - _BALL_EPS))


def _pairwise_poincare_distance_tangents(
    x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    x_point = _expmap0_tangent(x)
    y_point = _expmap0_tangent(y)
    x2 = (x_point * x_point).sum(dim=1, keepdim=True)
    y2 = (y_point * y_point).sum(dim=1).unsqueeze(0)
    dot = x_point @ y_point.t()
    difference2 = (x2 + y2 - 2.0 * dot).clamp_min(0.0)
    denominator = (1.0 - 2.0 * dot + x2 * y2).clamp_min(
        torch.finfo(x.dtype).tiny
    )
    mobius_norm = torch.sqrt(difference2 / denominator)
    return 2.0 * torch.atanh(mobius_norm.clamp(max=1.0 - _BALL_EPS))


def _hyperbolic_residual(
    residual_tangent: torch.Tensor, code_tangent: torch.Tensor
) -> torch.Tensor:
    residual_point = _expmap0_tangent(residual_tangent)
    code_point = _expmap0_tangent(code_tangent)
    difference = _mobius_add_unit_ball(-code_point, residual_point)
    return _logmap0_point(difference)


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
    ):
        super().__init__()
        self.dim = int(codebook_dim)
        self.n_embed = int(codebook_size)
        self.beta = float(beta)
        self.use_sk = sk_epsilon > 0 and sk_iters > 0
        self.sk_epsilon = float(sk_epsilon)
        self.sk_iters = int(sk_iters)
        self.embed = nn.Embedding(self.n_embed, self.dim)

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
        return _pairwise_poincare_distance_tangents(
            latent, self.get_code_embs()
        )

    def _balanced_assignments(self, distances: torch.Tensor) -> torch.Tensor:
        centered = self.center_distance(distances).double()
        assignments = self.sinkhorn(centered, self.sk_epsilon, self.sk_iters)
        if not torch.isfinite(assignments).all():
            raise RuntimeError("Sinkhorn assignment returned NaN or infinity.")
        return assignments

    def _indices(self, distances: torch.Tensor, infer_use_sk: bool) -> torch.Tensor:
        if (self.training and self.use_sk) or (self.use_sk and infer_use_sk):
            return self._balanced_assignments(distances).argmax(dim=-1)
        return torch.argmin(distances, dim=-1)

    @torch.no_grad()
    def assignment_diagnostics(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = x.reshape(-1, self.dim)
        assignments = self._balanced_assignments(self._distances(latent))
        ids = assignments.argmax(dim=-1)
        probabilities = assignments / assignments.sum(
            dim=-1, keepdim=True
        ).clamp_min(torch.finfo(assignments.dtype).tiny)
        entropy = -(
            probabilities
            * probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()
        ).sum(dim=-1).mean()
        usage = torch.bincount(ids, minlength=self.n_embed)
        return ids.view(*x.shape[:-1]), usage, entropy

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


        codebook_loss = _poincare_distance_tangent_pairs(
            x.detach(), x_q
        ).square().mean()
        commitment_loss = _poincare_distance_tangent_pairs(
            x, x_q.detach()
        ).square().mean()
        quant_loss = codebook_loss + self.beta * commitment_loss
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
        embed_ind = self._indices(self._distances(x), infer_use_sk=True).view(*x.shape[:-1])
        return _hyperbolic_residual(x, self.embed_code(embed_ind))


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
    """TIGER residual stack with fixed-curvature Poincaré VQ geometry."""
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
        for size in self.codebook_sizes:
            if self.vq_type == "vq":
                layer = VQLayer(
                    size, self.codebook_dim, self.vq_beta,
                    self.sk_epsilon, self.sk_iters,
                )
            elif self.vq_type == "ema":
                layer = EMAVQLayer(
                    size, self.codebook_dim, self.vq_beta,
                    self.sk_epsilon, self.sk_iters, config.ema_decay,
                )
            elif self.vq_type == "simvq":
                layer = SimVQLayer(
                    size, self.codebook_dim, self.vq_beta,
                    self.sk_epsilon, self.sk_iters, config.fix_code_embs,
                )
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
            residual = _hyperbolic_residual(residual, quant)
            quantized_x = quantized_x + quant
            sum_quant_loss = sum_quant_loss + quant_loss
            num_unused_codes += unused
            output[:, level] = indices
        return quantized_x, sum_quant_loss / self.codebook_num, num_unused_codes, output

    @torch.no_grad()
    def get_indices_with_stats(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        tokens = torch.empty(
            x.shape[0], self.codebook_num, dtype=torch.long, device=x.device
        )
        residual = x
        stats = []
        for level, layer in enumerate(self.vq_layers):
            indices, usage, entropy = layer.assignment_diagnostics(residual)
            tokens[:, level] = indices
            stats.append(
                {
                    "usage_counts": usage,
                    "assignment_entropy_nats": entropy,
                }
            )
            residual = _hyperbolic_residual(residual, layer.embed_code(indices))
        return tokens, stats

    def init_codebook(self, x: torch.Tensor, device: torch.device) -> torch.Tensor:
        residual = x
        for vq_layer in self.vq_layers:
            residual = vq_layer.init_codebook(residual, device)
        return residual
