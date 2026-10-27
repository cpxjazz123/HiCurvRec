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


def ball_project(values: torch.Tensor, curvature: float = 1.0) -> torch.Tensor:
    """The projection of a Euclidean vector into the Poincare ball."""
    norm = values.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = torch.clamp((1.0 - 1e-5) / (norm * (curvature ** 0.5)), max=1.0)
    return values * scale


def ball_add(x: torch.Tensor, y: torch.Tensor, curvature: float = 1.0) -> torch.Tensor:
    """The Mobius addition in the Poincare ball."""
    inner = (x * y).sum(-1, keepdim=True)
    numerator = ((1.0 + 2.0 * curvature * inner + curvature * y.square().sum(-1, keepdim=True)) * x
                 + (1.0 - curvature * x.square().sum(-1, keepdim=True)) * y)
    denominator = (1.0 + 2.0 * curvature * inner
                   + curvature * curvature * x.square().sum(-1, keepdim=True)
                   * y.square().sum(-1, keepdim=True)).clamp_min(1e-9)
    return numerator / denominator


def _conformal_factor(base: torch.Tensor, curvature: float = 1.0) -> torch.Tensor:
    """The conformal factor of the Poincare metric at a point: 2 / (1 - c ||x||^2)."""
    return 2.0 / (1.0 - curvature * base.square().sum(-1, keepdim=True)).clamp_min(1e-6)


def ball_log_map(base: torch.Tensor, point: torch.Tensor,
                 curvature: float = 1.0) -> torch.Tensor:
    """The tangent vector at ``base`` that reaches ``point``.

    The displacement is ``(-base) (+) point``: the point measured from the
    origin and brought back to the base. The logarithm at a base point is the
    origin logarithm scaled by the conformal factor of the metric at that base,
    which is what makes the iteration the Frechet mean.
    """
    difference = ball_add(-base, point, curvature)
    norm = difference.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    log_origin = (2.0 / (curvature ** 0.5)) * torch.atanh(
        (curvature ** 0.5) * norm.clamp(max=1.0 - 1e-5)) / norm * difference
    return log_origin * _conformal_factor(base, curvature)


def ball_exp_map(base: torch.Tensor, tangent: torch.Tensor,
                 curvature: float = 1.0) -> torch.Tensor:
    """The point of the ball that the tangent vector at ``base`` reaches.

    The tangent vector at a base point is the origin tangent scaled down by the
    conformal factor of the metric at that base.
    """
    norm = tangent.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    scaled = (torch.tanh((curvature ** 0.5) * norm / 2.0) / ((curvature ** 0.5) * norm)) * tangent
    return ball_add(base, scaled / _conformal_factor(base, curvature), curvature)


def frechet_mean(points: torch.Tensor, weights: torch.Tensor,
                 curvature: float = 1.0, iterations: int = 8) -> torch.Tensor:
    """The Frechet (Karcher) mean of points in the Poincare ball.

    The iteration is the Riemannian one: the weighted mean of the tangent vectors
    at the current estimate, taken back into the ball. This is the barycentre the
    proposal asks for, not a mean in the origin tangent space.
    """
    weights = weights.unsqueeze(-1)
    total = weights.sum(dim=-2).clamp_min(1e-9)
    estimate = (points * weights).sum(dim=-2) / total
    norm = estimate.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    estimate = estimate / norm * torch.tanh((curvature ** 0.5) * norm / 2.0) / (curvature ** 0.5)
    for _ in range(iterations):
        tangent = ball_log_map(estimate.unsqueeze(-2), points, curvature)
        step = (tangent * weights).sum(dim=-2) / total
        estimate = ball_exp_map(estimate, step, curvature)
    return estimate


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    """The ascending average rank of every candidate, ties averaged.

    Fully vectorised: the tie groups are found from the sorted values, their
    sizes and position sums are accumulated with two scatter_adds, and the
    average position of each group is scattered back. A Python loop over the
    rows and the codes cost more than the rest of the step.
    """
    order = values.argsort(dim=-1)
    sorted_values = values.gather(-1, order)
    is_new = torch.ones_like(sorted_values, dtype=torch.bool)
    is_new[:, 1:] = sorted_values[:, 1:] != sorted_values[:, :-1]
    group = torch.cumsum(is_new.to(torch.int64), dim=-1) - 1
    ones = torch.ones_like(sorted_values)
    sizes = torch.zeros_like(sorted_values)
    sizes.scatter_add_(1, group, ones)
    positions = torch.arange(values.shape[-1], device=values.device,
                             dtype=values.dtype).expand_as(sorted_values)
    sums = torch.zeros_like(sorted_values)
    sums.scatter_add_(1, group, positions)
    average = sums / sizes.clamp_min(1.0)
    ranks = torch.empty_like(values)
    ranks.scatter_(-1, order, average)
    return ranks


def behaviour_distance(encoded: torch.Tensor, codes: torch.Tensor,
                       context: torch.Tensor, mode: str = "both_on",
                       curvature: float = 1.0):
    """The hyperbolic distance of the code-conditioned point from the context.

    ``q_ik = Proj_B(z_i + c_k - b_i)`` and ``h_ik = d_{-1}(q_ik, b_i)``: the
    distance compares each candidate's behaviour-context displacement instead
    of repeating the same distance for every code.
    """
    if mode == "geometry_off":
        # The Euclidean counterpart: no ball, no projection, anywhere.
        raw_context = context
        raw_conditioned = (encoded[:, None, :]
                           + codes[None, :, :].to(encoded.dtype)
                           - raw_context[:, None, :])
        return (raw_conditioned - raw_context[:, None, :]).square().sum(-1)
    point = ball_project(encoded, curvature)
    context_point = ball_project(context, curvature)
    conditioned = ball_project(
        point[:, None, :] + codes[None, :, :].to(point.dtype) - context_point[:, None, :],
        curvature)
    # The Poincare distance is measured on the Mobius displacement
    # (-b) (+) q, not on the Euclidean difference of the two points.
    displacement = ball_add(-context_point[:, None, :], conditioned, curvature)
    delta = displacement.square().sum(-1).clamp_min(1e-12)
    denominator = ((1.0 - curvature * context_point.square().sum(-1))[:, None]
                   * (1.0 - curvature * conditioned.square().sum(-1))).clamp_min(1e-6)
    argument = (2.0 * curvature * delta / denominator).clamp_min(0.0)
    # acosh(1 + z) = 2 asinh(sqrt(z / 2)): the squared form has a finite
    # derivative at z = 0, while acosh itself has an infinite one, so a
    # conditioned point that coincides with the context would otherwise put a
    # NaN into the gradient in float32.
    stable = 4.0 * torch.asinh(torch.sqrt(argument * 0.5)).square()
    return stable / curvature


def rank_assignment_scores(encoded: torch.Tensor, codes: torch.Tensor,
                           context: torch.Tensor, mode: str = "both_on",
                           curvature: float = 1.0):
    """The rank sum that decides the first-level code, and its gradient path.

    For every candidate code two quantities are ranked among the candidates:
    the Euclidean reconstruction error ``e_ik = ||z_i - c_k||^2`` and the
    hyperbolic distance ``h_ik = d_{-1}(q_ik, b_i)`` of the code-conditioned
    point ``q_ik = Proj_B(z_i + c_k - b_i)``. The code with the lowest sum of
    ranks wins, with the lowest index breaking a tie.

    The ranks themselves are not differentiable, so the function also returns the
    differentiable scores of the same two quantities: the hard choice is made by
    the ranks, and the straight-through path of the quantiser carries the
    gradient of the distances. Nothing is scaled by a tuned constant.
    """
    euclidean = (encoded.square().sum(-1)[:, None] + codes.square().sum(-1)[None, :]
                 - 2.0 * (encoded @ codes.t())).clamp_min(0.0)
    if mode == "behaviour_off":
        index = euclidean.argmin(dim=-1)
        return index, -euclidean
    hyperbolic = behaviour_distance(encoded, codes, context, mode, curvature)
    total = _average_ranks(euclidean) + _average_ranks(hyperbolic)
    index = total.argmin(dim=-1)
    scores = -(euclidean + hyperbolic)
    return index, scores


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
        code_embs = self.get_code_embs()
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
        assignment_scores: torch.Tensor | None = None,
        assignment_index: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        latent = x.view(-1, self.dim)
        if assignment_scores is not None:
            # The coupled score decides the code: the behaviour and the geometry
            # are on the same path as the assignment, not beside it. The index
            # comes from the rank sum, the scores carry the gradient.
            embed_ind = assignment_index if assignment_index is not None \
                else assignment_scores.argmax(-1)
        else:
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
        quant_loss = F.mse_loss(x_q, x.detach()) + self.beta * F.mse_loss(x, x_q.detach())
        x_q = x + (x_q - x).detach()
        if assignment_scores is not None:
            # The soft candidate distribution of the coupled score replaces the
            # hard code in the straight-through estimate, so the behaviour and
            # the curvature reach the encoder and the codebook through the
            # gradient and not only through the choice.
            soft = (assignment_scores.softmax(dim=-1) @ self.get_code_embs()).view(x.shape)
            quant_loss = quant_loss + F.mse_loss(soft, x.detach())
            x_q = x_q + soft - soft.detach()
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
        for level, size in enumerate(self.codebook_sizes):
            epsilon = self.sk_epsilon if level == self.codebook_num - 1 else -1.0
            iterations = self.sk_iters if level == self.codebook_num - 1 else -1
            if self.vq_type == "vq":
                layer = VQLayer(size, self.codebook_dim, self.vq_beta, epsilon, iterations)
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
        assignment_scores: torch.Tensor | None = None,
        assignment_index: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
        quantized_x = torch.zeros(x.shape[0], self.codebook_dim, device=x.device)
        sum_quant_loss: torch.Tensor | float = 0.0
        num_unused_codes = 0.0
        output = torch.empty(x.shape[0], self.codebook_num, dtype=torch.long, device=x.device)
        residual = x
        for level, vq_layer in enumerate(self.vq_layers):
            quant, quant_loss, unused, indices = vq_layer(
                residual, infer_use_sk,
                assignment_scores=assignment_scores if level == 0 else None,
                assignment_index=assignment_index if level == 0 else None)
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
