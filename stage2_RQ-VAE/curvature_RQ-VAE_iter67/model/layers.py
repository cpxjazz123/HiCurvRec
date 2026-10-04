"""RQ-VAE layers with adaptive-curvature Poincare geometry.

The implementation retains TIGER's codebook initialization, balanced
geodesic Sinkhorn assignment, and straight-through losses. Each VQ layer keeps
origin-tangent code vectors and residuals and carries a runtime-settable
curvature buffer that the residual-scale controller retunes. The HG-Rec runner
consumes generated integer SIDs without changing T5.
"""

from __future__ import annotations

from typing import Any
import math


import torch
import torch.distributed as distributed
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans


CURVATURE = 1.0
_BALL_EPS = 1e-6
# Element budget for the padded block-diagonal Sinkhorn solve, which is
# n_buckets x width x n_codes in float64 and is held across all 50
# iterations. At 100M elements the block is ~320 MB, small against a 44 GB
# card, and it covers training (width ~9) and full-corpus evaluation
# (width ~100-200) alike. It exists because a batch that piles nearly all rows
# into one bucket reaches width ~1024, i.e. 67M elements, and took the run out
# with an OOM once the rest of the card was already full. Only such batches
# fall back to the global solve; normal ones are untouched.
_MAX_BUCKET_ELEMENTS = 100_000_000


def _curvature_like(
    curvature: torch.Tensor | float, reference: torch.Tensor
) -> torch.Tensor:
    return torch.as_tensor(
        curvature, device=reference.device, dtype=reference.dtype
    ).clamp_min(torch.finfo(reference.dtype).tiny)


def _expmap0_tangent(
    x: torch.Tensor, curvature: torch.Tensor | float = CURVATURE
) -> torch.Tensor:
    c = _curvature_like(curvature, x)
    sqrt_c = c.sqrt()
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    scaled_norm = sqrt_c * norm
    scale = torch.tanh(scaled_norm) / scaled_norm.clamp_min(
        torch.finfo(x.dtype).eps
    )
    point = scale * x
    point_norm = torch.linalg.vector_norm(point, dim=-1, keepdim=True)
    max_norm = (1.0 - _BALL_EPS) / sqrt_c
    return point * torch.clamp(
        max_norm / point_norm.clamp_min(torch.finfo(x.dtype).eps), max=1.0
    )


def _logmap0_point(
    x: torch.Tensor, curvature: torch.Tensor | float = CURVATURE
) -> torch.Tensor:
    c = _curvature_like(curvature, x)
    sqrt_c = c.sqrt()
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    norm_safe = norm.clamp_min(torch.finfo(x.dtype).eps)
    scaled = (sqrt_c * norm).clamp(max=1.0 - _BALL_EPS)
    return (torch.atanh(scaled) / (sqrt_c * norm_safe)) * x


def _mobius_add(
    x: torch.Tensor,
    y: torch.Tensor,
    curvature: torch.Tensor | float = CURVATURE,
) -> torch.Tensor:
    c = _curvature_like(curvature, x)
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    numerator = (1.0 + 2.0 * c * xy + c * y2) * x + (1.0 - c * x2) * y
    denominator = 1.0 + 2.0 * c * xy + c.square() * x2 * y2
    return numerator / denominator.clamp_min(torch.finfo(x.dtype).tiny)


def _poincare_distance_tangent_pairs(
    x: torch.Tensor,
    y: torch.Tensor,
    curvature: torch.Tensor | float = CURVATURE,
) -> torch.Tensor:
    c = _curvature_like(curvature, x)
    sqrt_c = c.sqrt()
    x_point = _expmap0_tangent(x, c)
    y_point = _expmap0_tangent(y, c)
    difference = _mobius_add(-x_point, y_point, c)
    norm = torch.linalg.vector_norm(difference, dim=-1)
    return (2.0 / sqrt_c) * torch.atanh(
        (sqrt_c * norm).clamp(max=1.0 - _BALL_EPS)
    )


def _pairwise_poincare_distance_tangents(
    x: torch.Tensor,
    y: torch.Tensor,
    curvature: torch.Tensor | float = CURVATURE,
) -> torch.Tensor:
    c = _curvature_like(curvature, x)
    sqrt_c = c.sqrt()
    x_point = _expmap0_tangent(x, c)
    y_point = _expmap0_tangent(y, c)
    x2 = (x_point * x_point).sum(dim=1, keepdim=True)
    y2 = (y_point * y_point).sum(dim=1).unsqueeze(0)
    dot = x_point @ y_point.t()
    difference2 = (x2 + y2 - 2.0 * dot).clamp_min(0.0)
    denominator = (1.0 - 2.0 * c * dot + c.square() * x2 * y2).clamp_min(
        torch.finfo(x.dtype).tiny
    )
    mobius_norm = sqrt_c * torch.sqrt(difference2.clamp_min(1e-12) / denominator)
    return (2.0 / sqrt_c) * torch.atanh(
        mobius_norm.clamp(max=1.0 - _BALL_EPS)
    )


def _hyperbolic_residual(
    residual_tangent: torch.Tensor,
    code_tangent: torch.Tensor,
    curvature: torch.Tensor | float = CURVATURE,
) -> torch.Tensor:
    residual_point = _expmap0_tangent(residual_tangent, curvature)
    code_point = _expmap0_tangent(code_tangent, curvature)
    difference = _mobius_add(-code_point, residual_point, curvature)
    return _logmap0_point(difference, curvature)


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
    """Poincare VQ with a runtime-settable curvature and balanced Sinkhorn.

    The curvature is a buffer rather than a Python float so the adaptive
    controller can retune the ball geometry mid-run without rebuilding the
    module and without adding anything to the optimizer's parameter list. It
    never receives a gradient: the controller is the only writer.
    """

    def __init__(
        self,
        codebook_size: int,
        codebook_dim: int,
        beta: float = 0.25,
        sk_epsilon: float = 0.003,
        sk_iters: int = 50,
        curvature: float = 1.0,
    ):
        super().__init__()
        if curvature <= 0.0:
            raise ValueError("Curvature must be positive.")
        if sk_epsilon <= 0.0 or sk_iters <= 0:
            raise ValueError("Sinkhorn epsilon and iteration count must be positive.")
        self.dim = int(codebook_dim)
        self.n_embed = int(codebook_size)
        self.beta = float(beta)
        self.use_sk = True
        self.sk_epsilon = float(sk_epsilon)
        self.sk_iters = int(sk_iters)
        self.register_buffer("curvature", torch.tensor(float(curvature)))
        self.embed = nn.Embedding(self.n_embed, self.dim)

    def get_code_embs(self) -> nn.Parameter:
        return self.embed.weight

    def _copy_init_embed(self, init_embed: torch.Tensor) -> None:
        self.embed.weight.data.copy_(init_embed)

    def get_curvature(self) -> torch.Tensor:
        return self.curvature

    def set_curvature(self, curvature: float) -> None:
        """Retune this level's ball geometry; called by the adaptive controller."""
        if not curvature > 0.0:
            raise ValueError("Curvature must be positive.")
        self.curvature.fill_(float(curvature))

    def get_effective_epsilon(self) -> float:
        return self.sk_epsilon

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
        iterations: int = 50,
    ) -> torch.Tensor:
        # float64 keeps the alternating normalization from losing precision on
        # small buckets, where epsilon=0.003 amplifies the distances ~333x.
        log_q = -distances.double() / float(epsilon)
        batch_size, codebook_size = log_q.shape
        log_q = log_q - torch.logsumexp(log_q, dim=(0, 1), keepdim=True)
        log_codebook_size = math.log(codebook_size)
        log_batch_size = math.log(batch_size)
        for _ in range(iterations):
            log_q = log_q - torch.logsumexp(log_q, dim=0, keepdim=True)
            log_q = log_q - log_codebook_size
            log_q = log_q - torch.logsumexp(log_q, dim=1, keepdim=True)
            log_q = log_q - log_batch_size
        return torch.exp(log_q + log_batch_size)

    def _distances(self, latent: torch.Tensor) -> torch.Tensor:
        return _pairwise_poincare_distance_tangents(
            latent, self.get_code_embs(), self.get_curvature()
        )

    def _balanced_assignments(self, distances: torch.Tensor) -> torch.Tensor:
        centered = self.center_distance(distances).double()
        assignments = self.sinkhorn(
            centered, self.get_effective_epsilon(), self.sk_iters
        )
        if not torch.isfinite(assignments).all():
            raise RuntimeError("Sinkhorn assignment returned NaN or infinity.")
        return assignments

    def _bucket_balanced_assignments(
        self, distances: torch.Tensor, bucket: torch.Tensor
    ) -> torch.Tensor:
        """Balance codes inside each preceding-code bucket.

        Plain Sinkhorn spreads the same total mass over every code in the whole
        batch. That is the wrong constraint once residual quantization is
        involved: an item's reachable codes at this level are already
        determined by the code it received at the previous level, so a code
        that only fits the residuals of some other bucket is being asked to
        compete for mass it can never win.

        Measured on the trained checkpoint, that mismatch cost most of the
        second level's capacity: L0 used all 256 codes, but each L0 bucket only
        reached ~37 of the 256 L1 codes, so (L0, L1) produced 9370 distinct
        prefixes instead of the 65536 the codebook allows.

        All buckets are then solved together in one batched Sinkhorn. The
        grouping uses a sort plus split rather than a per-bucket boolean mask:
        256 masked gathers measured 48 ms on their own against 0.76 ms for the
        sort, which is the difference between a run that finishes in an hour
        and one that needs four days.
        """
        n_codes = distances.shape[1]
        n_rows = distances.shape[0]
        centered = self.center_distance(distances).double()
        assignment = torch.zeros_like(centered)

        # Group rows by bucket in one pass: sort once, then split by counts.
        order = torch.argsort(bucket, stable=True)
        n_buckets = int(bucket.max()) + 1
        counts = torch.bincount(bucket, minlength=n_buckets)
        sizes = counts.tolist()
        width = max(sizes)
        if width == 0:
            return assignment
        # Cap the element count, not the width: a wide bucket is fine while few
        # buckets are populated, which is the normal training case, and only a
        # batch that piles many rows into one bucket needs the fallback.
        if n_buckets * width * n_codes > _MAX_BUCKET_ELEMENTS:
            return self.sinkhorn(
                centered, self.get_effective_epsilon(), self.sk_iters
            )

        # Build the (n_buckets, width) layout with one scatter instead of a
        # Python loop over buckets: the loop cost dominated everything else.
        # `within` is each row's position inside its own bucket, measured on the
        # sorted order, so the scatter below consumes the sorted rows too.
        slot = torch.arange(width, device=distances.device)
        block_index = torch.repeat_interleave(
            torch.arange(n_buckets, device=distances.device),
            counts,
        )
        starts = (torch.cumsum(counts, 0) - counts).tolist()
        sorted_position = torch.arange(n_rows, device=distances.device)
        within = slot[
            sorted_position
            - torch.as_tensor(starts, device=distances.device)[block_index]
        ]

        padded = centered.new_zeros(n_buckets, width, n_codes)
        valid = torch.zeros(
            n_buckets, width, dtype=torch.bool, device=distances.device
        )
        padded[block_index, within] = centered[order]
        valid[block_index, within] = True

        log_q = -padded / float(self.get_effective_epsilon())
        # The initial shift must ignore padding rows too, otherwise the zeros
        # they contribute leak into every real row of the block.
        initial = log_q.masked_fill(~valid.unsqueeze(-1), -torch.finfo(torch.float64).max)
        log_q = log_q - torch.logsumexp(initial, dim=(1, 2), keepdim=True)
        log_codebook_size = math.log(n_codes)
        block_rows = valid.sum(dim=1).clamp_min(1).to(torch.float64)
        big = torch.finfo(torch.float64).max
        for _ in range(self.sk_iters):
            # Column normalization per block, restricted to that block's rows.
            masked = log_q.masked_fill(~valid.unsqueeze(-1), -big)
            log_q = log_q - torch.logsumexp(masked, dim=1, keepdim=True)
            log_q = log_q - log_codebook_size
            log_q = log_q - torch.logsumexp(log_q, dim=2, keepdim=True)
            log_q = log_q - torch.log(block_rows).view(-1, 1, 1)

        # Undo the sort in one gather rather than a per-bucket scatter.
        probabilities = torch.exp(log_q)[block_index, within]
        assignment[order] = probabilities

        if not torch.isfinite(assignment).all():
            raise RuntimeError("Bucketed Sinkhorn assignment returned NaN or infinity.")
        return assignment

    def _indices(
        self,
        distances: torch.Tensor,
        infer_use_sk: bool,
        bucket: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not (self.use_sk and (self.training or infer_use_sk)):
            return torch.argmin(distances, dim=-1)
        if bucket is None:
            return self._balanced_assignments(distances).argmax(dim=-1)
        return self._bucket_balanced_assignments(distances, bucket).argmax(dim=-1)

    @torch.no_grad()
    def assignment_diagnostics(
        self, x: torch.Tensor, bucket: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = x.reshape(-1, self.dim)
        distances = self._distances(latent)
        assignments = (
            self._balanced_assignments(distances)
            if bucket is None
            else self._bucket_balanced_assignments(distances, bucket)
        )
        ids = assignments.argmax(dim=-1)
        # Shannon entropy of the assignment distribution per item. Terms with
        # zero probability contribute 0, so they are dropped rather than clamped:
        # clamping to the dtype's smallest normal makes log() return a huge
        # negative and the whole entropy becomes -inf.
        probabilities = assignments / assignments.sum(
            dim=-1, keepdim=True
        ).clamp_min(torch.finfo(assignments.dtype).tiny)
        positive = probabilities > 0
        log_probabilities = torch.where(
            positive, probabilities.log(), torch.zeros_like(probabilities)
        )
        entropy = -(
            probabilities * log_probabilities
        ).sum(dim=-1).mean()
        usage = torch.bincount(ids, minlength=self.n_embed)
        return ids.view(*x.shape[:-1]), usage, entropy

    def forward(
        self,
        x: torch.Tensor,
        infer_use_sk: bool = False,
        bucket: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        latent = x.view(-1, self.dim)
        curvature = self.get_curvature()
        embed_ind = self._indices(self._distances(latent), infer_use_sk, bucket)
        onehot = F.one_hot(embed_ind, self.n_embed)
        used = onehot.sum(0)
        if (
            distributed.is_initialized()
            and infer_use_sk is False
            and not getattr(self, "_skip_ddp_reduce", False)
        ):
            distributed.all_reduce(used, op=distributed.ReduceOp.SUM)
        unused_codes = int((used == 0).sum().item())
        x_q = F.embedding(embed_ind, self.get_code_embs()).view(x.shape)
        codebook_loss = _poincare_distance_tangent_pairs(
            x.detach(), x_q, curvature
        ).square().mean()
        commitment_loss = _poincare_distance_tangent_pairs(
            x, x_q.detach(), curvature
        ).square().mean()
        quant_loss = codebook_loss + self.beta * commitment_loss
        x_q = x + (x_q - x).detach()
        return x_q, quant_loss, unused_codes, embed_ind.view(*x.shape[:-1])

    def embed_code(self, embed_id: torch.Tensor) -> torch.Tensor:
        return F.embedding(embed_id, self.get_code_embs())

    def init_codebook(
        self,
        x: torch.Tensor,
        device: torch.device,
        bucket: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fit this level's codebook by KMeans; return the residual and the codes.

        The codes are returned so the next level can balance inside the same
        buckets instead of treating the whole batch as one pool.
        """
        kmeans = KMeans(n_clusters=self.n_embed, n_init="auto").fit(
            x.detach().cpu().numpy()
        )
        centers = torch.tensor(
            kmeans.cluster_centers_, dtype=torch.float32, device=device
        )
        if distributed.is_initialized():
            distributed.broadcast(centers, 0)
        self._copy_init_embed(centers.clone())
        embed_ind = self._indices(
            self._distances(x), infer_use_sk=True, bucket=bucket
        ).view(*x.shape[:-1])
        residual = _hyperbolic_residual(
            x, self.embed_code(embed_ind), self.get_curvature()
        )
        return residual, embed_ind




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
    """TIGER residual stack in the Poincare ball with adaptive curvature.

    The adaptive controller is a population-level feedback loop, not a second
    normalization: it reads only the median residual norm of a level and
    rescales that level's curvature so the tangent working point
    ``s = sqrt(c) * ||r||`` returns to a fixed target. It never touches an
    individual sample's magnitude, which is exactly the mechanism the pinned
    parent removes and this one deliberately keeps.
    """

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
        curvatures = [float(value) for value in config.layer_curvatures]
        if len(curvatures) != self.codebook_num:
            raise ValueError("layer_curvatures must match codebook_num")
        # Optional per-level working radius in the tangent frame, i.e. the
        # working point s = sqrt(c) * ||r|| that a level quantizes at. A level
        # with a non-zero target overwrites the magnitude of its input to
        # target / sqrt(c) before the Poincare map and maps the result back
        # afterwards, so the encoder cannot shrink ||z|| to pull the level back
        # to the ball centre the way it cancels a plain multiplicative scale.
        # 0.0 (or absent) keeps the previous code path.
        radii = getattr(config, "layer_working_radii", None)
        if radii is None:
            radii = [0.0] * self.codebook_num
        radii = [float(value) for value in radii]
        if len(radii) != self.codebook_num:
            raise ValueError("layer_working_radii must match codebook_num")
        if any(not (0.0 <= value < 1.0) for value in radii):
            raise ValueError("layer_working_radii must satisfy 0 <= r < 1")
        self.working_radii = tuple(radii)
        self.codebook_sizes = sizes
        self.vq_type = str(config.vq_type)
        self.vq_beta = float(config.beta)
        self.sk_epsilon = float(config.sk_epsilon)
        self.sk_iters = int(config.sk_iters)
        if self.vq_type != "vq":
            raise ValueError("This model requires TIGER's trainable VQ codebooks")
        self.target_s = float(getattr(config, "adaptive_target_s", 0.0))
        self.adaptive_ema_decay = float(
            getattr(config, "adaptive_ema_decay", 0.0)
        )
        self.adaptive_curv_min = float(
            getattr(config, "adaptive_curvature_min", 0.0)
        )
        self.adaptive_curv_max = float(
            getattr(config, "adaptive_curvature_max", 0.0)
        )
        if self.target_s <= 0.0:
            raise ValueError("adaptive_target_s must be positive")
        if not 0.0 <= self.adaptive_ema_decay < 1.0:
            raise ValueError("adaptive_ema_decay must satisfy 0 <= decay < 1")
        if not 0.0 < self.adaptive_curv_min < self.adaptive_curv_max:
            raise ValueError(
                "adaptive curvature bounds must satisfy 0 < min < max"
            )
        self.vq_layers = nn.ModuleList(
            [
                VQLayer(
                    codebook_size=size,
                    codebook_dim=self.codebook_dim,
                    beta=self.vq_beta,
                    sk_epsilon=self.sk_epsilon,
                    sk_iters=self.sk_iters,
                    curvature=curvatures[level],
                )
                for level, size in enumerate(self.codebook_sizes)
            ]
        )
        # Median residual norm per level, in the encoder's tangent frame. Seeded
        # from the initial curvature and target so the first retune is a no-op.
        self.register_buffer(
            "residual_norm_ema",
            torch.tensor(
                [
                    self.target_s / math.sqrt(curvatures[level])
                    for level in range(self.codebook_num)
                ],
                dtype=torch.float32,
            ),
        )
        self.last_level_residual_medians = self.residual_norm_ema.clone()

    def get_curvatures(self) -> torch.Tensor:
        return torch.stack([layer.get_curvature() for layer in self.vq_layers])

    def get_curvature_values(self) -> list[float]:
        return [float(layer.get_curvature().item()) for layer in self.vq_layers]

    def set_curvatures(self, curvatures: list[float] | tuple[float, ...]) -> None:
        if len(curvatures) != self.codebook_num:
            raise ValueError("curvatures must match codebook_num")
        for layer, value in zip(self.vq_layers, curvatures):
            layer.set_curvature(value)

    def get_residual_norm_ema(self) -> torch.Tensor:
        return self.residual_norm_ema

    @torch.no_grad()
    def observe_residual_norms(self, residual_norms: torch.Tensor) -> None:
        """Fold one retune window's median residual norms into the EMA."""
        medians = residual_norms.detach().to(
            device=self.residual_norm_ema.device,
            dtype=self.residual_norm_ema.dtype,
        )
        if not torch.isfinite(medians).all() or bool((medians <= 0.0).any()):
            raise RuntimeError(
                "Residual-norm medians must be finite and positive before the "
                "adaptive curvature update."
            )
        self.residual_norm_ema.mul_(self.adaptive_ema_decay).add_(
            medians, alpha=1.0 - self.adaptive_ema_decay
        )

    @torch.no_grad()
    def retune_curvatures(self) -> list[float]:
        """Set ``c_l = (s* / EMA[P50(||r_l||)])^2`` for every level."""
        ema = self.residual_norm_ema
        target = torch.as_tensor(
            self.target_s, device=ema.device, dtype=ema.dtype
        )
        curvatures = (target / ema).square().clamp(
            self.adaptive_curv_min, self.adaptive_curv_max
        )
        self.set_curvatures([float(value) for value in curvatures])
        return self.get_curvature_values()

    def effective_working_s(self) -> torch.Tensor:
        """The controlled quantity ``sqrt(c_l) * EMA[P50(||r_l||)]`` per level."""
        return self.get_curvatures().sqrt() * self.residual_norm_ema

    def get_effective_epsilons(self) -> list[float]:
        return [layer.get_effective_epsilon() for layer in self.vq_layers]

    def get_working_radii(self) -> tuple[float, ...]:
        return self.working_radii

    def _radius_for_level(
        self,
        level: int,
        curvature: torch.Tensor | float,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        """Tangent norm target for this level, shaped (rows, 1).

        The target is the configured working radius, so the level quantizes at a
        pinned ``s`` whatever the encoder emits. A per-bucket statistic such as
        the bucket's own mean residual norm was tried and rejected: it leaves the
        buckets exactly as spread out as they were.
        """
        sqrt_c = _curvature_like(curvature, residual).sqrt()
        return (self.working_radii[level] / sqrt_c.clamp_min(
            torch.finfo(residual.dtype).tiny
        )).to(dtype=residual.dtype).expand(residual.shape[0], 1)

    def _pin_to_radius(
        self,
        residual: torch.Tensor,
        target_norm: torch.Tensor,
    ) -> torch.Tensor:
        """Overwrite the magnitude, keep only the direction."""
        norm = torch.linalg.vector_norm(residual, dim=-1, keepdim=True)
        return residual * (target_norm / norm.clamp_min(
            torch.finfo(residual.dtype).tiny
        ))

    def _restore_norm(
        self,
        mapped: torch.Tensor,
        source: torch.Tensor,
        target_norm: torch.Tensor,
    ) -> torch.Tensor:
        """Inverse of the magnitude overwrite, back into the encoder's norms."""
        norm = torch.linalg.vector_norm(source, dim=-1, keepdim=True)
        return mapped * (norm / target_norm.clamp_min(
            torch.finfo(mapped.dtype).tiny
        ))

    def forward(
        self,
        x: torch.Tensor,
        infer_use_sk: bool = False,
    ):
        quantized_x = torch.zeros(
            x.shape[0], self.codebook_dim, device=x.device, dtype=x.dtype
        )
        sum_quant_loss: torch.Tensor | float = 0.0
        num_unused_codes = 0.0
        output = torch.empty(
            x.shape[0], self.codebook_num, dtype=torch.long, device=x.device
        )
        residual = x
        previous_codes: torch.Tensor | None = None
        # Detached per-level medians for the adaptive controller. Collected here
        # because the level's residual exists only inside this loop, and kept out
        # of the graph so the controller never back-propagates.
        level_medians: list[torch.Tensor] = []
        for level, vq_layer in enumerate(self.vq_layers):
            curvature = vq_layer.get_curvature()
            level_medians.append(
                torch.quantile(
                    torch.linalg.vector_norm(residual.detach(), dim=-1), 0.5
                )
            )
            if self.working_radii[level] == 0.0:
                quant, quant_loss, unused, indices = vq_layer(
                    residual, infer_use_sk, previous_codes
                )
                residual = _hyperbolic_residual(residual, quant, curvature)
                quantized_x = quantized_x + quant
            else:
                source = residual
                target_norm = self._radius_for_level(level, curvature, source)
                pinned = self._pin_to_radius(source, target_norm)
                quant, quant_loss, unused, indices = vq_layer(
                    pinned, infer_use_sk, previous_codes
                )
                # The code was chosen in the pinned frame, so the residual
                # subtraction and the decoder input are computed there too and
                # then mapped back into the encoder's own norms.
                residual = self._restore_norm(
                    _hyperbolic_residual(pinned, quant, curvature),
                    source, target_norm,
                )
                quantized_x = quantized_x + self._restore_norm(
                    quant, source, target_norm
                )
            previous_codes = indices
            sum_quant_loss = sum_quant_loss + quant_loss
            num_unused_codes += unused
            output[:, level] = indices
        self.last_level_residual_medians = torch.stack(level_medians)
        return (
            quantized_x,
            sum_quant_loss / self.codebook_num,
            num_unused_codes,
            output,
        )

    @torch.no_grad()
    def get_indices_with_stats(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        tokens = torch.empty(
            x.shape[0], self.codebook_num, dtype=torch.long, device=x.device
        )
        residual = x
        stats = []
        previous_codes: torch.Tensor | None = None
        for level, layer in enumerate(self.vq_layers):
            curvature = layer.get_curvature()
            natural_norm = torch.linalg.vector_norm(residual, dim=-1)
            working_s = curvature.sqrt() * natural_norm
            ball_radius = torch.tanh(working_s)
            quantiles = residual.new_tensor((0.25, 0.50, 0.75, 0.95, 0.99))
            natural_norm_quantiles = torch.quantile(natural_norm, quantiles)
            working_s_quantiles = torch.quantile(working_s, quantiles)
            ball_radius_quantiles = torch.quantile(ball_radius, quantiles)
            if self.working_radii[level] == 0.0:
                indices, usage, entropy = layer.assignment_diagnostics(
                    residual, previous_codes
                )
                residual = _hyperbolic_residual(
                    residual, layer.embed_code(indices), curvature
                )
            else:
                source = residual
                target_norm = self._radius_for_level(level, curvature, source)
                pinned = self._pin_to_radius(source, target_norm)
                indices, usage, entropy = layer.assignment_diagnostics(
                    pinned, previous_codes
                )
                residual = self._restore_norm(
                    _hyperbolic_residual(
                        pinned, layer.embed_code(indices), curvature
                    ),
                    source, target_norm,
                )
            tokens[:, level] = indices
            previous_codes = indices
            stats.append(
                {
                    "usage_counts": usage,
                    "assignment_entropy_nats": entropy,
                    "curvature": layer.get_curvature().detach(),
                    "effective_epsilon": torch.tensor(
                        layer.get_effective_epsilon(),
                        device=x.device,
                        dtype=x.dtype,
                    ),
                    "natural_residual_norm_quantiles": natural_norm_quantiles,
                    "working_s_quantiles": working_s_quantiles,
                    "ball_radius_quantiles": ball_radius_quantiles,
                }
            )
        return tokens, stats

    def init_codebook(self, x: torch.Tensor, device: torch.device) -> torch.Tensor:
        residual = x
        previous_codes: torch.Tensor | None = None
        for level, vq_layer in enumerate(self.vq_layers):
            curvature = vq_layer.get_curvature()
            if self.working_radii[level] == 0.0:
                residual, previous_codes = vq_layer.init_codebook(
                    residual, device, previous_codes
                )
                continue
            # This level's frame has a pinned tangent norm, so the codebook is
            # fitted on the pinned residuals and the residual handed to the next
            # level is mapped back to the encoder's norms.
            source = residual
            target_norm = self._radius_for_level(level, curvature, source)
            pinned = self._pin_to_radius(source, target_norm)
            fitted, previous_codes = vq_layer.init_codebook(
                pinned, device, previous_codes
            )
            residual = self._restore_norm(fitted, source, target_norm)
        return residual
