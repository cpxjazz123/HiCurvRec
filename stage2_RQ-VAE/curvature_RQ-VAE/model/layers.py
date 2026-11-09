"""RQ-VAE layers migrated from RecBole3.0.

The implementation retains TIGER's model, codebook initialization, balanced
geodesic Sinkhorn assignment, and straight-through losses. Each VQ layer uses
cyclic learnable Poincare curvature with origin-tangent code vectors/residuals.
The HG-Rec runner consumes generated integer SIDs without changing T5.
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


def _pairwise_euclid_distance_tangents(
    left: torch.Tensor,
    right: torch.Tensor,
    curvature: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Flat limit of the pairwise ball distance."""
    del curvature
    return torch.cdist(left, right)


def _euclid_distance_tangent_pairs(
    left: torch.Tensor,
    right: torch.Tensor,
    curvature: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Flat limit of the geodesic distance between tangent vectors."""
    del curvature
    return torch.linalg.vector_norm(left - right, dim=-1)


def _euclid_residual(
    residual_tangent: torch.Tensor,
    code_tangent: torch.Tensor,
    curvature: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Flat limit of the Mobius subtraction: plain tangent subtraction."""
    del curvature
    return residual_tangent - code_tangent


def _resolve_geometry(geometry: str):
    """Map a geometry name onto (pairwise, pairs, residual).

    The curvature argument is threaded through uniformly so the call sites do
    not branch; the flat limit simply ignores it.
    """
    if geometry == "poincare":
        return (
            _pairwise_poincare_distance_tangents,
            _poincare_distance_tangent_pairs,
            _hyperbolic_residual,
        )
    if geometry == "euclid":
        return (
            _pairwise_euclid_distance_tangents,
            _euclid_distance_tangent_pairs,
            _euclid_residual,
        )
    raise ValueError(f"Unknown geometry: {geometry}")


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
    """Poincare VQ at a fixed curvature with balanced Sinkhorn assignment."""

    def __init__(
        self,
        codebook_size: int,
        codebook_dim: int,
        beta: float = 0.25,
        sk_epsilon: float = 0.003,
        sk_iters: int = 50,
        curvature: float = 1.0,
        geometry: str = "poincare",
        assignment_mode: str = "bucket",
        smoothness_weight: float = 0.0,
        smoothness_scope: str = "global",
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
        self.curvature = float(curvature)
        self.geometry = str(geometry)
        if str(assignment_mode) not in ("bucket", "global", "argmin"):
            raise ValueError(
                "assignment_mode must be 'bucket', 'global' or 'argmin', got "
                f"{assignment_mode!r}"
            )
        self.assignment_mode = str(assignment_mode)
        if smoothness_weight < 0.0:
            raise ValueError("smoothness_weight must not be negative.")
        self.smoothness_weight = float(smoothness_weight)
        if str(smoothness_scope) not in ("global", "prefix"):
            raise ValueError(
                "smoothness_scope must be 'global' or 'prefix', got "
                f"{smoothness_scope!r}"
            )
        # "global" draws pairs from the whole batch; "prefix" draws them from
        # inside one preceding-code bucket, so the term constrains the levels
        # that carry a hierarchy instead of the batch at large.
        self.smoothness_scope = str(smoothness_scope)
        (
            self._pairwise_fn,
            self._pair_fn,
            self._residual_fn,
        ) = _resolve_geometry(self.geometry)
        self.embed = nn.Embedding(self.n_embed, self.dim)

    def get_code_embs(self) -> nn.Parameter:
        return self.embed.weight

    def _copy_init_embed(self, init_embed: torch.Tensor) -> None:
        self.embed.weight.data.copy_(init_embed)

    def get_curvature(self) -> torch.Tensor:
        return self.embed.weight.new_tensor(self.curvature)

    def get_effective_epsilon(self) -> float:
        return self.sk_epsilon

    @staticmethod
    def center_distance(
        distances: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Scale the distance matrix into [0, 1] and hand back the amplitude.

        The degeneracy check is left to the caller because reading the
        amplitude is a host sync: the bucketed assignment already reads one
        device scalar per level and validates the amplitude in that same read.
        """
        max_distance = distances.max()
        min_distance = distances.min()
        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        return (distances - middle) / amplitude, amplitude

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
        return self._pairwise_fn(
            latent, self.get_code_embs(), self.get_curvature()
        )

    def _balanced_assignments(self, distances: torch.Tensor) -> torch.Tensor:
        centered, amplitude = self.center_distance(distances)
        if not bool(amplitude > 0):
            raise ValueError("Cannot center a constant distance matrix.")
        assignments = self.sinkhorn(
            centered.double(), self.get_effective_epsilon(), self.sk_iters
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

        # Group rows by bucket in one pass: sort once, then split by counts.
        order = torch.argsort(bucket, stable=True)
        # bincount already returns max(bucket) + 1 entries, so the bucket count
        # is a shape on the host and not another device read.
        counts = torch.bincount(bucket)
        n_buckets = counts.numel()
        centered, amplitude = self.center_distance(distances)
        centered = centered.double()
        assignment = torch.zeros_like(centered)

        # The only host read this level needs: the widest bucket, which sizes
        # the padded layout, and the centering amplitude, which is the
        # degeneracy guard. Both are read together so the level costs one sync
        # per step instead of four.
        amplitude_value, width_value = torch.stack(
            (amplitude, counts.max().to(amplitude.dtype))
        ).tolist()
        if not amplitude_value > 0:
            raise ValueError("Cannot center a constant distance matrix.")
        width = int(width_value)
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
        # The bucket starts stay on the device: they are only ever used as a
        # gather index, so reading them back to the host and straight back to
        # the device was a sync that bought nothing.
        starts = torch.cumsum(counts, 0) - counts
        sorted_position = torch.arange(n_rows, device=distances.device)
        within = slot[sorted_position - starts[block_index]]

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
        """Pick one code per row.

        ``bucket`` balances codes inside each preceding-code bucket, ``global``
        balances over the whole batch ignoring the preceding code, and
        ``argmin`` takes the nearest code with no balancing at all. The bucket
        mode spreads the items of one preceding code across as many codes as it
        can, which is the opposite of what shared semantic prefixes need.
        """
        if self.assignment_mode == "argmin":
            return torch.argmin(distances, dim=-1)
        if not (self.use_sk and (self.training or infer_use_sk)):
            return torch.argmin(distances, dim=-1)
        if self.assignment_mode == "global" or bucket is None:
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

    def _prefix_partners(
        self, bucket: torch.Tensor, count: int, device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pair every row with the next row inside its own bucket, cyclically.

        One stable sort groups the rows, and a cyclic shift inside each group
        gives every row a partner that shares its preceding code. The loss is a
        mean, so computing it in the grouped order changes nothing.
        """
        order = torch.argsort(bucket, stable=True)
        counts = torch.bincount(bucket, minlength=1)
        starts = torch.cumsum(counts, 0) - counts
        repeated_starts = torch.repeat_interleave(starts, counts)
        repeated_counts = torch.repeat_interleave(counts, counts)
        local = torch.arange(order.numel(), device=device) - repeated_starts
        partner = repeated_starts + (local + 1) % repeated_counts
        return order, partner

    def _smoothness_loss(
        self,
        latent: torch.Tensor,
        quantized: torch.Tensor,
        bucket: torch.Tensor | None = None,
        pairs: int = 64,
    ) -> torch.Tensor:
        """Keep the quantiser a smooth function of the input, in the arm's metric.

        Stage3 has to learn content -> SID from the item's text embedding, and the
        SID is the quantiser's output. Nothing about the partition changes here:
        the term only asks that two latents the metric calls close are not sent to
        code vectors the metric calls far apart, which is what makes that map
        learnable. The distances are the arm's own, so in the hyperbolic arm the
        requirement is expressed in hyperbolic units and grows where the metric
        grows, while in the euclidean arm it is the plain Euclidean statement.
        """
        if self.smoothness_weight <= 0.0 or latent.shape[0] < 2:
            return latent.sum() * 0.0
        curvature = self.get_curvature()
        count = latent.shape[0]
        if self.smoothness_scope == "prefix" and bucket is not None:
            order, partner_index = self._prefix_partners(
                bucket.reshape(-1), count, latent.device
            )
            grouped = latent[order]
            grouped_q = quantized[order]
            source = grouped
            partner = grouped[partner_index]
            quantized = grouped_q
            quantized_partner = grouped_q[partner_index]
            latent_distance = self._pair_fn(source, partner, curvature).square()
            quantized_distance = self._pair_fn(
                quantized, quantized_partner, curvature
            ).square()
            with torch.no_grad():
                scale = latent_distance.median().clamp_min(1e-8)
            weight = torch.exp(-latent_distance / scale).detach()
            return (weight * quantized_distance).mean()
        pairs = min(int(pairs), count - 1)
        index = torch.randint(0, count, (count, pairs), device=latent.device)
        source = latent.unsqueeze(1).expand(count, pairs, latent.shape[-1])
        partner = latent[index]
        latent_distance = self._pair_fn(source, partner, curvature).square()
        quantized_distance = self._pair_fn(
            quantized.unsqueeze(1).expand(count, pairs, quantized.shape[-1]),
            quantized[index],
            curvature,
        ).square()
        with torch.no_grad():
            scale = latent_distance.median().clamp_min(1e-8)
        weight = torch.exp(-latent_distance / scale).detach()
        return (weight * quantized_distance).mean()

    def forward(
        self,
        x: torch.Tensor,
        infer_use_sk: bool = False,
        bucket: torch.Tensor | None = None,
        return_code: bool = False,
    ):
        latent = x.view(-1, self.dim)
        curvature = self.get_curvature()
        # The assignment is a discrete argmax, so nothing can flow back through
        # the distance matrix: computing it outside autograd drops its graph
        # without touching a single assigned code. The per-step cross-rank
        # reduction that used to sit here only fed an `unused_codes` counter
        # nobody read; the real usage counts come from the diagnostic pass.
        with torch.no_grad():
            embed_ind = self._indices(self._distances(latent), infer_use_sk, bucket)
        x_q = F.embedding(embed_ind, self.get_code_embs()).view(x.shape)
        codebook_loss = self._pair_fn(
            x.detach(), x_q, curvature
        ).square().mean()
        commitment_loss = self._pair_fn(
            x, x_q.detach(), curvature
        ).square().mean()
        quant_loss = codebook_loss + self.beta * commitment_loss
        if self.smoothness_weight > 0.0:
            quant_loss = quant_loss + self.smoothness_weight * self._smoothness_loss(
                latent, x_q.view(-1, self.dim), bucket
            )
        # The straight-through output is what the decoder consumes, and it
        # deliberately detaches the codebook: gradients reach the codebook only
        # through quant_loss. ``x_q_code`` carries the same forward value with
        # the codebook path intact, so a supervision term placed on it actually
        # updates the codebooks instead of only the encoder.
        code_output = x_q
        x_q = x + (x_q - x).detach()
        if return_code:
            return x_q, quant_loss, embed_ind.view(*x.shape[:-1]), code_output
        return x_q, quant_loss, embed_ind.view(*x.shape[:-1])

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
        residual = self._residual_fn(
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = x.view(-1, self.dim)
        with torch.no_grad():
            embed_ind = self._indices(self._distances(latent), infer_use_sk)
        x_q = F.embedding(embed_ind, self.get_code_embs()).view(x.shape)

        if self.training:
            onehot = F.one_hot(embed_ind, self.n_embed).type(latent.dtype)
            used = onehot.sum(0)
            embed_sum = onehot.t() @ latent
            if distributed.is_initialized():
                distributed.all_reduce(used, op=distributed.ReduceOp.SUM)
                distributed.all_reduce(embed_sum, op=distributed.ReduceOp.SUM)
            self.cluster_size.data.mul_(self.decay).add_(used, alpha=1 - self.decay)
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            norm_w = n * (self.cluster_size + self.eps) / (n + self.n_embed * self.eps)
            self.embed.data.copy_(self.embed_avg / norm_w.unsqueeze(1))

        quant_loss = self.beta * F.mse_loss(x, x_q.detach())
        x_q = x + (x_q - x).detach()
        return x_q, quant_loss, embed_ind.view(*x.shape[:-1])


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
    """TIGER residual stack in the Poincare ball at a fixed curvature."""

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
        # Whether the per-level pin is expressed in metric working-point
        # coordinates (s = sqrt(c) * ||v||, curvature-compensated so s is
        # constant across curvatures) or as a plain tangent norm (||v|| fixed,
        # so s = sqrt(c) * radius grows with c). True is the frozen protocol's
        # definition and the default; False is opt-in and is the only way to
        # change curvature without the depth moving to compensate.
        self.pin_in_s_coordinates = bool(
            getattr(config, "pin_in_s_coordinates", True)
        )
        self.codebook_sizes = sizes
        # Geometry plug-in: "poincare" is the frozen protocol, "euclid" is its
        # flat limit. Nothing else about the layer changes between them.
        self.geometry = str(getattr(config, "geometry", "poincare"))
        # Per-level assignment rule. The default is the frozen protocol; the
        # second level can be moved to global or nearest-only balancing to test
        # whether bucket balancing is what keeps prefixes item-unique.
        modes = getattr(config, "layer_assignment_modes", None)
        if modes is None:
            modes = ["bucket"] * self.codebook_num
        if len(modes) != self.codebook_num:
            raise ValueError(
                "layer_assignment_modes must have one entry per quantization level"
            )
        assignment_modes = [str(mode) for mode in modes]
        self.vq_type = str(config.vq_type)
        self.vq_beta = float(config.beta)
        self.sk_epsilon = float(config.sk_epsilon)
        self.sk_iters = int(config.sk_iters)
        # A per-level weight lets the same term be applied to one level only, which
        # is what separates "smoothing the prefix" from "smoothing the level that
        # disambiguates"; the scalar stays the default so existing runs are
        # unchanged.
        per_level = getattr(config, "smoothness_weights", None)
        if per_level is None:
            per_level = (getattr(config, "smoothness_weight", 0.0),) * self.codebook_num
        if len(per_level) != self.codebook_num:
            raise ValueError("smoothness_weights must have one entry per level")
        self.smoothness_weights = tuple(float(value) for value in per_level)
        self.smoothness_weight = self.smoothness_weights[0]
        scope = str(getattr(config, "smoothness_scope", "global"))
        if self.vq_type != "vq":
            raise ValueError("This model requires TIGER's trainable VQ codebooks")
        self.vq_layers = nn.ModuleList(
            [
                VQLayer(
                    codebook_size=size,
                    codebook_dim=self.codebook_dim,
                    beta=self.vq_beta,
                    sk_epsilon=self.sk_epsilon,
                    sk_iters=self.sk_iters,
                    curvature=curvatures[level],
                    geometry=self.geometry,
                    assignment_mode=assignment_modes[level],
                    smoothness_weight=self.smoothness_weights[level],
                    smoothness_scope=scope,
                )
                for level, size in enumerate(self.codebook_sizes)
            ]
        )

    def get_curvatures(self) -> torch.Tensor:
        return torch.stack([layer.get_curvature() for layer in self.vq_layers])

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

        With ``pin_in_s_coordinates`` (the default, and the frozen protocol),
        the target is scaled by 1/sqrt(c) so the metric working point
        s = sqrt(c) * ||v|| stays constant across curvatures. That makes a
        curvature change a pure geometry change: the depth in the ball is held
        fixed and only the metric factor moves.

        With the flag off, ||v|| is held at the configured radius instead, so
        s = sqrt(c) * radius grows with c. The depth and the metric factor then
        both move, which is why it is opt-in rather than a consequence of the
        constant.
        """
        radius = self.working_radii[level]
        if self.geometry == "euclid":
            # No conformal factor in the flat limit, so the tangent norm itself
            # is the working point. With unit curvatures this is the same 0.2 the
            # Poincare arm pins to, so only the metric differs between arms.
            return torch.full_like(residual[:, :1], radius)
        if not self.pin_in_s_coordinates:
            target = torch.full_like(residual[:, :1], radius)
        else:
            sqrt_c = _curvature_like(curvature, residual).sqrt()
            target = (
                radius / sqrt_c.clamp_min(torch.finfo(residual.dtype).tiny)
            ).to(dtype=residual.dtype)
        return target.expand(residual.shape[0], 1)

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
        return_prefixes: bool = False,
    ):
        """Quantize residually.

        With ``return_prefixes`` the per-level running sums are returned as an
        extra list. Entry ``l`` is the representation the decoder would receive
        if it stopped after level ``l``: it is the same accumulation the model
        feeds the decoder, not a sum of raw codebook vectors. The last entry is
        therefore identical to the returned ``quantized_x``, and the list
        carries gradient to exactly the codebooks that formed it.
        """
        quantized_x = torch.zeros(
            x.shape[0], self.codebook_dim, device=x.device, dtype=x.dtype
        )
        # Supervision path: identical forward value, codebooks in the graph.
        code_prefix: torch.Tensor | None = (
            torch.zeros_like(quantized_x) if return_prefixes else None
        )
        prefixes: list[torch.Tensor] = []
        sum_quant_loss: torch.Tensor | float = 0.0
        output = torch.empty(
            x.shape[0], self.codebook_num, dtype=torch.long, device=x.device
        )
        residual = x
        previous_codes: torch.Tensor | None = None
        for level, vq_layer in enumerate(self.vq_layers):
            curvature = vq_layer.get_curvature()
            if self.working_radii[level] == 0.0:
                # return_code only aliases the pre-straight-through tensor,
                # so asking for it always costs nothing and keeps both paths on
                # one code path.
                quant, quant_loss, indices, code_quant = vq_layer(
                    residual, infer_use_sk, previous_codes, return_code=True
                )
                residual = vq_layer._residual_fn(residual, quant, curvature)
                quantized_x = quantized_x + quant
                if return_prefixes:
                    code_prefix = code_prefix + code_quant
            else:
                source = residual
                target_norm = self._radius_for_level(level, curvature, source)
                pinned = self._pin_to_radius(source, target_norm)
                quant, quant_loss, indices, code_quant = vq_layer(
                    pinned, infer_use_sk, previous_codes, return_code=True
                )
                # The code was chosen in the pinned frame, so the residual
                # subtraction and the decoder input are computed there too and
                # then mapped back to the encoder's own norms.
                residual = self._restore_norm(
                    vq_layer._residual_fn(pinned, quant, curvature),
                    source, target_norm,
                )
                quantized_x = quantized_x + self._restore_norm(
                    quant, source, target_norm
                )
                if return_prefixes:
                    code_prefix = code_prefix + self._restore_norm(
                        code_quant, source, target_norm
                    )
            previous_codes = indices
            sum_quant_loss = sum_quant_loss + quant_loss
            output[:, level] = indices
            if return_prefixes:
                prefixes.append(code_prefix)
        if return_prefixes:
            return quantized_x, sum_quant_loss / self.codebook_num, output, prefixes
        return quantized_x, sum_quant_loss / self.codebook_num, output

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
            if self.working_radii[level] == 0.0:
                indices, usage, entropy = layer.assignment_diagnostics(
                    residual, previous_codes
                )
                residual = layer._residual_fn(
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
                    layer._residual_fn(
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
