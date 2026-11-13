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

# The losses are their own modules; a quantiser reports the two errors it measured
# and never builds the scalar itself.
from losses.codebook import codebook_loss
from losses.commitment import commitment_loss

# The metric is its own module; the quantiser only calls what it was handed.
from distances import build_distance


#: The tree's ONE geometry implementation, imported lazily so that a candidate tree
#: whose geometry module is missing fails where it is used, not at import time.
_GEOMETRY = None


def geometry_module():
    """Return the top-level `factor_share_geometry` module, or fail loud."""
    global _GEOMETRY
    if _GEOMETRY is None:
        try:
            import factor_share_geometry as module
        except ImportError as error:  # pragma: no cover - depends on sys.path
            raise RuntimeError(
                "the product-manifold geometry module factor_share_geometry is not "
                "importable from the quantiser (%s); run from the candidate tree" % (error,)
            ) from error
        _GEOMETRY = module
    return _GEOMETRY


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
        distance=None,
    ):
        super().__init__()
        self.dim = int(codebook_dim)
        self.n_embed = int(codebook_size)
        self.beta = float(beta)
        self.use_sk = sk_epsilon > 0 and sk_iters > 0
        self.sk_epsilon = float(sk_epsilon)
        self.sk_iters = int(sk_iters)
        # The metric is injected; a quantiser without one falls back to the registry's
        # default so existing callers (a probe, a framework check) keep working.
        self.distance = distance if distance is not None else build_distance()
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

    def _distances(self, latent, p=None, mode: str = "both_on") -> torch.Tensor:
        """The metric is the object handed in; this layer owns no formula.

        Every registry entry is called with the same positional shape
        `(latent, codebook, p, mode)`: the Euclidean entry ignores the last two, and
        the product entry uses them. `latent` is either the 32-vector
        `(r^E | h1 | h2)` or the already charted triple `(r^E, x_hi, x_lo)`.
        """
        return self.distance(latent, self.get_code_embs(), p, mode)

    def _product_blocks(self, latent: torch.Tensor, x_hi, x_lo, mode: str):
        """The distance argument: the charted triple when we have one.

        `geometry_off` is the tangent branch by definition, so it keeps the 32-vector.
        """
        if x_hi is None or x_lo is None or mode == "geometry_off":
            return latent
        return (latent[:, :geometry_module().D_E], x_hi, x_lo)

    def _geometric_parts(self, latent, x_hi, x_lo, p, mode: str, index):
        """The two geometric errors of `quantization`: E side, hyperbolic side, split."""
        geometry = geometry_module()
        rE, _, _ = geometry.split_tangent(latent)
        if x_hi is None or x_lo is None:
            x_hi, x_lo = geometry.item_chart_head(latent)
        values = geometry.share(p, "ref" if mode == "behaviour_off" else "struct")
        codebook = self.get_code_embs()
        v_hi, v_lo = geometry.codeword_chart(codebook, values)
        return geometry.quant_parts(rE, x_hi, x_lo, v_hi, v_lo,
                                   codebook[:, :geometry.D_E], index)

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
        p=None,
        mode: str = "both_on",
        x_hi=None,
        x_lo=None,
        infer_use_sk: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        """One level of the product manifold.

        `x` is `(r^E | h1 | h2)`; `p` is the per-item share `p_i` (mandatory for the
        product metric); `x_hi`/`x_lo` are the charted product point when the caller
        already ran the chart head. `x` itself is never updated here -- only
        `RQLayer` advances the Euclidean block.
        """
        latent = x.view(-1, self.dim)
        embed_ind = self._indices(
            self._distances(self._product_blocks(latent, x_hi, x_lo, mode), p, mode),
            infer_use_sk,
        )
        onehot = F.one_hot(embed_ind, self.n_embed)
        used = onehot.sum(0)
        # Skip the DDP all_reduce when running inference on rank 0 only:
        # the unused-code accounting is a per-rank stat, and a single-rank
        # all_reduce would deadlock waiting for the other 3 ranks to join.
        if distributed.is_initialized() and infer_use_sk is False and not getattr(self, "_skip_ddp_reduce", False):
            distributed.all_reduce(used, op=distributed.ReduceOp.SUM)
        unused_codes = int((used == 0).sum().item())

        x_q = F.embedding(embed_ind, self.get_code_embs()).view(x.shape)
        if p is None or mode == "geometry_off":
            # The baseline's two terms, unchanged: geometry_off has to be elementwise
            # identical to the frozen Euclidean RQ-VAE.
            quant_parts = (commitment_loss(latent, x_q), codebook_loss(latent, x_q))
        else:
            # The formula lives in factor_share_geometry.quant_parts; this layer only
            # reports the two errors it measured.
            quant_parts = self._geometric_parts(latent, x_hi, x_lo, p, mode, embed_ind)
        x_q = x + (x_q - x).detach()
        return x_q, quant_parts, unused_codes, embed_ind.view(*x.shape[:-1])

    def embed_code(self, embed_id: torch.Tensor) -> torch.Tensor:
        return F.embedding(embed_id, self.get_code_embs())

    def init_codebook(self, x: torch.Tensor, device: torch.device, p=None,
                      mode: str = "both_on", x_hi=None, x_lo=None) -> torch.Tensor:
        """KMeans centres, but the assignment that builds the residual uses the SAME D."""
        kmeans = KMeans(n_clusters=self.n_embed, n_init="auto").fit(x.detach().cpu().numpy())
        centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device)
        if distributed.is_initialized():
            distributed.broadcast(centers, 0)
        self._copy_init_embed(centers.clone())
        latent = x.view(-1, self.dim)
        embed_ind = torch.argmin(
            self._distances(self._product_blocks(latent, x_hi, x_lo, mode), p, mode),
            dim=-1,
        ).view(*x.shape[:-1])
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
        distance=None,
    ):
        super().__init__(codebook_size, codebook_dim, beta, sk_epsilon, sk_iters,
                         distance=distance)
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

        # The EMA variant has no gradient term for the entries: they move by the moving
        # average, so only the commitment term is scored, and its weight is beta.
        quant_parts = (commitment_loss(latent, x_q), None)
        x_q = x + (x_q - x).detach()
        return x_q, quant_parts, unused_codes, embed_ind.view(*x.shape[:-1])


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
        distance=None,
    ):
        super().__init__(codebook_size, codebook_dim, beta, sk_epsilon, sk_iters,
                         distance=distance)
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

    def __init__(self, config: Any, distance=None):
        super().__init__()
        self.config = config
        # One metric object for the whole residual stack; the levels share it.
        self.distance = distance if distance is not None else build_distance()
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
        # The frozen config says 'vq'. EMA / SimVQ do not implement the product-side
        # chart, so they would silently drop the mechanism: fail loud instead.
        if self.vq_type != "vq":
            raise ValueError(
                "the factor_share arm requires vq_type='vq' (the frozen config's value); "
                "got %r -- EMA/SimVQ have no product-side geometry and would train the "
                "baseline silently" % (self.vq_type,)
            )
        self.vq_beta = float(config.beta)
        self.sk_epsilon = float(config.sk_epsilon)
        self.sk_iters = int(config.sk_iters)

        layers: list[nn.Module] = []
        for level, size in enumerate(self.codebook_sizes):
            epsilon = self.sk_epsilon if level == self.codebook_num - 1 else -1.0
            iterations = self.sk_iters if level == self.codebook_num - 1 else -1
            layer = VQLayer(size, self.codebook_dim, self.vq_beta, epsilon, iterations,
                            distance=self.distance)
            layers.append(layer)
        self.vq_layers = nn.ModuleList(layers)

    def _blocks(self, flat: torch.Tensor, p, mode: str, x_hi, x_lo):
        """Split the 32-vector into the three blocks of `M`.

        Returns `(geometric, rE, h1, h2, x_hi, x_lo)`. `geometric` is False for the
        `geometry_off` branch (and for a caller that passes no share at all), where the
        frozen baseline's single 32-dim residual chain is kept verbatim.
        """
        geometry = geometry_module()
        geometric = p is not None and mode != "geometry_off"
        if not geometric:
            return (False, None, None, None, x_hi, x_lo)
        rE = flat[:, :geometry.D_E]
        h1 = flat[:, geometry.D_E:geometry.D_E + geometry.D_H1]
        h2 = flat[:, geometry.D_E + geometry.D_H1:]
        if x_hi is None or x_lo is None:
            x_hi, x_lo = geometry.item_chart_head(flat)
        return (True, rE, h1, h2, x_hi, x_lo)

    def forward(
        self,
        x: torch.Tensor,
        p=None,
        mode: str = "both_on",
        x_hi=None,
        x_lo=None,
        infer_use_sk: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
        """Quantize `(r^{E,l} | h1 | h2)`, advancing ONLY the Euclidean block.

        The two hyperbolic blocks are one product point each, reused at every level
        (`math.json` `geometric_quantization`): they are never part of the residual.
        """
        geometry = geometry_module()
        flat = x.view(-1, self.codebook_dim)
        geometric, rE, h1, h2, x_hi, x_lo = self._blocks(flat, p, mode, x_hi, x_lo)
        quantized_x = torch.zeros(flat.shape[0], self.codebook_dim, device=flat.device)
        per_level_parts: list = []
        num_unused_codes = 0.0
        output = torch.empty(flat.shape[0], self.codebook_num, dtype=torch.long,
                             device=flat.device)
        residual = flat
        for level, vq_layer in enumerate(self.vq_layers):
            level_input = torch.cat([rE, h1, h2], dim=-1) if geometric else residual
            quant, quant_parts, unused, indices = vq_layer(
                level_input, p=p, mode=mode, x_hi=x_hi, x_lo=x_lo,
                infer_use_sk=infer_use_sk,
            )
            if geometric:
                # only the Euclidean block carries the level's residual
                rE = rE - quant[:, :geometry.D_E]
            else:
                residual = residual - quant
            quantized_x = quantized_x + quant
            per_level_parts.append(quant_parts)
            num_unused_codes += unused
            output[:, level] = indices
        # The levels are reported, not summed: the scalar belongs to the loss module.
        return quantized_x, per_level_parts, num_unused_codes, output

    def init_codebook(self, x: torch.Tensor, device: torch.device, p=None,
                      mode: str = "both_on", x_hi=None, x_lo=None) -> torch.Tensor:
        geometry = geometry_module()
        flat = x.view(-1, self.codebook_dim)
        geometric, rE, h1, h2, x_hi, x_lo = self._blocks(flat, p, mode, x_hi, x_lo)
        residual = flat
        for vq_layer in self.vq_layers:
            level_input = torch.cat([rE, h1, h2], dim=-1) if geometric else residual
            updated = vq_layer.init_codebook(level_input, device, p=p, mode=mode,
                                             x_hi=x_hi, x_lo=x_lo)
            if geometric:
                rE = updated[:, :geometry.D_E]
            else:
                residual = updated
        return torch.cat([rE, h1, h2], dim=-1) if geometric else residual
