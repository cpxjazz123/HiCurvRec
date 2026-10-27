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


def fine_quantize(vq_layer: nn.Module, residual: torch.Tensor, anchors: torch.Tensor,
                  curvature: float, geometry_on: bool = True,
                  use_sinkhorn: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign a code to every residual and return the ambient contribution.

    With the geometry on, the residual is written in the tangent space of its
    anchor, the code vectors are mapped into the same tangent space, and the
    reconstruction is the exponential map of the selected code; the
    contribution is its displacement from the anchor. With the geometry off the
    residual is shifted by the anchor and quantized in the ordinary Euclidean
    way, which is the control the specification asks for.
    """
    from .hyperbolic import _sinkhorn_indices, tangent_quantize

    codes = vq_layer.get_code_embs()
    if geometry_on:
        indices, contribution, reconstructed = tangent_quantize(
            residual, codes, anchors, curvature, use_sinkhorn=use_sinkhorn,
            sk_epsilon=float(vq_layer.sk_epsilon), sk_iters=int(vq_layer.sk_iters))
        return indices, contribution, reconstructed
    shifted = residual - anchors
    distances = (shifted.square().sum(-1, keepdim=True)
                 - 2.0 * (shifted @ codes.t())
                 + codes.square().sum(-1).unsqueeze(0)).clamp_min(0.0)
    if use_sinkhorn:
        indices = _sinkhorn_indices(distances, float(vq_layer.sk_epsilon),
                                     int(vq_layer.sk_iters))
    else:
        indices = torch.argmin(distances, dim=-1)
    # The Euclidean counterpart: the contribution is the selected code, and the
    # reconstruction is the residual shifted by the anchor with the selected
    # code subtracted in that frame, so the anchor enters the reconstruction
    # and not only the assignment.
    return indices, F.embedding(indices, codes), shifted - F.embedding(indices, codes)


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
        fine_context: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
        quantized_x = torch.zeros(x.shape[0], self.codebook_dim, device=x.device)
        sum_quant_loss: torch.Tensor | float = 0.0
        num_unused_codes = 0.0
        output = torch.empty(x.shape[0], self.codebook_num, dtype=torch.long, device=x.device)
        residual = x
        for level, vq_layer in enumerate(self.vq_layers):
            if fine_context is not None and level >= 1:
                if level == 1:
                    # The anchors need the coarse residual, so they are built
                    # here and not before the stack.
                    fine_context = self._prepare_anchors(residual, fine_context)
                (quant, quant_loss, unused, indices,
                 contribution, reconstruction) = self._fine_forward(
                    vq_layer, residual, level, fine_context, infer_use_sk)
                # The residual update of the specification: the selected code is
                # subtracted in the tangent space of the anchor and the
                # exponential map is applied, so the reconstruction is the next
                # residual. The coarse level keeps the baseline's subtraction.
                residual = reconstruction
            else:
                quant, quant_loss, unused, indices = vq_layer(residual, infer_use_sk)
                residual = residual - quant
            quantized_x = quantized_x + quant
            sum_quant_loss = sum_quant_loss + quant_loss
            num_unused_codes += unused
            output[:, level] = indices
        return quantized_x, sum_quant_loss / self.codebook_num, num_unused_codes, output

    def _prepare_anchors(self, residual: torch.Tensor, context: dict) -> dict:
        """The branch anchors of the fine levels.

        The anchor of a branch is the Frechet mean of the residuals of the
        successors of that branch's items, taken over the training edges of the
        current batch and weighted by the edge counts. The successors need
        their own residual, so their coarse-level quantization is computed here
        as well; it is a context, not a training signal, and carries no
        gradient.
        """
        from .hyperbolic import branch_frechet_means

        curvature = float(context["curvature"])
        geometry_on = bool(context.get("geometry_on", True))
        behavior_on = bool(context.get("behavior_on", True))
        branches = context["branch_ids"]
        edge_sources = context["edge_sources"]
        edge_targets = context["edge_targets"]
        edge_weights = context["edge_weights"]
        target_branches = context["target_branches"]
        n_branches = int(context["n_branches"])
        with torch.no_grad():
            target_encoded = context["target_encoded"]
            target_quant, _, _, _ = self.vq_layers[0](target_encoded, False)
            target_residual = target_encoded - target_quant
        if behavior_on:
            source_branch = branches[edge_sources]
            target_branch = target_branches
        else:
            # The behaviour-off control of the specification: one global branch.
            source_branch = torch.zeros_like(edge_sources)
            target_branch = torch.zeros_like(target_branches)
        first = branch_frechet_means(target_residual[edge_targets], source_branch,
                                     edge_weights, n_branches, curvature, 3,
                                     geometry_on=geometry_on)
        with torch.no_grad():
            target_anchor = first[target_branch]
            _, _target_contribution, target_reconstruction = fine_quantize(
                self.vq_layers[1], target_residual, target_anchor, curvature,
                geometry_on=geometry_on, use_sinkhorn=False)
            target_residual_next = target_reconstruction
        second = branch_frechet_means(target_residual_next[edge_targets], source_branch,
                                      edge_weights, n_branches, curvature, 3,
                                      geometry_on=geometry_on)
        prepared = dict(context)
        if behavior_on:
            prepared["anchors"] = [first[branches], second[branches]]
        else:
            # The behaviour-off control: one global anchor for every item.
            global_anchor = first[0].unsqueeze(0).expand_as(first[branches])
            global_second = second[0].unsqueeze(0).expand_as(second[branches])
            prepared["anchors"] = [global_anchor, global_second]
        return prepared

    def _fine_forward(
        self,
        vq_layer: nn.Module,
        residual: torch.Tensor,
        level: int,
        context: dict,
        infer_use_sk: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
        """Quantize a fine level in the tangent space of the branch anchor.

        ``context`` carries the anchor of every item, the curvature and the two
        switches of the specification. With the geometry off the residual is
        shifted by the anchor and quantized in the ordinary Euclidean way, so
        the only difference between the two configurations is the geometry and
        not the branch labels.
        """
        codes = vq_layer.get_code_embs()
        anchors = context["anchors"][level - 1]
        use_sk = bool(vq_layer.use_sk) and (self.training or infer_use_sk)
        indices, contribution, reconstruction = fine_quantize(
            vq_layer, residual, anchors, float(context["curvature"]),
            geometry_on=bool(context.get("geometry_on", True)),
            use_sinkhorn=use_sk)
        quant_loss = (F.mse_loss(contribution, residual.detach())
                      + float(vq_layer.beta) * F.mse_loss(residual, contribution.detach()))
        x_q = residual + (contribution - residual).detach()
        onehot = F.one_hot(indices, vq_layer.n_embed)
        used = onehot.sum(0)
        # The flag that keeps a rank-0-only evaluation pass from waiting in a
        # collective is set on the quantizer layers, so it is read there.
        skip_reduce = (getattr(vq_layer, "_skip_ddp_reduce", False)
                       or getattr(self, "_skip_ddp_reduce", False))
        if distributed.is_initialized() and infer_use_sk is False and not skip_reduce:
            distributed.all_reduce(used, op=distributed.ReduceOp.SUM)
        unused_codes = int((used == 0).sum().item())
        return (x_q, quant_loss, unused_codes, indices.view(*residual.shape[:-1]),
                contribution, reconstruction)

    def init_codebook(self, x: torch.Tensor, device: torch.device) -> torch.Tensor:
        residual = x
        for vq_layer in self.vq_layers:
            residual = vq_layer.init_codebook(residual, device)
        return residual
