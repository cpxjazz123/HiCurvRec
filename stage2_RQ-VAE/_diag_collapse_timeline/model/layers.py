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

from . import stratum


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

        # --- the mechanism: the behaviour-stratified level-1 assignment ------
        # Only the builder's switch decides whether the mechanism's path runs; a
        # model configured without it is the baseline, bit for bit.
        self.stratified_enabled = bool(getattr(config, "stratified_enabled", False))
        self.level1_count = int(self.codebook_sizes[0])
        if self.stratified_enabled:
            # The only new parameter: one stratum logit per level-1 codeword,
            # initialised at the quantile of the behaviour rank it is meant to own.
            self.stratum_alpha = nn.Parameter(stratum.stratum_init(self.level1_count))
            # The measured scale of the offset: a scalar of the run, kept in the
            # checkpoint so a saved model states the scale it ran with.
            self.register_buffer("stratum_beta", torch.zeros(()), persistent=True)
        # The frozen rank table: data, not weights, and recomputable from the
        # training split, so the checkpoint does not carry it.  Registered even
        # when the mechanism is off, where it stays empty - a non-persistent
        # buffer adds no key to the state, so a switched-off model's state is the
        # baseline's state exactly.
        self.register_buffer("stratum_rank", torch.zeros(0), persistent=False)
        self.register_buffer("stratum_item_ids", torch.zeros(0, dtype=torch.long),
                             persistent=False)
        #: Whether the next forward also records the read-out diagnostics.
        self.observe_level1 = False
        #: The level-1 distance of every row at its chosen codeword, taken in a
        #: pass with the offset at exactly zero (so the cost IS the distance):
        #: this is what the trainer measures beta from, once.
        self.level1_chosen_distance = None
        #: The diagnostics of the last observed forward.
        self.level1_readout = None

    def forward(
        self,
        x: torch.Tensor,
        infer_use_sk: bool = False,
        item_ids: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
        quantized_x = torch.zeros(x.shape[0], self.codebook_dim, device=x.device)
        sum_quant_loss: torch.Tensor | float = 0.0
        num_unused_codes = 0.0
        output = torch.empty(x.shape[0], self.codebook_num, dtype=torch.long, device=x.device)
        residual = x
        for level, vq_layer in enumerate(self.vq_layers):
            if level == 0 and self.stratified_enabled:
                quant, quant_loss, unused, indices = self._stratified_level(
                    residual, vq_layer, item_ids)
            else:
                quant, quant_loss, unused, indices = vq_layer(residual, infer_use_sk)
            # The residual chain is a VALUE chain, not a gradient path, and the
            # detach states that.  For the baseline's own quantizer it changes
            # nothing: its output is ``x + (x_q - x).detach()``, whose derivative
            # with respect to its input is exactly the identity, so ``r - quant``
            # already had a zero derivative.  For the mechanism it is what keeps
            # the level-1 plan's gradient alive: without it the mechanism's
            # ``+quant`` into the decoder cancels against its ``-quant`` in the
            # residual - the levels above re-add the residual with derivative one
            # - and the codebook, the strata and the encoder receive exactly zero
            # (measured).  The decoder keeps the plan's gradient, which is the
            # path the specification's gradient rule names.
            residual = (residual - quant).detach()
            quantized_x = quantized_x + quant
            sum_quant_loss = sum_quant_loss + quant_loss
            num_unused_codes += unused
            output[:, level] = indices
        return quantized_x, sum_quant_loss / self.codebook_num, num_unused_codes, output

    # --- the mechanism's level 1 ------------------------------------------
    @torch.no_grad()
    def set_stratum_table(self, rank: torch.Tensor, item_ids: torch.Tensor) -> None:
        """Install the frozen per-item behaviour rank, indexed by global item id.

        The statistic itself is the specification's and is computed once from the
        training split; this method only carries it.  The ids must be ascending,
        so a lookup is a search, not a scan.
        """
        values = torch.as_tensor(rank, dtype=torch.float32).reshape(-1)
        ids = torch.as_tensor(item_ids, dtype=torch.long).reshape(-1)
        if values.numel() != ids.numel():
            raise ValueError("the rank table and the item ids disagree in length")
        if ids.numel() > 1 and bool((ids[1:] <= ids[:-1]).any()):
            raise ValueError("the rank table's item ids must be ascending")
        if values.numel() and (bool((values < 0).any()) or bool((values > 1).any())):
            raise ValueError("a behaviour rank lives in [0, 1]")
        self.stratum_rank = values.to(self.stratum_beta.device)
        self.stratum_item_ids = ids.to(self.stratum_beta.device)

    @torch.no_grad()
    def set_stratum_beta(self, beta: float) -> None:
        """Freeze the measured scale of the behaviour offset."""
        if not self.stratified_enabled:
            return
        value = float(beta)
        if not (value >= 0.0) or not value < float("inf"):
            raise ValueError("beta must be a finite non-negative distance")
        self.stratum_beta.fill_(value)

    def _gather_rank(self, item_ids: torch.Tensor, rows: int,
                     device: torch.device) -> torch.Tensor:
        """``rho_i`` for the batch, gathered by global item id.

        An item the statistic never saw - or one with no training edge - takes the
        neutral rank, never an invented order.
        """
        neutral = torch.full((rows,), stratum.NEUTRAL_RANK,
                             dtype=self.stratum_rank.dtype, device=device)
        if item_ids is None or self.stratum_rank.numel() == 0:
            return neutral
        ids = torch.as_tensor(item_ids, dtype=torch.long, device=device).reshape(-1)
        if ids.numel() != rows:
            raise ValueError("every row of the batch needs its item id")
        table = self.stratum_item_ids
        position = torch.searchsorted(table, ids).clamp(max=table.numel() - 1)
        found = table[position] == ids
        return torch.where(found, self.stratum_rank[position], neutral)

    def _stratified_level(self, residual: torch.Tensor, vq_layer: VQLayer,
                          item_ids: torch.Tensor):
        """The level-1 assignment: the baseline's plan over the mechanism's cost."""
        latent = residual.view(-1, self.codebook_dim)
        codebook = vq_layer.get_code_embs()
        rank = self._gather_rank(item_ids, latent.shape[0], latent.device)
        strata = torch.sigmoid(self.stratum_alpha)
        beta = float(self.stratum_beta.detach())
        cost = stratum.level1_cost(latent, codebook, stratum.curvature(), rank, strata, beta)
        plan_matrix = stratum.plan(cost, self.sk_epsilon, self.sk_iters)
        indices = stratum.assignment(plan_matrix)
        # The measurement pass runs with beta unset, so the offset is exactly zero
        # and the cost at the chosen codeword is D_{i,k1(i)}.
        self.level1_chosen_distance = (
            cost.detach().gather(1, indices[:, None]).squeeze(1) if beta == 0.0 else None)
        selector = stratum.straight_through(plan_matrix.to(cost.dtype), indices)
        quant = (selector @ codebook).view(residual.shape)
        chosen = F.embedding(indices, codebook)
        onehot = F.one_hot(indices, vq_layer.n_embed)
        used = onehot.sum(0)
        # The same inference guard the baseline's quantizer uses: a rank-0
        # evaluation must not wait for the ranks that are sitting in a barrier.
        if distributed.is_initialized() and not getattr(vq_layer, "_skip_ddp_reduce", False):
            distributed.all_reduce(used, op=distributed.ReduceOp.SUM)
        unused_codes = int((used == 0).sum().item())
        quant_loss = (F.mse_loss(chosen, latent.detach())
                      + vq_layer.beta * F.mse_loss(latent, chosen.detach()))
        if self.observe_level1:
            self.level1_readout = self._readout(cost, rank, strata, beta, indices)
        return quant, quant_loss, unused_codes, indices

    @torch.no_grad()
    def _readout(self, cost: torch.Tensor, rank: torch.Tensor, strata: torch.Tensor,
                 beta: float, indices: torch.Tensor) -> dict:
        """The mechanism's own read-out: how much the behaviour moves, and where.

        ``changed_fraction`` is the share of rows whose codeword differs between
        the mechanism's cost and the distance alone - the switch's effect measured
        on real data, not asserted.  ``offset_abs_mean`` is the offset's scale in
        the distance's units.
        """
        offset = stratum.level1_offset(rank, strata, beta)
        distance_only = cost.detach() + offset
        plain = stratum.assignment(stratum.plan(distance_only, self.sk_epsilon, self.sk_iters))
        return {"offset_abs_mean": float(offset.abs().mean()),
                "changed_fraction": float((plain != indices).to(torch.float64).mean())}

    def init_codebook(self, x: torch.Tensor, device: torch.device) -> torch.Tensor:
        residual = x
        for vq_layer in self.vq_layers:
            residual = vq_layer.init_codebook(residual, device)
        return residual
