"""The native mixed codeword space: one joint code, E16 x H16, curvature c=1.

A token is a single index into one codebook whose first 16 coordinates live in
Euclidean space and whose last 16 are tangent-space parameters that the
canonical :mod:`poincare_contract` maps into the ball. The two halves are never
selected separately: the cost of a candidate code is one number,

    C = 2 * [ alpha * ||rE - eE||^2 + (1 - alpha) * d_c(rH, eH)^2 / 4 ],

so a scalar weight still decides between two codes whose halves disagree.

Everything discrete stays where the baseline put it: the assignment rule is the
baseline's own rule, with this cost in place of the Euclidean one. The first two
levels take the nearest code and the last level takes the balanced Sinkhorn in
training, exactly as the frozen tree does. Nothing here replaces that rule.

The residual is native:

    rE_l = rE_{l-1} - eE_{l,k}                        (Euclidean subtraction)
    rH_l = (-eH_{l,k}) (+) rH_{l-1}                   (Moebius addition)

and the code that is subtracted is detached, so the next level's commitment
term still reaches the encoder while the codebook itself is updated by this
level's term.

The readout of a complete three-token code is

    phi(q) = [qE, log_0(qH)],   qE = sum_k eE_k,   qH = eH_1 (+) (eH_2 (+) eH_3),

right-nested because Moebius addition is not associative.

The Euclidean limit is explicit: at ``c = 0`` the canonical distance is
``2||u - v||``, so ``d^2 / 4`` is the squared Euclidean distance and the cost of
an ``alpha = 1/2`` code is the plain 32-dimensional squared distance.
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from .poincare_contract import expmap0, logmap0, mobius_add, project, squared_distance

#: The curvature of the mixed space. Fixed: the mechanism never learns or scans it.
CURVATURE = 1.0
#: The width of each half.
EUCLIDEAN_DIM = 16
HYPERBOLIC_DIM = 16
#: The width of one joint codeword.
TOTAL_DIM = EUCLIDEAN_DIM + HYPERBOLIC_DIM
#: The commitment weight of the frozen recipe.
BETA = 0.25
#: The bounded, symmetric candidate set of the outer loop.
ALPHA_CANDIDATES = (0.25, 0.5, 0.75)
#: The starting weight of every level before the outer loop runs.
ALPHA_INITIAL = 0.5
#: The tolerance of an accepted outer-loop step, relative to the incumbent.
ALPHA_TOLERANCE = 1e-8


def split(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The Euclidean half and the tangent half of a 32-dimensional vector."""
    return z[..., :EUCLIDEAN_DIM], z[..., EUCLIDEAN_DIM:]


def mixed_cost(residual_e: torch.Tensor, residual_h: torch.Tensor,
               codes_e: torch.Tensor, codes_h: torch.Tensor,
               alpha: torch.Tensor, curvature: float = CURVATURE) -> torch.Tensor:
    """``C`` for every (item, code) pair, in the convention of the spec.

    ``residual_h`` is already in the ball. ``codes_h`` are tangent-space
    parameters and are mapped with ``exp_0`` before the distance is taken.
    ``alpha`` is shape ``[B]`` (one weight per item) and is broadcast over the
    codes.
    """
    ball = expmap0(codes_h, curvature) if float(curvature) > 0 else codes_h
    euclidean = (residual_e.unsqueeze(1) - codes_e.unsqueeze(0)).pow(2).sum(-1)
    curved = squared_distance(residual_h.unsqueeze(1), ball.unsqueeze(0), curvature)
    alpha = alpha.view(-1, 1)
    return 2.0 * (alpha * euclidean + (1.0 - alpha) * curved / 4.0)


def quantized_readout(codes_e: list[torch.Tensor], codes_h: list[torch.Tensor],
                      curvature: float = CURVATURE) -> torch.Tensor:
    """``phi(q)`` for a complete code: the Euclidean sum and the nested ball sum."""
    q_e = torch.stack(codes_e, dim=0).sum(0)
    q_h = codes_h[-1]
    for code in reversed(codes_h[:-1]):
        q_h = mobius_add(code, q_h, curvature) if float(curvature) > 0 else code + q_h
    if float(curvature) > 0:
        q_h = project(q_h, curvature)
        tangent = logmap0(q_h, curvature)
    else:
        tangent = q_h
    return torch.cat([q_e, tangent], dim=-1)


class MixedVQLevel(nn.Module):
    """One level of the native mixed quantizer.

    The cost matrix replaces the Euclidean one; the assignment rule, the
    unused-code accounting and the straight-through treatment of the decoded
    vector are the baseline's.
    """

    def __init__(self, codebook_size: int, curvature: float = CURVATURE,
                 beta: float = BETA, sk_epsilon: float = -1.0, sk_iters: int = -1):
        super().__init__()
        self.n_embed = int(codebook_size)
        self.curvature = float(curvature)
        self.beta = float(beta)
        self.use_sk = sk_epsilon > 0 and sk_iters > 0
        self.sk_epsilon = float(sk_epsilon)
        self.sk_iters = int(sk_iters)
        # One joint table: columns 0..15 are E, columns 16..31 are tangent H.
        self.embed = nn.Embedding(self.n_embed, TOTAL_DIM)

    # -- the baseline's assignment rule, unchanged -------------------------
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
    def sinkhorn(self, distances: torch.Tensor, epsilon: float = 0.003,
                 iterations: int = 5) -> torch.Tensor:
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

    def _indices(self, distances: torch.Tensor, infer_use_sk: bool) -> torch.Tensor:
        if (self.training and self.use_sk) or (self.use_sk and infer_use_sk):
            centered = self.center_distance(distances).double()
            q = self.sinkhorn(centered, self.sk_epsilon, self.sk_iters)
            if torch.isnan(q).any() or torch.isinf(q).any():
                raise RuntimeError("Sinkhorn assignment returned NaN or infinity.")
            return torch.argmax(q, dim=-1)
        return torch.argmin(distances, dim=-1)

    # -- the mixed cost ----------------------------------------------------
    def get_code_embs(self) -> torch.Tensor:
        return self.embed.weight

    def distances(self, residual_e: torch.Tensor, residual_h: torch.Tensor,
                  alpha: torch.Tensor) -> torch.Tensor:
        codes = self.get_code_embs()
        codes_e, codes_h = split(codes)
        return mixed_cost(residual_e, residual_h, codes_e, codes_h, alpha, self.curvature)

    def forward(self, residual_e: torch.Tensor, residual_h: torch.Tensor,
                alpha: torch.Tensor, infer_use_sk: bool = False):
        """Quantize one level and return the new residual, the losses and the code.

        ``residual_e`` is ``[B, 16]``, ``residual_h`` is ``[B, 16]`` in the ball,
        ``alpha`` is ``[B]``.
        """
        cost = self.distances(residual_e, residual_h, alpha)
        index = self._indices(cost, infer_use_sk)
        codes = self.get_code_embs()
        codes_e, codes_h_tangent = split(codes)
        picked_e = codes_e[index]
        picked_h_tangent = codes_h_tangent[index]
        picked_h = expmap0(picked_h_tangent, self.curvature) if self.curvature > 0 else picked_h_tangent
        # Unused-code accounting, the baseline's shape: a one-hot column sum,
        # reduced across ranks when DDP is up.
        onehot = F.one_hot(index, self.n_embed)
        used = onehot.sum(0).float()
        unused = int((used == 0).sum().item())
        if (torch.distributed.is_available() and torch.distributed.is_initialized()
                and infer_use_sk is False and not getattr(self, "_skip_ddp_reduce", False)):
            torch.distributed.all_reduce(used, op=torch.distributed.ReduceOp.SUM)
            unused = int((used == 0).sum().item())
        # The commitment (code detached) and the codebook term (residual
        # detached), both in the mixed metric, divided by the joint width.
        commitment = self._cost_of(residual_e, residual_h,
                                    picked_e.detach(), picked_h.detach(), alpha)
        codebook_term = self._cost_of(residual_e.detach(), residual_h.detach(),
                                      picked_e, picked_h, alpha)
        quant_loss = (codebook_term + self.beta * commitment).mean() / float(TOTAL_DIM)
        # The residual update detaches only the chosen code.
        det_e = picked_e.detach()
        det_h = picked_h.detach()
        next_e = residual_e - det_e
        if self.curvature > 0:
            next_h = project(mobius_add(-det_h, residual_h, self.curvature), self.curvature)
        else:
            next_h = residual_h - det_h
        return (next_e, next_h), quant_loss, unused, index, picked_e, picked_h

    def _cost_of(self, residual_e: torch.Tensor, residual_h: torch.Tensor,
                 picked_e: torch.Tensor, picked_h: torch.Tensor,
                 alpha: torch.Tensor) -> torch.Tensor:
        return point_cost(residual_e, residual_h, picked_e, picked_h, alpha, self.curvature)

    @torch.no_grad()
    def init_codebook(self, residual_e: torch.Tensor, residual_h: torch.Tensor,
                      device: torch.device) -> None:
        """Rank-zero KMeans, broadcast before any rank computes its residual.

        DDP synchronises parameters at construction, before this initialisation.
        Its later gradient reduction cannot repair independently sampled centres.
        """
        from .utils import kmeans
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        if not distributed or torch.distributed.get_rank() == 0:
            codes_e = kmeans(residual_e.detach().float(), self.n_embed)
            tangent = (logmap0(residual_h.detach().float(), self.curvature)
                       if self.curvature > 0 else residual_h.detach().float())
            codes_h = kmeans(tangent, self.n_embed)
            self.embed.weight[:, :EUCLIDEAN_DIM].copy_(codes_e)
            self.embed.weight[:, EUCLIDEAN_DIM:].copy_(codes_h)
        if distributed:
            torch.distributed.broadcast(self.embed.weight, src=0)


class AlphaTable:
    """The discrete geometry weights the outer loop owns.

    ``1`` is the root, ``2`` is keyed by the first token and ``3`` by the first
    two, so a table entry adapts one region of the codebook rather than the
    whole space. A prefix with no entry inherits its nearest ancestor, which is
    the deterministic rule a new prefix follows at inference as well.
    """

    def __init__(self, levels: int = 3, initial: float = ALPHA_INITIAL):
        self.levels = {level: {} for level in range(1, int(levels) + 1)}
        self.initial = float(initial)
        self.levels[1][()] = float(initial)

    def get(self, level: int, prefix: tuple, override: dict | None = None) -> float:
        """Exact entry, this layer's global entry, then the parent layer.

        A trial override uses the very same lookup as a committed entry,
        including its effect on descendants inheriting from that entry.
        Per-layer global entries are used by G; local tables use prefix parents.
        """
        level = int(level)
        prefix = tuple(prefix)
        override = override or {}
        while level >= 1:
            table = self.levels.get(level, {})
            for key in (prefix, ()) if prefix else ((),):
                if (level, key) in override:
                    return float(override[(level, key)])
                if key in table:
                    return float(table[key])
            prefix = prefix[:-1]
            level -= 1
        return float(self.initial)

    def set(self, level: int, prefix: tuple, value: float) -> None:
        self.levels[int(level)][tuple(prefix)] = float(value)

    def lookups(self, level: int, tokens: list[torch.Tensor],
                override: dict | None = None) -> torch.Tensor:
        """The weight of every item, given the tokens of the earlier levels."""
        if level == 1 or not tokens:
            return torch.full((tokens[0].shape[0],) if tokens else (1,),
                              self.get(level, (), override),
                              dtype=torch.float32)
        prefixes = torch.stack(tokens[:level - 1], dim=1).detach().cpu().tolist()
        row = [self.get(level, tuple(int(x) for x in prefix), override) for prefix in prefixes]
        return torch.tensor(row, dtype=torch.float32)

    def to_dict(self) -> dict:
        return {str(level): {"|".join(str(part) for part in key): value
                             for key, value in sorted(table.items())}
                for level, table in self.levels.items()}

    @classmethod
    def from_dict(cls, payload: dict, levels: int = 3,
                  initial: float = ALPHA_INITIAL) -> "AlphaTable":
        table = cls(levels=levels, initial=initial)
        for level, entries in payload.items():
            for key, value in entries.items():
                parts = tuple(int(part) for part in key.split("|")) if key else ()
                table.set(int(level), parts, float(value))
        return table


class MixedRQ(nn.Module):
    """Three levels of the native mixed quantizer over one joint codebook each."""

    def __init__(self, sizes: Iterable[int], alpha_table: AlphaTable,
                 curvature: float = CURVATURE, beta: float = BETA,
                 sk_epsilon: float = -1.0, sk_iters: int = -1):
        super().__init__()
        sizes = [int(size) for size in sizes]
        self.curvature = float(curvature)
        self.alpha_table = alpha_table
        #: Per-level soft prefix embeddings of the last forward (issue #157).
        self.last_prefixes: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.levels = nn.ModuleList()
        for index, size in enumerate(sizes):
            last = index == len(sizes) - 1
            self.levels.append(MixedVQLevel(
                size, curvature=curvature, beta=beta,
                sk_epsilon=sk_epsilon if last else -1.0,
                sk_iters=sk_iters if last else -1,
            ))

    def get_extra_state(self) -> dict:
        """The non-parameter alpha table is part of a deployable tokenizer."""
        return {"schema": "mixed-rq-state/1", "curvature": self.curvature,
                "initial": self.alpha_table.initial,
                "alpha_table": self.alpha_table.to_dict()}

    def set_extra_state(self, state: dict) -> None:
        if state.get("schema") != "mixed-rq-state/1":
            raise ValueError("unsupported mixed-RQ checkpoint state")
        if float(state["curvature"]) != self.curvature:
            raise ValueError("checkpoint and tokenizer curvatures differ")
        self.alpha_table = AlphaTable.from_dict(state["alpha_table"], levels=len(self.levels),
                                                initial=float(state["initial"]))

    @property
    def vq_layers(self) -> nn.ModuleList:
        """The name the frozen trainer uses for the per-level modules."""
        return self.levels

    def forward(self, z: torch.Tensor, infer_use_sk: bool = False):
        """Quantize a 32-dimensional tangent-space code and read it back out."""
        z_e, z_t = split(z)
        residual_e = z_e
        residual_h = expmap0(z_t, self.curvature) if self.curvature > 0 else z_t
        codes_e: list[torch.Tensor] = []
        codes_h: list[torch.Tensor] = []
        tokens: list[torch.Tensor] = []
        total_loss: torch.Tensor | float = 0.0
        unused_total = 0
        from .rank_objective import compose_prefix, mixed_distance, soft_prefix
        prefixes: list[tuple[torch.Tensor, torch.Tensor]] = []
        running: tuple[torch.Tensor, torch.Tensor] | None = None
        for level, layer in enumerate(self.levels, start=1):
            alpha = self.alpha_table.lookups(level, tokens).to(z.device)
            codes_e_all = layer.embed.weight[:, :EUCLIDEAN_DIM]
            codes_h_all = layer.embed.weight[:, EUCLIDEAN_DIM:]
            rows = residual_e.shape[0]
            pre_cost = mixed_distance(
                residual_e[:, None, :].expand(rows, codes_e_all.shape[0], -1),
                residual_h[:, None, :].expand(rows, codes_h_all.shape[0], -1),
                codes_e_all[None, :, :].expand(rows, -1, -1),
                codes_h_all[None, :, :].expand(rows, -1, -1), self.curvature)
            running = compose_prefix(running, soft_prefix(pre_cost, codes_e_all, codes_h_all),
                                     level, self.curvature)
            prefixes.append(running)
            (residual_e, residual_h), quant_loss, unused, index, picked_e, picked_h = layer(
                residual_e, residual_h, alpha, infer_use_sk=infer_use_sk)
            codes_e.append(picked_e)
            codes_h.append(picked_h)
            tokens.append(index)
            total_loss = total_loss + quant_loss
            unused_total += int(unused)
        self.last_prefixes = prefixes
        self.last_codes = list(tokens)
        readout = quantized_readout(codes_e, codes_h, self.curvature)
        stacked = torch.stack(tokens, dim=1)
        return readout, total_loss / len(self.levels), unused_total, stacked

    @torch.no_grad()
    def init_codebooks(self, z: torch.Tensor, device: torch.device) -> None:
        z_e, z_t = split(z)
        residual_e = z_e
        residual_h = expmap0(z_t, self.curvature) if self.curvature > 0 else z_t
        tokens: list[torch.Tensor] = []
        for level, layer in enumerate(self.levels, start=1):
            layer.init_codebook(residual_e, residual_h, device)
            alpha = self.alpha_table.lookups(level, tokens).to(device)
            (residual_e, residual_h), _, _, index, _, _ = layer(residual_e, residual_h, alpha)
            tokens.append(index)


def encode_tokens(z: torch.Tensor, rq: MixedRQ, override: dict | None = None) -> torch.Tensor:
    """The hard three-token codes of a batch of tangent codes, without gradients.

    ``override`` maps ``(level, prefix)`` to a weight, which is how the outer
    loop evaluates a candidate without touching the table itself.
    """
    with torch.no_grad():
        z_e, z_t = split(z)
        residual_e = z_e
        residual_h = expmap0(z_t, rq.curvature) if rq.curvature > 0 else z_t
        tokens: list[torch.Tensor] = []
        for level, layer in enumerate(rq.levels, start=1):
            if override:
                rows = _override_rows(level, tokens, override, rq.alpha_table)
                alpha = rows.to(z.device)
            else:
                alpha = rq.alpha_table.lookups(level, tokens).to(z.device)
            cost = layer.distances(residual_e, residual_h, alpha)
            index = torch.argmin(cost, dim=-1)
            codes = layer.get_code_embs()
            codes_e, codes_h_tangent = split(codes)
            picked_e = codes_e[index]
            picked_h = expmap0(codes_h_tangent[index], rq.curvature) if rq.curvature > 0 else codes_h_tangent[index]
            residual_e = residual_e - picked_e
            residual_h = (project(mobius_add(-picked_h, residual_h, rq.curvature), rq.curvature)
                          if rq.curvature > 0 else residual_h - picked_h)
            tokens.append(index)
        return torch.stack(tokens, dim=1)


def _override_rows(level: int, tokens: list[torch.Tensor], override: dict,
                   table: AlphaTable) -> torch.Tensor:
    """Per-item weights under a candidate override, falling back to the table."""
    return table.lookups(level, tokens, override=override)


def point_cost(residual_e: torch.Tensor, residual_h: torch.Tensor,
               code_e: torch.Tensor, code_h: torch.Tensor,
               alpha, curvature=CURVATURE) -> torch.Tensor:
    """``C`` for one item and one code, the closed form the contract freezes.

    One vectorised call :func:`mixed_cost` is what training uses; this scalar
    path exists so the framework can compare the mechanism against a reference
    rebuilt from SymPy statements, which a batched matrix cannot express.
    ``code_h`` is the ball point, not the tangent parameter: the tangent-to-ball
    map ``exp_0`` is a convention of the codebook, declared in the math spec, and
    left out of this surface because its ``tanh(z)/z`` form has no removable
    singularity the statement language can express.
    """
    # One value per item: the reduction is over the coordinates, never over the
    # batch, because the reference this is compared with returns one value per
    # sampled row.
    euclidean = (residual_e - code_e).pow(2).sum(dim=-1)
    # The canonical c=0 distance is 2||u-v|| as well; never drop its factor 4.
    curved = squared_distance(residual_h, code_h, curvature)
    weight = alpha if not torch.is_tensor(alpha) else alpha.reshape(-1)
    return 2.0 * (weight * euclidean + (1.0 - weight) * curved / 4.0)

