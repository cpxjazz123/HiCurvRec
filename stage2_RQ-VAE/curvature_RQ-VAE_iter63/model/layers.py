from __future__ import annotations

import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans


CURVATURE_MIN = 0.05
CURVATURE_MAX = 1.5
_BALL_EPS = 1e-6


def _curvature_like(curvature, reference):
    return torch.as_tensor(curvature, device=reference.device, dtype=reference.dtype).clamp_min(
        torch.finfo(reference.dtype).tiny
    )


def _expand_curvature(curvature, reference):
    value = _curvature_like(curvature, reference)
    while value.ndim < reference.ndim:
        value = value.unsqueeze(-1)
    return value


def _expmap0_tangent(x, curvature):
    c = _expand_curvature(curvature, x)
    sqrt_c = c.sqrt()
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    scaled = sqrt_c * norm
    factor = torch.tanh(scaled) / scaled.clamp_min(torch.finfo(x.dtype).eps)
    point = factor * x
    point_norm = torch.linalg.vector_norm(point, dim=-1, keepdim=True)
    max_norm = (1.0 - _BALL_EPS) / sqrt_c
    return point * torch.clamp(max_norm / point_norm.clamp_min(torch.finfo(x.dtype).eps), max=1.0)


def _logmap0_point(x, curvature):
    c = _expand_curvature(curvature, x)
    sqrt_c = c.sqrt()
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    norm_safe = norm.clamp_min(torch.finfo(x.dtype).eps)
    scaled = (sqrt_c * norm).clamp(max=1.0 - _BALL_EPS)
    return (torch.atanh(scaled) / (sqrt_c * norm_safe)) * x


def _mobius_add(x, y, curvature):
    c = _expand_curvature(curvature, x)
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    numerator = (1.0 + 2.0 * c * xy + c * y2) * x + (1.0 - c * x2) * y
    denominator = 1.0 + 2.0 * c * xy + c.square() * x2 * y2
    return numerator / denominator.clamp_min(torch.finfo(x.dtype).tiny)


def _poincare_item_code_distances(x, codebook, curvature):
    """One query's curvature is shared by its latent and every code vector."""
    c = _curvature_like(curvature, x).reshape(x.shape[0], 1)
    sqrt_c = c.sqrt()
    x_norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    code_norm = torch.linalg.vector_norm(codebook, dim=-1).unsqueeze(0)
    x_scaled = sqrt_c * x_norm
    code_scaled = sqrt_c * code_norm
    x_factor = torch.tanh(x_scaled) / x_scaled.clamp_min(torch.finfo(x.dtype).eps)
    code_factor = torch.tanh(code_scaled) / code_scaled.clamp_min(torch.finfo(x.dtype).eps)
    max_norm = (1.0 - _BALL_EPS) / sqrt_c
    x_factor = x_factor * torch.clamp(
        max_norm / (x_norm * x_factor).clamp_min(torch.finfo(x.dtype).eps), max=1.0
    )
    code_factor = code_factor * torch.clamp(
        max_norm / (code_norm * code_factor).clamp_min(torch.finfo(x.dtype).eps), max=1.0
    )
    x2 = (x_norm * x_factor).square()
    code2 = (code_norm * code_factor).square()
    dot = (x @ codebook.t()) * x_factor * code_factor
    difference2 = (x2 + code2 - 2.0 * dot).clamp_min(1e-12)
    denominator = (1.0 - 2.0 * c * dot + c.square() * x2 * code2).clamp_min(
        torch.finfo(x.dtype).tiny
    )
    scaled_norm = (sqrt_c * torch.sqrt(difference2 / denominator)).clamp(
        max=1.0 - _BALL_EPS
    )
    return (2.0 / sqrt_c) * torch.atanh(scaled_norm)


def _poincare_row_distances(x, y, curvature):
    c = _curvature_like(curvature, x).reshape(x.shape[0], 1)
    x_point = _expmap0_tangent(x, c)
    y_point = _expmap0_tangent(y, c)
    difference = _mobius_add(-x_point, y_point, c)
    norm = torch.linalg.vector_norm(difference, dim=-1, keepdim=True).clamp_min(1e-12)
    sqrt_c = c.sqrt()
    return (2.0 / sqrt_c) * torch.atanh((sqrt_c * norm).clamp(max=1.0 - _BALL_EPS))




def _hyperbolic_residual(residual_tangent, code_tangent, curvature):
    residual_point = _expmap0_tangent(residual_tangent, curvature)
    code_point = _expmap0_tangent(code_tangent, curvature)
    difference = _mobius_add(-code_point, residual_point, curvature)
    return _logmap0_point(difference, curvature)


class MLP(nn.Module):
    def __init__(self, sizes, dropout=0.0):
        super().__init__()
        modules = []
        for input_size, output_size in zip(sizes[:-1], sizes[1:]):
            modules.extend((nn.Dropout(p=dropout), nn.Linear(input_size, output_size), nn.ReLU()))
        if modules:
            modules.pop()
        self.mlp = nn.Sequential(*modules)

    def forward(self, x):
        return self.mlp(x)


class VQLayer(nn.Module):
    def __init__(self, codebook_size, codebook_dim, beta, sk_epsilon, sk_iters):
        super().__init__()
        self.n_embed = int(codebook_size)
        self.dim = int(codebook_dim)
        self.beta = float(beta)
        self.sk_epsilon = float(sk_epsilon)
        self.sk_iters = int(sk_iters)
        self.embed = nn.Embedding(self.n_embed, self.dim)

    def get_code_embs(self):
        return self.embed.weight

    @staticmethod
    def _center_distance(distances):
        high = distances.max()
        low = distances.min()
        middle = (high + low) / 2.0
        amplitude = high - middle + 1e-5
        return (distances - middle) / amplitude

    @torch.no_grad()
    def _balanced_assignments(self, distances):
        log_q = -self._center_distance(distances).double() / self.sk_epsilon
        batch_size, codebook_size = log_q.shape
        log_q = log_q - torch.logsumexp(log_q, dim=(0, 1), keepdim=True)
        log_codebook_size = math.log(codebook_size)
        log_batch_size = math.log(batch_size)
        for _ in range(self.sk_iters):
            log_q = log_q - torch.logsumexp(log_q, dim=0, keepdim=True)
            log_q = log_q - log_codebook_size
            log_q = log_q - torch.logsumexp(log_q, dim=1, keepdim=True)
            log_q = log_q - log_batch_size
        result = torch.exp(log_q + log_batch_size)
        if not torch.isfinite(result).all():
            raise RuntimeError("Sinkhorn assignment returned NaN or infinity")
        return result

    def _indices(self, distances, balanced=True):
        if balanced:
            return self._balanced_assignments(distances).argmax(dim=-1)
        return torch.argmin(distances, dim=-1)

    def _distances(self, x, curvature):
        return _poincare_item_code_distances(x, self.get_code_embs(), curvature)

    def forward(self, x, curvature, infer_use_sk=False):
        latent = x.reshape(-1, self.dim)
        c = curvature.reshape(-1)
        indices = self._indices(self._distances(latent, c), balanced=self.training or infer_use_sk)
        used = F.one_hot(indices, self.n_embed).sum(0)
        if dist.is_initialized() and not getattr(self, "_skip_ddp_reduce", False):
            dist.all_reduce(used, op=dist.ReduceOp.SUM)
        unused = int((used == 0).sum().item())
        quantized = F.embedding(indices, self.get_code_embs()).view_as(x)
        codebook_loss = _poincare_row_distances(latent.detach(), quantized.reshape(-1, self.dim), c).square().mean()
        commitment_loss = _poincare_row_distances(latent, quantized.detach().reshape(-1, self.dim), c).square().mean()
        quant_loss = codebook_loss + self.beta * commitment_loss
        quantized = x + (quantized - x).detach()
        return quantized, quant_loss, unused, indices.view(*x.shape[:-1])

    def init_codebook(self, x, curvature, device):
        if not dist.is_initialized() or dist.get_rank() == 0:
            km = KMeans(n_clusters=self.n_embed, n_init="auto").fit(x.detach().cpu().numpy())
            centers = torch.as_tensor(km.cluster_centers_, dtype=torch.float32, device=device)
        else:
            centers = torch.empty(self.n_embed, self.dim, dtype=torch.float32, device=device)
        if dist.is_initialized():
            dist.broadcast(centers, src=0)
        self.embed.weight.data.copy_(centers)
        indices = self._indices(self._distances(x, curvature), balanced=True)
        return _hyperbolic_residual(x, self.embed(indices), curvature)

    @torch.no_grad()
    def assignment_diagnostics(self, x, curvature):
        probabilities = self._balanced_assignments(self._distances(x, curvature))
        ids = probabilities.argmax(dim=-1)
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(probabilities.dtype).tiny
        )
        entropy = -(probabilities * probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()).sum(dim=-1).mean()
        return ids, torch.bincount(ids, minlength=self.n_embed), entropy


class RQLayer(nn.Module):
    def __init__(self, codebook_sizes, codebook_dim, beta=0.25, sk_epsilon=0.003, sk_iters=50):
        super().__init__()
        self.codebook_sizes = list(codebook_sizes)
        self.codebook_dim = int(codebook_dim)
        self.vq_layers = nn.ModuleList(
            VQLayer(size, codebook_dim, beta, sk_epsilon, sk_iters) for size in self.codebook_sizes
        )

    def forward(self, x, curvature, infer_use_sk=False):
        if curvature.ndim != 2 or curvature.shape != (x.shape[0], len(self.vq_layers)):
            raise ValueError("Expected one fixed curvature per item and residual level")
        quantized_sum = torch.zeros_like(x)
        residual = x
        quant_losses = []
        tokens = torch.empty(x.shape[0], len(self.vq_layers), dtype=torch.long, device=x.device)
        unused_codes = 0
        for level, layer in enumerate(self.vq_layers):
            level_curvature = curvature[:, level]
            quantized, loss, unused, indices = layer(
                residual, level_curvature, infer_use_sk=infer_use_sk
            )
            residual = _hyperbolic_residual(residual, quantized, level_curvature)
            quantized_sum = quantized_sum + quantized
            quant_losses.append(loss)
            unused_codes += unused
            tokens[:, level] = indices
        return quantized_sum, torch.stack(quant_losses).mean(), unused_codes, tokens

    def init_codebook(self, x, curvature, device):
        if curvature.ndim != 2 or curvature.shape != (x.shape[0], len(self.vq_layers)):
            raise ValueError("Expected one fixed curvature per item and residual level")
        residual = x
        for level, layer in enumerate(self.vq_layers):
            residual = layer.init_codebook(residual, curvature[:, level], device)
        return residual

    @torch.no_grad()
    def get_indices_with_stats(self, x, curvature):
        if curvature.ndim != 2 or curvature.shape != (x.shape[0], len(self.vq_layers)):
            raise ValueError("Expected one fixed curvature per item and residual level")
        residual = x
        tokens = torch.empty(x.shape[0], len(self.vq_layers), dtype=torch.long, device=x.device)
        stats = []
        for level, layer in enumerate(self.vq_layers):
            level_curvature = curvature[:, level]
            ids, usage, entropy = layer.assignment_diagnostics(residual, level_curvature)
            tokens[:, level] = ids
            stats.append({"usage_counts": usage, "assignment_entropy_nats": entropy})
            residual = _hyperbolic_residual(residual, layer.embed(ids), level_curvature)
        return tokens, stats
