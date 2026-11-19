"""Uniform semantic-neighbourhood KL for the RQ-VAE arm.

The target graph is fixed before training: for every item the K nearest sentence-T5
neighbours are positives and K deterministic non-neighbours are negatives.  The target
distribution over that fixed support is a scale-free softmax of cosine distance, using the
mean positive distance of the item as its temperature.

The query distribution uses the same support and the *hard* quantised vectors stored in a
memory bank for the candidates.  The anchor uses a straight-through soft readout of the
codebooks, so the KL term reaches both the encoder (through the distances) and the codebook
entries (through the soft mixture).  The forward value of the anchor is still the hard q.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import logsumexp
from sklearn.neighbors import NearestNeighbors

NAME = "uniform_knn"
DESCRIPTION = "KL between fixed sentence-T5 KNN and quantised KNN, with negative samples"

KNN_NEIGHBOURS = 10
KNN_NEGATIVES = 10
KNN_SEED = 42
KNN_TAU_FLOOR = 1e-4
KNN_PROB_FLOOR = 1e-3
#: The KL term is O(1); the frozen RQ-VAE loss is ~3e-4 at its end.  The 2026-11-17
#: audit measured a KL/base parameter-gradient ratio of ~6206 at the final checkpoint,
#: so 1e-3 made the KL gradient ~6x the base gradient even though the scalar values
#: looked comparable.  This fixed 1e-4 is derived from that measurement (not a sweep)
#: and puts the KL gradient below, but in the same order as, the base gradient.
KNN_WEIGHT = 0.0

__all__ = [
    "KNN_NEIGHBOURS", "KNN_NEGATIVES", "KNN_SEED", "KNN_TAU_FLOOR",
    "KNN_PROB_FLOOR", "KNN_WEIGHT", "build_knn_targets",
    "knn_neighbourhood_kl", "soft_quantized_readout", "hard_quantized_from_tokens",
    "hard_quantized_from_embeddings", "QuantizedMemory",
]


def build_knn_targets(
    embeddings: np.ndarray,
    *,
    k: int = KNN_NEIGHBOURS,
    negatives: int = KNN_NEGATIVES,
    seed: int = KNN_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return candidates, log target probabilities and per-item temperature.

    The first ``k`` columns are the exact cosine neighbours; the remaining ``negatives``
    columns are random items that are neither the anchor nor a positive.  The target is
    normalised over the whole ``k + negatives`` support, so negatives carry a real (small)
    probability mass and therefore a real gradient.
    """
    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 2 or len(values) <= k + negatives:
        raise ValueError("embeddings must be a 2-D matrix with more than k+negatives rows")
    n = len(values)
    k = int(k)
    negatives = int(negatives)
    if k < 1 or negatives < 1:
        raise ValueError("k and negatives must both be positive")

    neighbour = NearestNeighbors(
        n_neighbors=k + 1, metric="cosine", algorithm="brute", n_jobs=-1
    ).fit(values)
    distances, indices = neighbour.kneighbors(values)
    positives = np.empty((n, k), dtype=np.int64)
    positive_distances = np.empty((n, k), dtype=np.float64)
    for item in range(n):
        row = indices[item]
        dist = distances[item]
        keep = row != item
        row = row[keep]
        dist = dist[keep]
        if len(row) < k:
            raise ValueError(f"only {len(row)} non-self neighbours for item {item}")
        positives[item] = row[:k]
        positive_distances[item] = dist[:k]

    rng = np.random.default_rng(int(seed))
    negative = np.empty((n, negatives), dtype=np.int64)
    for item in range(n):
        banned = set(int(value) for value in positives[item])
        banned.add(item)
        # Sampling a handful more than needed makes rejection cheap and deterministic.
        draw = rng.choice(n, size=max(4 * negatives, negatives + 8), replace=False)
        chosen = [int(value) for value in draw if int(value) not in banned]
        if len(chosen) < negatives:
            remaining = np.array(
                [value for value in range(n) if value not in banned], dtype=np.int64
            )
            extra = rng.choice(remaining, size=negatives - len(chosen), replace=False)
            chosen.extend(int(value) for value in extra)
        negative[item] = chosen[:negatives]

    unit = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    negative_distances = 1.0 - np.einsum("ij,ikj->ik", unit, unit[negative])
    negative_distances = np.maximum(negative_distances, 0.0)
    all_distances = np.concatenate([positive_distances, negative_distances], axis=1)
    candidates = np.concatenate([positives, negative], axis=1)
    tau = np.maximum(positive_distances.mean(axis=1), KNN_TAU_FLOOR)
    logits = -all_distances / tau[:, None]
    log_target = logits - logsumexp(logits, axis=1, keepdims=True)
    return candidates.astype(np.int64), log_target.astype(np.float32), tau.astype(np.float32)


def knn_neighbourhood_kl(
    q_anchor: torch.Tensor,
    q_candidates: torch.Tensor,
    log_target: np.ndarray | torch.Tensor,
    *,
    prob_floor: float = KNN_PROB_FLOOR,
) -> torch.Tensor:
    """KL(P_target || P_quantised) over the fixed positive/negative support.

    ``q_anchor`` is [B, D], ``q_candidates`` is [B, support, D].  The first
    ``KNN_NEIGHBOURS`` candidates are positive; the query temperature is the mean
    positive squared distance and is detached, so the *scale* of q is not part of the
    objective.  The probability floor bounds the KL for exactly duplicate q vectors.
    """
    if q_anchor.ndim != 2 or q_candidates.ndim != 3:
        raise ValueError("q_anchor must be [B,D] and q_candidates [B,S,D]")
    if q_candidates.shape[0] != q_anchor.shape[0] or q_candidates.shape[2] != q_anchor.shape[1]:
        raise ValueError("q_anchor and q_candidates disagree on batch or dimension")
    support = q_candidates.shape[1]
    positive = min(KNN_NEIGHBOURS, support)
    target = torch.as_tensor(log_target, dtype=q_anchor.dtype, device=q_anchor.device)
    if target.shape != (q_anchor.shape[0], support):
        raise ValueError(f"log_target shape {tuple(target.shape)} does not match support {support}")

    distance = ((q_anchor[:, None, :] - q_candidates) ** 2).sum(dim=-1)
    positive_distance = distance[:, :positive]
    # The temperature is a normaliser, not a learned scale: detach it.  Without
    # this, the q norm can be reduced by shrinking tau rather than by improving
    # the neighbour ordering, which is exactly the drift observed in the first run.
    tau = positive_distance.mean(dim=1, keepdim=True).detach().clamp_min(1e-8)
    logits = -distance / tau
    log_q = F.log_softmax(logits, dim=1)
    floor = float(prob_floor) / support
    log_q = torch.log((1.0 - float(prob_floor)) * torch.exp(log_q) + floor)
    p = torch.exp(target)
    terms = torch.where(p > 0, p * (target - log_q), torch.zeros_like(p))
    return terms.sum(dim=1).mean()


def hard_quantized_from_tokens(model, tokens: torch.Tensor) -> torch.Tensor:
    """Sum the selected codebook entries for already-assigned SID tokens."""
    total = None
    for level, layer in enumerate(model.rq.vq_layers):
        index = tokens[:, level].to(device=layer.get_code_embs().device, dtype=torch.long)
        value = F.embedding(index, layer.get_code_embs())
        total = value if total is None else total + value
    if total is None:
        raise ValueError("the model has no quantisation levels")
    return total


@torch.no_grad()
def hard_quantized_from_embeddings(
    model,
    embeddings: torch.Tensor,
    *,
    assignment: str = "export",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return hard q and tokens using the *export* assignment by default.

    ``assignment="export"`` follows the Stage2 export path exactly: argmin at every
    residual level, no Sinkhorn.  This is deliberately different from the training
    forward's last-level Sinkhorn assignment; the KL target and q memory use the
    exported path so that they cannot disagree with the SID that is actually scored.
    """
    if assignment != "export":
        raise ValueError("hard_quantized_from_embeddings only supports the export assignment")
    latent = model.encoder(embeddings)
    residual = latent
    total = None
    tokens = []
    for layer in model.rq.vq_layers:
        distances = layer._distances(residual)
        index = distances.argmin(dim=1)
        hard = F.embedding(index, layer.get_code_embs())
        total = hard if total is None else total + hard
        residual = residual - hard
        tokens.append(index)
    if total is None:
        raise ValueError("the model has no quantisation levels")
    return total.detach(), torch.stack(tokens, dim=1)


class QuantizedMemory:
    """Detached export-path q cache with explicit age accounting.

    The cache is refreshed from the current model at every epoch start and the current
    mini-batch is recomputed *after* ``optimizer.step()``.  Items not in the current
    batch therefore have a bounded age (at most one epoch); ``stats()`` makes that
    staleness visible instead of silently mixing representations from different steps.
    """

    def __init__(self, vectors: torch.Tensor, *, step: int = 0):
        self.vectors = vectors.detach().clone()
        self.ages = torch.zeros(len(vectors), dtype=torch.long, device=vectors.device)
        self.step = int(step)

    @classmethod
    def from_model(cls, model, embeddings: torch.Tensor, *, step: int = 0):
        vectors, _ = hard_quantized_from_embeddings(model, embeddings)
        return cls(vectors, step=step)

    @torch.no_grad()
    def refresh(self, model, embeddings: torch.Tensor, *, step: int) -> None:
        vectors, _ = hard_quantized_from_embeddings(model, embeddings)
        self.vectors.copy_(vectors)
        self.ages.zero_()
        self.step = int(step)

    @torch.no_grad()
    def update_batch(self, model, embeddings: torch.Tensor, item_ids: torch.Tensor,
                     *, step: int) -> None:
        vectors, _ = hard_quantized_from_embeddings(model, embeddings)
        self.update_items(item_ids, vectors, step=step)

    @torch.no_grad()
    def update_items(self, item_ids: torch.Tensor, vectors: torch.Tensor, *, step: int) -> None:
        """Update exactly the rows supplied; used after DDP all-gather."""
        self.ages += 1
        self.vectors[item_ids] = vectors
        self.ages[item_ids] = 0
        self.step = int(step)

    def index(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self.vectors[item_ids]

    def stats(self) -> dict:
        return {
            "items": int(len(self.vectors)),
            "step": int(self.step),
            "mean_age": float(self.ages.float().mean()),
            "max_age": int(self.ages.max()),
            "stale_items": int((self.ages > 0).sum()),
        }


def soft_quantized_readout(
    model,
    embeddings: torch.Tensor,
    tokens: torch.Tensor | None = None,
    *,
    assignment: str = "export",
    tau_floor: float = KNN_TAU_FLOOR,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (straight-through soft q, hard q) for a raw, non-DDP RQVAE.

    ``assignment="export"`` (the default) uses argmin at every level, matching the
    SID export.  Pass ``assignment="model"`` and the model's own tokens only for the
    explicit training/export consistency audit.  The soft mixture is computed from
    the same residual path, so changing the codebook or the encoder changes the
    returned proxy; its temperature is detached.
    """
    if assignment not in {"export", "model"}:
        raise ValueError("assignment must be 'export' or 'model'")
    if assignment == "model" and tokens is None:
        raise ValueError("assignment='model' requires the model's tokens")
    latent = model.encoder(embeddings)
    residual = latent
    hard_total = torch.zeros_like(latent)
    soft_total = torch.zeros_like(latent)
    for level, layer in enumerate(model.rq.vq_layers):
        distances = layer._distances(residual)
        tau = distances.mean(dim=1, keepdim=True).detach().clamp_min(float(tau_floor))
        weights = torch.softmax(-distances / tau, dim=1)
        codebook = layer.get_code_embs()
        soft = weights @ codebook
        if assignment == "export":
            index = distances.argmin(dim=1)
        else:
            index = tokens[:, level].to(device=embeddings.device, dtype=torch.long)
        hard = F.embedding(index, codebook.detach())
        hard_total = hard_total + hard
        soft_total = soft_total + soft
        residual = residual - hard

    proxy = hard_total + (soft_total - soft_total.detach())
    return proxy, hard_total
