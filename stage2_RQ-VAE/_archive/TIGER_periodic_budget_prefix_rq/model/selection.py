"""Frozen TIGER hard assignment with an identical-cost backward surrogate.

There is no cross-rank balancing: the frozen Sinkhorn is rank-local. Training
balances only layers whose frozen configuration enables it; inference always
uses argmin. Hard selections are not differentiated. The q gradient is solely
the derivative of P(C) @ E, including its soft codebook contribution, while
`hard` remains available for the original commitment/codebook losses.
"""
from functools import lru_cache

import torch


SOFT_TEMPERATURE = 0.003


def center_cost(cost):
    """Exactly the frozen global min/max normalization (including 1e-5)."""
    maximum, minimum = cost.max(), cost.min()
    middle = (maximum + minimum) / 2
    return (cost - middle) / (maximum - middle + 1e-5)


def _log_sinkhorn(centered, epsilon, iterations):
    log_q = -centered.double() / epsilon
    batch_size, codebook_size = log_q.shape
    log_q = log_q - torch.logsumexp(log_q.reshape(-1), dim=0)
    # Algebraically identical to the frozen probability-domain normalization,
    # in its original column-then-row order and for the original iteration count.
    log_batch = log_q.new_tensor(batch_size).log()
    log_codes = log_q.new_tensor(codebook_size).log()
    for _ in range(iterations):
        log_q = log_q - torch.logsumexp(log_q, dim=0, keepdim=True) - log_codes
        log_q = log_q - torch.logsumexp(log_q, dim=1, keepdim=True) - log_batch
    return torch.exp(log_q + log_batch).to(centered.dtype)


@lru_cache(maxsize=1)
def _cuda_sinkhorn():
    from tools.issue253.cuda_math import compile_probabilities
    return compile_probabilities(_sinkhorn)


@lru_cache(maxsize=1)
def _cuda_softmax():
    from tools.issue253.cuda_math import compile_probabilities
    return compile_probabilities(_softmax)


def _sinkhorn(centered, epsilon, iterations):
    return _log_sinkhorn(centered, epsilon, iterations)


def _softmax(centered):
    return torch.softmax(-centered / SOFT_TEMPERATURE, dim=-1)


def probabilities(cost, *, balanced, epsilon, iterations, training=True):
    """Smooth same-cost probabilities; usable directly in finite differences."""
    if balanced and training and (epsilon <= 0 or iterations <= 0):
        raise ValueError("Balanced selection requires positive epsilon and iterations.")
    # The centring stays outside the compiled region: its minimum and maximum
    # are device values, and handing them to a kernel as scalars would read
    # them back from the device.
    centered = center_cost(cost)
    if balanced and training:
        function = (_cuda_sinkhorn() if centered.is_cuda and centered.dtype == torch.float32
                    else _sinkhorn)
        return function(centered, epsilon, iterations)
    function = (_cuda_softmax() if centered.is_cuda and centered.dtype == torch.float32
                else _softmax)
    return function(centered)


@torch.no_grad()
def _hard_ids(cost, balanced, epsilon, iterations, training):
    if not (balanced and training):
        return cost.argmin(dim=-1)
    # Keep the actual frozen arithmetic for hard IDs, not a newly substituted
    # rule or an argmin of the surrogate. Double is the frozen implementation.
    q = torch.exp(-center_cost(cost).double() / epsilon)
    batch_size, codebook_size = q.shape
    q = q / q.sum(-1, keepdim=True).sum(-2, keepdim=True)
    for _ in range(iterations):
        q = q / q.sum(0, keepdim=True) / codebook_size
        q = q / q.sum(1, keepdim=True) / batch_size
    q = q * batch_size
    if not bool(torch.isfinite(q).all()):
        raise RuntimeError("Frozen hard Sinkhorn returned a nonfinite assignment.")
    return q.argmax(dim=-1)


def select(cost, codewords, *, balanced, epsilon, iterations, training=True, hard_ids=None):
    """Return exact-hard q, actual selected codewords, IDs and smooth P.

    `soft + (hard-soft).detach()` can lose hard-value bits through cancellation.
    The equivalent derivative is evaluated as
    `hard.detach() + (soft-soft.detach())`: forward is bit-exact hard, and backward
    is only the soft assignment pathway, without an extra hard-codeword gradient.
    The separate `hard` tensor is *not* detached for original quantization losses.
    """
    p = probabilities(cost, balanced=balanced, epsilon=epsilon,
                      iterations=iterations, training=training)
    ids = (_hard_ids(cost, balanced, epsilon, iterations, training)
           if hard_ids is None else hard_ids)
    hard = codewords[ids]
    soft = p.to(codewords.dtype) @ codewords
    q = hard.detach() + (soft - soft.detach())
    return {"q": q, "hard": hard, "ids": ids, "probabilities": p}
