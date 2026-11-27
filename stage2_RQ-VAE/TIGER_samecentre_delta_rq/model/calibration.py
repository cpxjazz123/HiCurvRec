"""One initial global real-loss norm budget; no training-step work or free weight."""
import math

import torch
import torch.distributed as dist


def norm_budget_weight(budgets):
    ratios = []
    for name, (original, curvature) in budgets.items():
        if not all(math.isfinite(x) and x >= 0 for x in (original, curvature)):
            raise ValueError('invalid gradient norm: ' + name)
        if curvature == 0:
            continue
        if original == 0:
            raise RuntimeError('degenerate active original gradient: ' + name)
        ratios.append(original / curvature)
    if not ratios:
        raise RuntimeError('no active curvature module')
    weight = min(ratios)
    if not math.isfinite(weight) or weight <= 0:
        raise RuntimeError('initial real-loss gradient calibration is degenerate')
    return weight


def mean_gradients(gradients):
    """Reduce VJP vectors in place, before any module norm is taken."""
    if dist.is_available() and dist.is_initialized():
        world = dist.get_world_size()
        for gradient in gradients:
            dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
            gradient.div_(world)


def gradient_norm(gradients):
    return math.hypot(*(float(torch.linalg.vector_norm(g, dtype=torch.float64))
                        for g in gradients))


def coupled_groups(model):
    """Partition actual parameter elements: encoder and the three codebooks.

    Issue #268 removed the trainable gate, so no gate or query group exists on
    the curvature path.
    """
    groups = {'encoder': tuple(model.encoder.parameters())}
    for level, book in enumerate(model.rq.vq_layers):
        groups['codebook%d' % level] = tuple(book.parameters())
    return groups


def module_norms(model, gradients):
    result = {}
    for name, members in coupled_groups(model).items():
        values = []
        for member in members:
            parameter, index = member if isinstance(member, tuple) else (member, None)
            gradient = gradients[id(parameter)]
            values.append(gradient if index is None else gradient[index])
        result[name] = gradient_norm(values)
    return result
