
import torch


def _busemann(x, xi, c):
    """The Busemann function of the base point xi, in the Lorentz model."""
    dxy = ((x - xi) ** 2).sum(-1)
    inner = 1.0 - 1.0 / c
    numerator = dxy - inner * (1.0 - (x ** 2).sum(-1))
    denominator = dxy - inner * (1.0 - (xi ** 2).sum(-1))
    return torch.log((numerator / denominator).clamp(min=1e-15)) / c ** 0.5


def mechanism_loss(source, target, codebook, mask, **parameters):
    c = float(parameters.get("c", 1.0))
    weight = float(parameters.get("weight", 0.0))
    base = torch.zeros_like(codebook[0])
    distance = _busemann(source.unsqueeze(1), base.unsqueeze(0), c)
    picked = distance.argmin(dim=-1)
    gathered = distance.gather(1, picked.unsqueeze(1)).squeeze(1)
    return (gathered * mask).mean() + weight * (codebook ** 2).sum()
