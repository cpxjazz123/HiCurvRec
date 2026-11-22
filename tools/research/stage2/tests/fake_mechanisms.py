"""Wrong mechanisms, so the reference check has something to refuse.

Two defects the check must catch, and it must catch them for the right reason:

* ``wrong_distance`` computes the Mobius sum with the point NOT negated - the
  formula is off by a sign, so the values themselves disagree;
* ``wrong_gradient`` returns the right number but scales its gradient by three -
  the forward pass agrees, so only the gradient comparison and gradcheck see it.
"""
import torch

from tools.research.geometry import poincare_contract as geometry


def wrong_distance(x, y, c):
    """``(-x) (+) y`` written without the negation: a genuine formula error."""
    curvature = float(c)
    x2 = (x ** 2).sum(dim=-1, keepdim=True)
    y2 = (y ** 2).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    scale = 1 + 2 * curvature * xy + curvature * y2
    first = scale * x + (1 - curvature * x2) * y
    denominator = 1 + 2 * curvature * xy + curvature * curvature * x2 * y2
    mobius = first / denominator
    return 2 / curvature ** 0.5 * torch.atanh(curvature ** 0.5 * mobius.norm(dim=-1))


def wrong_gradient(x, y, c):
    """The right value, a gradient three times too large: only gradcheck sees it."""
    value = geometry.distance(x, y, c)
    return value.detach() + (value - value.detach()) * 3
