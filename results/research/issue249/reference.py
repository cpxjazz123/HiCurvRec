"""Independent implementation of Eq. A.1 for issue #249.

No import of the production modules: the distances are written out by hand in plain Python and the
statistic is summed with explicit loops over the reference set. Hand-computable small graphs only.
"""
from fractions import Fraction


def euclidean(x, y) -> float:
    return float(sum((a - b) ** 2 for a, b in zip(x, y)) ** 0.5)


def hyperbolic(x, y, kappa: float = 1.0) -> float:
    """Poincare ball distance from the closed form, computed independently of the production code."""
    import math

    def ball(v):
        norm = sum(a * a for a in v) ** 0.5
        radius = 1.0 / math.sqrt(kappa)
        limit = radius * (1.0 - 1e-3)
        if norm > limit:
            factor = limit / norm
            return [a * factor for a in v]
        return list(v)

    bx, by = ball(x), ball(y)
    x2 = sum(a * a for a in bx)
    y2 = sum(a * a for a in by)
    gap = sum((a - b) ** 2 for a, b in zip(bx, by))
    denominator = (1.0 - kappa * x2) * (1.0 - kappa * y2)
    argument = math.sqrt(kappa) * gap ** 0.5 / denominator ** 0.5
    return 2.0 / math.sqrt(kappa) * math.atanh(min(argument, 1.0 - 1e-7))


def spherical(x, y, kappa: float = 1.0) -> float:
    import math
    radius = 1.0 / math.sqrt(kappa)
    x2 = sum(a * a for a in x)
    y2 = sum(a * a for a in y)
    distance2 = sum((a - b) ** 2 for a, b in zip(x, y))
    inner = 1.0 - 2.0 * distance2 / ((1.0 + x2) * (1.0 + y2))
    return radius * math.acos(max(-1.0, min(1.0, inner)))


def sectional(d_am, d_ab, d_ac, d_bc, weights=None) -> float:
    """Eq. A.1 summed with explicit loops; the mean is over the reference set."""
    total = 0.0
    mass = 0.0
    for index in range(len(d_am)):
        weight = 1.0 if weights is None else weights[index]
        if weight <= 0:
            continue
        term = d_am[index] ** 2 + d_bc ** 2 / 4.0 - (d_ab[index] ** 2 + d_ac[index] ** 2) / 2.0
        total += weight * term
        mass += weight
    if mass <= 0:
        raise ValueError('no weight mass')
    return total / mass


def graph_statistic(histogram_s2, s2_b, s2_c, g_bc, n_references) -> float:
    """The closed form of the #248 statistic, recomputed from the distance histogram."""
    return (histogram_s2 + n_references * g_bc ** 2 / 4.0 - (s2_b + s2_c) / 2.0) / n_references


def exact_fraction_statistic(d_am, d_ab, d_ac, d_bc) -> Fraction:
    """Eq. A.1 in exact rational arithmetic, for the hand-computed graphs."""
    total = Fraction(0)
    for index in range(len(d_am)):
        term = (Fraction(d_am[index]) ** 2 + Fraction(d_bc) ** 2 / 4
                - (Fraction(d_ab[index]) ** 2 + Fraction(d_ac[index]) ** 2) / 2)
        total += term
    return total / len(d_am)
