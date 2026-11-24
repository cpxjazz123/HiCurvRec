"""Independent one-variable linear-program oracle; no production imports."""
import math

import numpy as np
from scipy.optimize import linprog


def reference_weight(budgets):
    constraints = []
    for name, (original, curvature) in budgets.items():
        if not all(math.isfinite(x) and x >= 0 for x in (original, curvature)):
            raise ValueError('invalid gradient norm: ' + name)
        if curvature == 0:
            continue
        if original == 0:
            raise ValueError('degenerate active original gradient: ' + name)
        constraints.append(curvature / original)
    if not constraints:
        raise ValueError('no active curvature module')
    solution = linprog([-1.0], A_ub=np.asarray(constraints)[:, None],
                       b_ub=np.ones(len(constraints)), bounds=[(0, None)],
                       method='highs')
    if not solution.success or not math.isfinite(solution.x[0]) or solution.x[0] <= 0:
        raise ValueError('reference LP has no finite positive optimum')
    return float(solution.x[0])


def reference_global_norm(rank_vectors):
    """Independent NumPy reduction retains cancellation before Euclidean norm."""
    vectors = np.asarray(rank_vectors, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError('rank vectors must be [ranks, parameter_elements]')
    mean = np.array([math.fsum(column) / len(vectors) for column in vectors.T])
    return math.sqrt(math.fsum(float(x * x) for x in mean))
