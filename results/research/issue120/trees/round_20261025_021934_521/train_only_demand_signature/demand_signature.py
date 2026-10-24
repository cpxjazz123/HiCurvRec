"""Train-only demand-shape SID suffix; pure Python/Numpy for testability."""
import math
import numpy as np


def fit_demand_suffix(frame, n_items):
    """Return one suffix token per item using only supplied training user/item rows."""
    counts = np.zeros(n_items, dtype=np.int64)
    users = [set() for _ in range(n_items)]
    repeated = np.zeros(n_items, dtype=np.int64)
    per_user = {}
    for user, item in zip(frame['user_id'].to_numpy(), frame['target'].to_numpy(dtype=np.int64)):
        if item < 0 or item >= n_items:
            continue
        counts[item] += 1
        users[item].add(user)
        key = (item, user)
        per_user[key] = per_user.get(key, 0) + 1
    for (item, user), count in per_user.items():
        if count >= 2:
            repeated[item] += 1
    u = np.asarray([len(x) for x in users], dtype=np.int64)
    r = np.divide(repeated, u + 1, dtype=float)
    p = np.zeros(n_items, dtype=float)
    valid = u > 0
    p[valid] = np.log1p(counts[valid] / u[valid])
    # Deciles based on observed training items. Equal-frequency assignment is
    # deterministic under ties; all bin boundaries are fit from training only.
    active = np.flatnonzero(valid)
    tokens = np.zeros(n_items, dtype=np.int64)
    if not len(active):
        return tokens
    order = active[np.argsort(counts[active], kind='stable')]
    decile = np.empty(n_items, dtype=np.int64)
    decile[:] = -1
    for rank, item in enumerate(order):
        decile[item] = min(9, rank * 10 // len(order))
    for d in range(10):
        group = active[decile[active] == d]
        if not len(group):
            continue
        # rank-based quantiles avoid duplicate-edge ambiguity for tied values.
        for values, bins, offset, multiplier in ((r, 3, 0, 2), (p, 2, 0, 1)):
            sorted_group = group[np.argsort(values[group], kind='stable')]
            b = np.empty(len(sorted_group), dtype=np.int64)
            for rank, item in enumerate(sorted_group):
                b[rank] = min(bins - 1, rank * bins // len(sorted_group))
            if bins == 3:
                rb = np.empty(n_items, dtype=np.int64)
                rb[sorted_group] = b
            else:
                pb = np.empty(n_items, dtype=np.int64)
                pb[sorted_group] = b
        tokens[group] = 6 * rb[group] + pb[group]
    return tokens
