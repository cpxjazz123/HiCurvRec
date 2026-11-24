"""Independent ambient geometry and explicit scalar attention; no production imports."""
import math
import numpy as np

CURVATURES = (-1, 0, 1)

def ambient_squared(x, y, *, scale=1.0, curvatures=CURVATURES):
    x, y = np.asarray(x, dtype=np.float64)/scale, np.asarray(y, dtype=np.float64)/scale
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1] or not np.isfinite(scale) or scale <= 0:
        raise ValueError('positive scale and matching two-dimensional tangents required')
    rx, ry = np.linalg.norm(x, axis=1), np.linalg.norm(y, axis=1)
    ux = np.divide(x, rx[:, None], out=np.zeros_like(x), where=rx[:, None] != 0)
    uy = np.divide(y, ry[:, None], out=np.zeros_like(y), where=ry[:, None] != 0)
    result = []
    for c in curvatures:
        if c == 0:
            value = np.sum((x[:, None, :]-y[None, :, :])**2, axis=-1)
        elif c < 0:
            k = math.sqrt(-c)
            time = np.outer(np.cosh(k*rx), np.cosh(k*ry))
            space = (np.sinh(k*rx)[:, None]*ux) @ (np.sinh(k*ry)[:, None]*uy).T
            value = np.arccosh(np.maximum(time-space, 1.0))**2/(-c)
        else:
            k = math.sqrt(c)
            ax = np.concatenate((np.cos(k*rx)[:, None], np.sin(k*rx)[:, None]*ux), axis=-1)
            ay = np.concatenate((np.cos(k*ry)[:, None], np.sin(k*ry)[:, None]*uy), axis=-1)
            value = np.arccos(np.clip(ax @ ay.T, -1.0, 1.0))**2/c
        result.append(value)
    return np.stack(result, axis=-1)

def scalar_attention(tokens, mask, query, key_weight, key_bias, value_weight, value_bias, score_weight, score_bias, output_weight, output_bias):
    """Loop over examples/experts/tokens; softmaxes are independently normalized."""
    tokens, mask = np.asarray(tokens), np.asarray(mask, dtype=bool)
    all_weights, all_attention = [], []
    for row, valid in zip(tokens, mask):
        active = list(np.flatnonzero(valid))
        extended = [v if ok else np.zeros_like(v) for v, ok in zip(row, valid)] + [np.zeros(row.shape[-1])]
        if not active: active = [len(row)]
        contexts, probabilities = [], []
        for q in query:
            logits = [float(np.dot(q, key_weight@extended[j]+key_bias))/math.sqrt(row.shape[-1]) for j in active]
            p = np.exp(np.asarray(logits)-max(logits)); p /= p.sum()
            attention = np.zeros(len(extended)); attention[active] = p
            context = sum(float(a)*(value_weight@extended[j]+value_bias) for j, a in zip(active, p))
            contexts.append(float(output_weight@np.tanh(score_weight@context+score_bias)+output_bias))
            probabilities.append(attention)
        weights = np.exp(np.asarray(contexts)-max(contexts)); weights /= weights.sum()
        all_weights.append(weights); all_attention.append(probabilities)
    return np.asarray(all_weights), np.asarray(all_attention)
