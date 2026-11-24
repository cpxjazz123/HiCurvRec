"""Independent numpy reference for the layer-wise gate and the prefix proxy.

Deliberately different derivations from the production code: the gate is
recomputed with explicit loops over levels, experts and token slots, and the
A.1 proxy is computed by an explicit mean over the reference set instead of
the moment identity used in production.
"""
import numpy as np


def _linear(x, weight, bias):
    return x @ weight.T + bias


def reference_gate_weights(tokens, mask, query, key_w, key_b, value_w, value_b,
                           score_w0, score_b0, score_w2, score_b2, null_token):
    """Return (weights[N,L,E], attention[N,L,E,T+1]) from an explicit loop."""
    tokens = np.asarray(tokens, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    count, slots, dim = tokens.shape
    levels, experts = query.shape[0], query.shape[1]
    null_valid = ~mask.any(axis=1)
    effective = np.concatenate((mask, null_valid[:, None]), axis=1)
    extended = np.concatenate((np.where(mask[:, :, None], tokens, 0.0),
                               np.broadcast_to(null_token.reshape(1, 1, dim), (count, 1, dim))), axis=1)
    keys = _linear(extended.reshape(-1, dim), key_w, key_b).reshape(count, slots + 1, dim)
    values = _linear(extended.reshape(-1, dim), value_w, value_b).reshape(count, slots + 1, dim)
    weights = np.zeros((count, levels, experts))
    attention = np.zeros((count, levels, experts, slots + 1))
    for row in range(count):
        live = np.flatnonzero(effective[row])
        for level in range(levels):
            for expert in range(experts):
                logits = np.array([query[level, expert] @ keys[row, slot] / np.sqrt(dim)
                                   for slot in live])
                shifted = np.exp(logits - logits.max())
                shares = shifted / shifted.sum()
                summary = sum(shares[k] * values[row, live[k]] for k in range(len(live)))
                hidden = np.tanh(score_w0 @ summary + score_b0)
                score = float(score_w2 @ hidden + score_b2)
                attention[row, level, expert, live] = shares
                weights[row, level, expert] = score
    for level in range(levels):
        for row in range(count):
            row_values = weights[row, level]
            row_values = row_values - row_values.max()
            row_values = np.exp(row_values)
            weights[row, level] = row_values / row_values.sum()
    return weights, attention


def reference_prefixes(codes):
    """The exact running sum of the hard codewords, level by level."""
    codes = [np.asarray(code) for code in codes]
    out = np.zeros_like(codes[0])
    result = []
    for code in codes:
        out = out + code
        result.append(out.copy())
    return np.stack(result, axis=0)


def reference_layer_proxy(q, triples, references, squared):
    """The A.1 proxy by an explicit mean over the reference set (no moment identity).

    ``squared`` is the frozen distance primitive; the aggregation here is an
    explicit per-reference mean, which is a different derivation from the
    production moment identity.

    K = mean_a[ d(q_a,q_m) ] + d(q_b,q_c)/4 - ( mean_a[ d(q_a,q_b) ] + mean_a[ d(q_a,q_c) ] )/2
    """
    q = np.asarray(q)
    triples = np.asarray(triples, dtype=np.int64)
    references = np.asarray(references, dtype=np.int64)

    def distance(left, right):
        return float(np.asarray(squared(q[left][None, :], q[right][None, :])).reshape(-1)[0])

    values = []
    for centre, b, c in triples:
        first = float(np.mean([distance(a, centre) for a in references]))
        second = distance(b, c) / 4.0
        third = (float(np.mean([distance(a, b) for a in references]))
                 + float(np.mean([distance(a, c) for a in references]))) / 2.0
        values.append(first + second - third)
    return np.asarray(values, dtype=np.float64)
