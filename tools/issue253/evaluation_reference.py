"""Independent C4/C16 oracles, never importing candidate geometry or learned gates.

Matrix callbacks return [N,M], paired callbacks [N]; both accept scale= exactly
like issue251.full_evaluation.full_evaluate. The joint interface shares ambient
embeddings between fixed3/fixed5 without storing a catalogue distance matrix.
Only the explicitly requested metrics are computed (native fixed3 never ±3).
"""
import math

import numpy as np


def fixed_metrics_numpy(x, y, scale=1., *, paired=False, metrics=(3, 5)):
    """Ambient Lorentz/sphere identities in float64, evaluated in caller blocks.

    This finite-domain oracle deliberately does not use production's stable
    radial/log hyperbolic or sphere chord/atan2 formulas. Large radii that make
    ambient cancellation inaccurate require a separate stable crosscheck, not
    silent clipping of the tangents. The domain guard also prevents overflow.
    """
    if not np.isscalar(scale) or not np.isfinite(scale) or scale <= 0:
        raise ValueError('scale must be finite and strictly positive')
    if not metrics or len(set(metrics)) != len(metrics) or any(n not in (3, 5) for n in metrics):
        raise ValueError('request fixed three and/or five metrics exactly once')
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if (x.ndim != 2 or y.ndim != 2 or not x.shape[1] or x.shape[1] != y.shape[1]
            or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError('matching finite tangent matrices required')
    if paired and x.shape != y.shape:
        raise ValueError('paired coordinates must have the same shape')
    x, y = x/scale, y/scale
    rx, ry = np.linalg.norm(x, axis=1), np.linalg.norm(y, axis=1)
    largest_k = math.sqrt(3) if 5 in metrics else 1.
    if not np.isfinite(rx).all() or not np.isfinite(ry).all() or max(
            rx.max(initial=0.), ry.max(initial=0.))*largest_k > 300:
        raise ValueError('ambient reference exceeds its finite numerical domain')
    ux = np.divide(x, rx[:, None], out=np.zeros_like(x), where=rx[:, None] != 0)
    uy = np.divide(y, ry[:, None], out=np.zeros_like(y), where=ry[:, None] != 0)
    shape = (len(x),) if paired else (len(x), len(y))

    def inner(left, right):
        return np.einsum('ij,ij->i', left, right) if paired else left@right.T

    def temporal(left, right):
        return left*right if paired else np.outer(left, right)

    # Euclidean term via dot/norm identity, avoiding [N,M,D] allocation.
    euclidean = np.maximum(temporal(rx*rx, np.ones_like(ry))
                           + temporal(np.ones_like(rx), ry*ry)-2*inner(x, y), 0.)
    total = euclidean.copy()
    result = {}
    for magnitude in (1., 3.) if 5 in metrics else (1.,):
        k = math.sqrt(magnitude)
        space_x, space_y = np.sinh(k*rx)[:, None]*ux, np.sinh(k*ry)[:, None]*uy
        lorentz = temporal(np.cosh(k*rx), np.cosh(k*ry))-inner(space_x, space_y)
        hyperbolic = np.arccosh(np.maximum(lorentz, 1.))**2/magnitude
        sphere_x = np.column_stack((np.cos(k*rx), np.sin(k*rx)[:, None]*ux))
        sphere_y = np.column_stack((np.cos(k*ry), np.sin(k*ry)[:, None]*uy))
        spherical = np.arccos(np.clip(inner(sphere_x, sphere_y), -1., 1.))**2/magnitude
        total += hyperbolic+spherical
        n = 3 if magnitude == 1. else 5
        if n in metrics:
            value = total/n
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError('invalid ambient distance output')
            result['fixed'+str(n)] = value
    return result


def fixed_three_squared_numpy(x, y, scale=1.):
    return fixed_metrics_numpy(x, y, scale, metrics=(3,))['fixed3']


def fixed_three_paired_numpy(x, y, scale=1.):
    return fixed_metrics_numpy(x, y, scale, paired=True, metrics=(3,))['fixed3']


def fixed_five_squared_numpy(x, y, scale=1.):
    return fixed_metrics_numpy(x, y, scale, metrics=(5,))['fixed5']


def fixed_five_paired_numpy(x, y, scale=1.):
    return fixed_metrics_numpy(x, y, scale, paired=True, metrics=(5,))['fixed5']


def matrix_blocks(x, y, *, scale, chunk=512, metrics=(3, 5)):
    """Yield (left_slice,right_slice,metric_dict), bounded O(chunk²+chunk*D).

    Independent row/column blocks also support rectangular heldout/reference
    sets. A production component-moment coordinator may visit upper blocks only
    and accumulate the transpose, exactly as issue251's _moments does.
    """
    if type(chunk) is not int or chunk <= 0:
        raise ValueError('chunk must be a positive integer')
    for start in range(0, len(x), chunk):
        left = slice(start, min(start+chunk, len(x)))
        for offset in range(0, len(y), chunk):
            right = slice(offset, min(offset+chunk, len(y)))
            yield left, right, fixed_metrics_numpy(x[left], y[right], scale,
                                                   metrics=metrics)


def a1_predictions(coordinates, triples, component_labels, *, scale, experts=3):
    """Small-table oracle: explicitly enumerate every original reference row.

    Deliberately not the streamed production moment reduction. Triple order and
    repetitions are retained; endpoints and centre must share a full component.
    """
    q, triples, labels = np.asarray(coordinates), np.asarray(triples), np.asarray(component_labels)
    if (triples.ndim != 2 or triples.shape[1] != 3 or triples.dtype.kind not in 'iu'
            or labels.shape != (len(q),) or labels.dtype.kind not in 'iu'
            or np.any(triples < 0) or np.any(triples >= len(q))):
        raise ValueError('lawful global triple and component identities required')
    result = []
    for centre, b, c in triples:
        if not labels[centre] == labels[b] == labels[c]:
            raise ValueError('triple crosses frozen components')
        refs = np.flatnonzero(labels == labels[centre])
        metric = 'fixed'+str(experts)
        distance = fixed_metrics_numpy(q[refs], q[[centre, b, c]], scale,
                                       metrics=(experts,))[metric]
        pair = fixed_metrics_numpy(q[[b]], q[[c]], scale, metrics=(experts,))[metric][0, 0]
        result.append(math.fsum(float(row[0]-(row[1]+row[2])/2) for row in distance)/len(refs)+pair/4)
    return np.asarray(result, dtype=np.float64)


def reduce_curvature(target, predicted, centres):
    """Independent scalar grouping oracle, no production metric reducer imports."""
    target, predicted, centres = np.asarray(target), np.asarray(predicted), np.asarray(centres)
    if (target.ndim != 1 or not len(target) or predicted.shape != target.shape
            or centres.shape != target.shape or centres.dtype.kind not in 'iu'
            or not np.isfinite(target).all() or not np.isfinite(predicted).all()):
        raise ValueError('aligned nonempty finite curvature rows required')
    groups = {}
    sign_counts = {name: [0, 0] for name in ('negative', 'zero', 'positive')}
    for truth, prediction, item in zip(target, predicted, centres):
        groups.setdefault(str(int(item)), []).append(abs(float(truth)-float(prediction)))
        name = 'negative' if truth < 0 else 'positive' if truth > 0 else 'zero'
        sign_counts[name][0] += 1
        sign_counts[name][1] += int(np.sign(truth) == np.sign(prediction))
    per_item = {item: {'count': len(errors), 'mean_abs_error': math.fsum(errors)/len(errors)}
                for item, errors in groups.items()}
    for item, row in per_item.items():
        positions = [i for i, centre in enumerate(centres) if str(int(centre)) == item]
        row['strata'], row['sign'] = {}, {'recall': {}}
        for sign, name in ((-1, 'negative'), (0, 'zero'), (1, 'positive')):
            selected = [i for i in positions if np.sign(target[i]) == sign]
            if not selected:
                row['strata'][name] = {'status': 'NA', 'reason': 'target sign absent'}
                row['sign']['recall'][name] = {'status': 'NA', 'reason': 'target sign absent'}
                continue
            error = math.fsum(abs(float(predicted[i])-float(target[i])) for i in selected)/len(selected)
            row['strata'][name] = {'status': 'OK', 'count': len(selected),
                                   'macro_mae': error, 'micro_mae': error}
            row['sign']['recall'][name] = {'status': 'OK', 'count': len(selected),
                'recall': sum(np.sign(predicted[i]) == sign for i in selected)/len(selected)}
    recalls = {name: {'status': 'OK', 'count': count, 'recall': correct/count}
               if count else {'status': 'NA', 'reason': 'target sign absent'}
               for name, (count, correct) in sign_counts.items()}
    present = [name for name, (count, _) in sign_counts.items() if count]
    return {'status': 'OK', 'count': len(target), 'centres': len(groups),
            'micro_mae': math.fsum(math.fsum(errors) for errors in groups.values())/len(target),
            'macro_mae': math.fsum(row['mean_abs_error'] for row in per_item.values())/len(groups),
            'per_item': per_item, 'sign': {'recall': recalls, 'present_signs': present,
                'balanced_accuracy': math.fsum(recalls[name]['recall'] for name in present)/len(present)}}


def reduce_user_ranks(ranks):
    """Scalar oracle over persisted zero-based target ranks; misses stay None."""
    ranks = list(ranks)
    if not ranks or any(rank is not None and (type(rank) is not int or not 0 <= rank < 20)
                        for rank in ranks):
        raise ValueError('nonempty lawful beam20 target ranks required')
    result = {'n_eval': len(ranks)}
    for k in (5, 10):
        gains = [1/math.log2(rank+2) if rank is not None and rank < k else 0. for rank in ranks]
        result['test_recall@'+str(k)] = sum(gain > 0 for gain in gains)/len(ranks)
        result['test_ndcg@'+str(k)] = math.fsum(gains)/len(ranks)
    return result
