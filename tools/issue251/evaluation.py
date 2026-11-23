"""Independent hard-SID and continuous-encoder diagnostics.

The caller injects the candidate's frozen ``fixed_squared`` geometry, independent
of trainable gates. Hard SID is decoded from actual codebooks; continuous z has
its own entry point and never acquires SID or prefix metrics. Every curvature
evaluation requires graph/component evidence for exact full-reference equality.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from scipy.sparse import issparse
from scipy.sparse.csgraph import connected_components, shortest_path

from tools.graph_infra import fidelity, tiger_compare
from tools.graph_infra.metrics import average_ranks, spearman

_REF_CHUNK = 512
_VERTEX_CHUNK = 64
_PAIR_CHUNK = 2048
_SIGNS = ((-1, 'negative'), (0, 'zero'), (1, 'positive'))


def _na(reason):
    return {'status': 'NA', 'reason': reason}


def _numpy(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _indices(value, width, items, name):
    rows = _numpy(value)
    if rows.ndim != 2 or rows.shape[1] != width or rows.dtype.kind not in 'iu':
        raise ValueError(f'{name} must be an integer (N,{width}) array')
    if np.any(rows < 0) or np.any(rows >= items):
        raise ValueError(f'{name} index outside the item table')
    return rows


def _decode(sid, codebooks):
    rows = _numpy(sid)
    if rows.ndim != 2 or rows.shape[1] not in (3, 4) or not len(rows):
        raise ValueError('SID must have three semantic columns and optional fourth token')
    if rows.dtype.kind not in 'iu' or np.any(rows < 0) or len(codebooks) != 3:
        raise ValueError('nonnegative integer SID and exactly three codebooks required')
    books = [_numpy(book) for book in codebooks]
    dimension = books[0].shape[1] if books[0].ndim == 2 else 0
    for level, book in enumerate(books):
        if (book.ndim != 2 or not len(book) or not dimension or
                book.shape[1] != dimension or not np.isfinite(book).all()):
            raise ValueError('codebooks must have matching finite nonempty dimensions')
        if np.any(rows[:, level] >= len(book)):
            raise ValueError('SID index outside its codebook')
    # Index each actual codebook, not encoder output or a quantizer surrogate.
    q = fidelity.decode_hard_sid(rows, books)
    return rows, books, torch.from_numpy(q)


def _positive(value, name):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and strictly positive')


def _squared(x, y, fixed_squared, scale):
    with torch.no_grad():
        values = _numpy(fixed_squared(x, y, scale=scale)).astype(np.float64, copy=False)
    expected = np.broadcast_shapes(tuple(x.shape[:-1]), tuple(y.shape[:-1]))
    if values.shape != expected or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('fixed_squared must return finite nonnegative broadcast squared distances')
    return values


def _pair_squared(q, pairs, fixed_squared, scale):
    result = np.empty(len(pairs), dtype=np.float64)
    for start in range(0, len(pairs), _PAIR_CHUNK):
        block = pairs[start:start + _PAIR_CHUNK]
        result[start:start + len(block)] = _squared(
            q[block[:, 0]][:, None, :], q[block[:, 1]][:, None, :],
            fixed_squared, scale).reshape(-1)
    return result


def _pair_distances(q, pairs, fixed_squared, scale):
    return np.sqrt(_pair_squared(q, pairs, fixed_squared, scale))


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return _na('no observations')
    return {'count': int(values.size), 'mean': float(values.mean()),
            'std': float(values.std()), 'min': float(values.min()),
            'q25': float(np.quantile(values, .25)), 'median': float(np.median(values)),
            'q75': float(np.quantile(values, .75)), 'max': float(values.max())}


def _errors(target, predicted, centres):
    if not len(target):
        return _na('no valid triples in this stratum')
    error = np.abs(predicted - target)
    _, inverse, counts = np.unique(centres, return_inverse=True, return_counts=True)
    macro = np.bincount(inverse, weights=error) / counts
    return {'status': 'OK', 'count': int(len(target)), 'centres': int(len(counts)),
            'micro_mae': float(error.mean()), 'macro_mae': float(macro.mean())}


def _curvature_report(target, predicted, centres, excluded):
    report = _errors(target, predicted, centres)
    report.update({'excluded': int(excluded), 'target': target.tolist(),
                   'predicted': predicted.tolist(), 'per_item': {}})
    for item in np.unique(centres):
        mask = centres == item
        report['per_item'][str(int(item))] = {
            'count': int(mask.sum()), 'target': _summary(target[mask]),
            'predicted': _summary(predicted[mask]),
            'mean_signed_error': float((predicted[mask] - target[mask]).mean()),
            'mean_abs_error': float(np.abs(predicted[mask] - target[mask]).mean())}
    # Use exact frozen target signs; no held-out-fitted zero threshold.
    target_sign, predicted_sign = np.sign(target), np.sign(predicted)
    recalls, strata, present = {}, {}, []
    for sign, name in _SIGNS:
        mask = target_sign == sign
        strata[name] = _errors(target[mask], predicted[mask], centres[mask])
        if mask.any():
            present.append(name)
            recalls[name] = {'status': 'OK', 'count': int(mask.sum()),
                             'recall': float((predicted_sign[mask] == sign).mean())}
        else:
            recalls[name] = _na('target sign absent')
    report['strata'] = strata
    report['sign'] = {'accuracy': float((target_sign == predicted_sign).mean())
                      if len(target) else None,
                      'balanced_accuracy': float(np.mean([recalls[s]['recall'] for s in present]))
                      if present else None, 'present_signs': present, 'recall': recalls,
                      'zero_tolerance': 0., 'averaging': 'macro over present target signs'}
    return report


def _components(value, items, component_labels=None):
    def checked(labels):
        labels = _numpy(labels)
        if labels.shape != (items,) or labels.dtype.kind not in 'iu':
            raise ValueError('integer component labels must align with the item table')
        return labels

    if issparse(value):
        if value.shape != (items, items):
            raise ValueError('graph shape must match the item table')
        inferred = connected_components(value, directed=False, return_labels=True)[1]
    elif value is not None and not isinstance(value, Mapping):
        inferred = checked(value)
    else:
        inferred = None
    labels = checked(component_labels) if component_labels is not None else inferred
    if labels is None:
        raise ValueError('component labels or graph required to prove full reference sets')
    if inferred is not None and component_labels is not None:
        # Component IDs may differ; each ID must map to exactly one other ID.
        pairs = np.unique(np.column_stack((labels, inferred)), axis=0)
        if len(pairs) != len(np.unique(labels)) or len(pairs) != len(np.unique(inferred)):
            raise ValueError('component labels conflict with supplied graph/components')
    return labels


def _predict(q, triples, references, labels, fixed_squared, scale):
    """Exact full-reference A.1, reusing distance moments at repeated centres."""
    predicted = np.empty(len(triples), dtype=np.float64)
    for centre in np.unique(triples[:, 0]):
        positions = np.flatnonzero(triples[:, 0] == centre)
        local = triples[positions]
        refs = _numpy(references[int(centre)])
        if (refs.ndim != 1 or refs.dtype.kind not in 'iu' or not len(refs) or
                np.any(refs < 0) or np.any(refs >= len(q)) or
                len(np.unique(refs)) != len(refs)):
            raise ValueError('each reference set must contain unique lawful item indices')
        if not np.isin(local, refs).all():
            raise ValueError('reference set must include the centre and both endpoints')
        full_component = np.flatnonzero(labels == labels[centre])
        if not np.array_equal(np.sort(refs), full_component):
            raise ValueError('reference set must equal the full frozen centre component')
        vertices, inverse = np.unique(local, return_inverse=True)
        inverse = inverse.reshape(local.shape)
        means = np.empty(len(vertices), dtype=np.float64)
        for start in range(0, len(vertices), _VERTEX_CHUNK):
            block = vertices[start:start + _VERTEX_CHUNK]
            total = np.zeros(len(block), dtype=np.float64)
            for offset in range(0, len(refs), _REF_CHUNK):
                ref_block = refs[offset:offset + _REF_CHUNK]
                total += _squared(q[ref_block][:, None, :], q[block][None, :, :],
                                  fixed_squared, scale).sum(axis=0)
            means[start:start + len(block)] = total / len(refs)
        endpoint_squared = _pair_squared(q, local[:, 1:3], fixed_squared, scale)
        predicted[positions] = (means[inverse[:, 0]] - .5 * (
            means[inverse[:, 1]] + means[inverse[:, 2]]) + .25 * endpoint_squared)
    return predicted


def _distortion(q, triples, graph_or_pairs, fixed_squared, scale, distance_scale):
    if issparse(graph_or_pairs):
        pairs = np.unique(np.concatenate((triples[:, [0, 1]], triples[:, [0, 2]],
                                          triples[:, [1, 2]])), axis=0)
        real = np.empty(len(pairs), dtype=np.float64)
        for source in np.unique(pairs[:, 0]):
            mask = pairs[:, 0] == source
            distances = shortest_path(graph_or_pairs, directed=False, unweighted=True,
                                      indices=int(source))
            real[mask] = distances[pairs[mask, 1]]
        pair_definition = 'all unique ordered centre-endpoint and endpoint-endpoint triple pairs'
    elif isinstance(graph_or_pairs, Mapping):
        pairs = _indices(graph_or_pairs['pairs'], 2, len(q), 'distance pairs')
        real = np.asarray(graph_or_pairs['distances'], dtype=np.float64)
        pair_definition = 'all explicitly supplied reference distance pairs'
    else:
        return _na('reference graph or explicit pair distances not supplied')
    if real.shape != (len(pairs),) or np.any(real < 0) or np.isnan(real).any():
        raise ValueError('reference distances must align with pairs and be nonnegative')
    distance = _pair_distances(q, pairs, fixed_squared, scale)
    finite = np.isfinite(real)
    report = {'count': int(finite.sum()), 'unreachable': int((~finite).sum()),
              'pair_definition': pair_definition, 'distance_scale': float(distance_scale),
              'scale_fitted_on_evaluation': False}
    if finite.any():
        report['raw'] = fidelity.distance_distortion(real[finite], distance[finite])
        report['scale_fixed'] = fidelity.distance_distortion(
            real[finite], distance[finite] * distance_scale)
    else:
        report['raw'] = report['scale_fixed'] = _na('no reachable reference pairs')
    return report

def _curvature_metrics(q, triples, targets, reference_sets, graph_or_pairs, scale,
                       fixed_squared, component_labels):
    triples = _indices(triples, 3, len(q), 'triples')
    if isinstance(targets, Mapping):
        values = np.asarray(targets['values'], dtype=np.float64)
        valid = np.asarray(targets.get('valid', np.ones(len(triples), dtype=bool)))
        if valid.dtype.kind != 'b':
            raise ValueError('target validity mask must be boolean')
    else:
        values = np.asarray(targets, dtype=np.float64)
        valid = np.ones(len(triples), dtype=bool)
    if values.shape != (len(triples),) or valid.shape != values.shape:
        raise ValueError('targets and validity mask must align with triples')
    if not np.isfinite(values[valid]).all():
        raise ValueError('valid targets must be finite')
    labels = _components(graph_or_pairs, len(q), component_labels)
    lawful = triples[valid]
    predicted = _predict(q, lawful, reference_sets, labels, fixed_squared, scale)
    report = {
        'metric': 'equal_weight_mean_of_five_squared_expert_distances',
        'expert_curvatures': [-3, -1, 0, 1, 3], 'tangent_scale': float(scale),
        'curvature_definition': 'GraphMoRE A.1 discrete proxy, not intrinsic curvature',
        'curvature': _curvature_report(values[valid], predicted, lawful[:, 0],
                                       (~valid).sum())}
    return report, lawful


def evaluate_continuous(z, triples, targets, reference_sets,
                        components_or_graph_optional=None, scale=1., *,
                        fixed_squared, component_labels=None):
    """Evaluate actual continuous encoder z separately, never as decoded SID.

    Uses the same frozen metric, full-component evidence, masks and curvature
    summaries as :func:`evaluate`. Optional explicit pair distances or a graph
    produce distance distortion with unit distance multiplier; no scale is fit.
    """
    _positive(scale, 'scale')
    values = _numpy(z)
    if (values.ndim != 2 or not len(values) or not values.shape[1] or
            values.dtype.kind not in 'fiu' or not np.isfinite(values).all()):
        raise ValueError('continuous z must be a finite nonempty item matrix')
    q = torch.from_numpy(values.astype(np.float64, copy=False))
    report, lawful = _curvature_metrics(
        q, triples, targets, reference_sets, components_or_graph_optional, scale,
        fixed_squared, component_labels)
    report.update({
        'representation': 'continuous_encoder_z',
        'distance_distortion': _distortion(q, lawful, components_or_graph_optional,
                                           fixed_squared, scale, 1.)})
    return report


def evaluate_reconstruction(embeddings, reconstruction, *, representation):
    """Aligned actual decoder errors, labelled by the decoded representation."""
    target, actual = _numpy(embeddings), _numpy(reconstruction)
    if (target.ndim != 2 or target.shape != actual.shape or not len(target) or
            not target.shape[1] or not np.isfinite(target).all() or not np.isfinite(actual).all()):
        raise ValueError('reconstruction and embeddings must be aligned finite item matrices')
    difference = actual.astype(np.float64) - target
    l2 = np.linalg.norm(difference, axis=1)
    return {'status': 'OK', 'source': 'caller-supplied '+representation+' reconstruction',
            'mean_squared_error': float(np.square(difference).mean()),
            'mean_l2_error': float(l2.mean()), 'per_item_l2': l2.tolist()}



def evaluate(sid, codebooks, triples, targets, reference_sets,
             components_or_graph_optional=None, scale=1.0, *, fixed_squared,
             distance_scale=1.0, embeddings=None, reconstruction=None, component_labels=None):
    """Evaluate actual three-codeword q, independently of the train loss.

    ``triples`` are global (centre,b,c) IDs. ``targets`` is a vector or a mapping
    with ``values`` and optional boolean ``valid``; masked rows never enter any
    metric. ``reference_sets[centre]`` contains the full frozen component.
    Graph/component labels must prove full-component equality. A mapping
    ``{'pairs': ..., 'distances': ...}`` supplies distortion distances only and
    therefore requires separate ``component_labels``. These labels are never
    passed to a model. When both graph and labels are given they must agree.
    ``scale`` is the pre-frozen tangent scale; ``distance_scale`` is a separately
    pre-frozen distance-unit multiplier; neither scale is fitted here.
    Reconstruction arrays, if supplied, must already have been computed from q.
    They are reported separately and can never replace decoded-SID geometry.
    """
    _positive(scale, 'scale')
    _positive(distance_scale, 'distance_scale')
    rows, books, q = _decode(sid, codebooks)
    report, lawful = _curvature_metrics(
        q, triples, targets, reference_sets, components_or_graph_optional, scale,
        fixed_squared, component_labels)
    occupancy = fidelity.codebook_occupancy(rows)
    for level, item in enumerate(occupancy):
        item.update({'capacity': len(books[level]),
                     'occupancy': item['used'] / len(books[level])})
    report.update({
        'representation': 'hard_sid_decoded', 'decoded_q': q.tolist(),
        'prefix': {f'L{level}': fidelity.prefix_collision(rows, level)
                   for level in (1, 2, 3)},
        'prefix_sharing': fidelity.prefix_sharing(rows),
        'codebook_occupancy': occupancy,
        'distance_distortion': _distortion(q, lawful, components_or_graph_optional,
                                           fixed_squared, scale, distance_scale),
        'reconstruction': _na('aligned embedding and hard-q reconstruction not supplied')})
    if (embeddings is None) != (reconstruction is None):
        raise ValueError('embedding and hard-q reconstruction must be supplied together')
    if embeddings is not None:
        target = _numpy(embeddings)
        if target.ndim != 2 or len(target) != len(q):
            raise ValueError('reconstruction and embeddings must align with SID items')
        report['reconstruction'] = evaluate_reconstruction(
            target, reconstruction, representation='hard-q')
    return report


def evaluate_heldout(sid, codebooks, heldout, scale=1.0, *, fixed_squared):
    """Consume only HeldOutEvaluation.strict_banks/load_strict, never model inputs.

    Banks need ``pairs``. Existing banks have topology fields, not class labels;
    their decoded distance summaries and frozen-field rank associations are
    reported directly. Optional binary ``labels`` permit a tie-correct distance
    AUC; absence is NA, never an inferred negative label or trained probe.
    """
    _, _, q = _decode(sid, codebooks)
    return _evaluate_heldout_coordinates(q, heldout, scale, fixed_squared,
                                         'hard_sid_decoded')


def evaluate_continuous_heldout(z, heldout, scale=1.0, *, fixed_squared):
    """Use actual continuous coordinates on the very same frozen strict banks."""
    return _evaluate_heldout_coordinates(z, heldout, scale, fixed_squared,
                                         'continuous_encoder_z')


def _evaluate_heldout_coordinates(values, heldout, scale, fixed_squared, representation):
    _positive(scale, 'scale')
    values = _numpy(values)
    if (values.ndim != 2 or not len(values) or not values.shape[1] or
            values.dtype.kind not in 'fiu' or not np.isfinite(values).all()):
        raise ValueError('held-out coordinates must be a finite nonempty item matrix')
    q = torch.from_numpy(values.astype(np.float64, copy=False))
    report = {'representation': representation, 'probe_trained': False,
              'labels_passed_to_model': False, 'tangent_scale': float(scale), 'banks': {}}
    for name in heldout.strict_banks():
        entry = heldout.entries.get(name, {})
        if not name.startswith('strict_') or entry.get('use') != 'evaluation':
            raise ValueError('only strict evaluation banks are permitted')
        bank = heldout.load_strict(name)
        pairs = _indices(bank['pairs'], 2, len(q), 'held-out pairs')
        distance = _pair_distances(q, pairs, fixed_squared, scale)
        result = {'pairs': len(pairs), 'sha256': entry.get('sha256'),
                  'distance': _summary(distance), 'topology': {},
                  'distance_ranking_auc': _na('binary labels not supplied')}
        for field in ('hop_full_visible', 'hop_directed_src_dst', 'hop_subgraph',
                      'two_hop_count', 'jaccard_out', 'jaccard_in', 'bhattacharyya_out'):
            if field not in bank:
                continue
            values = np.asarray(bank[field], dtype=np.float64)
            if values.shape != distance.shape:
                raise ValueError('held-out topology field must align with its pairs')
            valid = np.isfinite(values) & (values >= 0)
            result['topology'][field] = {
                'count': int(valid.sum()), 'missing': int((~valid).sum()),
                'distance_spearman': spearman(distance[valid], values[valid]),
                'target': _summary(values[valid]),
                'interpretation': 'association on strict pairs; visible-field name retained'}
        if 'labels' in bank:
            labels = np.asarray(bank['labels'])
            if labels.shape != distance.shape or not np.isin(labels, [0, 1]).all():
                raise ValueError('held-out labels must be aligned binary values')
            for label, key in ((1, 'positive'), (0, 'negative')):
                selected = distance[labels == label]
                result[key] = {'count': int(len(selected)),
                               'mean_distance': float(selected.mean()) if len(selected) else None}
            positives = int(labels.sum())
            negatives = len(labels) - positives
            if positives and negatives:
                ranks = average_ranks(-distance)
                result['distance_ranking_auc'] = float((ranks[labels == 1].sum() -
                    positives * (positives + 1) / 2) / (positives * negatives))
        report['banks'][name] = result
    return report


def compare_historical_tiger(repo):
    """Reuse the actual historical SID; unbound historical geometry stays NA."""
    return tiger_compare.compare(repo)
