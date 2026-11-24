"""Mechanism metrics for issue #255: SID structure, edge L1 sharing, strict
held-out groups, reconstruction error and the layer-wise A.1 curvature proxy.

All geometry uses the frozen issue #253 independent numpy distance kernel,
with its pre-frozen common_fixed3 scale. Training-native scales and learned
gates never enter this evaluation metric. Catalogue extraction lives in
evaluate_arm; this module has no training, asset-building or GPU side effects.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

from tools.issue253.evaluation_reference import fixed_metrics_numpy

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'stage2_RQ-VAE').is_dir())

#: The archived cross-model tangent scale of the frozen #253 ``common_fixed3``
#: cell.  Both arms are measured with this one scale, so the two arms are
#: comparable; the scale was frozen before the evaluation and is never fitted
#: here.
CROSS_MODEL_TANGENT_SCALE = 0.08433422446250916

#: The frozen catalogue: 24587 items, three levels of 256 codewords plus the
#: untouched fourth collision counter.
ITEMS = 24587
CODEBOOK_SIZE = 256
LEVELS = (1, 2, 3)

#: The frozen GraphMoRE A.1 target population.
EXPECTED_TRIPLES = 28391948

#: The three-expert curvatures of the frozen equal-weight symmetric distance.
EXPERT_CURVATURES = (-1, 0, 1)

_MOMENT_CHUNK = 512
_TRIPLE_CHUNK = 16384


def require(condition, message):
    """Fail closed with a message instead of producing a partial number."""
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    """The SHA-256 of one file, read in bounded blocks."""
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def file_record(path):
    """The path and SHA-256 of one input, for the provenance block."""
    path = Path(path)
    require(path.is_file(), 'missing input file: ' + str(path))
    return {'path': str(path.resolve()), 'sha256': sha256_file(path)}


# --------------------------------------------------------------------------
# SID structure
# --------------------------------------------------------------------------

def cumulative_prefixes(codebooks, rows):
    """The hard cumulative prefixes ``q^(l) = sum_{j<=l} e_{i,j}``.

    Accumulation preserves the checkpoint dtype and level order, including
    float32 rounding. Converting to float64 first would not prove bitwise
    identity to the production hard forward.
    """
    books = [np.asarray(book) for book in codebooks]
    rows = np.asarray(rows, dtype=np.int64)
    require(len(books) == 3, 'exactly three codebooks are required')
    require(rows.ndim == 2 and rows.shape[1] in (3, 4) and len(rows),
            'SID must be an (items, 3|4) integer table')
    dimension = books[0].shape[1]
    require(all(book.ndim == 2 and book.shape[1] == dimension and np.isfinite(book).all()
                for book in books), 'codebooks must be finite and of one dimension')
    require(all(book.dtype == books[0].dtype for book in books), 'codebook dtypes differ')
    running = np.zeros((len(rows), dimension), dtype=books[0].dtype)
    prefixes = []
    for level, book in enumerate(books):
        selected = rows[:, level]
        require(selected.min() >= 0 and selected.max() < len(book),
                'SID level %d outside its codebook' % (level + 1))
        running = running + book[selected]
        prefixes.append(running.copy())
    return np.stack(prefixes, axis=0)


def assert_prefix_identity(prefixes, rows, codebooks):
    """Check every hard prefix, not merely the final decoded representation."""
    actual = np.asarray(prefixes)
    decoded = cumulative_prefixes(codebooks, rows)
    require(actual.dtype == decoded.dtype and actual.shape == decoded.shape,
            'hard prefix shape/dtype differs from checkpoint decoding')
    for level in range(3):
        require(np.array_equal(actual[level], decoded[level]),
                'level-%d prefix differs from the actual SID' % (level + 1))
    return True


def l3_collision_group_sizes(rows):
    """The size of the L3 collision group every item belongs to."""
    rows = np.asarray(rows, dtype=np.int64)
    _, inverse, counts = np.unique(rows[:, :3], axis=0, return_inverse=True, return_counts=True)
    return counts[inverse].astype(np.int64)


def sid_structure(rows):
    """The SID structure of one arm.

    Reports the distinct L1 codes, the distinct L2 codes and L2 prefixes, the
    distinct L3 prefixes and triples, the three-code collision rate
    ``1 - distinct_triples / items``, the fourth-token occupancy and the
    per-item L3 collision-group size.
    """
    rows = np.asarray(rows, dtype=np.int64)
    require(rows.ndim == 2 and rows.shape[1] in (3, 4) and len(rows),
            'SID must be an (items, 3|4) integer table')
    count = len(rows)
    l1 = rows[:, 0]
    l2 = rows[:, 1]
    distinct_l1 = int(len(np.unique(l1)))
    distinct_l2 = int(len(np.unique(l2)))
    distinct_l2_prefix = int(len(np.unique(rows[:, :2], axis=0)))
    distinct_l3_prefix = int(len(np.unique(rows[:, :3], axis=0)))
    group_sizes = l3_collision_group_sizes(rows)
    fourth = rows[:, 3] if rows.shape[1] == 4 else np.zeros(count, dtype=np.int64)
    fourth_values, fourth_counts = np.unique(fourth, return_counts=True)
    histogram = {str(int(value)): int(total) for value, total in
                 zip(fourth_values, fourth_counts)}
    return {
        'items': count,
        'distinct_l1_codes': distinct_l1,
        'distinct_l2_codes': distinct_l2,
        'distinct_l2_prefixes': distinct_l2_prefix,
        'distinct_l3_prefixes': distinct_l3_prefix,
        'distinct_l3_triples': distinct_l3_prefix,
        'collision_l1': 1.0 - distinct_l1 / count,
        'collision_l2_prefix': 1.0 - distinct_l2_prefix / count,
        'collision_l3_prefix': 1.0 - distinct_l3_prefix / count,
        'three_code_collision': 1.0 - distinct_l3_prefix / count,
        'codebook_usage': [
            {'level': level + 1, 'capacity': CODEBOOK_SIZE,
             'used': int(len(np.unique(rows[:, level]))),
             'occupancy': float(len(np.unique(rows[:, level])) / CODEBOOK_SIZE),
             'counts': np.bincount(rows[:, level], minlength=CODEBOOK_SIZE).tolist()}
            for level in range(3)],
        'l3_group_size': {
            'min': int(group_sizes.min()), 'max': int(group_sizes.max()),
            'mean': float(group_sizes.mean()), 'median': float(np.median(group_sizes)),
            'items_in_collision_groups': int((group_sizes > 1).sum()),
            'singletons': int((group_sizes == 1).sum()),
        },
        'fourth_token': {
            'distinct': int(len(fourth_values)),
            'occupancy_share': float(len(fourth_values) / count),
            'max_count': int(fourth_counts.max()),
            'mean_count': float(fourth_counts.mean()),
            'median_count': float(np.median(fourth_counts)),
            'count_histogram': histogram,
        },
        'per_item_l3_group_size': group_sizes,
    }


# --------------------------------------------------------------------------
# Training-visible directed edges and the L1 sharing rate
# --------------------------------------------------------------------------

def load_visible_edges(path):
    """The frozen training-visible directed edges as ``(source, target)``.

    The asset is a CSR whose rows are the sources; every stored index is one
    directed edge.  The file is read only.
    """
    with np.load(Path(path), allow_pickle=False) as blob:
        indptr = np.asarray(blob['indptr'], dtype=np.int64)
        indices = np.asarray(blob['indices'], dtype=np.int64)
        shape = np.asarray(blob['shape'], dtype=np.int64)
    require(len(indptr) == int(shape[0]) + 1 and int(shape[0]) == int(shape[1]),
            'the visible-edge asset must be a square CSR')
    counts = np.diff(indptr)
    source = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
    return source, indices


def edge_l1_sharing(rows, source, target, *, items=None):
    """The L1 sharing rate over the training-visible directed edges.

    ``per_edge`` is the share of covered edges whose two endpoints carry the
    same L1 code.  The per-centre macro rates are computed in both directions:
    over the out-neighbours of a centre and over its in-neighbours.  The
    coverage is the share of edges whose two endpoints have a legal SID.
    """
    rows = np.asarray(rows, dtype=np.int64)
    source = np.asarray(source, dtype=np.int64)
    target = np.asarray(target, dtype=np.int64)
    require(source.shape == target.shape, 'edge endpoints must be aligned')
    count = len(rows)
    if items is None:
        items = count
    legal = ((source >= 0) & (source < items) & (target >= 0) & (target < items))
    require(items == count and bool(legal.all()),
            'all directed edges must have catalogue SID coverage')
    covered = int(legal.sum())
    total = len(source)
    shared = np.zeros(total, dtype=bool)
    if covered:
        left = source[legal]
        right = target[legal]
        shared[legal] = rows[left, 0] == rows[right, 0]
    shared_edges = int(shared.sum())
    per_edge_rate = float(shared_edges / covered) if covered else None

    def direction(index, other):
        degrees = np.zeros(items, dtype=np.int64)
        counts_shared = np.zeros(items, dtype=np.int64)
        if covered:
            np.add.at(degrees, index, 1)
            np.add.at(counts_shared, index, shared[legal].astype(np.int64))
        present = degrees > 0
        rates = np.zeros(items, dtype=np.float64)
        rates[present] = counts_shared[present] / degrees[present]
        return {
            'centres': int(present.sum()),
            'edges': int(degrees.sum()),
            'shared_edges': int(counts_shared.sum()),
            'macro_rate': float(rates[present].mean()) if present.any() else None,
            'micro_rate': float(counts_shared.sum() / degrees.sum()) if degrees.sum() else None,
            'min_rate': float(rates[present].min()) if present.any() else None,
            'max_rate': float(rates[present].max()) if present.any() else None,
            'per_item_degree': degrees,
            'per_item_shared': counts_shared,
            'per_item_rate': rates,
        }, other

    out_direction, _ = direction(source[legal], None)
    in_direction, _ = direction(target[legal], None)
    return {
        'edges_total': total,
        'edges_covered': covered,
        'coverage_rate': float(covered / total) if total else None,
        'shared_edges': shared_edges,
        'per_edge_rate': per_edge_rate,
        'out_direction': out_direction,
        'in_direction': in_direction,
        'per_edge_shared': shared,
    }


# --------------------------------------------------------------------------
# The layer-wise A.1 curvature proxy
# --------------------------------------------------------------------------

def component_moments(values, labels, *, scale=CROSS_MODEL_TANGENT_SCALE,
                      chunk=_MOMENT_CHUNK):
    """The mean distance from every item to its frozen component.

    This is the frozen #253 estimator's moment identity: the mean is taken over
    the entire frozen component of the item, the centre included.  The block
    traversal, the chunk size and the symmetrisation are the #253 ones, so the
    numbers are the #253 numbers.
    """
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels)
    require(values.ndim == 2 and len(values) == len(labels) and np.isfinite(values).all(),
            'finite aligned coordinates and component labels are required')
    require(type(chunk) is int and chunk > 0, 'chunk must be a positive integer')
    require(np.isfinite(scale) and scale > 0, 'scale must be finite and positive')
    moments = np.zeros(len(values), dtype=np.float64)
    try:
        from threadpoolctl import threadpool_limits
        limits = threadpool_limits(limits=1)
    except Exception:                                   # pragma: no cover
        limits = None
    try:
        for component in np.unique(labels):
            indices = np.flatnonzero(labels == component)
            for start in range(0, len(indices), chunk):
                left = indices[start:start + chunk]
                for offset in range(start, len(indices), chunk):
                    right = indices[offset:offset + chunk]
                    block = fixed_metrics_numpy(values[left], values[right], scale,
                                                metrics=(3,))['fixed3']
                    moments[left] += block.sum(axis=1) / len(indices)
                    if offset != start:
                        moments[right] += block.sum(axis=0) / len(indices)
    finally:
        if limits is not None:
            limits.restore_original_limits()
    return moments


def paired_fixed3(left, right, *, scale=CROSS_MODEL_TANGENT_SCALE):
    """The frozen three-expert ambient squared distance of aligned rows."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    require(left.shape == right.shape and left.ndim == 2,
            'paired coordinates must be aligned matrices')
    return fixed_metrics_numpy(left, right, scale, paired=True, metrics=(3,))['fixed3']


def a1_proxy(values, triples, labels, *, scale=CROSS_MODEL_TANGENT_SCALE,
             moments=None, chunk=_TRIPLE_CHUNK):
    """The GraphMoRE A.1 proxy of every triple, by the frozen moment identity.

    ``K = mean_a[d(q_a, q_m)] + d(q_b, q_c)/4 - (mean_a[d(q_a, q_b)] +
    mean_a[d(q_a, q_c)])/2`` where ``a`` runs over the whole frozen component of
    the centre.  This is exactly the estimator the frozen #253 cell used, and it
    is identical to the explicit-mean oracle of
    ``stage2_RQ-VAE/TIGER_prefix_curvature_gate_rq/tests/reference_layer_gate.py``.
    """
    triples = np.asarray(triples, dtype=np.int64)
    require(triples.ndim == 2 and triples.shape[1] == 3, 'triples must be (n, 3)')
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels)
    require(labels.shape == (len(values),) and np.isfinite(values).all(),
            'aligned finite catalogue and components required')
    require(np.all(triples >= 0) and np.all(triples < len(values)),
            'triple outside catalogue')
    require(np.all(labels[triples[:, 0]] == labels[triples[:, 1]]) and
            np.all(labels[triples[:, 0]] == labels[triples[:, 2]]),
            'triple crosses frozen reference components')
    if moments is None:
        moments = component_moments(values, labels, scale=scale)
    centre, b, c = triples[:, 0], triples[:, 1], triples[:, 2]
    predicted = np.empty(len(triples), dtype=np.float64)
    for start in range(0, len(triples), chunk):
        end = min(start + chunk, len(triples))
        pair = paired_fixed3(values[b[start:end]], values[c[start:end]], scale=scale)
        predicted[start:end] = (moments[centre[start:end]]
                                - 0.5 * (moments[b[start:end]] + moments[c[start:end]])
                                + 0.25 * pair)
    return predicted


def a1_proxy_explicit(values, triples, references, squared):
    """The same proxy by an explicit mean over the reference set.

    ``squared`` is the frozen distance primitive; the aggregation here is an
    explicit per-reference mean, which is a different derivation from the
    moment identity and therefore an independent oracle.
    """
    values = np.asarray(values, dtype=np.float64)
    triples = np.asarray(triples, dtype=np.int64)
    references = np.asarray(references, dtype=np.int64)

    def distance(left, right):
        return float(np.asarray(squared(values[left][None, :], values[right][None, :])).reshape(-1)[0])

    result = []
    for centre, b, c in triples:
        first = float(np.mean([distance(a, centre) for a in references]))
        second = distance(b, c) / 4.0
        third = (float(np.mean([distance(a, b) for a in references]))
                 + float(np.mean([distance(a, c) for a in references]))) / 2.0
        result.append(first + second - third)
    return np.asarray(result, dtype=np.float64)


def reduce_curvature_report(target, predicted, centres):
    """The frozen curvature reduction: macro/micro MAE, sign recalls, balance.

    The definitions are the #253 ones: the micro MAE is the mean absolute
    error over every valid triple, the macro MAE is the mean over the centres
    of the per-centre mean absolute error, and the sign recalls are macro
    averaged over the target signs that occur.
    """
    target = np.asarray(target, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    centres = np.asarray(centres, dtype=np.int64)
    require(target.ndim == 1 and len(target) and predicted.shape == target.shape
            and centres.shape == target.shape and np.isfinite(target).all()
            and np.isfinite(predicted).all(),
            'aligned nonempty finite curvature rows are required')
    error = np.abs(predicted - target)
    signed = predicted - target
    _, inverse, counts = np.unique(centres, return_inverse=True, return_counts=True)
    per_item = np.bincount(inverse, weights=error) / counts
    truth_sign = np.sign(target)
    prediction_sign = np.sign(predicted)
    recalls, strata, present = {}, {}, []
    for sign, name in ((-1, 'negative'), (0, 'zero'), (1, 'positive')):
        selected = truth_sign == sign
        number = int(selected.sum())
        if not number:
            recalls[name] = {'status': 'NA', 'reason': 'target sign absent'}
            strata[name] = {'status': 'NA', 'reason': 'no valid triples in this stratum'}
            continue
        present.append(name)
        recalls[name] = {'status': 'OK', 'count': number,
                         'recall': float((prediction_sign[selected] == sign).mean())}
        local_centres, local_inverse, local_counts = np.unique(
            centres[selected], return_inverse=True, return_counts=True)
        local_mae = np.bincount(local_inverse, weights=error[selected]) / local_counts
        strata[name] = {'status': 'OK', 'count': number, 'centres': len(local_centres),
                        'micro_mae': float(error[selected].mean()),
                        'macro_mae': float(local_mae.mean())}
    return {
        'status': 'OK',
        'count': int(len(target)),
        'centres': int(len(counts)),
        'micro_mae': float(error.mean()),
        'macro_mae': float(per_item.mean()),
        'mean_signed_error': float(signed.mean()),
        'sign': {
            'accuracy': float((prediction_sign == truth_sign).mean()),
            'balanced_accuracy': float(np.mean([recalls[name]['recall'] for name in present])),
            'present_signs': present,
            'recall': recalls,
            'zero_tolerance': 0.0,
        },
        'strata': strata,
        'per_item_macro_mae': per_item,
    }


def per_item_curvature_rows(target, predicted, centres):
    """The per-item distribution summary, one row per centre.

    The rows assume the triples of a centre are contiguous, which is how the
    frozen shards store them.
    """
    target = np.asarray(target, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    centres = np.asarray(centres, dtype=np.int64)
    require(len(target) == len(predicted) == len(centres) and len(target),
            'aligned nonempty curvature rows are required')
    boundaries = np.flatnonzero(centres[1:] != centres[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(centres)]))
    require(len(np.unique(centres)) == len(starts),
            'each centre must occupy exactly one contiguous range')
    truth_sign = np.sign(target)
    prediction_sign = np.sign(predicted)
    rows = []
    for start, end in zip(starts, ends):
        local_target = target[start:end]
        local_predicted = predicted[start:end]
        local_truth = truth_sign[start:end]
        local_prediction = prediction_sign[start:end]
        row = {
            'item': int(centres[start]),
            'count': int(end - start),
            'target_mean': float(local_target.mean()),
            'target_std': float(local_target.std()),
            'target_min': float(local_target.min()),
            'target_q25': float(np.quantile(local_target, .25)),
            'target_median': float(np.median(local_target)),
            'target_q75': float(np.quantile(local_target, .75)),
            'target_max': float(local_target.max()),
            'predicted_mean': float(local_predicted.mean()),
            'predicted_std': float(local_predicted.std()),
            'predicted_min': float(local_predicted.min()),
            'predicted_q25': float(np.quantile(local_predicted, .25)),
            'predicted_median': float(np.median(local_predicted)),
            'predicted_q75': float(np.quantile(local_predicted, .75)),
            'predicted_max': float(local_predicted.max()),
            'mean_signed_error': float((local_predicted - local_target).mean()),
            'mean_abs_error': float(np.abs(local_predicted - local_target).mean()),
        }
        for sign, name in ((-1, 'negative'), (0, 'zero'), (1, 'positive')):
            selected = local_truth == sign
            number = int(selected.sum())
            row['recall_' + name] = (float((local_prediction[selected] == sign).mean())
                                     if number else None)
            row['count_' + name] = number
            row['mae_' + name] = (float(np.abs(local_predicted[selected] -
                                                local_target[selected]).mean())
                                  if number else None)
        rows.append(row)
    return rows


def distribution_summary(values):
    """A small distribution summary of a numeric vector."""
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {'count': 0}
    return {
        'count': int(values.size),
        'mean': float(values.mean()),
        'std': float(values.std()),
        'min': float(values.min()),
        'q25': float(np.quantile(values, .25)),
        'median': float(np.median(values)),
        'q75': float(np.quantile(values, .75)),
        'max': float(values.max()),
    }


class CurvatureAccumulator:
    """Reduce complete centre-owned shards without retaining triple arrays."""

    def __init__(self):
        self.seen = set()
        self.count = 0
        self.absolute_sum = 0.
        self.macro_sum = 0.
        self.signed_sum = 0.
        self.strata = {name: {'count': 0, 'correct': 0., 'absolute_sum': 0.,
                              'macro_sum': 0., 'centres': 0}
                       for name in ('negative', 'zero', 'positive')}
        self.item_mae = []

    def add(self, target, predicted, centres):
        require(np.isfinite(target).all() and np.isfinite(predicted).all(),
                'nonfinite curvature rows')
        rows = per_item_curvature_rows(target, predicted, centres)
        for row in rows:
            item, count = row['item'], row['count']
            require(item not in self.seen, 'centre appears in multiple shard ranges')
            self.seen.add(item)
            self.count += count
            self.absolute_sum += row['mean_abs_error'] * count
            self.signed_sum += row['mean_signed_error'] * count
            self.macro_sum += row['mean_abs_error']
            self.item_mae.append(row['mean_abs_error'])
            for name, total in self.strata.items():
                number = row['count_' + name]
                if number:
                    total['count'] += number
                    total['correct'] += number * row['recall_' + name]
                    total['absolute_sum'] += number * row['mae_' + name]
                    total['macro_sum'] += row['mae_' + name]
                    total['centres'] += 1
        return rows

    def finish(self, *, expected_count, expected_centres):
        require(self.count == expected_count and len(self.seen) == expected_centres,
                'partial curvature population is forbidden')
        require(self.count > 0, 'empty curvature population')
        recalls, strata, present = {}, {}, []
        for name, total in self.strata.items():
            if not total['count']:
                recalls[name] = strata[name] = {'status': 'NA', 'reason': 'target sign absent'}
                continue
            present.append(name)
            recalls[name] = {'status': 'OK', 'count': total['count'],
                             'recall': total['correct'] / total['count']}
            strata[name] = {'status': 'OK', 'count': total['count'],
                            'centres': total['centres'],
                            'micro_mae': total['absolute_sum'] / total['count'],
                            'macro_mae': total['macro_sum'] / total['centres']}
        return {'status': 'OK', 'count': self.count, 'centres': len(self.seen),
                'micro_mae': self.absolute_sum / self.count,
                'macro_mae': self.macro_sum / len(self.seen),
                'mean_signed_error': self.signed_sum / self.count,
                'per_item_mae_distribution': distribution_summary(self.item_mae),
                'strata': strata,
                'sign': {'accuracy': sum(v['correct'] for v in self.strata.values()) / self.count,
                         'balanced_accuracy': float(np.mean([recalls[n]['recall'] for n in present])),
                         'recall': recalls, 'present_signs': present, 'zero_tolerance': 0.}}


def strict_prefix_report(rows, heldout):
    """Use existing strict banks verbatim; never generate groups or controls."""
    rows = np.asarray(rows)
    banks = {}
    for name in heldout.strict_banks():
        entry = heldout.entries[name]
        require(name.startswith('strict_') and entry.get('use') == 'evaluation',
                'only frozen strict evaluation banks are permitted')
        pairs = np.asarray(heldout.load_strict(name)['pairs'])
        require(pairs.ndim == 2 and pairs.shape[1] == 2 and pairs.dtype.kind in 'iu'
                and np.all(pairs >= 0) and np.all(pairs < len(rows)),
                'invalid frozen strict pair identities')
        banks[name] = {
            'pairs': len(pairs), 'sha256': entry['sha256'], 'split': _bank_split(name),
            'prefix': {'L%d' % level: {
                'count': len(pairs),
                'shared': int(np.all(rows[pairs[:, 0], :level] ==
                                     rows[pairs[:, 1], :level], axis=1).sum()),
                'rate': float(np.all(rows[pairs[:, 0], :level] ==
                                     rows[pairs[:, 1], :level], axis=1).mean())
                        if len(pairs) else None}
                       for level in LEVELS}}
    controls = {}
    for split in ('val', 'test'):
        control_name = 'strict_struct_%s_degree_matched' % split
        require(control_name in banks and banks[control_name]['pairs'] > 0,
                'missing frozen degree-matched control: ' + split)
        control = banks[control_name]['prefix']['L1']['rate']
        groups = {}
        for name, bank in banks.items():
            if bank['split'] == split and name != control_name:
                rate = bank['prefix']['L1']['rate']
                groups[name] = {'pairs': bank['pairs'], 'l1_rate': rate,
                                'l1_difference': rate - control if rate is not None else None}
        require(bool(groups), 'missing strict structure groups: ' + split)
        controls[split] = {'control_bank': control_name, 'control_l1_rate': control,
                           'groups': groups}
    return {'banks': banks, 'degree_matched_control': controls,
            'generated_groups': False, 'suffix_excluded': True}


# --------------------------------------------------------------------------
# The strict held-out structure groups
# --------------------------------------------------------------------------

def _bank_split(name):
    for split in ('val', 'test'):
        if '_%s_' % split in name:
            return split
    return 'unknown'


# --------------------------------------------------------------------------
# Reconstruction
# --------------------------------------------------------------------------

def reconstruction_report(embeddings, reconstruction, *, representation):
    """The reconstruction error against the frozen 768-d embeddings.

    The mean squared error is taken over the 768 dimensions of every item and
    then over the items; the per-item L2 error is the raw table.
    """
    from tools.issue251.evaluation import evaluate_reconstruction

    report = evaluate_reconstruction(embeddings, reconstruction, representation=representation)
    report['source'] = 'caller-supplied ' + representation + ' reconstruction'
    return report
