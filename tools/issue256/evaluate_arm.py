"""Complete R01 training-curvature and frozen validation diagnostics; no test rows."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from tools.issue256.identity import ROOT, RUN_ID, EVIDENCE
from tools.issue256.extract_catalogues import context, record
from tools.issue252.final_evaluation import _coverage
from tools.issue255 import mechanism_metrics as metrics

OUTPUT = EVIDENCE / 'evaluation'
CPU_THREADS = 1
ASSETS = ROOT / 'assets'
EMBEDDINGS = ROOT / 'stage1_GeneEmbedding/output/sentence_t5.npy'


def write_json(path, value):
    partial = path.with_name(path.name + '.partial')
    partial.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    partial.replace(path)


def validation_prefixes(rows, manifest):
    entries = [e for e in manifest['assets'] if e['name'].startswith('strict_struct_val_')]
    expected = {'strict_struct_val_' + suffix for suffix in
                ('direct', 'two_hop', 'shared_in', 'shared_out', 'degree_matched')}
    metrics.require({e['name'] for e in entries} == expected, 'validation bank set drift')
    banks = {}
    for entry in entries:
        path = ROOT / entry['path']
        metrics.require(entry['use'] == 'evaluation' and record(path)['sha256'] == entry['sha256'],
                        'validation asset identity changed')
        with np.load(path, allow_pickle=False) as blob:
            pairs = blob['pairs']
        metrics.require(pairs.ndim == 2 and pairs.shape[1] == 2 and pairs.dtype.kind in 'iu'
                        and len(pairs) > 0 and pairs.min() >= 0 and pairs.max() < len(rows),
                        'invalid complete validation pairs')
        banks[entry['name']] = {'split': 'val', 'pairs': len(pairs), 'asset': record(path),
            'prefix': {'L%d' % level: {
                'shared': int(np.all(rows[pairs[:, 0], :level] == rows[pairs[:, 1], :level], axis=1).sum()),
                'rate': float(np.all(rows[pairs[:, 0], :level] == rows[pairs[:, 1], :level], axis=1).mean())}
                       for level in metrics.LEVELS}}
    control = banks['strict_struct_val_degree_matched']['prefix']['L1']['rate']
    return {'banks': banks, 'degree_matched_control': {'val': {
        'control_bank': 'strict_struct_val_degree_matched', 'control_l1_rate': control,
        'groups': {name: {'pairs': bank['pairs'], 'l1_rate': bank['prefix']['L1']['rate'],
                         'l1_difference': bank['prefix']['L1']['rate'] - control}
                   for name, bank in banks.items() if not name.endswith('degree_matched')}}},
        'generated_groups': False, 'suffix_excluded': True, 'test_banks_opened': False}


def evaluate_layer(values, labels, coverage, level):
    values = np.asarray(values, dtype=np.float64)
    moments = metrics.component_moments(values, labels)
    accumulator = metrics.CurvatureAccumulator()
    path = OUTPUT / ('r01_L%d_items.jsonl' % level)
    partial = path.with_name(path.name + '.partial')
    excluded = 0
    with partial.open('w') as handle:
        for shard in coverage['shards']:
            with np.load(ASSETS / 'shards' / shard['name'], allow_pickle=False) as blob:
                valid = blob['valid']
                metrics.require(int(valid.sum()) == shard['valid']
                                and int((~valid).sum()) == shard['excluded'], 'triple coverage changed')
                triples = np.column_stack([blob[key][valid] for key in ('centre', 'b', 'c')])
                target = blob['curvature'][valid]
            excluded += shard['excluded']
            predicted = metrics.a1_proxy(values, triples, labels, moments=moments, chunk=16384)
            for row in accumulator.add(target, predicted, triples[:, 0]):
                row.update(run_id=RUN_ID, level=level)
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
            print('[issue256] L%d %s triples=%d' % (level, shard['name'], len(triples)), flush=True)
    reduced = accumulator.finish(expected_count=metrics.EXPECTED_TRIPLES, expected_centres=24326)
    metrics.require(excluded == coverage['excluded_triples'], 'excluded population changed')
    partial.replace(path)
    reduced.update(excluded=excluded, per_item=record(path),
                   items_without_lawful_triples=sorted(set(range(metrics.ITEMS)) - accumulator.seen))
    return {'representation': 'hard_cumulative_prefix', 'level': level,
        'metric': 'common_fixed3', 'expert_curvatures': [-1, 0, 1],
        'tangent_scale': metrics.CROSS_MODEL_TANGENT_SCALE, 'scale_fitted_on_evaluation': False,
        'precision': 'float32 hard-prefix accumulation; float64 independent metric',
        'reference_scope': 'entire original frozen component including centre',
        'population': 'training graph, in-sample; not held-out structure generalization',
        'curvature': reduced}


def main():
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=CPU_THREADS)
    started = time.monotonic()
    _, _, _, identity = context()
    pointer_path = OUTPUT / 'EXTRACTION.json'
    pointer = json.loads(pointer_path.read_text())
    metrics.require(pointer['context'] == identity, 'CUDA extraction belongs to another generation')
    catalogue = Path(pointer['catalogue']['path'])
    metrics.require(record(catalogue) == pointer['catalogue'], 'extracted catalogue changed')
    receipt_path = Path(pointer['receipt']['path'])
    metrics.require(record(receipt_path) == pointer['receipt'], 'extraction receipt changed')
    with np.load(catalogue, allow_pickle=False) as blob:
        arrays = {key: blob[key] for key in blob.files}
    manifest_path = ASSETS / 'manifest.json'
    manifest_record = record(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    metrics.require(manifest_record['sha256'] == identity['checkpoint_provenance']['asset_manifest_sha256'],
                    'pre-training manifest changed')
    inputs = {}
    for entry in manifest['assets']:
        descriptor = record(ROOT / entry['path'])
        metrics.require(descriptor['sha256'] == entry['sha256'], 'frozen asset changed: ' + entry['path'])
        inputs[entry['path']] = descriptor
    labels, coverage = _coverage(ASSETS, metrics.ITEMS)
    rows = arrays['sid']
    source, target = metrics.load_visible_edges(ASSETS / 'train/visible_forward.npz')
    structure = metrics.sid_structure(rows)
    edge = metrics.edge_l1_sharing(rows, source, target)
    reconstruction = metrics.reconstruction_report(
        np.load(EMBEDDINGS, mmap_mode='r', allow_pickle=False), arrays['reconstruction'],
        representation='hard_q_L3')
    per_item = OUTPUT / 'r01_sid_items.npz'
    np.savez(per_item, item=np.arange(metrics.ITEMS), sid=rows,
        l3_collision_group_size=structure.pop('per_item_l3_group_size'),
        out_degree=edge['out_direction'].pop('per_item_degree'),
        out_shared=edge['out_direction'].pop('per_item_shared'),
        out_rate=edge['out_direction'].pop('per_item_rate'),
        in_degree=edge['in_direction'].pop('per_item_degree'),
        in_shared=edge['in_direction'].pop('per_item_shared'),
        in_rate=edge['in_direction'].pop('per_item_rate'),
        reconstruction_l2=np.asarray(reconstruction.pop('per_item_l2')))
    edge.pop('per_edge_shared')
    gates = []
    for level in range(3):
        weights, attention = arrays['weights'][:, level], arrays['attention'][:, level]
        gates.append({'level': level + 1, 'weights_mean': weights.mean(0).tolist(),
            'weights_distribution': [metrics.distribution_summary(weights[:, k]) for k in range(3)],
            'gate_entropy_nats': metrics.distribution_summary(
                -np.sum(weights * np.log(np.maximum(weights, 1.e-30)), axis=-1)),
            'attention_mean_expert_by_token': attention.mean(0).tolist(),
            'attention_entropy_nats_per_expert': [metrics.distribution_summary(
                -np.sum(attention[:, k] * np.log(np.maximum(attention[:, k], 1.e-30)), axis=-1))
                for k in range(3)]})
    generation = {'catalogue': record(catalogue), 'sid_structure': structure,
        'visible_directed_l1_sharing': edge, 'strict_heldout': validation_prefixes(rows, manifest),
        'reconstruction': reconstruction, 'per_item_sid': record(per_item),
        'gates': {'shared_across_levels': False, 'levels': gates,
            'token_order': ['one_hop', 'two_hop', 'in_only', 'out_only', 'null'],
            'valid_count_by_token': arrays['token_mask'].sum(0).tolist()},
        'prefix_identity': {'all_levels_bitwise_equal': True, 'actual_sid_reexport_equal': True},
        'layers': {}}
    for level in metrics.LEVELS:
        generation['layers']['L%d' % level] = evaluate_layer(arrays['prefixes'][level - 1], labels, coverage, level)
    metrics.require(context()[-1] == identity, 'checkpoint/source identity changed during evaluation')
    metrics.require(record(manifest_path) == manifest_record, 'manifest changed during evaluation')
    for descriptor in inputs.values():
        metrics.require(record(descriptor['path']) == descriptor, 'asset changed during evaluation')
    metrics.require(record(catalogue) == pointer['catalogue'] and record(receipt_path) == pointer['receipt'],
                    'catalogue/extraction changed during evaluation')
    report = {'schema': 'genrec.issue256.mechanism_evaluation/1', 'status': 'COMPLETE',
        'run_id': RUN_ID, 'context': identity, 'generation': generation, 'coverage': coverage,
        'comparison': {'metric': 'common_fixed3', 'tangent_scale': metrics.CROSS_MODEL_TANGENT_SCALE,
                       'all_levels_same_units': True, 'scale_fitted_on_evaluation': False},
        'provenance': {'manifest': manifest_record, 'frozen_assets': inputs,
            'extraction': record(pointer_path), 'evaluator': record(__file__),
            'metric_implementation': record(metrics.__file__),
            'population_reader': record(ROOT / 'tools/issue252/final_evaluation.py')},
        'test_access': False, 'elapsed_wall_seconds': time.monotonic() - started}
    write_json(OUTPUT / 'FINAL_EVALUATION.json', report)
    print(json.dumps({'status': 'COMPLETE', 'report': record(OUTPUT / 'FINAL_EVALUATION.json'),
                      'elapsed_wall_seconds': report['elapsed_wall_seconds']}, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
