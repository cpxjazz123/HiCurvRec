"""Read-only, no-argument, post-training Issue255 mechanism evaluation.

Run with genrec_env_v2/bin/python3.9 -B -m tools.issue255.evaluate_arm.
No training, SID replacement, asset generation, scale fitting or Stage3 call.
Only OUTPUT is writable. This CPU consumer requires the separately completed,
CUDA/source-bound EXTRACTION.json and catalogues; it never quantizes a model.
FINAL_EVALUATION.json is published last, only after all six complete cells and
input rechecks succeed. A failed/partial run has no published success report.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

from tools.issue255 import mechanism_metrics as metrics
from tools.issue255.identity import ROOT, ARM, RUN_ID, RUN, EVIDENCE
from tools.issue252.final_evaluation import load_bound_checkpoint, load_sid, _coverage
from tools.issue255 import historical
from tools.issue252.run_workspace import load_run
from tools.issue253.final_evaluation import _books, read_json

ASSETS = ROOT / 'assets'
OUTPUT = EVIDENCE / 'evaluation'
OLD_ARM = 'TIGER_three_expert_attention_rq'
COMPARISON = ROOT / 'results/research/issue253/FINAL_EVALUATION.json'
COMPARISON_SUMMARY = ROOT / 'results/research/issue253/FINAL_EVALUATION_SUMMARY.json'
COMPARISON_RECEIPT = ROOT / 'results/research/issue253/FORMAL_RUN.json'
EMBEDDINGS = ROOT / 'stage1_GeneEmbedding/output/sentence_t5.npy'
TRIPLE_CHUNK = 16384
EXPECTED_CENTRES = 24326
COMPARISON_SHA256 = '7492f837377cd01dad566c4950538ee65056d8532eefbcf58d34bdf2f37896ed'
TIGER_SID_SHA256 = '025ad8bd5809ce6296047514a80a4fedd012f3145d184eb4babd51c76c140059'
REPORT = OUTPUT / 'FINAL_EVALUATION.json'
TABLE = OUTPUT / 'METRICS.tsv'

require = metrics.require
record = metrics.file_record


def write_json(path, value):
    path = Path(path)
    partial = path.with_name(path.name + '.partial')
    partial.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    partial.replace(path)


def _relative(name):
    value = Path(name)
    require(not value.is_absolute() and '..' not in value.parts, 'unsafe provenance path')
    return ROOT / value


def comparison_generation():
    """Read the actual frozen report header without loading its 681-MB rows."""
    require(record(COMPARISON)['sha256'] == COMPARISON_SHA256, 'frozen #253 report changed')
    with COMPARISON.open() as handle:
        header = handle.read(65536)
    before, separator, _ = header.partition('"curvature":')
    require(bool(separator), 'missing frozen #253 common-cell header')
    # The frozen schema starts root/generations/issue253/hard_q/cells/common_fixed3.
    prefix = json.loads(before + '"curvature":null' + '}' * 6)
    cell = prefix['generations']['issue253']['hard_q']['cells']['common_fixed3']
    require(prefix['schema'] == 'genrec.issue253.final_evaluation/1'
            and cell['expert_curvatures'] == [-1, 0, 1]
            and cell['reference_scope'] == 'entire frozen component'
            and cell['tangent_scale'] == metrics.CROSS_MODEL_TANGENT_SCALE,
            'common_fixed3 metric/scale drift in frozen report')
    summary = read_json(COMPARISON_SUMMARY)
    require(summary['full_report'] == record(COMPARISON)
            and summary['run_id'] == prefix['run_id'], 'comparison summary/report identity mismatch')
    compact_cell = summary['generations']['issue253']['hard_q']['cells']['common_fixed3']
    require(compact_cell['tangent_scale'] == cell['tangent_scale']
            and compact_cell['curvature']['count'] == metrics.EXPECTED_TRIPLES
            and compact_cell['curvature']['centres'] == EXPECTED_CENTRES,
            'comparison common cell is partial or uses another scale')
    receipt = read_json(COMPARISON_RECEIPT)
    require(receipt['arm'] == OLD_ARM and receipt['run_id'] == prefix['run_id'],
            'comparison dispatch identifies another generation')
    run = load_run(ROOT, receipt['run_id'], OLD_ARM)
    for key in ('source', 'result'):
        require(run[key] == receipt[key], 'comparison dispatch path changed')
    require(record(run['workspace'])['sha256'] == receipt['workspace_sha256'],
            'comparison workspace changed')
    require(summary['provenance']['formal_dispatch'] == record(COMPARISON_RECEIPT),
            'comparison dispatch differs from final evidence')
    return run, summary, cell


def artifact_paths(run):
    result = Path(run['result'])
    return {'checkpoint': result / 'rqvae_best.pth', 'sid': result / 'item_sids.json',
            'binding': result / 'archive_binding.json',
            'export_manifest': result / 'export_manifest.json'}


def validate_inputs(payloads, summary):
    """One shared full-reference population; old manifests need not equal v10."""
    files = {}

    def bound(name, digest):
        path = _relative(name)
        if name not in files:
            files[name] = record(path)
        require(files[name]['sha256'] == digest, 'frozen input changed: ' + name)

    manifest = read_json(ASSETS / 'manifest.json')
    manifest_record = record(ASSETS / 'manifest.json')
    require(payloads['issue255']['provenance']['asset_manifest_sha256'] == manifest_record['sha256'],
            'issue255 pre-training manifest changed')
    for entry in manifest['assets']:
        bound(entry['path'], entry['sha256'])
        require(payloads['issue255']['provenance']['assets'].get(entry['path']) == entry['sha256'],
                'issue255 checkpoint asset binding missing: ' + entry['path'])
    for payload in payloads.values():
        provenance = payload['provenance']
        for section in ('assets', 'training_inputs'):
            for name, digest in provenance[section].items():
                bound(name, digest)
        for name in ('assets/index.npz', 'assets/node_moments.npz'):
            require(name in provenance['assets'], 'checkpoint lacks full-reference identity')
        require('stage1_GeneEmbedding/output/sentence_t5.npy' in provenance['assets']
                and 'results/stage0_build_parquet/train.parquet' in provenance['training_inputs'],
                'checkpoint lacks embedding/training-input binding')
    for name, descriptor in summary['provenance']['frozen_inputs']['files'].items():
        bound(name, descriptor['sha256'])
    shared = ('assets/index.npz', 'assets/node_moments.npz',
              'assets/train/visible_forward.npz', 'assets/train/item_ids.npy')
    for name in shared:
        require(payloads['issue253']['provenance']['assets'][name] ==
                payloads['issue255']['provenance']['assets'][name], 'cross-model population differs: ' + name)
    target = np.load(EMBEDDINGS, mmap_mode='r', allow_pickle=False)
    require(target.shape == (metrics.ITEMS, 768) and target.dtype == np.float32
            and np.isfinite(target).all(), 'invalid frozen embedding catalogue')
    ids = np.load(ASSETS / 'train/item_ids.npy', allow_pickle=False)
    require(np.array_equal(ids, np.arange(metrics.ITEMS)), 'global item identities changed')
    return {'manifest': manifest_record, 'files': files}


def common_metric():
    return {'name': 'common_fixed3', 'expert_curvatures': [-1, 0, 1],
            'tangent_scale': metrics.CROSS_MODEL_TANGENT_SCALE,
            'scale_fitted_on_evaluation': False,
            'reference_scope': 'entire frozen component'}


def extraction_sources():
    """Bind the producer, consumer and framework code, not just array bytes."""
    paths = (Path(__file__), Path(metrics.__file__),
             ROOT / 'tools/issue255/identity.py',
             ROOT / 'tools/issue255/historical.py',
             ROOT / 'tools/issue255/extract_catalogues.py',
             ROOT / 'stage2_RQ-VAE' / ARM / 'catalogue_audit_entry.py',
             ROOT / 'tools/research/stage2/launcher.py',
             ROOT / 'tools/research/stage2/contract.py',
             ROOT / 'tools/research/stage2/anticollapse/patch_tree.py',
             ROOT / 'tools/issue252/run_workspace.py',
             ROOT / 'tools/issue252/archive_audit.py',
             ROOT / 'tools/issue252/final_evaluation.py',
             ROOT / 'tools/issue253/final_evaluation.py',
             ROOT / 'tools/issue253/evaluation_reference.py')
    return {str(path.relative_to(ROOT)): record(path) for path in paths}


def bound_generations():
    """Resolve only the frozen comparison and approved corrected generation."""
    old_run, summary, cell = comparison_generation()
    runs = {'issue253': old_run, 'issue255': load_run(ROOT, RUN_ID, ARM)}
    require(Path(runs['issue255']['source']) == RUN / 'src'
            and Path(runs['issue255']['result']) == RUN / 'result',
            'corrected generation paths differ from identity')
    paths, payloads, identities = {}, {}, {}
    for name, run in runs.items():
        paths[name] = artifact_paths(run)
        loader = historical.load_bound_checkpoint if name == 'issue253' else load_bound_checkpoint
        payload, binding = loader(tree=run['source'], **paths[name])
        require(payload['run_id'] == run['run_id'], 'formal checkpoint run mismatch')
        identities[name] = {'run_id': run['run_id'], 'arm': run['arm'],
            'source': run['source'], 'source_hashes': run['snapshot_sources'],
            'workspace': record(run['workspace']),
            'artifacts': {key: record(path) for key, path in paths[name].items()},
            'checkpoint_provenance': payload['provenance'], 'binding': binding,
            'workers': payload.get('workers'), 'sampling': payload.get('sampling')}
        payloads[name] = payload
    require(identities['issue253']['artifacts'] == summary['provenance']['new_artifacts'],
            'comparison checkpoint/SID differs from frozen #253 generation')
    inputs = validate_inputs(payloads, summary)
    context = {'generations': identities, 'inputs': inputs,
        'comparison_report': record(COMPARISON),
        'comparison_summary': record(COMPARISON_SUMMARY),
        'comparison_dispatch': record(COMPARISON_RECEIPT),
        'contract': record(EVIDENCE / 'CONTRACT.md'),
        'acceptance': record(EVIDENCE / 'ACCEPTANCE.json'),
        'extractor_sources': extraction_sources()}
    return runs, paths, payloads, summary, cell, context


def validate_catalogue(arrays, rows, books, *, embedding_shape, token_columns):
    """Validate hard values without executing an encoder or assignment rule.

    Shapes follow the explicit bound inputs, permitting tiny isolated consumer
    tests. Production obtains these inputs from the complete frozen catalogue.
    """
    require(set(arrays) == {'prefixes', 'sid', 'weights', 'attention',
                            'token_mask', 'reconstruction'}, 'catalogue schema mismatch')
    items, width = embedding_shape
    require(rows.shape == (items, 4) and rows.dtype == np.int64,
            'invalid expected SID catalogue')
    require(arrays['sid'].dtype == np.int64 and np.array_equal(arrays['sid'], rows),
            'catalogue differs from actual exported SID including fourth suffix')
    metrics.assert_prefix_identity(arrays['prefixes'], rows, books)
    require(arrays['prefixes'].dtype == np.float32, 'hard prefixes must retain float32')
    weights, attention, mask = (arrays[key] for key in ('weights', 'attention', 'token_mask'))
    require(weights.shape == (items, 3, 3)
            and attention.shape == (items, 3, 3, token_columns),
            'gate/attention catalogue shape mismatch')
    require(weights.dtype == attention.dtype == np.float32
            and np.isfinite(weights).all() and np.isfinite(attention).all()
            and np.all(weights >= 0) and np.all(attention >= 0)
            and np.max(np.abs(weights.sum(-1) - 1)) <= 1.e-6
            and np.max(np.abs(attention.sum(-1) - 1)) <= 1.e-6,
            'gates or token attention are not normalized float32')
    require(mask.shape == (items, token_columns) and mask.dtype == np.bool_
            and np.all(mask.any(-1))
            and np.all(np.where(mask[:, None, None, :], 0., attention) == 0),
            'invalid token mask or nonzero masked attention')
    reconstruction = arrays['reconstruction']
    require(reconstruction.shape == (items, width) and reconstruction.dtype == np.float32
            and np.isfinite(reconstruction).all(), 'invalid reconstruction catalogue')


def load_extraction(receipt_path, *, context, rows, books, embedding_shape, token_columns):
    """Fail closed on stale identities, artifact bytes, metric or hard prefixes."""
    receipt_path = Path(receipt_path)
    require(receipt_path.is_file(), 'missing CUDA EXTRACTION.json; run extract_catalogues first')
    receipt = read_json(receipt_path)
    require(receipt.get('schema') == 'genrec.issue255.catalogue_extraction/1'
            and receipt.get('status') == 'COMPLETE', 'incomplete CUDA extraction receipt')
    require(receipt.get('context') == context, 'extraction provenance differs from bound generation')
    require(receipt.get('metric') == common_metric(), 'extraction common_fixed3 metric differs')
    catalogues = receipt.get('catalogues', {})
    require(set(catalogues) == set(context['generations']) == set(rows) == set(books)
            == set(token_columns), 'extraction generation coverage differs')
    result = {}
    for name, entry in catalogues.items():
        execution = entry.get('execution', {})
        require(execution.get('device_type') == 'cuda'
                and execution.get('catalogue_passes') == 1
                and execution.get('no_grad') is True
                and execution.get('inference_mode') is False
                and execution.get('actual_sid_reexport_equal') is True
                and execution.get('all_levels_bitwise_equal') is True,
                'extraction lacks actual CUDA hard-SID evidence')
        path = receipt_path.parent / (name + '_catalogue.npz')
        require(path.is_file() and entry.get('artifact') == record(path),
                'catalogue artifact changed or is missing: ' + name)
        with np.load(path, allow_pickle=False) as blob:
            arrays = {key: blob[key] for key in blob.files}
        validate_catalogue(arrays, rows[name], books[name],
                           embedding_shape=embedding_shape, token_columns=token_columns[name])
        result[name] = arrays
    return result


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def evaluate_layer(values, labels, coverage, name, level):
    """Full component moments plus all frozen triple occurrences, bounded memory."""
    values = np.asarray(values, dtype=np.float64)
    moments = metrics.component_moments(values, labels)
    accumulator = metrics.CurvatureAccumulator()
    rows_path = OUTPUT / ('%s_L%d_items.jsonl' % (name, level))
    partial = rows_path.with_name(rows_path.name + '.partial')
    excluded = 0
    with partial.open('w') as handle:
        for shard in coverage['shards']:
            with np.load(ASSETS / 'shards' / shard['name'], allow_pickle=False) as blob:
                valid = blob['valid']
                require(int(valid.sum()) == shard['valid'] and int((~valid).sum()) == shard['excluded'],
                        'frozen shard coverage changed')
                triples = np.column_stack([blob[key][valid] for key in ('centre', 'b', 'c')])
                target = blob['curvature'][valid]
            excluded += shard['excluded']
            predicted = metrics.a1_proxy(values, triples, labels, moments=moments, chunk=TRIPLE_CHUNK)
            rows = accumulator.add(target, predicted, triples[:, 0])
            for row in rows:
                row.update(generation=name, level=level)
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
            print('[issue255-evaluation] %s L%d %s triples=%d' %
                  (name, level, shard['name'], len(triples)), flush=True)
    result = accumulator.finish(expected_count=metrics.EXPECTED_TRIPLES,
                                expected_centres=EXPECTED_CENTRES)
    require(excluded == coverage['excluded_triples'], 'excluded population drift')
    partial.replace(rows_path)
    result.update(excluded=excluded, per_item=record(rows_path),
                  items_without_lawful_triples=sorted(set(range(metrics.ITEMS)) - accumulator.seen))
    return {'representation': 'hard_cumulative_prefix', 'level': level,
            'metric': 'common_fixed3', 'expert_curvatures': [-1, 0, 1],
            'tangent_scale': metrics.CROSS_MODEL_TANGENT_SCALE,
            'scale_fitted_on_evaluation': False, 'distance_multiplier': 1.,
            'metric_units': 'GraphMoRE A.1 from equal-weight ambient squared distance',
            'precision': 'float32 hard-prefix accumulation; float64 independent metric',
            'reference_scope': 'entire original frozen component including centre',
            'curvature': result}


def evaluate_generation(name, catalogue, arrays, labels, coverage, source, target, heldout):
    rows = arrays['sid']
    prefixes = arrays['prefixes']
    weights, attention = arrays['weights'], arrays['attention']
    mask = arrays['token_mask']
    reconstruction = metrics.reconstruction_report(
        np.load(EMBEDDINGS, mmap_mode='r', allow_pickle=False), arrays['reconstruction'],
        representation='hard_q_L3')
    structure = metrics.sid_structure(rows)
    edge = metrics.edge_l1_sharing(rows, source, target)
    per_item = OUTPUT / (name + '_sid_items.npz')
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
    gate_summary = []
    for level in range(3):
        gate_summary.append({'level': level + 1, 'weights_mean': weights[:, level].mean(0).tolist(),
            'weights_distribution': [metrics.distribution_summary(weights[:, level, k]) for k in range(3)],
            'gate_entropy_nats': metrics.distribution_summary(
                -np.sum(weights[:, level] * np.log(np.maximum(weights[:, level], 1.e-30)), axis=-1)),
            'attention_mean_expert_by_token': attention[:, level].mean(0).tolist(),
            'attention_entropy_nats_per_expert': [metrics.distribution_summary(
                -np.sum(attention[:, level, k] * np.log(np.maximum(attention[:, level, k], 1.e-30)), axis=-1))
                for k in range(3)]})
    result = {'catalogue': record(catalogue), 'catalogue_schema': {
        'prefixes': '[3,N,32] float32 hard forward in level order',
        'sid': '[N,4] int64 actual export including untouched suffix',
        'weights': '[N,3,3] float32, expert order -1,0,+1',
        'attention': '[N,3,3,T+1] float32, last token is explicit null',
        'token_mask': '[N,T+1] bool', 'reconstruction': '[N,768] float32 decoder(hard q L3)'},
        'prefix_identity': {'all_levels_bitwise_equal': True, 'actual_sid_reexport_equal': True,
                            'catalogue_passes': 1, 'no_grad': True, 'device_type': 'cuda'},
        'sid_structure': structure, 'visible_directed_l1_sharing': edge,
        'strict_heldout': metrics.strict_prefix_report(rows, heldout),
        'reconstruction': reconstruction, 'per_item_sid': record(per_item),
        'gates': {'shared_across_levels': name == 'issue253', 'levels': gate_summary,
                  'token_order': (['one_hop', 'two_hop'] if name == 'issue253'
                                  else ['one_hop', 'two_hop', 'in_only', 'out_only']) + ['null'],
                  'valid_count_by_token': mask.sum(0).tolist()}, 'layers': {}}
    for level in metrics.LEVELS:
        result['layers']['L%d' % level] = evaluate_layer(prefixes[level - 1], labels, coverage, name, level)
    return result


def baseline(summary, source, target, heldout):
    from tools.graph_infra.tiger_compare import STAGE3_SID
    path = ROOT / STAGE3_SID
    require(record(path)['sha256'] == TIGER_SID_SHA256 == summary['historical_tiger']['sid_sha256'],
            'frozen TIGER Stage3 SID changed')
    rows = load_sid(path, metrics.ITEMS, [256] * 3)
    structure = metrics.sid_structure(rows)
    structure.pop('per_item_l3_group_size')
    edge = metrics.edge_l1_sharing(rows, source, target)
    per_item = OUTPUT / 'tiger_sid_items.npz'
    np.savez(per_item, sid=rows, item=np.arange(metrics.ITEMS),
             out_degree=edge['out_direction'].pop('per_item_degree'),
             out_shared=edge['out_direction'].pop('per_item_shared'),
             out_rate=edge['out_direction'].pop('per_item_rate'),
             in_degree=edge['in_direction'].pop('per_item_degree'),
             in_shared=edge['in_direction'].pop('per_item_shared'),
             in_rate=edge['in_direction'].pop('per_item_rate'))
    edge.pop('per_edge_shared')
    return {'sid': record(path), 'sid_structure': structure, 'per_item_sid': record(per_item),
            'visible_directed_l1_sharing': edge,
            'strict_heldout': metrics.strict_prefix_report(rows, heldout),
            'geometry_and_reconstruction': summary['historical_tiger']['geometry_metrics'],
            'recall': {'status': 'NA', 'reason': 'no same-source historical Stage3 rank/checkpoint binding'}}


def main():
    require(len(sys.argv) == 1, 'this evaluator accepts no arguments; source constants only')
    sys.dont_write_bytecode = True
    import torch
    from tools.graph_infra.heldout_eval import HeldOutEvaluation
    torch.set_num_threads(1)
    runs, paths, payloads, summary, cell, context = bound_generations()
    identities, inputs = context['generations'], context['inputs']
    comparison_summary_record = context['comparison_summary']
    receipt_path = OUTPUT / 'EXTRACTION.json'
    catalogues = load_extraction(receipt_path, context=context,
        rows={name: load_sid(paths[name]['sid'], metrics.ITEMS, [256] * 3) for name in runs},
        books={name: _books(payloads[name]) for name in runs},
        embedding_shape=(metrics.ITEMS, 768), token_columns={'issue253': 3, 'issue255': 5})
    extraction_record = record(receipt_path)
    catalogue_records = {name: record(OUTPUT / (name + '_catalogue.npz')) for name in runs}
    labels, coverage = _coverage(ASSETS, metrics.ITEMS)
    require(coverage == summary['coverage'], 'evaluation must retain the frozen #253 full population')
    heldout = HeldOutEvaluation(ROOT, ASSETS)
    isolation = heldout.is_isolated_from_training()
    require(all(isolation[key] == 0 for key in ('visible_intersect_struct_val',
            'visible_intersect_struct_test', 'struct_val_intersect_struct_test')), 'held-out users are not isolated')
    source, target = metrics.load_visible_edges(ASSETS / 'train/visible_forward.npz')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    # Never leave an older report masquerading as a successful partial rerun.
    if REPORT.exists():
        REPORT.unlink()
    reports = {}
    for name in runs:
        catalogue = OUTPUT / (name + '_catalogue.npz')
        reports[name] = evaluate_generation(name, catalogue, catalogues.pop(name),
                                           labels, coverage, source, target, heldout)
    tiger = baseline(summary, source, target, heldout)
    # Recheck all source/checkpoint/SID and input bytes before publication.
    for name, run in runs.items():
        load_run(ROOT, run['run_id'], run['arm'])
        loader = historical.load_bound_checkpoint if name == 'issue253' else load_bound_checkpoint
        loader(tree=run['source'], **paths[name])
        require(identities[name]['artifacts'] == {key: record(path) for key, path in paths[name].items()},
                'formal artifacts changed during evaluation')
    for descriptor in inputs['files'].values():
        require(record(descriptor['path']) == descriptor, 'input changed during evaluation')
    require(record(ASSETS / 'manifest.json') == inputs['manifest'], 'asset manifest changed during evaluation')
    require(record(COMPARISON)['sha256'] == COMPARISON_SHA256, 'comparison report changed during evaluation')
    require(record(COMPARISON_SUMMARY) == comparison_summary_record,
            'comparison summary changed during evaluation')
    require(record(tiger['sid']['path']) == tiger['sid'], 'frozen TIGER SID changed during evaluation')
    require(record(receipt_path) == extraction_record, 'CUDA extraction receipt changed during evaluation')
    for name, descriptor in catalogue_records.items():
        require(record(OUTPUT / (name + '_catalogue.npz')) == descriptor,
                'CUDA catalogue changed during evaluation')
    rows = ['generation\tlevel\ttriples\tcentres\tmacro_mae\tmicro_mae\tbalanced_sign_accuracy\tnegative_recall\tzero_recall\tpositive_recall\tdirected_l1_micro\tout_centre_l1_macro\tprefix_collision\tcodebook_used\treconstruction_mse']
    for name, generation in list(reports.items()) + [('frozen_tiger', tiger)]:
        for number in metrics.LEVELS:
            level = 'L%d' % number
            if 'layers' in generation:
                curvature = generation['layers'][level]['curvature']
                recalls = curvature['sign']['recall']
                fields = [name, level, curvature['count'], curvature['centres'], curvature['macro_mae'],
                          curvature['micro_mae'], curvature['sign']['balanced_accuracy']]
                fields.extend(recalls[key].get('recall', 'NA') for key in ('negative', 'zero', 'positive'))
            else:
                fields = [name, level] + ['NA'] * 8
            sid = generation['sid_structure']
            collision = sid['collision_l1' if number == 1 else 'collision_l%d_prefix' % number]
            edge = generation['visible_directed_l1_sharing']
            fields.extend([edge['per_edge_rate'], edge['out_direction']['macro_rate'], collision,
                           sid['codebook_usage'][number - 1]['used'],
                           generation.get('reconstruction', {}).get('mean_squared_error', 'NA')])
            rows.append('\t'.join(map(str, fields)))
    TABLE.write_text('\n'.join(rows) + '\n')
    report = {'schema': 'genrec.issue255.mechanism_evaluation/1', 'status': 'COMPLETE',
        'generations': reports, 'historical_tiger': tiger, 'coverage': coverage,
        'comparison': {'metric': 'common_fixed3', 'tangent_scale': cell['tangent_scale'],
                       'scale_fitted_on_evaluation': False, 'native_metrics_mixed': False,
                       'all_levels_same_units': True},
        'populations': {'items': metrics.ITEMS, 'visible_directed_edges': len(source),
            'curvature_triples_per_cell': metrics.EXPECTED_TRIPLES, 'cells': 6,
            'macro': 'unweighted mean of lawful-centre mean absolute errors',
            'micro': 'all frozen valid triple occurrences, multiplicity preserved',
            'sign': 'exact zero tolerance; balanced over present target signs',
            'edge_macro': 'unweighted nonempty source centres; isolated centres excluded',
            'per_item_edge_arrays': 'all N centres; degree=0 implies unavailable rate (stored 0)',
            'heldout': 'existing strict banks and existing degree-matched controls only',
            'heldout_isolation': isolation, 'sid': 'first three semantic codes; suffix excluded'},
        'units': {'sharing_collision_occupancy_sign_recall': 'dimensionless fractions',
                  'reconstruction_mse': 'mean squared error per embedding coordinate, 768 dimensions',
                  'reconstruction_l2': 'Euclidean norm in original embedding units',
                  'gate_attention_entropy': 'nats',
                  'curvature': 'raw GraphMoRE A.1; common_fixed3 tangent scale; no target-scale division'},
        'stage3': {'status': 'NA', 'reason': 'separate frozen Stage3 evaluator; never inferred from Stage2'},
        'stage3_collision_gate': {'threshold_exclusive': .4,
            'collision': reports['issue255']['sid_structure']['three_code_collision'],
            'eligible': reports['issue255']['sid_structure']['three_code_collision'] < .4,
            'dispatch_performed': False},
        'provenance': {'runs': identities, 'inputs': inputs,
            'extraction': extraction_record, 'extraction_context': context,
            'comparison_report': record(COMPARISON), 'comparison_summary': comparison_summary_record,
            'comparison_dispatch': record(COMPARISON_RECEIPT),
            'contract': record(ROOT / 'results/research/issue255/CONTRACT.md'),
            'acceptance': record(ROOT / 'results/research/issue255/ACCEPTANCE.json'),
            'evaluator_sources': {str(path.relative_to(ROOT)): record(path) for path in
                (Path(__file__), Path(metrics.__file__), ROOT / 'tools/issue253/evaluation_reference.py',
                 ROOT / 'tools/issue253/final_evaluation.py', ROOT / 'tools/issue252/final_evaluation.py',
                 ROOT / 'tools/issue252/run_workspace.py', ROOT / 'tools/issue252/archive_audit.py',
                 ROOT / 'tools/issue251/evaluation.py', ROOT / 'tools/graph_infra/heldout_eval.py',
                 ROOT / 'tools/graph_infra/tiger_compare.py')}}, 'table': record(TABLE)}
    write_json(REPORT, _json_safe(report))
    print(json.dumps({'schema': report['schema'], 'status': 'COMPLETE',
                      'report': record(REPORT), 'table': record(TABLE), 'cells': 6}), flush=True)
    return report


if __name__ == '__main__':
    main()
