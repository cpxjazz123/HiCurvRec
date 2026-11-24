"""Independent arithmetic/protocol/archive reconciliation for exact Issue253 run.

Never imports a candidate model, trains, changes a ledger, or selects a newest run.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path

import numpy as np

from tools.issue252.reconcile_results import remeasure_test, bound_archive
from tools.issue252.layer_metrics import catalogue_metrics
from tools.issue253.final_evaluation import (ROOT, ARM, EVIDENCE, REPORT, LEDGER,
    EXPECTED_TRIPLES, ITEMS, WIDTH, ASSETS, EMBEDDINGS, HISTORICAL_ARM,
    HISTORICAL_SHA256, load_bound_checkpoint, load_sid, resolve_formal_run,
    record, read_json, write_json, require, r2_api, load_historical_checkpoint,
    _state_scalar, _coverage)

OUTPUT = EVIDENCE/'RECONCILIATION.json'


def close(actual, expected, label):
    require(np.isscalar(actual) and np.isscalar(expected)
            and math.isfinite(actual) and math.isfinite(expected)
            and math.isclose(actual, expected, abs_tol=1e-9, rel_tol=1e-9),
            'independent reduction disagrees: '+label)


def reconcile_curvature(report, *, expected_count=EXPECTED_TRIPLES):
    require(type(expected_count) is int and expected_count > 0, 'invalid full coverage count')
    rows = list(report.get('per_item', {}).values())
    require(rows and all(type(row.get('count')) is int and row['count'] > 0 for row in rows),
            'missing/invalid per-centre coverage counts')
    count = sum(row['count'] for row in rows)
    require(count == report.get('count') == expected_count, 'full curvature coverage count differs')
    macro = math.fsum(row['mean_abs_error'] for row in rows)/len(rows)
    micro = math.fsum(row['mean_abs_error']*row['count'] for row in rows)/count
    close(macro, report['macro_mae'], 'curvature macro MAE')
    close(micro, report['micro_mae'], 'curvature micro MAE')
    recalls, strata, present = {}, {}, []
    for sign in ('negative', 'zero', 'positive'):
        selected = []
        for row in rows:
            local = row.get('strata', {}).get(sign)
            require(isinstance(local, dict), 'missing independently reducible per-centre strata')
            if local.get('status') != 'NA':
                require(type(local.get('count')) is int and local['count'] > 0,
                        'invalid per-centre sign count')
                recall = row['sign']['recall'][sign]
                require(recall['count'] == local['count'] and 0 <= recall['recall'] <= 1,
                        'per-centre recall/count mismatch')
                selected.append((local, recall))
        n = sum(local['count'] for local, _ in selected)
        published = report['sign']['recall'][sign]
        if not n:
            require(published.get('status') == 'NA', 'absent target sign was not NA')
            recalls[sign] = {'status': 'NA', 'reason': 'target sign absent'}
            continue
        present.append(sign)
        require(published.get('count') == n, 'global sign coverage differs')
        recall = math.fsum(local['count']*r['recall'] for local, r in selected)/n
        close(recall, published['recall'], sign+' recall')
        recalls[sign] = {'status': 'OK', 'count': n, 'recall': recall}
        strata[sign] = {'status': 'OK', 'count': n,
            'macro_mae': math.fsum(local['macro_mae'] for local, _ in selected)/len(selected),
            'micro_mae': math.fsum(local['micro_mae']*local['count'] for local, _ in selected)/n}
        if 'strata' in report:
            require(report['strata'][sign]['count'] == n, 'global stratum count differs')
            for field in ('macro_mae', 'micro_mae'):
                close(strata[sign][field], report['strata'][sign][field], sign+' '+field)
    require(sum(r.get('count', 0) for r in recalls.values()) == count, 'sign strata do not cover all rows')
    for row in rows:
        require(sum(local.get('count', 0) for local in row['strata'].values()) == row['count'],
                'per-centre sign strata do not cover every row')
    require(report['sign']['present_signs'] == present, 'present-sign order/population differs')
    balanced = math.fsum(recalls[sign]['recall'] for sign in present)/len(present)
    close(balanced, report['sign']['balanced_accuracy'], 'balanced accuracy')
    accuracy = math.fsum(r.get('count', 0)*r.get('recall', 0.) for r in recalls.values())/count
    if 'accuracy' in report['sign']:
        close(accuracy, report['sign']['accuracy'], 'sign accuracy')
    return {'count': count, 'centres': len(rows), 'macro_mae': macro, 'micro_mae': micro,
            'balanced_accuracy': balanced, 'accuracy': accuracy, 'recall': recalls, 'strata': strata}


def inclusive_window(rows):
    from tools.issue253.performance import summarize_window
    result = summarize_window(rows)
    # Keep report field names while using the single canonical inclusive
    # distribution: first end-start, then end_i-end_(i-1), including gaps.
    return {**result, 'wall_seconds': result['window_seconds'],
            'p95_method': result['p95_definition'],
            'boundary_duration_sum': result['sum_epoch_seconds'],
            'inter_boundary_wait_seconds': result['interepoch_seconds']}


def reconcile_per_user(shards, *, expected_users):
    require(type(expected_users) is int and expected_users > 0, 'invalid full test user population')
    actual = remeasure_test(shards)
    require(actual['n_eval'] == expected_users, 'full per-user evaluation population differs')
    for shard in shards:
        expected = len(range(shard['rank'], expected_users, 4))
        require(shard['n_eval'] == expected, 'rank-local original strided user population differs')
    return actual


def _stream(path):
    import json
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _stage3_final():
    """Same unique-generation rule as the official archive uploader, no timestamps."""
    base = ROOT/'results/stage3_T5Train'/ARM
    candidates = set()
    if (base/'test_final.json').is_file():
        candidates.add((base/'test_final.json').resolve())
    candidates.update(path.resolve() for path in base.glob('logs/*/*/test_final.json'))
    require(len(candidates) == 1, 'missing or ambiguous exact Stage3 final generation')
    return next(iter(candidates))


def _check_descriptor(descriptor):
    require(isinstance(descriptor, dict) and descriptor == record(descriptor.get('path', '')),
            'missing/changed evidence descriptor')
    return Path(descriptor['path'])


def _admission(run, train_start):
    """Validate actual persisted dispatch proof, not a post-training rerun."""
    from tools.issue252.training_admission import check_receipt
    credential = run.get('training_admission')
    require(isinstance(credential, dict), 'formal workspace has no original training admission')
    admission_path = _check_descriptor(credential)
    require(check_receipt(ROOT, Path(run['admitted_tree'])) == credential,
            'original training admission no longer matches current actual source')
    require(admission_path.stat().st_mtime <= train_start['timestamp'],
            'training admission was generated after formal launch')
    state_path = Path.home()/'.cache/genrec-mcp'/('stage2__'+ARM+'.json')
    state = read_json(state_path)
    for key, expected in (('run_id', run['run_id']), ('source_tree', run['source']),
                          ('result_dir', run['result']), ('source_digest', run['source_digest']),
                          ('checks_fingerprint', run['checks_fingerprint']), ('graph_identity', run['graph_identity'])):
        require(state.get(key) == expected, 'persisted public dispatch generation differs: '+key)
    machine = state.get('admission', {})
    require(machine.get('passed') is True and {c['name']: c['status'] for c in machine.get('checks', [])}
            == {'entry': 'PASS', 'behavior_graph': 'PASS'}, 'public pre-dispatch entry/graph proof missing')
    require(machine.get('source_digest') == run['source_digest']
            and machine.get('graph_identity') == run['graph_identity']
            and machine.get('checks_fingerprint') == run['checks_fingerprint'],
            'machine pre-dispatch proof belongs to another generation')
    require(type(state.get('pid')) is int and type(state.get('worker_pid')) is int
            and isinstance(state.get('launched_at'), (int, float))
            and state['launched_at'] <= train_start['timestamp'], 'real worker/launch identities missing')
    return {'training_admission': record(admission_path), 'public_dispatch_state': record(state_path),
            'machine': machine, 'worker_pid': state['worker_pid'], 'trainer_pid': state['pid'],
            'launched_at': state['launched_at']}


def _reconstruction(report, inputs, actual):
    require(inputs.shape == actual.shape and inputs.ndim == 2 and len(inputs),
            'aligned actual reconstruction arrays missing')
    squared_parts, l2_parts = [], []
    for start in range(0, len(inputs), 1024):
        target = np.asarray(inputs[start:start+1024], dtype=np.float64)
        reconstructed = np.asarray(actual[start:start+1024], dtype=np.float64)
        require(np.isfinite(target).all() and np.isfinite(reconstructed).all(),
                'nonfinite actual reconstruction')
        difference = reconstructed-target
        squared_parts.append(float(np.square(difference).sum()))
        l2_parts.append(float(np.linalg.norm(difference, axis=1).sum()))
    squared = math.fsum(squared_parts)/inputs.size
    mean_l2 = math.fsum(l2_parts)/len(inputs)
    close(squared, report['mean_squared_error'], 'reconstruction MSE')
    close(mean_l2, report['mean_l2_error'], 'reconstruction mean L2')
    return {'mean_squared_error': squared, 'mean_l2_error': mean_l2}


def reconcile_distortion(cell):
    rows = [row['distance_distortion'] for row in cell['curvature']['per_item'].values()]
    require(all(type(row['count']) is int and row['count'] > 0
                and type(row['relative_count']) is int and 0 <= row['relative_count'] <= row['count']
                and math.isfinite(row['absolute_sum']) and row['absolute_sum'] >= 0
                and math.isfinite(row['relative_sum']) and row['relative_sum'] >= 0 for row in rows),
            'invalid per-centre distortion sufficient statistics')
    count = sum(row['count'] for row in rows)
    published = cell['distance_distortion']
    require(count == cell['curvature']['count'] == published['count'], 'distortion occurrence coverage differs')
    absolute = math.fsum(row['absolute_sum'] for row in rows)/count
    close(absolute, published['mean_abs_error'], 'distance mean absolute error')
    relative_count = sum(row['relative_count'] for row in rows)
    relative = math.fsum(row['relative_sum'] for row in rows)/relative_count if relative_count else None
    if relative is None:
        require(published['mean_relative_error'] is None, 'undefined relative distortion must be null')
    else:
        close(relative, published['mean_relative_error'], 'distance relative error')
    return {'count': count, 'mean_abs_error': absolute, 'mean_relative_error': relative,
            'positive_graph_distance_count': relative_count}


def _historical_recall(full, expected_users):
    descriptor = full['historical_stage3']['archive_recall']
    path = _check_descriptor(descriptor)
    old = read_json(path)
    require(old['stage2_checkpoint_sha256'] == HISTORICAL_SHA256
            and old['stage3_completed_epochs'] == 150 and old['stage3_world_size'] == 4,
            'historical Stage3 archive generation/protocol differs')
    config = old['stage3_input_config']
    require(all(config.get(key) == value for key, value in
                {'num_epochs': 150, 'seed': 42, 'beam_size': 20, 'no_eval': True, 'skip_test': False}.items()),
            'historical recommendation protocol differs')
    stage3 = Path(old['stage3_run_id']).resolve()
    require(stage3.is_relative_to(ROOT/'results/stage3_T5Train'/HISTORICAL_ARM),
            'historical per-user path crosses arm generations')
    paths = [stage3/'_test_shards'/('rank%d.json' % rank) for rank in range(4)]
    metrics = full['historical_stage3']['metrics']
    for key, value in metrics.items():
        close(value, old[key], 'historical archived '+key)
    if not all(path.is_file() for path in paths):
        return {'archive_metrics': metrics, 'archive_source': descriptor,
                'per_user_recompute': {'status': 'NA', 'reason': 'original historical rank files unavailable'}}
    require(read_json(stage3/'test_final.json') == {k: v for k, v in old.items() if k != 'stage3_archive_evidence'},
            'historical local final record differs from immutable archive')
    recomputed = reconcile_per_user([read_json(path) for path in paths], expected_users=expected_users)
    for key, value in recomputed.items():
        close(value, old[key], 'historical original per-user '+key)
    return {'archive_metrics': metrics, 'archive_source': descriptor,
            'per_user_recompute': {'status': 'OK', 'metrics': recomputed,
                                   'shards': [record(path) for path in paths]}}


def run_reconciliation():
    import pyarrow.parquet as parquet
    run, dispatch = resolve_formal_run()
    result = Path(run['result'])
    paths = {'checkpoint': result/'rqvae_best.pth', 'sid': result/'item_sids.json',
             'binding': result/'archive_binding.json', 'export_manifest': result/'export_manifest.json'}
    payload, binding = load_bound_checkpoint(tree=run['source'], **paths)
    require(payload['run_id'] == run['run_id'], 'final checkpoint crosses dispatch generations')
    rows = load_sid(paths['sid'], ITEMS, [256]*3)
    catalogue = catalogue_metrics(rows)
    require(catalogue['l3_collision'] < .4, 'Stage3 collision hard gate failed')
    layer_path = ROOT/'results/stage2_RQ-VAE'/ARM/'stage2_metrics'/(record(paths['sid'])['sha256']+'.json')
    layer = read_json(layer_path)
    require(layer.get('sid') == record(paths['sid']) and layer.get('run_id') == run['run_id']
            and layer.get('arm') == ARM and layer.get('read_stage3_scores') is False,
            'layered SID metrics source/run/prospective binding missing')
    for key, value in catalogue.items():
        if isinstance(value, list):
            require(value == layer['catalogue'][key], 'layer occupancy differs')
        else:
            close(value, layer['catalogue'][key], 'layer '+key)
    for split in ('train', 'valid'):
        require(layer['inputs'][split] == record(ROOT/'results/stage0_build_parquet'/(split+'.parquet')),
                'layer record input source differs')
    metrics_path = result/'logs/training_metrics.jsonl'
    metrics = _stream(metrics_path)
    require(metrics and all(row.get('run_id') == run['run_id'] for row in metrics), 'Stage2 metrics run differs')
    trains = [row for row in metrics if row['event'] == 'train']
    boundaries = [row for row in metrics if row['event'] == 'epoch_boundary']
    require([row['epoch'] for row in trains] == list(range(1, 3001))
            and [row['epoch'] for row in boundaries] == list(range(1, 3001)), 'Stage2 budget is not exactly3000')
    require(metrics[-1]['event'] == 'train_end' and metrics[-1]['checkpoint_selection'] == 'final_epoch',
            'Stage2 final completion/selection missing')
    starts = [row for row in metrics if row['event'] == 'train_start']
    require(len(starts) == 1, 'ambiguous Stage2 train_start')
    start = starts[0]
    require(all(start.get(key) == value for key, value in
                {'epochs': 3000, 'batch_size_per_rank': 1024, 'total_batch_size': 4096,
                 'seed': 42, 'codebook_dim': 32, 'codebook_size': [256]*3}.items()), 'Stage2 frozen recipe differs')
    for row in trains:
        require(all(math.isfinite(row[key]) for key in ('loss', 'recon', 'curvature_estimate',
                                                       'gradient_norm_before_clip')), 'nonfinite formal loss/gradient')
        require(len(row.get('gate_mean', [])) == 3, 'formal gate expert population differs')
    for boundary in boundaries:
        coverage = boundary.get('structural_coverage', [])
        require(len(coverage) == 4 and all(row.get('epoch_updates') == 6 for row in coverage),
                'original four-rank structural steps were skipped')
    require(all(row.get('updates') == 18000 for row in boundaries[-1]['structural_coverage']),
            'full18000 optimizer-step structural coverage missing')
    calibration = [row for row in metrics if row['event'] == 'curvature_calibration']
    require(len(calibration) == 1 and all(math.isfinite(calibration[0][key]) and calibration[0][key] > 0
        for key in ('reconstruction_gradient', 'curvature_gradient', 'lambda', 'tangent_scale', 'target_scale')),
        'initial real-loss calibration is absent/repeated/degenerate')
    close(calibration[0]['lambda'], _state_scalar(payload, 'curvature_weight'), 'frozen calibration lambda')
    close(calibration[0]['tangent_scale'], _state_scalar(payload, 'tangent_scale'), 'frozen native scale')
    admission = _admission(run, start)
    timing = {'formal_1_3000': inclusive_window(boundaries), 'formal_31_60': inclusive_window(boundaries[30:60]),
              'formal_31_3000': inclusive_window(boundaries[30:]),
              'initialization_seconds': next(row['initialization_seconds'] for row in metrics
                                             if row['event'] == 'initialization_complete'),
              'trainer_total_seconds': metrics[-1]['total_wall_time_s'],
              'last_boundary_to_end_seconds': metrics[-1]['timestamp']-boundaries[-1]['timestamp'],
              'historical_same_scope_seconds': .6405808043}
    measured = timing['formal_31_60']['mean_seconds']
    timing['performance'] = {'measured_seconds_per_epoch': measured, 'target_seconds': .4,
        'expected_max_seconds': .5, 'hard_max_seconds': .8, 'target_met': measured <= .4,
        'expected_met': measured <= .5, 'formal_ceiling_met': measured <= .8}

    final_path = _stage3_final()
    final = read_json(final_path)
    stage3 = final_path.parent
    for name, path in (('sid', paths['sid']), ('stage2_checkpoint', paths['checkpoint']),
                       ('archive_binding', paths['binding'])):
        key = 'sid_path' if name == 'sid' else name
        require(Path(final[key]).resolve() == path.resolve() and final[name+'_sha256'] == record(path)['sha256'],
                'Stage3 input generation differs: '+name)
    require(final['stage2_run_id'] == run['run_id'] and final['stage3_run_id'] == str(stage3)
            and final['stage3_world_size'] == 4 and final['stage3_completed_epochs'] == 150,
            'Stage3 formal run/world/epoch identity differs')
    config = final['stage3_input_config']
    expected = {'num_epochs': 150, 'eval_interval': 5, 'eval_start_epoch': 75, 'seed': 42,
                'beam_size': 20, 'no_eval': True, 'skip_test': False}
    require(all(config.get(key) == value for key, value in expected.items())
            and list(config.get('early_eval_epochs', [])) == [25, 50, 75, 100, 125],
            'Stage3 frozen protocol differs')
    require(final['stage3_trainer_sha256'] == record(ROOT/'stage3_T5Train/train_HG-Rec.py')['sha256'],
            'Stage3 executed trainer changed')
    stage3_metrics_path = stage3/'training_metrics.jsonl'
    stage3_metrics = _stream(stage3_metrics_path)
    require([row['epoch'] for row in stage3_metrics if row['event'] == 'train'] == list(range(1, 151))
            and stage3_metrics[-1]['event'] == 'train_end', 'Stage3 budget/completion missing')
    stage3_start = next(row for row in stage3_metrics if row['event'] == 'train_start')
    require(layer_path.stat().st_mtime <= stage3_start['timestamp'], 'SID layer record was not prospective')
    stage3_state_path = Path.home()/'.cache/genrec-mcp'/('stage3__'+ARM+'.json')
    stage3_state = read_json(stage3_state_path)
    prospective_layer = stage3_state.get('layer_record', {})
    require(stage3_state.get('stage2_run_id') == run['run_id']
            and Path(stage3_state.get('sid', '')).resolve() == paths['sid'].resolve()
            and stage3_state.get('collision_gate_waived') is False
            and type(stage3_state.get('pid')) is int
            and 0 < stage3_state.get('launched_at', 0) <= stage3_start['timestamp'],
            'Stage3 persisted sanctioned launch generation/identity differs')
    require(prospective_layer.get('passed') is True
            and prospective_layer.get('sha256') == record(layer_path)['sha256']
            and Path(prospective_layer.get('report', '')).resolve() == layer_path.resolve(),
            'SHA-keyed SID layer record was not bound at actual Stage3 dispatch')
    shard_paths = [stage3/'_test_shards'/('rank%d.json' % rank) for rank in range(4)]
    test_path = ROOT/'results/stage0_build_parquet/test.parquet'
    expected_users = parquet.read_metadata(test_path).num_rows
    recall = reconcile_per_user([read_json(path) for path in shard_paths], expected_users=expected_users)
    require(recall['n_eval'] == final['n_eval'], 'test_final user denominator differs')
    for key, value in recall.items():
        close(value, final[key], key)

    full = read_json(REPORT)
    require(full.get('schema') == 'genrec.issue253.final_evaluation/1' and full['run_id'] == run['run_id']
            and full['provenance']['formal_dispatch'] == dispatch
            and full['provenance']['workspace'] == record(run['workspace'])
            and full['provenance']['new_binding'] == binding,
            'final evaluation belongs to another formal generation')
    for key, path in paths.items():
        require(full['provenance']['new_artifacts'][key] == record(path), 'full evaluation artifact changed')
    for descriptor in full['provenance']['evaluator_sources'].values():
        _check_descriptor(descriptor)
    for descriptor in full['provenance']['frozen_inputs']['files'].values():
        _check_descriptor(descriptor)
    _, actual_coverage = _coverage(ASSETS, ITEMS)
    require(actual_coverage == full['coverage'], 'actual full target/reference coverage changed')
    require(full['comparison']['source_checkpoint_sha256'] == HISTORICAL_SHA256
            and full['comparison']['scale_fitted_on_evaluation'] is False,
            'historical common-scale identity missing')
    historical_root = EVIDENCE/'historical_archive/100'
    historical = full['provenance']['historical_archive']
    old = load_historical_checkpoint(historical_root, full['provenance']['historical_binding'], historical['entry'])
    close(_state_scalar(old, 'tangent_scale'), full['comparison']['s_ref'], 'common historical scale')
    geometry, mse = {}, {}
    target = np.load(EMBEDDINGS, mmap_mode='r', allow_pickle=False)
    cells_count = 0
    for generation in ('issue253', 'issue252'):
        geometry[generation], mse[generation] = {}, {}
        extraction = _check_descriptor(full['provenance']['extractions'][generation])
        with np.load(extraction, allow_pickle=False) as blob:
            for representation, key in (('hard_q', 'q'), ('continuous_z', 'z')):
                view = full['generations'][generation][representation]
                require(view['representation'] == representation and blob[key].shape == (ITEMS, WIDTH),
                        'representation extraction label/shape mismatch')
                mse[generation][representation] = _reconstruction(view['reconstruction'], target, blob['reconstruction_'+key])
                expected_cells = {'common_fixed3', 'common_fixed5'}
                if generation == 'issue253':
                    expected_cells.add('native_fixed3')
                require(set(view['cells']) == expected_cells, 'prospective metric cell missing/extra')
                geometry[generation][representation] = {}
                for name, cell in view['cells'].items():
                    experts = 5 if name == 'common_fixed5' else 3
                    require(cell['expert_curvatures'] == ([-3, -1, 0, 1, 3] if experts == 5 else [-1, 0, 1]),
                            'metric cell expert population differs')
                    scale = _state_scalar(payload, 'tangent_scale') if name == 'native_fixed3' else full['comparison']['s_ref']
                    close(cell['tangent_scale'], scale, 'cell common/native scale')
                    require(cell['distance_distortion']['count'] == EXPECTED_TRIPLES
                            and cell['distance_distortion']['scale_fitted_on_evaluation'] is False
                            and cell['distance_distortion']['distance_scale'] == 1.,
                            'distance distortion coverage/fitted scale differs')
                    require(cell['curvature']['excluded'] == full['coverage']['excluded_triples']
                            and cell['shards'] == [row['name'] for row in full['coverage']['shards']],
                            'cell frozen exclusions/shards differ')
                    geometry[generation][representation][name] = {
                        'curvature': reconcile_curvature(cell['curvature']),
                        'distance_distortion': reconcile_distortion(cell)}
                    cells_count += 1
    require(cells_count == 10, 'all ten prospective cells are required')

    ledger = read_json(LEDGER)
    require(ledger.get('100') == historical['entry'], 'historical ledger100 evidence changed')
    historical_recall = _historical_recall(full, expected_users)
    key = ARM+'/'+record(paths['sid'])['sha256'][:12]
    number, archived = bound_archive(ledger, key, record(paths['sid'])['sha256'], record(paths['checkpoint'])['sha256'])
    files = archived['files']
    for group in ('candidate_sources', 'infrastructure_sources', 'repository_configs'):
        for name, digest in binding[group].items():
            require(files.get(ARM+'/'+name) == digest, 'R2 executed source hash differs: '+name)
    require(files.get(ARM+'/archive_binding.json') == record(paths['binding'])['sha256']
            and files.get(ARM+'/export_manifest.json') == record(paths['export_manifest'])['sha256'],
            'R2 binding/export generation differs')
    api = r2_api()
    api.verify_upload(number, files)
    remote = api.rclone(['cat', api.BUCKET+'/'+number+'/recall.json'], text=False)
    require(remote.returncode == 0 and hashlib.sha256(remote.stdout).hexdigest() == files['recall.json'],
            'R2 recall object hash differs')
    import json
    remote_recall = json.loads(remote.stdout)
    evidence = remote_recall.pop('stage3_archive_evidence', {})
    require(remote_recall == final, 'R2 recall differs from complete actual final-test record')
    for name in ('HG_Rec.log', 'training_metrics.jsonl'):
        data = (stage3/name).read_bytes()
        require(evidence.get(name) == {'content': data.decode(), 'sha256': hashlib.sha256(data).hexdigest()},
                'R2 complete Stage3 log evidence differs: '+name)
    receipt = {'number': number, 'target': api.BUCKET+'/'+number+'/', 'key': key,
               'remote_sha256_reverified_at': datetime.now(timezone.utc).isoformat(),
               'ledger': record(LEDGER), 'files': files}
    archive_receipt_path = EVIDENCE/'R2_RECEIPT.json'
    write_json(archive_receipt_path, receipt)
    require(resolve_formal_run()[1] == dispatch, 'formal dispatch changed during reconciliation')
    load_bound_checkpoint(tree=run['source'], **paths)
    output = {'schema': 'genrec.issue253.reconciliation/1', 'run_id': run['run_id'],
        'catalogue': catalogue, 'timing': timing, 'admission': admission,
        'recall_from_original_per_user_ranks': recall, 'geometry_from_per_centre_records': geometry,
        'reconstruction_from_bound_arrays': mse, 'historical_tiger': full['historical_tiger'],
        'historical_stage3': historical_recall,
        'scientific_threshold': {'test_R@10_greater_than': .065, 'met': recall['test_recall@10'] > .065,
            'interpretation': 'observational, no extra arm/seed/retraining'},
        'r2': record(archive_receipt_path),
        'evidence': {'formal_dispatch': dispatch, 'workspace': record(run['workspace']),
            'stage2_metrics': record(metrics_path), 'stage3_metrics': record(stage3_metrics_path),
            'stage3_dispatch_state': record(stage3_state_path),
            'test_final': record(final_path), 'test_shards': [record(path) for path in shard_paths],
            'test_input': record(test_path), 'layer_metrics': record(layer_path),
            'full_evaluation': record(REPORT), 'evaluator': record(__file__)}}
    write_json(OUTPUT, output)
    print(json.dumps({'report': record(OUTPUT), 'recall': recall, 'r2_number': number}), flush=True)
    return output


def main():
    run_reconciliation()


if __name__ == '__main__':
    main()
