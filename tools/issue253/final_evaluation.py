"""Source-bound CPU evaluation of the ten prospective Issue253 metric cells.

No training, SID inference, live historical imports, or evaluation-fitted scales.
The only formal-run selector is the once-written dispatch receipt.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import multiprocessing
from pathlib import Path, PurePosixPath
import resource
import time

import numpy as np
from threadpoolctl import threadpool_limits

from tools.issue252 import archive_audit
from tools.issue252.final_evaluation import (load_bound_checkpoint, load_sid,
    validate_final_checkpoint, _coverage, _compact, _prefix_banks)
from tools.issue253.evaluation_reference import fixed_metrics_numpy

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'stage2_RQ-VAE').is_dir())
ARM = 'TIGER_three_expert_attention_rq'
EVIDENCE = ROOT/'results/research/issue253'
FORMAL_RUN_RECEIPT = EVIDENCE/'FORMAL_RUN.json'
REPORT = EVIDENCE/'FINAL_EVALUATION.json'
SUMMARY = EVIDENCE/'FINAL_EVALUATION_SUMMARY.json'
ASSETS = ROOT/'assets'
EMBEDDINGS = ROOT/'stage1_GeneEmbedding/output/sentence_t5.npy'
LEDGER = ROOT/'results/research/issue139/r2_ledger.json'
HISTORICAL_ARM = 'TIGER_gnn_attention_sectional_rq'
HISTORICAL_SHA256 = '0ca5cd32c307460cad7ecd72734538dff2ca3427f64286063ece8c290f8e28f0'
HISTORICAL_SID_SHA256 = '037766704565c5143ab721fd5ca5f61bbba3e800baa9dd561d3ae5601987cff2'
HISTORICAL_BINDING_SHA256 = '549e38cdadf44478e400f45f7de2e934da83383513a5ba46902825be71d3c7ed'
HISTORICAL_TAR_SHA256 = '253944a0bd36499fbfc3cbf9235d173f1888f41c157061855e2b5f78fe03fe6a'
ITEMS, WIDTH, EXPECTED_TRIPLES = 24587, 32, 28391948


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, 'duplicate JSON key: '+key)
            value[key] = item
        return value
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique)


def record(path):
    path = Path(path).resolve()
    require(path.is_file(), 'missing evidence: '+str(path))
    return {'path': str(path), 'sha256': archive_audit.sha256(path)}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix+'.partial')
    partial.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    partial.replace(path)


def resolve_formal_run():
    """Never scan result mtimes; the dispatch receipt names one immutable run."""
    from tools.issue252.run_workspace import load_run
    receipt = read_json(FORMAL_RUN_RECEIPT)
    require(receipt.get('arm') == ARM, 'formal dispatch receipt arm mismatch')
    run = load_run(ROOT, receipt.get('run_id'), ARM)
    for key in ('source', 'result'):
        require(receipt.get(key) == run[key], 'formal dispatch receipt '+key+' mismatch')
    require(receipt.get('workspace_sha256') == archive_audit.sha256(run['workspace']),
            'formal dispatch workspace hash mismatch')
    require(receipt.get('asset_manifest_sha256') == archive_audit.sha256(ASSETS/'manifest.json'),
            'formal dispatch asset manifest changed')
    return run, record(FORMAL_RUN_RECEIPT)


def _positive(value, label):
    require(np.isscalar(value) and np.isfinite(value) and value > 0,
            label+' must be finite and positive')
    return float(value)


def _state_scalar(payload, key):
    import torch
    value = payload.get('state_dict', {}).get(key)
    require(isinstance(value, torch.Tensor) and value.numel() == 1,
            'checkpoint missing scalar '+key)
    return _positive(float(value), key)


def comparison_scale(payload, *, checkpoint_sha256):
    require(checkpoint_sha256 == HISTORICAL_SHA256, 'historical checkpoint hash mismatch')
    return _state_scalar(payload, 'tangent_scale')


def validate_representation(coordinates, *, representation, sid, codebooks):
    from tools.issue251.evaluation import _decode
    rows, books, decoded = _decode(sid, codebooks)
    value = np.asarray(coordinates)
    require(representation in ('hard_q', 'continuous_z'), 'unknown representation')
    require(value.shape == (len(rows), books[0].shape[1]) and np.isfinite(value).all(),
            'representation must cover the finite aligned catalogue')
    if representation == 'hard_q':
        require(np.array_equal(value, decoded.numpy()), 'hard_q differs from actual three-book SID decode')
    return value


def _relative(name):
    require(isinstance(name, str) and name, 'empty archive/source identity')
    path = PurePosixPath(name)
    require(not path.is_absolute() and '..' not in path.parts and str(path) == name,
            'unsafe archive/source identity: '+name)
    return path


def _archive_path(root, name):
    _relative(name)
    root = Path(root).resolve()
    path = root/name
    require(path.resolve().is_relative_to(root), 'archive path escapes its root')
    cursor = path
    while cursor != root:
        require(not cursor.is_symlink(), 'redirected archive source: '+name)
        cursor = cursor.parent
    return path


def validate_archived_sources(receipt, archive_root, *, files, prefix):
    """Validate historical bytes against BOTH receipt and ledger, never live code."""
    prefix = str(_relative(prefix))
    root = Path(archive_root).resolve()
    result = {}
    for group in ('candidate_sources', 'infrastructure_sources', 'repository_configs'):
        mapping = receipt.get(group)
        require(isinstance(mapping, dict), 'missing archived '+group)
        if group == 'candidate_sources':
            require('train_rqvae.py' in mapping and any(k.startswith('model/') for k in mapping),
                    'incomplete archived candidate sources')
        for name, digest in mapping.items():
            _relative(name)
            key = prefix+'/'+name
            path = _archive_path(root, key)
            require(path.is_file(), 'missing archived source: '+name)
            require(files.get(key) == digest == archive_audit.sha256(path),
                    'archived source/ledger hash mismatch: '+name)
        result[group] = dict(mapping)
    audit = receipt.get('audit', {})
    try:
        relative = Path(audit['path']).relative_to(Path(receipt['repository'])).as_posix()
    except (KeyError, ValueError):
        raise ValueError('invalid archived audit source identity') from None
    _relative(relative)
    key = prefix+'/'+relative
    require(files.get(key) == audit.get('sha256') and (root/key).is_file()
            and archive_audit.sha256(root/key) == audit['sha256'], 'archived audit hash mismatch')
    return result


def r2_api():
    spec = importlib.util.spec_from_file_location('issue253_r2_layout',
        ROOT/'results/research/issue139/r2_layout.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def historical_archive():
    """Fetch and reverify the exact numbered archive; no upload or ledger mutation."""
    api = r2_api()
    ledger = read_json(LEDGER)
    entry = ledger.get('100', {})
    require(entry.get('key') == HISTORICAL_ARM+'/'+HISTORICAL_SID_SHA256[:12],
            'historical ledger100 generation mismatch')
    files = entry.get('files', {})
    for key, digest in (('rqvae_best.pth', HISTORICAL_SHA256),
                        ('item_sids.json', HISTORICAL_SID_SHA256),
                        (HISTORICAL_ARM+'/archive_binding.json', HISTORICAL_BINDING_SHA256),
                        (HISTORICAL_ARM+'/stage2_source.tar.gz', HISTORICAL_TAR_SHA256)):
        require(files.get(key) == digest, 'historical ledger identity mismatch: '+key)
    api.verify_upload('100', files)
    directory = EVIDENCE/'historical_archive/100'
    for name, digest in files.items():
        _relative(name)
        path = directory/name
        require(not path.is_symlink() and path.resolve().is_relative_to(directory.resolve()),
                'redirected historical archive destination')
        if path.is_file() and archive_audit.sha256(path) == digest:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(path.suffix+'.partial')
        with partial.open('wb') as output:
            done = api.rclone(['cat', api.BUCKET+'/100/'+name], text=False, consume=output.write)
        require(done.returncode == 0 and archive_audit.sha256(partial) == digest,
                'historical remote object hash mismatch: '+name)
        partial.replace(path)
    binding_path = directory/HISTORICAL_ARM/'archive_binding.json'
    receipt = read_json(binding_path)
    validate_archived_sources(receipt, directory, files=files, prefix=HISTORICAL_ARM)
    return directory, receipt, entry


def _identity_path(value, base=None):
    """Resolve one stored identity that may be repository-relative or absolute.

    The archived binding stores absolute paths while the archived export
    manifest stores repository-relative ones; both name the same generation.
    """
    require(isinstance(value, str) and value.strip(), 'empty stored path identity')
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    root = Path(base).resolve() if base else ROOT
    return (root/path).resolve()


def load_historical_checkpoint(directory, receipt, entry):
    import torch
    directory = Path(directory)
    files = entry['files']
    binding_path = directory/HISTORICAL_ARM/'archive_binding.json'
    require(archive_audit.sha256(binding_path) == HISTORICAL_BINDING_SHA256
            == files.get(HISTORICAL_ARM+'/archive_binding.json')
            and read_json(binding_path) == receipt, 'historical archived receipt changed')
    require(entry.get('key') == HISTORICAL_ARM+'/'+HISTORICAL_SID_SHA256[:12]
            and files.get('rqvae_best.pth') == HISTORICAL_SHA256
            and files.get('item_sids.json') == HISTORICAL_SID_SHA256,
            'historical archive generation changed')
    validate_archived_sources(receipt, directory, files=files, prefix=HISTORICAL_ARM)
    checkpoint, sid = directory/'rqvae_best.pth', directory/'item_sids.json'
    require(archive_audit.sha256(checkpoint) == HISTORICAL_SHA256
            and archive_audit.sha256(sid) == HISTORICAL_SID_SHA256, 'historical checkpoint/SID changed')
    payload = validate_final_checkpoint(torch.load(checkpoint, map_location='cpu', weights_only=False))
    require(payload['run_id'] == receipt.get('run_id') == 'issue252_formal_r2'
            and payload['source_tree'] == receipt.get('source_tree')
            and payload['provenance'] == receipt.get('checkpoint_provenance'),
            'historical checkpoint/archived receipt mismatch')
    for name, digest in (('checkpoint', HISTORICAL_SHA256), ('sid', HISTORICAL_SID_SHA256)):
        require(receipt[name]['sha256'] == digest, 'historical receipt '+name+' mismatch')
    export = read_json(directory/HISTORICAL_ARM/'export_manifest.json')
    base = receipt.get('repository')
    require(export.get('schema') == 'genrec.stage2.export_manifest/1'
            and _identity_path(export.get('tree'), base=base)
            == _identity_path(receipt.get('source_tree'), base=base)
            and export.get('tokens_match') is True
            and export.get('same_l3') == 1. and export.get('differing_items') == 0,
            'historical exact export proof missing')
    for name in ('checkpoint', 'sid'):
        require(_identity_path(export.get(name), base=base)
                == _identity_path(receipt[name]['path'], base=base)
                and export.get(name+'_sha256') == receipt[name]['sha256'],
                'historical export '+name+' identity mismatch')
    export_path = directory/HISTORICAL_ARM/'export_manifest.json'
    require(archive_audit.sha256(export_path) == receipt['export_manifest']['sha256']
            == files[HISTORICAL_ARM+'/export_manifest.json'], 'historical export hash mismatch')
    return payload


def _books(payload):
    import torch
    state = payload['state_dict']
    expected = {f'rq.vq_layers.{level}.embed.weight' for level in range(3)}
    actual = {key for key in state if key.startswith('rq.vq_layers.') and key.endswith('.embed.weight')}
    require(actual == expected, 'checkpoint requires exactly three codebooks')
    books = []
    for level in range(3):
        value = state[f'rq.vq_layers.{level}.embed.weight']
        require(isinstance(value, torch.Tensor) and value.dtype == torch.float32
                and tuple(value.shape) == (256, WIDTH) and bool(torch.isfinite(value).all()),
                'invalid actual checkpoint codebook')
        books.append(value.detach().cpu().numpy())
    return books


def _moments(values, labels, specs, chunk):
    moments = {name: np.zeros(len(values), dtype=np.float64) for name in specs}
    groups = {}
    for name, (experts, scale) in specs.items():
        groups.setdefault(scale, []).append((name, experts))
    for component in np.unique(labels):
        indices = np.flatnonzero(labels == component)
        for start in range(0, len(indices), chunk):
            left = indices[start:start+chunk]
            for offset in range(start, len(indices), chunk):
                right = indices[offset:offset+chunk]
                for scale, cells in groups.items():
                    distances = fixed_metrics_numpy(values[left], values[right], scale,
                                                    metrics=tuple(sorted({e for _, e in cells})))
                    for name, experts in cells:
                        block = distances['fixed'+str(experts)]
                        moments[name][left] += block.sum(axis=1)/len(indices)
                        if offset != start:
                            moments[name][right] += block.sum(axis=0)/len(indices)
    return moments, groups


def _accumulator():
    return {'per_item': {}, 'count': 0, 'excluded': 0, 'absolute_sum': 0., 'macro_sum': 0.,
            'correct': 0., 'distortion_sum': 0., 'relative_sum': 0., 'relative_count': 0,
            'strata': {name: {'count': 0, 'absolute_sum': 0., 'macro_sum': 0., 'centres': 0, 'correct': 0.}
                       for name in ('negative', 'zero', 'positive')}}


def evaluate_cells(coordinates, assets, *, native_scale=None, common_scale, chunk=512, triple_chunk=16384):
    """One component-block traversal and one shard traversal for all requested cells."""
    from tools.issue251.evaluation import _curvature_report
    started = time.perf_counter()
    values = np.asarray(coordinates, dtype=np.float64)
    require(values.ndim == 2 and len(values) and values.shape[1] and np.isfinite(values).all(),
            'finite nonempty coordinate catalogue required')
    require(type(chunk) is int and chunk > 0 and type(triple_chunk) is int and triple_chunk > 0,
            'positive integer evaluation chunks required')
    specs = {'common_fixed3': (3, _positive(common_scale, 'common_scale')),
             'common_fixed5': (5, float(common_scale))}
    if native_scale is not None:
        specs['native_fixed3'] = (3, _positive(native_scale, 'native_scale'))
    assets = Path(assets)
    with np.load(assets/'node_moments.npz', allow_pickle=False) as blob:
        labels = blob['component']
    require(labels.shape == (len(values),) and labels.dtype.kind in 'iu' and np.all(labels >= 0),
            'components must cover the original full catalogue')
    with np.load(assets/'index.npz', allow_pickle=False) as blob:
        names = [str(x) for x in blob['shard_names']]
    require(names and len(names) == len(set(names)) and all(Path(n).name == n for n in names),
            'invalid frozen shard identities')
    with threadpool_limits(limits=1):
        moments, groups = _moments(values, labels, specs, chunk)
    totals = {name: _accumulator() for name in specs}
    for shard in names:
        with np.load(assets/'shards'/shard, allow_pickle=False) as blob:
            raw = {key: blob[key] for key in ('centre', 'b', 'c', 'valid', 'curvature', 'bc_dist_sq')}
        valid = raw['valid']
        require(valid.ndim == 1 and valid.dtype.kind == 'b', 'frozen validity must be boolean')
        require(all(v.shape == valid.shape for v in raw.values()), 'misaligned frozen target columns')
        for key in ('centre', 'b', 'c'):
            require(raw[key].dtype.kind in 'iu', 'integer frozen global IDs required')
        centre, b, c = (raw[k][valid] for k in ('centre', 'b', 'c'))
        target, graph_squared = raw['curvature'][valid], raw['bc_dist_sq'][valid]
        require(np.isfinite(target).all() and np.isfinite(graph_squared).all()
                and np.all(graph_squared >= 0), 'nonfinite/negative lawful frozen target')
        for ids in (centre, b, c):
            require(np.all(ids >= 0) and np.all(ids < len(values)), 'triple outside full catalogue')
        require(np.all((labels[centre] == labels[b]) & (labels[centre] == labels[c])),
                'triple crosses original frozen component')
        predicted = {name: np.empty(len(centre), dtype=np.float64) for name in specs}
        distance_errors = {name: np.empty(len(centre), dtype=np.float64) for name in specs}
        graph_distances = np.sqrt(graph_squared)
        for start in range(0, len(centre), triple_chunk):
            end = min(start+triple_chunk, len(centre))
            bb, cc = b[start:end], c[start:end]
            graph = graph_distances[start:end]
            for scale, cells in groups.items():
                paired = fixed_metrics_numpy(values[bb], values[cc], scale, paired=True,
                                             metrics=tuple(sorted({e for _, e in cells})))
                for name, experts in cells:
                    pair = paired['fixed'+str(experts)]
                    predicted[name][start:end] = (moments[name][centre[start:end]]
                        -.5*(moments[name][bb]+moments[name][cc])+.25*pair)
                    error = np.abs(np.sqrt(pair)-graph)
                    distance_errors[name][start:end] = error
                    total = totals[name]
                    total['distortion_sum'] += float(error.sum())
                    positive = graph > 0
                    total['relative_sum'] += float((error[positive]/graph[positive]).sum())
                    total['relative_count'] += int(positive.sum())
        items, starts, counts = np.unique(centre, return_index=True, return_counts=True)
        for name, total in totals.items():
            total['excluded'] += int((~valid).sum())
            require(np.isfinite(predicted[name]).all(), 'nonfinite full A.1 prediction')
            for item, start, count in zip(items, starts, counts):
                key = str(int(item))
                require(np.all(centre[start:start+count] == item) and key not in total['per_item'],
                        'centre rows must be contiguous and owned by exactly one shard')
                local = _curvature_report(target[start:start+count], predicted[name][start:start+count],
                                          centre[start:start+count], 0)
                per_item = local['per_item'][key]
                graph = graph_distances[start:start+count]
                error = distance_errors[name][start:start+count]
                positive = graph > 0
                per_item['distance_distortion'] = {'count': int(count),
                    'absolute_sum': float(error.sum()), 'relative_count': int(positive.sum()),
                    'relative_sum': float((error[positive]/graph[positive]).sum())}
                # Persist independent sign/stratum counts so reconciliation can
                # recompute ALL aggregate statistics, not merely trust recalls.
                per_item['sign'] = local['sign']
                per_item['strata'] = local['strata']
                total['per_item'][key] = per_item
                total['count'] += int(count)
                total['absolute_sum'] += local['micro_mae']*count
                total['macro_sum'] += local['macro_mae']
                total['correct'] += local['sign']['accuracy']*count
                for sign, accumulator in total['strata'].items():
                    stratum = local['strata'][sign]
                    if stratum['status'] == 'NA':
                        continue
                    n = stratum['count']
                    accumulator['count'] += n
                    accumulator['centres'] += 1
                    accumulator['absolute_sum'] += stratum['micro_mae']*n
                    accumulator['macro_sum'] += stratum['macro_mae']
                    accumulator['correct'] += local['sign']['recall'][sign]['recall']*n
        print('[issue253-evaluation] '+shard+' valid='+str(len(centre)), flush=True)
    result = {}
    for name, (experts, scale) in specs.items():
        total = totals[name]
        count = total['count']
        require(count > 0, 'no valid frozen triples')
        strata, recalls, present = {}, {}, []
        for sign, row in total['strata'].items():
            if row['count']:
                present.append(sign)
                strata[sign] = {'status': 'OK', 'count': row['count'],
                    'micro_mae': row['absolute_sum']/row['count'], 'macro_mae': row['macro_sum']/row['centres']}
                recalls[sign] = {'status': 'OK', 'count': row['count'], 'recall': row['correct']/row['count']}
            else:
                strata[sign] = recalls[sign] = {'status': 'NA', 'reason': 'target sign absent'}
        result[name] = {'metric': 'equal_weight_'+str(experts)+'_expert_squared_distance; independent ambient',
            'expert_curvatures': [-1, 0, 1] if experts == 3 else [-3, -1, 0, 1, 3],
            'tangent_scale': scale, 'reference_scope': 'entire frozen component', 'shards': names,
            'max_scaled_tangent_norm': float(np.linalg.norm(values, axis=1).max()/scale),
            'curvature': {'status': 'OK', 'count': count, 'excluded': total['excluded'],
                'centres': len(total['per_item']), 'per_item': total['per_item'],
                'micro_mae': float(total['absolute_sum']/count),
                'macro_mae': total['macro_sum']/len(total['per_item']), 'strata': strata,
                'sign': {'accuracy': float(total['correct']/count), 'recall': recalls,
                    'balanced_accuracy': float(np.mean([recalls[s]['recall'] for s in present])),
                    'present_signs': present, 'zero_tolerance': 0.}},
            'distance_distortion': {'count': count, 'mean_abs_error': total['distortion_sum']/count,
                'mean_relative_error': total['relative_sum']/total['relative_count'] if total['relative_count'] else None,
                'distance_scale': 1., 'scale_fitted_on_evaluation': False,
                'scope': 'all valid frozen triple (b,c) pairs, weighted by triple occurrence'}}
    return result


def _broadcast_metric(experts):
    def metric(x, y, scale=1.):
        left, right = np.broadcast_arrays(np.asarray(x), np.asarray(y))
        shape = left.shape[:-1]
        return fixed_metrics_numpy(left.reshape(-1, left.shape[-1]), right.reshape(-1, right.shape[-1]),
            scale, paired=True, metrics=(experts,))['fixed'+str(experts)].reshape(shape)
    return metric


def _extract_new_worker(tree, checkpoint, sid, binding, export, embeddings, output):
    import torch
    from tools.issue252.final_evaluation import _training_module
    from tools.graph_infra.fidelity import decode_hard_sid
    payload, _ = load_bound_checkpoint(checkpoint=checkpoint, sid=sid, tree=tree,
                                      binding=binding, export_manifest=export)
    torch.set_num_threads(1)
    training = _training_module(Path(tree))
    model = training.RQVAE(training._tokenizer_config(), in_dim=768, distance=training.DISTANCE).cpu()
    model.load_state_dict(payload['state_dict'], strict=True)
    model.eval()
    rows = load_sid(sid, ITEMS, [256]*3)
    q = decode_hard_sid(rows, _books(payload))
    target = np.load(embeddings, mmap_mode='r', allow_pickle=False)
    z = np.empty((ITEMS, WIDTH), dtype=np.float32)
    reconstructed_q, reconstructed_z = np.empty_like(target), np.empty_like(target)
    with torch.inference_mode():
        for start in range(0, ITEMS, 1024):
            end = min(start+1024, ITEMS)
            encoded = model.encoder(torch.from_numpy(np.array(target[start:end], copy=True)))
            z[start:end] = encoded.numpy()
            reconstructed_z[start:end] = model.decoder(encoded).numpy()
            reconstructed_q[start:end] = model.decoder(torch.from_numpy(q[start:end]).to(encoded.dtype)).numpy()
    np.savez(output, q=q, z=z, reconstruction_q=reconstructed_q, reconstruction_z=reconstructed_z)


def extract_new(run, paths):
    output = EVIDENCE/'extraction'/run['run_id']/'final_arrays.npz'
    output.parent.mkdir(parents=True, exist_ok=True)
    process = multiprocessing.get_context('spawn').Process(target=_extract_new_worker,
        args=(run['source'], paths['checkpoint'], paths['sid'], paths['binding'], paths['export_manifest'],
              EMBEDDINGS, output))
    process.start()
    while process.is_alive():
        process.join(30)
        if process.is_alive():
            print('[issue253-evaluation] extracting exact new checkpoint pid='+str(process.pid), flush=True)
    require(process.exitcode == 0 and output.is_file(), 'source-bound extraction subprocess failed')
    with np.load(output, allow_pickle=False) as blob:
        arrays = {name: blob[name] for name in blob.files}
    return arrays, record(output)


def _historical_mlp_source(tree):
    """Prove the executed MLP class from archived source, without importing it."""
    layers = ast.parse((tree/'model/layers.py').read_text())
    mlp = next(node for node in layers.body if isinstance(node, ast.ClassDef) and node.name == 'MLP')
    # Freeze exact semantics as AST, independent of whitespace/comments.
    expected = ast.parse('''class MLP(nn.Module):
    def __init__(self, hidden_sizes: list[int], dropout: float = 0.0):
        super().__init__()
        modules: list[nn.Module] = []
        for input_size, output_size in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            modules.extend((nn.Dropout(p=dropout), nn.Linear(input_size, output_size), nn.ReLU()))
        if modules:
            modules.pop()
        self.mlp = nn.Sequential(*modules)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
''').body[0]
    for method in ('__init__', 'forward'):
        actual = next(node for node in mlp.body if isinstance(node, ast.FunctionDef) and node.name == method)
        oracle = next(node for node in expected.body if isinstance(node, ast.FunctionDef) and node.name == method)
        require(ast.dump(actual, include_attributes=False) == ast.dump(oracle, include_attributes=False),
                'unsupported archived MLP '+method+' semantics')
    return mlp


def extract_historical(payload, directory):
    """Exact archived Linear/ReLU/Dropout(eval) graph, no live historical imports."""
    import torch
    import torch.nn.functional as functional
    from tools.graph_infra.fidelity import decode_hard_sid
    tree = Path(directory)/HISTORICAL_ARM
    mlp_ast = _historical_mlp_source(tree)
    config = read_json(tree/'config/train_config.json')
    sizes = [768]+config['HIDDEN_SIZES']+[WIDTH]
    require(config['CODEBOOK_DIM'] == WIDTH and config['CODEBOOK_SIZE'] == [256]*3,
            'historical archived architecture mismatch')
    state = payload['state_dict']
    torch.set_num_threads(1)
    graphs = {}
    for branch, widths in (('encoder', sizes), ('decoder', sizes[::-1])):
        keys = {f'{branch}.mlp.{1+3*level}.{field}' for level in range(len(widths)-1)
                for field in ('weight', 'bias')}
        require({key for key in state if key.startswith(branch+'.')} == keys,
                'unsupported historical '+branch+' state schema')
        parameters = []
        for level, (input_width, output_width) in enumerate(zip(widths[:-1], widths[1:])):
            weight, bias = (state[f'{branch}.mlp.{1+3*level}.{field}'] for field in ('weight', 'bias'))
            require(tuple(weight.shape) == (output_width, input_width) and tuple(bias.shape) == (output_width,)
                    and weight.dtype == bias.dtype == torch.float32
                    and bool(torch.isfinite(weight).all() and torch.isfinite(bias).all()),
                    'invalid actual historical affine state')
            parameters.append((weight, bias))
        graphs[branch] = parameters

    def forward(values, branch):
        for index, (weight, bias) in enumerate(graphs[branch]):
            values = functional.linear(values, weight, bias)
            if index+1 != len(graphs[branch]):
                values = functional.relu(values)
        return values

    # Execute ONLY the already inspected archived MLP class, not imports, trainer,
    # GNN, quantizer or changed infrastructure. Its initializer is not called:
    # build the equivalent modules directly, avoiding RNG-dependent initialization.
    namespace = {'nn': torch.nn, 'torch': torch}
    checked_class = ast.Module(body=[mlp_ast], type_ignores=[])
    exec(compile(checked_class, str(tree/'model/layers.py')+'#verified-MLP', 'exec'), namespace)
    for branch, widths in (('encoder', sizes), ('decoder', sizes[::-1])):
        original = namespace['MLP'].__new__(namespace['MLP'])
        torch.nn.Module.__init__(original)
        modules = []
        for weight, bias in graphs[branch]:
            affine = torch.nn.Linear.__new__(torch.nn.Linear)
            torch.nn.Module.__init__(affine)
            affine.in_features, affine.out_features = weight.shape[1], weight.shape[0]
            affine.weight, affine.bias = torch.nn.Parameter(weight, requires_grad=False), torch.nn.Parameter(bias, requires_grad=False)
            modules.extend((torch.nn.Dropout(0.), affine, torch.nn.ReLU()))
        original.mlp = torch.nn.Sequential(*modules[:-1])
        original.eval()
        fixture = torch.arange(3*widths[0], dtype=torch.float32).reshape(3, widths[0])/widths[0]
        with torch.inference_mode():
            require(torch.equal(original(fixture), forward(fixture, branch)),
                    'historical checkpoint-only extraction disagrees with archived MLP forward')
    rows = load_sid(Path(directory)/'item_sids.json', ITEMS, [256]*3)
    q = decode_hard_sid(rows, _books(payload))
    target = np.load(EMBEDDINGS, mmap_mode='r', allow_pickle=False)
    z, reconstruction_q, reconstruction_z = np.empty((ITEMS, WIDTH), np.float32), np.empty_like(target), np.empty_like(target)
    with torch.inference_mode():
        for start in range(0, ITEMS, 1024):
            end = min(start+1024, ITEMS)
            encoded = forward(torch.from_numpy(np.array(target[start:end], copy=True)), 'encoder')
            z[start:end] = encoded.numpy()
            reconstruction_z[start:end] = forward(encoded, 'decoder').numpy()
            reconstruction_q[start:end] = forward(torch.from_numpy(q[start:end]).float(), 'decoder').numpy()
    arrays = dict(q=q, z=z, reconstruction_q=reconstruction_q, reconstruction_z=reconstruction_z)
    output = EVIDENCE/'extraction/issue252_formal_r2/final_arrays.npz'
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **arrays)
    return arrays, record(output)


def frozen_inputs(new_payload, historical_payload):
    """Shared old leaves stay frozen; new assets bind only the new manifest."""
    intake = read_json(EVIDENCE/'asset_audit.json')
    expected = {entry['path']: entry['sha256'] for entry in intake['assets']}
    records = {}
    for name, digest in expected.items():
        item = record(ROOT/name)
        require(item['sha256'] == digest, 'intake input changed: '+name)
        records[name] = item
        for payload in (new_payload, historical_payload):
            bound = payload['provenance'].get('assets', {})
            if name not in {'results/stage0_build_parquet/'+split+'.parquet'
                            for split in ('train', 'valid', 'test')}:
                require(bound.get(name) == digest, 'checkpoint frozen asset mismatch: '+name)
    manifest = read_json(ASSETS/'manifest.json')
    provenance = new_payload['provenance']
    require(provenance.get('asset_manifest_sha256') == archive_audit.sha256(ASSETS/'manifest.json'),
            'new checkpoint pre-training manifest mismatch')
    for entry in manifest['assets']:
        name = str(_relative(entry['path']))
        actual = record(ROOT/name)
        require(actual['sha256'] == entry['sha256'] == provenance.get('assets', {}).get(name),
                'new registered asset source mismatch: '+name)
        records[name] = actual
    for payload in (new_payload, historical_payload):
        require(payload['provenance'].get('training_inputs', {}).get('results/stage0_build_parquet/train.parquet')
                == expected['results/stage0_build_parquet/train.parquet'], 'checkpoint train input mismatch')
        require(payload['provenance'].get('assets', {}).get('stage1_GeneEmbedding/output/sentence_t5.npy')
                == expected['stage1_GeneEmbedding/output/sentence_t5.npy'], 'checkpoint embedding input mismatch')
        for name in ('assets/index.npz', 'assets/node_moments.npz'):
            require(payload['provenance'].get('assets', {}).get(name) == expected[name],
                    'checkpoint full reference identity missing')
    require(_state_scalar(new_payload, 'target_scale') == _state_scalar(historical_payload, 'target_scale'),
            'frozen target scales differ')
    return {'asset_audit': record(EVIDENCE/'asset_audit.json'), 'files': records,
            'new_asset_manifest': record(ASSETS/'manifest.json'),
            'historical_asset_manifest_sha256': historical_payload['provenance']['asset_manifest_sha256']}


def run_final_evaluation():
    import torch
    from tools.graph_infra import fidelity
    from tools.graph_infra.heldout_eval import HeldOutEvaluation
    from tools.issue251.evaluation import (evaluate_reconstruction, evaluate_heldout,
        evaluate_continuous_heldout, compare_historical_tiger)
    started = time.perf_counter()
    torch.set_num_threads(1)
    run, dispatch = resolve_formal_run()
    result = Path(run['result'])
    paths = {'checkpoint': result/'rqvae_best.pth', 'sid': result/'item_sids.json',
             'binding': result/'archive_binding.json', 'export_manifest': result/'export_manifest.json'}
    new, receipt = load_bound_checkpoint(tree=run['source'], **paths)
    require(new['run_id'] == run['run_id'], 'new formal checkpoint run differs from dispatch')
    archive_start = time.perf_counter()
    archive, old_receipt, old_entry = historical_archive()
    old = load_historical_checkpoint(archive, old_receipt, old_entry)
    archive_seconds = time.perf_counter()-archive_start
    inputs = frozen_inputs(new, old)
    _, coverage = _coverage(ASSETS, ITEMS)
    target = np.load(EMBEDDINGS, mmap_mode='r', allow_pickle=False)
    require(target.shape == (ITEMS, 768) and target.dtype == np.float32 and np.isfinite(target).all(),
            'frozen embeddings must be finite float32 [24587,768]')
    extraction_start = time.perf_counter()
    new_arrays, new_extraction = extract_new(run, paths)
    new_extraction_seconds = time.perf_counter()-extraction_start
    extraction_start = time.perf_counter()
    old_arrays, old_extraction = extract_historical(old, archive)
    old_extraction_seconds = time.perf_counter()-extraction_start
    s_ref = comparison_scale(old, checkpoint_sha256=HISTORICAL_SHA256)
    heldout = HeldOutEvaluation(ROOT, ASSETS)
    require(bool(heldout.strict_banks()), 'strict heldout banks missing')
    reports = {}
    for generation, payload, sid, arrays, native in (
            ('issue253', new, paths['sid'], new_arrays, _state_scalar(new, 'tangent_scale')),
            ('issue252', old, archive/'item_sids.json', old_arrays, None)):
        books = _books(payload)
        rows = load_sid(sid, ITEMS, [256]*3)
        reports[generation] = {}
        for representation, key in (('hard_q', 'q'), ('continuous_z', 'z')):
            values = validate_representation(arrays[key], representation=representation, sid=rows, codebooks=books)
            cell_start = time.perf_counter()
            cells = evaluate_cells(values, ASSETS, native_scale=native, common_scale=s_ref)
            geometry_seconds = time.perf_counter()-cell_start
            heldout_start = time.perf_counter()
            for cell in cells.values():
                require(cell['curvature']['count'] == coverage['valid_triples']
                        and cell['curvature']['excluded'] == coverage['excluded_triples']
                        and cell['shards'] == [row['name'] for row in coverage['shards']],
                        'cell did not preserve full frozen coverage')
                experts = len(cell['expert_curvatures'])
                callback = _broadcast_metric(experts)
                cell['strict_heldout'] = (evaluate_heldout(rows, books, heldout,
                    scale=cell['tangent_scale'], fixed_squared=callback) if representation == 'hard_q'
                    else evaluate_continuous_heldout(values, heldout, scale=cell['tangent_scale'], fixed_squared=callback))
            view = {'representation': representation, 'cells': cells,
                'reconstruction': evaluate_reconstruction(target, arrays['reconstruction_'+key], representation=representation),
                'geometry_wall_seconds': geometry_seconds,
                'heldout_wall_seconds': time.perf_counter()-heldout_start,
                'joint_cells_and_heldout_wall_seconds': time.perf_counter()-cell_start}
            if representation == 'hard_q':
                view.update(prefix={f'L{k}': fidelity.prefix_collision(rows, k) for k in (1, 2, 3)},
                            prefix_sharing=fidelity.prefix_sharing(rows),
                            codebook_occupancy=fidelity.codebook_occupancy(rows), strict_prefix=_prefix_banks(rows, heldout))
                for level in view['codebook_occupancy']:
                    level.update(capacity=256, occupancy=level['used']/256)
            reports[generation][representation] = view
    historical = compare_historical_tiger(ROOT)
    historical['recall'] = {'status': 'NA', 'reason': 'no same-source historical Stage3 rank/checkpoint binding supplied'}
    old_recall_path = archive/'recall.json'
    require(record(old_recall_path)['sha256'] == old_entry['files']['recall.json'],
            'historical Stage3 recall object changed')
    old_recall = read_json(old_recall_path)
    require(old_recall['sid_sha256'] == HISTORICAL_SID_SHA256
            and old_recall['stage2_checkpoint_sha256'] == HISTORICAL_SHA256
            and old_recall['archive_binding_sha256'] == HISTORICAL_BINDING_SHA256
            and old_recall['stage2_run_id'] == old['run_id'],
            'historical Stage3/SID/checkpoint generation mismatch')
    sources = {str(p.relative_to(ROOT)): record(p) for p in (
        Path(__file__), ROOT/'tools/issue253/evaluation_reference.py', ROOT/'tools/issue253/reconcile_results.py',
        ROOT/'tools/issue251/evaluation.py', ROOT/'tools/issue251/full_evaluation.py',
        ROOT/'tools/graph_infra/fidelity.py', ROOT/'tools/graph_infra/heldout_eval.py')}
    report = {'schema': 'genrec.issue253.final_evaluation/1', 'run_id': run['run_id'],
        'generations': reports, 'historical_tiger': historical, 'coverage': coverage,
        'historical_stage3': {'archive_recall': record(old_recall_path),
            'stage3_run_id': old_recall['stage3_run_id'],
            'metrics': {key: old_recall[key] for key in
                ('n_eval', 'test_recall@5', 'test_recall@10', 'test_ndcg@5', 'test_ndcg@10')}},
        'comparison': {'s_ref': s_ref, 'source_checkpoint_sha256': HISTORICAL_SHA256,
                       'scale_fitted_on_evaluation': False, 'target_scale': _state_scalar(new, 'target_scale'),
                       'interpretation': 'observational single-arm differences, not expert-count causal proof'},
        'provenance': {'formal_dispatch': dispatch, 'workspace': record(run['workspace']),
            'new_binding': receipt, 'new_artifacts': {key: record(path) for key, path in paths.items()},
            'historical_binding': old_receipt, 'historical_archive': {'number': '100', 'entry': old_entry,
                'binding': record(archive/HISTORICAL_ARM/'archive_binding.json'), 'ledger': record(LEDGER)},
            'extractions': {'issue253': new_extraction, 'issue252': old_extraction},
            'frozen_inputs': inputs, 'evaluator_sources': sources},
        'phase_wall_seconds': {'historical_archive_fetch_verify': archive_seconds,
            'new_source_bound_extraction': new_extraction_seconds,
            'historical_source_bound_extraction': old_extraction_seconds},
        'total_wall_seconds': time.perf_counter()-started,
        'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    require(resolve_formal_run()[1] == dispatch, 'dispatch receipt changed during evaluation')
    load_bound_checkpoint(tree=run['source'], **paths)
    load_historical_checkpoint(archive, old_receipt, old_entry)
    require(frozen_inputs(new, old) == inputs, 'frozen evaluation inputs changed during pass')
    write_json(REPORT, report)
    compact = _compact(report)
    compact['full_report'] = record(REPORT)
    write_json(SUMMARY, compact)
    print(json.dumps({'run_id': run['run_id'], 'report': record(REPORT), 'cells': 10}), flush=True)
    return report


def main():
    run_final_evaluation()


if __name__ == '__main__':
    main()
