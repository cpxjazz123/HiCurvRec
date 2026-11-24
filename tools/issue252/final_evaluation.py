"""CPU-only final same-generation evaluation; never trains or regenerates SIDs.

Set the explicit FINAL_* constants to the completed immutable run before invoking
``python -m tools.issue252.final_evaluation``. There is no latest-run discovery.
Metric definitions belong to issue251's evaluators, not this orchestration layer.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys

import numpy as np

from tools.issue252 import archive_audit

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'stage2_RQ-VAE').is_dir())
FINAL_RUN = ROOT/'results/stage2_RQ-VAE/_runs/issue252_formal_r2/TIGER_gnn_attention_sectional_rq'
FINAL_TREE = FINAL_RUN/'src'
FINAL_RESULT = FINAL_RUN/'result'
FINAL_CHECKPOINT = FINAL_RESULT/'rqvae_best.pth'
FINAL_SID = FINAL_RESULT/'item_sids.json'
FINAL_BINDING = FINAL_RESULT/'archive_binding.json'
FINAL_EXPORT = FINAL_RESULT/'export_manifest.json'
ASSETS = ROOT/'assets'
EMBEDDINGS = ROOT/'stage1_GeneEmbedding/output/sentence_t5.npy'
REPORT = ROOT/'results/research/issue252/FINAL_EVALUATION.json'
SUMMARY = ROOT/'results/research/issue252/FINAL_EVALUATION_SUMMARY.json'
EXPECTED_ITEMS = 24587
EXPECTED_TRIPLES = 28391948
CODE_SIZES = (256, 256, 256)
BATCH = 1024


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON global identity/key: '+key)
        result[key] = value
    return result


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=_unique_object)


def load_sid(path, items, code_sizes):
    """Read exactly str(0)..str(N-1), explicitly ordered by global identity."""
    if type(items) is not int or items <= 0:
        raise ValueError('global catalogue size must be a positive integer')
    if len(code_sizes) != 3 or any(type(n) is not int or n <= 0 for n in code_sizes):
        raise ValueError('exactly three positive integer codebook sizes required')
    raw = _json(path)
    if not isinstance(raw, dict) or set(raw) != {str(i) for i in range(items)}:
        raise ValueError('SID global catalogue identities must be exactly str(0)..str(N-1)')
    rows = []
    for item in range(items):
        row = raw[str(item)]
        if not isinstance(row, list) or len(row) != 4 or any(type(x) is not int for x in row):
            raise ValueError('SID needs exactly four integer columns, not bool/float')
        if any(not 0 <= row[level] < capacity for level, capacity in enumerate(code_sizes)):
            raise ValueError('SID code outside actual codebook range')
        if not 0 <= row[3] <= np.iinfo(np.int64).max:
            raise ValueError('SID fourth collision counter must be nonnegative int64')
        rows.append(row)
    return np.asarray(rows, dtype=np.int64)


def validate_final_checkpoint(payload):
    """A final-epoch selection is not evidence of completion without epoch 3000."""
    if not isinstance(payload, dict) or type(payload.get('epoch')) is not int or payload['epoch'] != 3000:
        raise ValueError('formal completion requires exactly 3000 epochs; smoke is not final')
    for field in ('run_id', 'source_tree'):
        if not isinstance(payload.get(field), str) or not payload[field].strip():
            raise ValueError('final checkpoint missing '+field+' identity')
    selections = [payload[key] for key in ('selection', 'checkpoint_selection') if key in payload]
    if not selections or any(value != 'final_epoch' for value in selections):
        raise ValueError('final checkpoint requires final_epoch selection')
    provenance = payload.get('provenance')
    if (not isinstance(provenance, dict) or not provenance.get('candidate_sources')
            or not isinstance(provenance['candidate_sources'], dict)
            or not isinstance(provenance.get('infrastructure_sources'), dict)):
        raise ValueError('final checkpoint missing executed source provenance')
    if not isinstance(payload.get('state_dict'), dict) or not payload['state_dict']:
        raise ValueError('final checkpoint missing actual model state')
    return payload


def load_bound_checkpoint(*, checkpoint, sid, tree, binding, export_manifest):
    """Validate exact archive/export identities before importing any model code."""
    import torch

    checkpoint, sid, tree, binding, export_manifest = (
        Path(p).resolve() for p in (checkpoint, sid, tree, binding, export_manifest))
    receipt, _ = archive_audit.validate_binding(binding, sid, checkpoint, tree)
    if receipt['export_manifest']['path'] != str(export_manifest) or not export_manifest.is_file():
        raise ValueError('explicit export manifest differs from exact bound export')
    payload = validate_final_checkpoint(torch.load(checkpoint, map_location='cpu', weights_only=False))
    if payload['run_id'] != receipt['run_id']:
        raise ValueError('checkpoint run identity differs from archive binding')
    source = Path(payload['source_tree'])
    if not source.is_absolute():
        source = Path(receipt['repository'])/source
    if source.resolve() != tree:
        raise ValueError('checkpoint source tree differs from executed source binding')
    if payload['provenance'] != receipt['checkpoint_provenance']:
        raise ValueError('checkpoint provenance differs from bound source provenance')
    if archive_audit.sha256(checkpoint) != receipt['checkpoint']['sha256']:
        raise ValueError('checkpoint changed while reading final binding')
    return payload, receipt


def _record(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError('missing frozen input: '+str(path))
    return {'path': str(path), 'sha256': archive_audit.sha256(path)}


def _frozen_inputs(repo, assets, embeddings, provenance):
    acceptance_path = repo/'results/research/issue252/ACCEPTANCE.json'
    inputs = _json(acceptance_path)['inputs']
    records = {}
    for name, expected in inputs.items():
        actual = _record(repo/expected['path'])
        # Data remain bound to intake; versioned engineering assets are bound
        # to the manifest captured before this exact checkpoint's training.
        expected_digest = (provenance.get('asset_manifest_sha256')
                           if name == 'assets_manifest' else expected['sha256'])
        if actual['sha256'] != expected_digest:
            raise ValueError('frozen '+name+' input hash mismatch')
        records[name] = actual
    if (Path(records['embeddings']['path']) != embeddings
            or Path(records['assets_manifest']['path']) != assets/'manifest.json'):
        raise ValueError('evaluation inputs differ from frozen input paths')
    manifest = _json(assets/'manifest.json')
    if provenance.get('asset_manifest_sha256') != records['assets_manifest']['sha256']:
        raise ValueError('checkpoint asset manifest provenance mismatch')
    if provenance.get('training_inputs', {}).get(inputs['train']['path']) != records['train']['sha256']:
        raise ValueError('checkpoint training input provenance mismatch')
    leaves = {}
    for entry in manifest['assets']:
        relative = Path(entry['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('invalid frozen asset path')
        record = _record(repo/relative)
        if record['sha256'] != entry['sha256']:
            raise ValueError('frozen asset hash mismatch: '+entry['path'])
        if provenance.get('assets', {}).get(entry['path']) != record['sha256']:
            raise ValueError('checkpoint asset provenance mismatch: '+entry['path'])
        leaves[entry['path']] = record
    return {'acceptance': _record(acceptance_path), 'inputs': records, 'assets': leaves,
            'intake_asset_manifest_sha256': inputs['assets_manifest']['sha256'],
            'asset_manifest_authority': 'checkpoint_pre_training_provenance'}, manifest


def _coverage(assets, items):
    with np.load(assets/'node_moments.npz', allow_pickle=False) as blob:
        labels = blob['component']
    if labels.shape != (items,) or labels.dtype.kind not in 'iu' or np.any(labels < 0):
        raise ValueError('frozen component references must cover the entire catalogue')
    with np.load(assets/'index.npz', allow_pickle=False) as blob:
        names = [str(name) for name in blob['shard_names']]
    if len(set(names)) != len(names) or any(Path(name).name != name for name in names):
        raise ValueError('invalid frozen shard identity')
    count = excluded = 0
    rows = []
    for name in names:
        with np.load(assets/'shards'/name, allow_pickle=False) as blob:
            valid = blob['valid']
            if valid.ndim != 1 or valid.dtype.kind != 'b':
                raise ValueError('frozen validity mask must be boolean')
            for field in ('centre', 'b', 'c'):
                values = blob[field]
                if (values.shape != valid.shape or values.dtype.kind not in 'iu'
                        or np.any(values[valid] < 0) or np.any(values[valid] >= items)):
                    raise ValueError('invalid frozen triple global identities')
            for field in ('curvature', 'bc_dist_sq'):
                values = blob[field]
                if values.shape != valid.shape or not np.isfinite(values[valid]).all():
                    raise ValueError('invalid frozen triple target')
            lawful = int(valid.sum())
            invalid = int((~valid).sum())
        count += lawful
        excluded += invalid
        rows.append({'name': name, 'valid': lawful, 'excluded': invalid})
    if count != EXPECTED_TRIPLES:
        raise ValueError('full evaluation requires every frozen valid triple: '+str(count))
    components, sizes = np.unique(labels, return_counts=True)
    return labels, {'items': items, 'valid_triples': count, 'excluded_triples': excluded,
                    'shards': rows, 'references': 'entire original frozen component including centre',
                    'component_sizes': {str(k): int(v) for k, v in zip(components, sizes)},
                    'ordered_component_reference_pairs': sum(int(n)**2 for n in sizes),
                    'sampling': False, 'duplicate_triple_occurrences_preserved': True}


def _training_module(tree):
    # Standard candidate imports are intentionally used, but a previously imported
    # model from another run must never win through Python's module cache.
    for name, module in tuple(sys.modules.items()):
        if name.split('.')[0] not in ('model', 'losses', 'distances', 'train_config',
                                      'train_rqvae', 'assignment_floor'):
            continue
        filename = getattr(module, '__file__', None)
        if filename and not Path(filename).resolve().is_relative_to(tree):
            raise ValueError('conflicting already-imported candidate source: '+name)
    sys.path.insert(0, str(tree))
    training = importlib.import_module('train_rqvae')
    if Path(training.__file__).resolve() != tree/'train_rqvae.py':
        raise ValueError('model import did not use exact bound source tree')
    return training


class _GeometryCrosscheck:
    """Check production on every independent full-reference distance block."""
    def __init__(self, production):
        self.production = production
        self.report = {'matrix_calls': 0, 'paired_calls': 0, 'distance_values': 0,
                       'max_absolute_error': 0., 'rtol': 1e-9, 'atol': 1e-9,
                       'scope': 'every full component block and valid frozen triple pair'}

    def _check(self, x, y, scale, paired):
        import torch
        from tools.issue251.full_evaluation import fixed_squared_numpy, fixed_paired_numpy
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        independent = (fixed_paired_numpy if paired else fixed_squared_numpy)(x, y, scale=scale)
        with torch.inference_mode():
            left, right = torch.from_numpy(x), torch.from_numpy(y)
            if paired:
                actual = self.production(left[:, None, :], right[:, None, :], scale=scale).reshape(-1)
            else:
                actual = self.production(left, right, scale=scale)
            actual = actual.cpu().numpy()
        if actual.shape != independent.shape or not np.isfinite(actual).all():
            raise ValueError('invalid production geometry output')
        error = float(np.abs(actual-independent).max(initial=0.))
        self.report['max_absolute_error'] = max(self.report['max_absolute_error'], error)
        self.report['paired_calls' if paired else 'matrix_calls'] += 1
        self.report['distance_values'] += int(independent.size)
        if not np.allclose(actual, independent, rtol=self.report['rtol'], atol=self.report['atol']):
            raise ValueError('independent ambient geometry disagrees with production: '+str(error))
        return independent

    def matrix(self, x, y, scale=1.):
        return self._check(x, y, scale, False)

    def paired(self, x, y, scale=1.):
        return self._check(x, y, scale, True)


def _prefix_banks(sid, heldout):
    from tools.graph_infra.metrics import prefix_at_k
    result = {}
    for name in heldout.strict_banks():
        bank = heldout.load_strict(name)
        pairs = np.asarray(bank['pairs'])
        if (pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype.kind not in 'iu'
                or np.any(pairs < 0) or np.any(pairs >= len(sid))):
            raise ValueError('invalid strict bank global pair identities')
        result[name] = {'pairs': len(pairs), 'sha256': heldout.entries[name]['sha256'],
                        'representation': 'hard_sid_first_three_columns',
                        'prefix': {f'L{k}': prefix_at_k(((sid[a], sid[b]) for a, b in pairs), k)
                                   if len(pairs) else {'status': 'NA', 'reason': 'empty strict bank'}
                                   for k in (1, 2, 3)}}
    return result


def _compact(value):
    if isinstance(value, dict):
        return {key: _compact(item) for key, item in value.items()
                if key not in ('per_item', 'per_item_l2', 'decoded_q')}
    if isinstance(value, list):
        return [_compact(item) for item in value]
    return value


def run_final_evaluation(*, tree, checkpoint, sid, binding, export_manifest,
                         repo=ROOT, assets=ASSETS, embeddings=EMBEDDINGS,
                         output=REPORT, summary=SUMMARY):
    """Evaluate the explicit completed run; missing/crossed evidence fails closed."""
    import torch
    from tools.issue251.evaluation import (evaluate, evaluate_heldout,
        evaluate_continuous_heldout, evaluate_reconstruction, compare_historical_tiger)
    from tools.issue251.full_evaluation import full_evaluate
    from tools.graph_infra.heldout_eval import HeldOutEvaluation
    from tools.graph_infra import fidelity

    tree, checkpoint, sid, binding, export_manifest, repo, assets, embeddings, output, summary = (
        Path(p).resolve() for p in (tree, checkpoint, sid, binding, export_manifest,
                                   repo, assets, embeddings, output, summary))
    if output == summary or any(p.is_relative_to(tree) or p in (checkpoint, sid, binding, export_manifest)
                                for p in (output, summary)):
        raise ValueError('evaluation outputs cannot overwrite inputs or executed source')
    payload, receipt = load_bound_checkpoint(checkpoint=checkpoint, sid=sid, tree=tree,
                                             binding=binding, export_manifest=export_manifest)
    if Path(receipt['repository']).resolve() != repo:
        raise ValueError('archive binding belongs to another repository')
    inputs, _ = _frozen_inputs(repo, assets, embeddings, payload['provenance'])
    state = payload['state_dict']
    expected_books = {f'rq.vq_layers.{level}.embed.weight' for level in range(3)}
    actual_books = {key for key in state if key.startswith('rq.vq_layers.')
                    and key.endswith('.embed.weight')}
    if actual_books != expected_books:
        raise ValueError('checkpoint must contain exactly three actual codebooks')
    books = []
    for level, capacity in enumerate(CODE_SIZES):
        key = f'rq.vq_layers.{level}.embed.weight'
        book = state.get(key)
        if not isinstance(book, torch.Tensor) or tuple(book.shape) != (capacity, 32):
            raise ValueError('checkpoint missing actual frozen three codebooks: '+key)
        book = book.detach().cpu().numpy()
        if not np.isfinite(book).all():
            raise ValueError('nonfinite actual codebook')
        books.append(book)
    rows = load_sid(sid, EXPECTED_ITEMS, list(CODE_SIZES))
    labels, coverage = _coverage(assets, len(rows))
    target = np.load(embeddings, mmap_mode='r', allow_pickle=False)
    if target.shape != (EXPECTED_ITEMS, 768) or target.dtype != np.float32 or not np.isfinite(target).all():
        raise ValueError('frozen embedding catalogue must be finite float32 (24587,768)')
    torch.set_num_threads(1)
    training = _training_module(tree)
    model = training.RQVAE(training._tokenizer_config(), in_dim=768, distance=training.DISTANCE).cpu()
    model.load_state_dict(state, strict=True)
    model.eval()
    calibration = [float(getattr(model, key)) for key in ('tangent_scale', 'curvature_weight', 'target_scale')]
    if not np.isfinite(calibration).all() or min(calibration) <= 0:
        raise ValueError('checkpoint missing valid frozen calibration')
    scale = calibration[0]
    # The SID is never inferred here. Only actual exported indices select books.
    q = fidelity.decode_hard_sid(rows, books)
    z = np.empty((len(rows), 32), dtype=np.float32)
    reconstruction = np.empty_like(target)
    continuous_reconstruction = np.empty_like(target)
    with torch.inference_mode():
        for start in range(0, len(rows), BATCH):
            end = min(start+BATCH, len(rows))
            encoded = model.encoder(torch.from_numpy(np.array(target[start:end], copy=True)))
            z[start:end] = encoded.cpu().numpy()
            continuous_reconstruction[start:end] = model.decoder(encoded).cpu().numpy()
            hard = torch.from_numpy(q[start:end]).to(dtype=encoded.dtype)
            reconstruction[start:end] = model.decoder(hard).cpu().numpy()
    fixed_squared = importlib.import_module('model.geometry').fixed_squared
    empty = np.empty((0, 3), dtype=np.int64)
    hard = evaluate(rows, books, empty, np.empty(0), {}, component_labels=labels,
                    scale=scale, fixed_squared=fixed_squared,
                    embeddings=target, reconstruction=reconstruction)
    continuous = {'representation': 'continuous_encoder_z',
                  'reconstruction': evaluate_reconstruction(target, continuous_reconstruction,
                                                            representation='continuous-z')}
    agreements = {}
    for name, coordinates, report in (('hard_q', q, hard), ('continuous_z', z, continuous)):
        crosscheck = _GeometryCrosscheck(fixed_squared)
        full = full_evaluate(coordinates, assets, scale=scale,
                             metric=crosscheck.matrix, paired_metric=crosscheck.paired)
        if (full['curvature']['count'] != coverage['valid_triples']
                or full['curvature']['excluded'] != coverage['excluded_triples']
                or full['shards'] != [row['name'] for row in coverage['shards']]):
            raise ValueError('full evaluation did not preserve exact frozen coverage')
        report.update(full)
        agreements[name] = crosscheck.report
    heldout = HeldOutEvaluation(repo, assets)
    if not heldout.strict_banks():
        raise ValueError('strict frozen held-out banks are missing')
    hard['strict_heldout'] = evaluate_heldout(rows, books, heldout, scale=scale, fixed_squared=fixed_squared)
    hard['strict_prefix'] = _prefix_banks(rows, heldout)
    continuous['strict_heldout'] = evaluate_continuous_heldout(z, heldout, scale=scale, fixed_squared=fixed_squared)
    historical = compare_historical_tiger(repo)
    historical['recall'] = {'status': 'NA', 'reason': 'no same-generation historical recall binding supplied'}
    evaluator_paths = ('tools/issue252/final_evaluation.py', 'tools/issue252/archive_audit.py',
                       'tools/issue251/evaluation.py', 'tools/issue251/full_evaluation.py',
                       'tools/graph_infra/fidelity.py', 'tools/graph_infra/metrics.py',
                       'tools/graph_infra/tiger_compare.py', 'tools/graph_infra/heldout_eval.py')
    report = {'schema': 'genrec.issue252.final_evaluation/1', 'scope': 'formal final 3000-epoch same-run checkpoint',
              'hard_q': hard, 'continuous_z': continuous, 'historical_tiger': historical,
              'coverage': coverage, 'ambient_vs_production': agreements,
              'provenance': {'run_id': payload['run_id'], 'epoch': payload['epoch'],
                  'selection': 'final_epoch', 'source_tree': str(tree),
                  'candidate_sources': receipt['candidate_sources'],
                  'infrastructure_sources': receipt['infrastructure_sources'],
                  'checkpoint_provenance': payload['provenance'],
                  'artifacts': {name: _record(path) for name, path in
                      (('checkpoint', checkpoint), ('sid', sid), ('binding', binding),
                       ('export_manifest', export_manifest))},
                  'frozen_inputs': inputs,
                  'evaluator_sources': {name: _record(repo/name) for name in evaluator_paths},
                  'scale_fitted_on_evaluation': False, 'calibration': dict(zip(
                      ('tangent_scale', 'curvature_weight', 'target_scale'), calibration))}}
    # Recheck the binding after the long CPU pass; never publish crossed evidence.
    load_bound_checkpoint(checkpoint=checkpoint, sid=sid, tree=tree, binding=binding,
                          export_manifest=export_manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, separators=(',', ':'), allow_nan=False)+'\n', encoding='utf-8')
    compact = _compact(report)
    compact['full_report'] = {**_record(output), 'bytes': output.stat().st_size}
    summary.write_text(json.dumps(compact, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'run_id': payload['run_id'], 'full_report': compact['full_report'],
                      'summary': str(summary), 'valid_triples_per_representation': coverage['valid_triples']}))
    return report


def main():
    run_final_evaluation(tree=FINAL_TREE, checkpoint=FINAL_CHECKPOINT, sid=FINAL_SID,
                         binding=FINAL_BINDING, export_manifest=FINAL_EXPORT)


if __name__ == '__main__':
    main()
