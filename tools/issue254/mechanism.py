"""Read-only, CPU-only complete-catalogue mechanism diagnostics for Issue254."""
from __future__ import annotations

import csv
import gc
import hashlib
import importlib
import json
from pathlib import Path
import sys
import types

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'stage2_RQ-VAE').is_dir())
OUTPUT = ROOT / 'results/research/issue254'
ITEMS = 24587
SCALE = 0.08433422446250916


def require(condition, message):
    if not condition:
        raise ValueError(message)


def aggregate_gates(weights, curvatures):
    weights = np.asarray(weights, dtype=np.float64)
    curvatures = np.asarray(curvatures)
    require(weights.ndim == 2 and weights.shape[1] == len(curvatures), 'expert shape mismatch')
    require(np.isfinite(weights).all() and (weights >= 0).all(), 'invalid expert weights')
    require(np.allclose(weights.sum(1), 1, atol=2e-6, rtol=0), 'expert weights not normalized')
    terms = np.zeros_like(weights)
    np.log(weights, out=terms, where=weights > 0)
    return dict(negative=weights[:, curvatures < 0].sum(1), zero=weights[:, curvatures == 0].sum(1),
                positive=weights[:, curvatures > 0].sum(1), entropy=-(weights * terms).sum(1))


def summarize_attention(gates, attention, mask):
    gates, attention, mask = np.asarray(gates), np.asarray(attention), np.asarray(mask)
    require(attention.shape == gates.shape + (3,) and mask.shape == (len(gates), 3)
            and mask.dtype == bool, 'attention shape mismatch')
    require(np.isfinite(attention).all() and (attention >= 0).all(), 'invalid attention')
    require(np.allclose(attention.sum(2), 1, atol=2e-6, rtol=0), 'attention not normalized')
    require(np.all(mask[:, 2] == ~mask[:, :2].any(1)), 'null attention mask mismatch')
    require(np.all(np.where(mask[:, None, :], 0, attention) == 0), 'masked attention is nonzero')
    return np.einsum('ne,net->nt', gates, attention)


def join_curvature(per_item, item_ids):
    item_ids = np.asarray(item_ids)
    require(len(np.unique(item_ids)) == len(item_ids), 'duplicate catalogue item IDs')
    require(set(per_item).issubset({str(int(x)) for x in item_ids}), 'unknown curvature centre')
    mae, accuracy = np.full(len(item_ids), np.nan), np.full(len(item_ids), np.nan)
    count = np.zeros(len(item_ids), dtype=np.int64)
    for row, item in enumerate(item_ids):
        value = per_item.get(str(int(item)))
        if value is None:
            continue
        count[row] = value['count']
        require(count[row] > 0, 'stored curvature centre has no valid triples')
        mae[row], accuracy[row] = value['mean_abs_error'], value['sign']['accuracy']
        require(np.isfinite(mae[row]) and mae[row] >= 0 and 0 <= accuracy[row] <= 1,
                'invalid stored curvature metric')
    return mae, accuracy, count


def sid_diagnostics(sid, forward):
    """Compare equality relations only within this arm; missing edges stay NaN."""
    sid = np.asarray(sid)
    require(sid.ndim == 2 and sid.shape[1] >= 3 and forward.shape == (len(sid), len(sid)),
            'SID/graph catalogue mismatch')
    result = {}
    for level in (1, 2, 3):
        _, labels, counts = np.unique(sid[:, :level], axis=0, return_inverse=True, return_counts=True)
        result['sid_l%d_occupancy' % level] = counts[labels]
        for direction, graph in (('out', forward.tocsr()), ('in', forward.T.tocsr())):
            degree = np.diff(graph.indptr)
            rows = np.repeat(np.arange(len(sid)), degree)
            shared = np.bincount(rows, weights=(labels[rows] == labels[graph.indices]), minlength=len(sid))
            fraction = np.full(len(sid), np.nan)
            np.divide(shared, degree, out=fraction, where=degree > 0)
            result['edge_prefix_l%d_%s' % (level, direction)] = fraction
    result['collision_group_size'] = result['sid_l3_occupancy'].copy()
    return result


class Audit:
    def __init__(self):
        self.files = {}

    def check(self, path, digest):
        path = Path(path)
        if not path.is_absolute():
            path = ROOT / path
        path = path.resolve()
        key = str(path)
        if key not in self.files:
            h = hashlib.sha256()
            with path.open('rb') as source:
                for block in iter(lambda: source.read(8 << 20), b''):
                    h.update(block)
            self.files[key] = h.hexdigest()
        require(self.files[key] == digest, 'SHA256 mismatch: ' + key)
        return path

    def record(self, record):
        return self.check(record['path'], record['sha256'])


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON identity: ' + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def load_gate(tree, state, arm):
    """Import just the exact archived attention/token modules, never trainer/init."""
    import torch
    name = '_issue254_' + arm
    package = types.ModuleType(name)
    package.__path__ = [str(tree / 'model')]
    sys.modules[name] = package
    module = importlib.import_module(name + '.structure_gate')
    gate = module.TopologyAttention().cpu()
    selected = {k[len('structure.'):]: v for k, v in state.items() if k.startswith('structure.')}
    require(bool(selected), 'checkpoint lacks original structure weights')
    gate.load_state_dict(selected, strict=True)
    gate.requires_grad_(False)
    gate.eval()
    require(all(p.device.type == 'cpu' for p in gate.parameters()), 'non-CPU gate parameter')
    return gate


def extract_gates(gate):
    """Use original full static-operator CPU path, with complete egos per item."""
    import torch
    # The original token path only needs these three store attributes. Avoid EgoStore's
    # unrelated graph loads and any on-demand graph construction.
    class StaticStore:
        repo = ROOT
        root = ROOT / 'assets'
        n = ITEMS
    store = StaticStore()
    gates, attention, masks = [], [], []
    with torch.inference_mode():
        for start in range(0, ITEMS, 32):
            ids = np.arange(start, min(start + 32, ITEMS))
            _, weights = gate.gates_for_items(store, ids, torch.device('cpu'), chunk=32)
            gates.append(weights.numpy().copy())
            attention.append(gate.last_attention.numpy().copy())
            masks.append(gate.last_token_mask.numpy().copy())
        weights, attention, mask = np.concatenate(gates), np.concatenate(attention), np.concatenate(masks)
        fixed = np.array([0, 1, 127, 1024, ITEMS - 1, 1], dtype=np.int64)
        _, reference = gate.gates_for_items(store, fixed, torch.device('cpu'), chunk=len(fixed))
        ref_attention, ref_mask = gate.last_attention.clone(), gate.last_token_mask.clone()
        _, partitioned = gate.gates_for_items(store, fixed, torch.device('cpu'), chunk=1)
        error = max(float(np.max(np.abs(reference.numpy() - partitioned.numpy()))),
                    float(np.max(np.abs(ref_attention.numpy() - gate.last_attention.numpy()))),
                    float(np.max(np.abs(weights[fixed] - reference.numpy()))),
                    float(np.max(np.abs(attention[fixed] - ref_attention.numpy()))))
        require(error <= 2e-6, 'full/partitioned original CPU gate disagreement')
        require(np.array_equal(mask[fixed], ref_mask.numpy())
                and torch.equal(ref_mask, gate.last_token_mask), 'partitioned mask disagreement')
    return weights, attention, mask, {'item_ids': fixed.tolist(), 'max_abs_error': error, 'atol': 2e-6}


def main():
    import torch
    from scipy.sparse import csr_matrix
    torch.set_num_threads(1)
    audit = Audit()
    evidence = read_json(OUTPUT / 'EVIDENCE.json')
    evaluation = read_json(audit.record(evidence['issue253_final_evaluation']))
    manifest_path = audit.record(evidence['asset_manifest'])
    manifest = read_json(manifest_path)
    entries = {e['path']: e for e in manifest['assets']}
    ledger = read_json(audit.record(evidence['r2_ledger']))
    frozen = evaluation['provenance']['frozen_inputs']['files']
    ids = np.load(audit.record(frozen['assets/train/item_ids.npy']), allow_pickle=False)
    require(np.array_equal(ids, np.arange(ITEMS)), 'global catalogue order changed')
    embeddings = np.load(audit.record(frozen['stage1_GeneEmbedding/output/sentence_t5.npy']), mmap_mode='r')
    require(embeddings.shape == (ITEMS, 768), 'embedding shape mismatch')
    graph_path = 'assets/train/visible_forward.npz'
    with np.load(audit.check(graph_path, entries[graph_path]['sha256']), allow_pickle=False) as graph_blob:
        graph = csr_matrix((graph_blob['data'], graph_blob['indices'], graph_blob['indptr']),
                           shape=tuple(graph_blob['shape']))
    columns, raw, summaries = {}, {'item_ids': ids}, {}
    for arm in ('issue252', 'issue253'):
        identity = evidence['generations'][arm]
        checkpoint = audit.record(identity['stage2_checkpoint'])
        sid_path = audit.record(identity['stage2_sid'])
        binding = read_json(audit.record(identity['stage2_archive_binding']))
        evaluation_binding = evaluation['provenance']['historical_binding' if arm == 'issue252' else 'new_binding']
        require(evaluation_binding == binding, 'reused geometry belongs to a different source/checkpoint binding')
        export = read_json(audit.record(identity['stage2_export_manifest']))
        test = read_json(audit.record(identity['stage3_test_final']))
        for field, source in (('sid', 'stage2_sid'), ('checkpoint', 'stage2_checkpoint')):
            expected = identity[source]['sha256']
            require(binding[field]['sha256'] == expected and export[field + '_sha256'] == expected,
                    'binding/export identity mismatch')
            test_field = 'sid_sha256' if field == 'sid' else 'stage2_checkpoint_sha256'
            require(test[test_field] == expected, 'Stage3 mechanism identity mismatch')
            leaf = 'item_sids.json' if field == 'sid' else 'rqvae_best.pth'
            require(ledger[identity['r2_number']]['files'][leaf] == expected, 'R2 identity mismatch')
        require(test['archive_binding_sha256'] == identity['stage2_archive_binding']['sha256']
                and test['stage2_export_manifest_sha256'] == identity['stage2_export_manifest']['sha256'],
                'Stage3 binding/export mismatch')
        tree = Path(binding['source_tree'])
        if not tree.is_dir() and arm == 'issue252':
            tree = ROOT / 'results/research/issue253/historical_archive/100' / identity['arm']
        for relative, digest in binding['candidate_sources'].items():
            audit.check(tree / relative, digest)
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        require(payload['epoch'] == 3000 and payload['run_id'] == identity['stage2_run_id'], 'wrong final checkpoint')
        require(payload['provenance'] == binding['checkpoint_provenance'], 'checkpoint provenance mismatch')
        require(payload['provenance']['asset_manifest_sha256'] == evidence['asset_manifest']['sha256'],
                'asset manifest not checkpoint-bound')
        for relative, digest in payload['provenance']['assets'].items():
            audit.check(relative, digest)
        arrays_path = audit.record(evaluation['provenance']['extractions'][arm])
        with np.load(arrays_path, allow_pickle=False) as blob:
            arrays = {key: blob[key] for key in ('q', 'z', 'reconstruction_q', 'reconstruction_z')}
        require(arrays['q'].shape == arrays['z'].shape == (ITEMS, 32)
                and arrays['q'].dtype == np.float64 and arrays['z'].dtype == np.float32,
                'invalid source-bound coordinate arrays')
        for key in ('reconstruction_q', 'reconstruction_z'):
            require(arrays[key].shape == embeddings.shape, 'invalid reconstruction dimensions')
        require(all(np.isfinite(a).all() for a in arrays.values()), 'nonfinite source-bound extraction')
        sid_json = read_json(sid_path)
        require(set(sid_json) == {str(i) for i in ids}, 'SID catalogue mismatch')
        sid = np.asarray([sid_json[str(i)] for i in ids], dtype=np.int64)
        require(sid.shape == (ITEMS, 4) and ((sid[:, :3] >= 0) & (sid[:, :3] < 256)).all(),
                'invalid SID code range')
        q_check = np.zeros((ITEMS, 32), dtype=np.float64)
        for level in range(3):
            book = payload['state_dict'][f'rq.vq_layers.{level}.embed.weight'].numpy()
            require(book.shape == (256, 32), 'invalid checkpoint codebook')
            q_check += book[sid[:, level]].astype(np.float64)
        require(np.array_equal(q_check, arrays['q']), 'hard q is not SID-selected codebook sum')
        # Verify the exact transitive CPU dependencies before importing the gate.
        infrastructure = ['tools/issue252/ego_operators.py', 'tools/issue252/host_transfer.py']
        infrastructure += ['tools/' + arm + '/' + leaf + '.py'
                           for leaf in ('ego_kernel', 'sparse_adjoint', 'static_features')]
        for relative in infrastructure:
            require(relative in binding['infrastructure_sources'], 'unbound CPU infrastructure: ' + relative)
            audit.check(relative, binding['infrastructure_sources'][relative])
        gate = load_gate(tree, payload['state_dict'], arm)
        weights, attention, mask, partition = extract_gates(gate)
        expert_curvatures = [-3, -1, 0, 1, 3] if arm == 'issue252' else [-1, 0, 1]
        grouped = aggregate_gates(weights, expert_curvatures)
        values = {'gate_' + key: value for key, value in grouped.items()}
        weighted = summarize_attention(weights, attention, mask)
        values.update({'attention_' + label: weighted[:, i] for i, label in enumerate(('1hop', '2hop', 'null'))})
        for rep in ('q', 'z'):
            values['reconstruction_mse_' + rep] = np.mean(
                (arrays['reconstruction_' + rep].astype(np.float64) - embeddings) ** 2, axis=1)
        values['quantization_mse'] = np.mean((arrays['q'] - arrays['z']) ** 2, axis=1)
        counts = []
        for rep, key in (('q', 'hard_q'), ('z', 'continuous_z')):
            cell = evaluation['generations'][arm][key]['cells']['common_fixed3']
            require(cell['tangent_scale'] == SCALE and cell['expert_curvatures'] == [-1, 0, 1],
                    'common curvature metric changed')
            curvature = cell['curvature']
            require(curvature['count'] == 28391948 and curvature['excluded'] == 0, 'incomplete curvature population')
            mae, accuracy, count = join_curvature(curvature['per_item'], ids)
            require(int(count.sum()) == curvature['count'], 'curvature item denominator mismatch')
            values['curvature_mae_' + rep] = mae
            values['curvature_sign_accuracy_' + rep] = accuracy
            counts.append(count)
        require(np.array_equal(*counts), 'q/z curvature populations differ')
        values['valid_curvature'] = counts[0] > 0
        values['curvature_triple_count'] = counts[0]
        values.update(sid_diagnostics(sid, graph))
        columns.update({key + '_' + arm: value for key, value in values.items()})
        raw.update({'gates_' + arm: weights, 'attention_' + arm: attention,
                    'attention_mask_' + arm: mask, 'sid_' + arm: sid})
        summaries[arm] = {'items': ITEMS, 'experts': weights.shape[1], 'attention_shape': list(attention.shape),
                          'expert_curvatures': expert_curvatures,
                          'attention_axis': ['1hop', '2hop', 'null'], 'partition_consistency': partition,
                          'extraction_reference': evaluation['provenance']['extractions'][arm],
                          'common_fixed3_scale': SCALE, 'valid_curvature_items': int(values['valid_curvature'].sum()),
                          'mean_gate_negative': float(grouped['negative'].mean()),
                          'mean_gate_zero': float(grouped['zero'].mean()),
                          'mean_gate_positive': float(grouped['positive'].mean()),
                          'hard_q_sid_codebook_exact': True,
                          'used_l1_codes': int(len(np.unique(sid[:, 0]))),
                          'used_l2_codes': int(len(np.unique(sid[:, 1]))),
                          'used_l3_codes': int(len(np.unique(sid[:, 2])))}
        Audit().check(checkpoint, identity['stage2_checkpoint']['sha256'])
        del gate, payload, arrays
        gc.collect()
    tables = OUTPUT / 'tables'
    tables.mkdir(parents=True, exist_ok=True)
    csv_path, npz_path = tables / 'mechanism_items.csv', tables / 'mechanism_arrays.npz'
    with csv_path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['item_id'] + list(columns))
        for row, item in enumerate(ids):
            writer.writerow([int(item)] + ['' if isinstance(v[row], (float, np.floating)) and not np.isfinite(v[row])
                                          else v[row] for v in columns.values()])
    np.savez_compressed(npz_path, **raw)
    with np.load(npz_path, allow_pickle=False) as loaded:
        require(all(np.array_equal(loaded[key], value) for key, value in raw.items()), 'mechanism reload mismatch')
    with csv_path.open(newline='') as stream:
        reloaded = list(csv.DictReader(stream))
    require([int(row['item_id']) for row in reloaded] == ids.tolist(), 'CSV catalogue reload mismatch')
    output_records = {}
    for path in (csv_path, npz_path):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        output_records[path.name] = {'path': str(path), 'sha256': digest, 'bytes': path.stat().st_size}
    summary = {'schema': 'genrec.issue254.mechanism/1', 'purpose': 'evaluation', 'device': 'cpu',
               'items': ITEMS, 'arms': summaries, 'inputs_sha256': audit.files, 'outputs': output_records,
               'csv_columns': ['item_id'] + list(columns), 'reload_verified': True,
               'definitions': {'gate_entropy': 'Shannon entropy in nats over original 5/3 experts',
                               'sid_lN_occupancy': 'number of catalogue items sharing first N tokens within arm',
                               'edge_prefix': 'unweighted fraction of frozen directed neighbours sharing within-arm prefix; empty is missing',
                               'curvature': 'existing full-triple common_fixed3 per-centre metrics; not intrinsic sectional curvature'},
               'limitations': ['Posthoc association is not single-factor causal attribution.',
                               'q/z/reconstruction arrays are hash-bound references, not duplicated in this NPZ.']}
    (OUTPUT / 'MECHANISM_SUMMARY.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
