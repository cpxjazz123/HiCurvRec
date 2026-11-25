"""Discriminate static-cache, affine and reduction drift; never admission/training."""
import hashlib
import json
import os
import sys
import time
import traceback

from tools.issue256.r03.identity import ROOT, SOURCE, EVIDENCE, PYTHON
from tools.research.stage2 import launcher


def child():
    import numpy as np
    import torch
    sys.path.insert(0, str(SOURCE))
    import train_rqvae as training
    import model.structure as structure
    from tools.graph_infra.ego_gnn import EgoStore
    from benchmark_entry import configure, source_binding
    run_id, output = configure('static_consumers')
    rank = int(os.environ['RANK'])
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True, warn_only=True)
    report = {'run_id': run_id, 'rank': rank, 'worker_pid': os.getpid(),
              'formal_training': False, 'admission_credential': False,
              'source_hashes': source_binding('static_consumers'),
              'dtype': 'float32', 'status': 'RUNNING'}
    started = time.monotonic()

    def compare(left, right):
        if left.shape != right.shape or left.dtype != right.dtype:
            raise AssertionError('component shapes/dtypes must match')
        a, b = left.reshape(-1), right.reshape(-1)
        mismatch, maximum = 0, 0.0
        hashes = [hashlib.sha256(), hashlib.sha256()]
        for begin in range(0, a.numel(), 262144):
            x = a[begin:begin + 262144].detach().cpu().numpy()
            y = b[begin:begin + 262144].detach().cpu().numpy()
            hashes[0].update(x.tobytes())
            hashes[1].update(y.tobytes())
            mismatch += int(np.count_nonzero(x != y))
            if x.size and x.dtype != np.bool_:
                maximum = max(maximum, float(np.max(np.abs(x.astype(np.float64) - y))))
        return {'shape': list(left.shape), 'dtype': str(left.dtype),
                'different': mismatch, 'max_abs': maximum,
                'left_sha256': hashes[0].hexdigest(), 'right_sha256': hashes[1].hexdigest()}

    originals = {}
    trace = []
    phase = 'left'
    cursor = 0
    stages = []

    def intercept(name, function):
        def wrapper(*args, **kwargs):
            nonlocal cursor
            result = function(*args, **kwargs)
            values = result if isinstance(result, tuple) else (result,)
            if phase == 'left':
                trace.append((name, tuple(v.detach().cpu().clone() for v in values)))
            else:
                expected_name, expected = trace[cursor]
                if expected_name != name:
                    raise AssertionError('component execution order changed')
                stages.append({'phase': phase, 'call': cursor, 'stage': name,
                               'outputs': [compare(a, b) for a, b in zip(expected, values)]})
                cursor += 1
            return result
        return wrapper

    try:
        store = EgoStore(ROOT, ROOT / 'assets')
        left = structure.SubgraphTokens().to(device).train()
        right = structure.SubgraphTokens().to(device).train()
        right.load_state_dict(left.state_dict())
        with torch.no_grad():
            ll, lf = left._cached_store(store)
            rl, rf = right._cached_store(store)
            report['features'] = compare(lf, rf)
            lm = left._feature_cache[store]['devices'][str(device)]['means'][lf.dtype]
            rm = right._feature_cache[store]['devices'][str(device)]['means'][rf.dtype]
            report['static_means'] = []
            for index, (a, b) in enumerate(zip(lm, rm)):
                report['static_means'].append({'layout': structure.TOKEN_TYPES[index],
                    **{name: compare(getattr(a, name), getattr(b, name)) for name in
                       ('incoming', 'outgoing', 'in_nonempty', 'out_nonempty')}})
            report['first_affine_parameters'] = [compare(a, b) for a, b in
                                                 zip(left._first_affine(), right._first_affine())]
            compiled = structure._compiled_affine_layer()
            originals['_compiled_affine_layer'] = structure._compiled_affine_layer
            traced_compiled = intercept('first_affine', compiled)
            structure._compiled_affine_layer = lambda: traced_compiled
            for name in ('directional_means', '_second_affine', '_pool_readout'):
                originals[name] = getattr(structure, name)
                setattr(structure, name, intercept(name, originals[name]))
            ids = np.arange(256, dtype=np.int64)
            first = left.tokens_for_items(store, ids, device)
            phase, cursor = 'right', 0
            second = right.tokens_for_items(store, ids, device)
            if cursor != len(trace):
                raise AssertionError('fresh consumer omitted component calls')
            phase, cursor = 'left_repeat', 0
            repeat = left.tokens_for_items(store, ids, device)
            if cursor != len(trace):
                raise AssertionError('repeat consumer omitted component calls')
            report['token_outputs'] = {'fresh': compare(first[0], second[0]),
                                       'same_consumer_repeat': compare(first[0], repeat[0]),
                                       'masks': compare(first[1], second[1])}
            report['stages'] = stages
            report['item_ids'] = {'start': 0, 'stop': 256, 'all_four_complete_ego_layouts': True}
        report['status'] = 'COLLECTED_NOT_ADMISSION'
    except Exception:
        report['status'] = 'FAIL'
        report['error'] = traceback.format_exc()
        raise
    finally:
        for name, original in originals.items():
            setattr(structure, name, original)
        report['wall_seconds'] = time.monotonic() - started
        (output / ('rank_%d.json' % rank)).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print('static-consumers', rank, report['status'], run_id, flush=True)


def main():
    return launcher.launch(SOURCE / 'static_consumers_entry.py',
        EVIDENCE / 'static_consumers_launcher' / str(os.getpid()), python=PYTHON,
        log_name='static_consumers.log')


if __name__ == '__main__':
    raise SystemExit(main())
