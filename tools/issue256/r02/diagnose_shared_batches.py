"""Measure batch-shape/precision drift on the same actual post-epoch weights."""
import json
import os
import sys
import time
import traceback

from tools.issue256.r02.identity import SOURCE, EVIDENCE, PYTHON
from tools.research.stage2 import launcher


def child():
    import numpy as np
    import torch
    sys.path.insert(0, str(SOURCE))
    import train_rqvae as training
    from benchmark_entry import configure, source_binding
    from validation_entry import digest
    run_id, output = configure('shared_batches')
    rank = int(os.environ['RANK'])
    report = {'run_id': run_id, 'rank': rank, 'worker_pid': os.getpid(),
              'formal_training': False, 'admission_credential': False,
              'source_hashes': source_binding('shared_batches'), 'status': 'RUNNING'}
    arrays = {}
    started = time.monotonic()

    def comparison(left, right):
        difference = np.abs(left.astype(np.float64)-right)
        tolerance = 1e-5+1e-4*np.abs(right)
        return {'shape': list(left.shape), 'different': int(np.count_nonzero(left != right)),
                'max_abs': float(difference.max()) if difference.size else 0.,
                'outside_frozen_forward_tolerance': int(np.count_nonzero(difference > tolerance))}

    class Observer:
        def __call__(self, event, **fields):
            if event == 'first_step_before_backward':
                self.ids = np.unique(np.asarray(fields['item_ids'].cpu(), dtype=np.int64))
                self.aux = np.unique(np.asarray(fields['structural_sample']['ids'], dtype=np.int64))
            if event != 'diagnostic_complete':
                return
            model, catalogue = fields['model'], fields['catalogue']
            union = np.unique(np.concatenate((self.ids, self.aux)))
            indices = {'main': np.searchsorted(union, self.ids),
                       'aux': np.searchsorted(union, self.aux)}
            before = digest(model.named_parameters())
            previous = torch.get_float32_matmul_precision()
            report['policies'] = {}
            for policy in ('high', 'highest'):
                torch.set_float32_matmul_precision(policy)
                captured = {}
                phase = 'union'
                hooks = []
                def hook(name):
                    def measured(owner, inputs, result):
                        captured.setdefault(phase, {})[name] = result.detach().cpu().numpy().copy()
                    return measured
                for name, module in model.structure.attention.named_modules():
                    if name in ('key', 'value', 'score.0', 'score.1', 'score.2'):
                        hooks.append(module.register_forward_hook(hook(name)))
                try:
                    _, weights = model.structure.gates_for_items(model.store, union, catalogue.device)
                    captured['union']['tokens'] = model.structure.last_tokens.detach().cpu().numpy().copy()
                    captured['union']['weights'] = weights.detach().cpu().numpy().copy()
                    union_mask = model.structure.last_token_mask[:, :model.structure.last_tokens.shape[1]].detach().clone()
                    rows = {}
                    for label, ids in (('main', self.ids), ('aux', self.aux)):
                        phase = label
                        _, weights = model.structure.gates_for_items(model.store, ids, catalogue.device)
                        captured[label]['tokens'] = model.structure.last_tokens.detach().cpu().numpy().copy()
                        captured[label]['weights'] = weights.detach().cpu().numpy().copy()
                        rows[label] = {key: comparison(captured['union'][key][indices[label]], value)
                                       for key, value in captured[label].items()}
                    # Isolate attention arithmetic with exactly the union's saved GNN tokens.
                    union_tokens = torch.tensor(captured['union']['tokens'], device=catalogue.device)
                    mask = union_mask
                    phase = 'fixed_union'
                    union_weights = model.structure.attention(union_tokens, mask)[0]
                    fixed_union = union_weights.detach().cpu().numpy()
                    fixed = {}
                    for label in ('main', 'aux'):
                        phase = 'fixed_'+label
                        index = torch.tensor(indices[label], device=catalogue.device)
                        value = model.structure.attention(union_tokens[index], mask[index])[0]
                        fixed[label] = comparison(fixed_union[indices[label]], value.detach().cpu().numpy())
                    report['policies'][policy] = {'end_to_end': rows, 'fixed_identical_token_attention': fixed}
                    for label, values in captured.items():
                        for key, value in values.items():
                            arrays[policy+'_'+label+'_'+key.replace('.', '_')] = value
                finally:
                    for handle in hooks:
                        handle.remove()
            torch.set_float32_matmul_precision(previous)
            report['parameters_unchanged'] = digest(model.named_parameters()) == before
            report['parameters_sha256'] = before
            report['population'] = {'union': len(union), 'main': len(self.ids), 'aux': len(self.aux)}
            report['completed_epoch'] = fields['completed_epoch']

    training.DIAGNOSTIC_OBSERVER = Observer()
    try:
        training.main(diagnostic_epochs=1)
        report['status'] = 'COLLECTED_NOT_ADMISSION'
    except Exception:
        report['status'] = 'FAIL'
        report['error'] = traceback.format_exc()
        raise
    finally:
        path = output/('rank_%d_arrays.npz'%rank)
        np.savez(path, **arrays)
        report['arrays'] = str(path)
        report['wall_seconds'] = time.monotonic()-started
        (output/('rank_%d.json'%rank)).write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print('shared-batches', rank, report['status'], run_id, flush=True)


def main():
    return launcher.launch(SOURCE/'shared_batches_entry.py',
        EVIDENCE/'shared_batches_launcher'/str(os.getpid()), python=PYTHON,
        log_name='shared_batches.log')


if __name__ == '__main__':
    raise SystemExit(main())
