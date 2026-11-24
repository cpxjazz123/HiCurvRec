"""CPU-only NumPy reconciliation of saved batch/precision output arrays."""
from pathlib import Path
import hashlib
import json
import math

import numpy as np
import pandas as pd
import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'assets/manifest.json').is_file())
EVIDENCE = Path(__file__).resolve().parent
TRACE = EVIDENCE/'shared_batches/issue256_r02_shared_batches_2601733_586369709'
SAMPLES = EVIDENCE/'validation/issue256_r02_validation_2567507_586345783'
TRAIN = ROOT/'results/stage0_build_parquet/train.parquet'


def binding(path):
    return {'path': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    train = np.unique(pd.read_parquet(TRAIN, columns=['target'])['target'].to_numpy(dtype=np.int64))
    # Independent deterministic DistributedSampler(epoch0, seed42, world4) indices.
    order = torch.randperm(len(train), generator=torch.Generator(device='cpu').manual_seed(42)).numpy()
    total = 4*math.ceil(len(train)/4)
    order = np.concatenate((order, order[:total-len(order)]))
    rows = []
    for rank in range(4):
        report_path = TRACE/('rank_%d.json'%rank)
        report = json.loads(report_path.read_text())
        sample_path = SAMPLES/('rank_%d.json'%rank)
        sample = json.loads(sample_path.read_text())['first_structural_sample']
        populations = {'main': np.sort(train[order[rank:total:4][:1024]]),
                       'aux': np.unique(np.asarray(sample['ids'], dtype=np.int64))}
        union = np.union1d(populations['main'], populations['aux'])
        assert report['population'] == dict(union=len(union), **{k: len(v) for k,v in populations.items()})
        arrays_path = TRACE/('rank_%d_arrays.npz'%rank)
        comparisons = 0
        with np.load(arrays_path, allow_pickle=False) as arrays:
            for policy, values in report['policies'].items():
                for label, population in populations.items():
                    index = np.searchsorted(union, population)
                    for key, expected in values['end_to_end'][label].items():
                        suffix = key.replace('.', '_')
                        left = arrays[policy+'_union_'+suffix][index]
                        right = arrays[policy+'_'+label+'_'+suffix]
                        difference = np.abs(left.astype(np.float64)-right)
                        observed = {'shape': list(left.shape),
                            'different': int(np.count_nonzero(left != right)),
                            'max_abs': float(difference.max()) if difference.size else 0.,
                            'outside_frozen_forward_tolerance': int(np.count_nonzero(difference > 1e-5+1e-4*np.abs(right)))}
                        assert observed == expected, (rank, policy, label, key, observed, expected)
                        comparisons += 1
        rows.append({'rank': rank, 'report': binding(report_path), 'arrays': binding(arrays_path),
                     'structural_sample_source': binding(sample_path), 'exact_comparisons': comparisons})
    result = {'schema': 'genrec.issue256.batch_trace_reconciliation/1',
              'passed': True, 'cpu_only': True, 'production_module_imports': False,
              'train': binding(TRAIN), 'ranks': rows,
              'total_exact_comparisons': sum(r['exact_comparisons'] for r in rows),
              'reducer': binding(Path(__file__).resolve()),
              'qualification': 'All end-to-end saved stage statistics independently reproduced; separate fixed-token attention returned weights were not persisted and are not claimed independently recomputed.'}
    (EVIDENCE/'BATCH_TRACE_RECONCILIATION.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'passed': True, 'exact_comparisons': result['total_exact_comparisons'], 'cpu_only': True}))


if __name__ == '__main__':
    main()
