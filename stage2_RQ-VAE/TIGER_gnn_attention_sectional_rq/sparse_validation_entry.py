"""Four-rank CUDA resource regression only; never train or export a checkpoint."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
import unittest

import torch
import benchmark_entry as common


def main():
    run_id, output = common.configure('sparse_validation')
    rank, local_rank = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    report = {'run_id': run_id, 'rank': rank, 'local_rank': local_rank,
              'worker_pid': os.getpid(), 'controller_pid': os.getppid(),
              'formal_training': False, 'diagnostic_epochs': 0,
              'source_hashes': common.source_binding('sparse_validation'), 'status': 'running',
              'world_size': 4, 'port': 50202,
              'test': 'test_production_backward_avoids_native_csr_transpose_conversion'}
    start = time.monotonic()
    try:
        torch.cuda.set_device(local_rank)
        report['device'] = str(torch.device('cuda', local_rank))
        path = Path(__file__).resolve().parent / 'tests/test_sparse_adjoint.py'
        spec = importlib.util.spec_from_file_location('sparse_adjoint252_validation', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.SparseAdjointTests.production_device = torch.device('cuda', local_rank)
        suite = unittest.TestSuite([module.SparseAdjointTests(report['test'])])
        result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
        torch.cuda.synchronize(local_rank)
        report.update(tests_run=result.testsRun,
                      failures=[{'test': str(test), 'traceback': trace} for test, trace in result.failures],
                      errors=[{'test': str(test), 'traceback': trace} for test, trace in result.errors],
                      skipped=[{'test': str(test), 'reason': reason} for test, reason in result.skipped])
        report['status'] = 'pass' if result.wasSuccessful() and result.testsRun == 1 and not result.skipped else 'fail'
        return 0 if report['status'] == 'pass' else 1
    except Exception:
        report['status'] = 'error'
        report['error'] = traceback.format_exc()
        raise
    finally:
        report['wall_seconds'] = time.monotonic() - start
        (output / ('rank_%d.json' % rank)).write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    raise SystemExit(main())
