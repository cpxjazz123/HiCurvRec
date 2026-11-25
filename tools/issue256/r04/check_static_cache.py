"""Run only static-cache regressions beneath the standard four-card launcher."""
import json
import os
import sys
import time

from tools.issue256.r04.identity import ROOT, SOURCE, EVIDENCE, PYTHON
from tools.research.stage2 import launcher


def child():
    import torch
    import pytest
    torch.cuda.set_device(int(os.environ['RANK']))
    torch.set_num_threads(1)
    sys.path.insert(0, str(SOURCE))
    from benchmark_entry import configure, source_binding
    run_id, output = configure('static_cache_check')
    rank = int(os.environ['RANK'])
    args = [str(SOURCE/'tests/test_static_reproducibility.py'), '-q',
            '--basetemp='+str(output/('pytest_rank%d'%rank))]
    failures = []
    class Results:
        def pytest_runtest_logreport(self, report):
            if report.failed:
                failures.append({'nodeid': report.nodeid, 'phase': report.when,
                                 'failure': str(report.longrepr)})
    started = time.monotonic()
    binding = source_binding('static_cache_check')
    code = int(pytest.main(args, plugins=[Results()]))
    report = {'run_id': run_id, 'rank': rank, 'worker_pid': os.getpid(),
              'exit_code': code, 'failures': failures, 'pytest_args': args,
              'wall_seconds': time.monotonic()-started, 'source_hashes': binding,
              'formal_training': False, 'complete_admission': False}
    (output/('rank_%d.json'%rank)).write_text(json.dumps(report, indent=2)+'\n')
    return code


def main():
    return launcher.launch(SOURCE/'static_cache_check_entry.py',
        EVIDENCE/'static_cache_check_launcher'/str(os.getpid()), python=PYTHON,
        log_name='static_cache_check.log')


if __name__ == '__main__':
    raise SystemExit(main())
