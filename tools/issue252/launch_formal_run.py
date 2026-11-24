"""Start the sanctioned four-card Stage2 run for the Issue252 candidate.

The run is bound to the immutable source snapshot and the independent result
directory of one run id, and it may only start after the machine admission
checks, the four test suites and the inclusive performance window have passed
and been written into the training receipt.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'tools/research/stage2/contract.py').is_file())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.issue252 import run_workspace, training_admission
from tools.research.stage2 import launcher

ARM = 'TIGER_gnn_attention_sectional_rq'
TREE = ROOT / 'stage2_RQ-VAE' / ARM
PYTHON = '/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9'


def latest(pattern):
    candidates = sorted((ROOT / 'results/research/issue252').glob(pattern),
                         key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise SystemExit('missing evidence: ' + pattern)
    return candidates[-1]


def main():
    tests = latest('contract_tests/*')
    validation = latest('validation/*')
    benchmark = latest('benchmark/issue252_benchmark_*')
    descriptor = training_admission.create_receipt(
        ROOT, TREE,
        validation_reports=sorted(validation.glob('rank_*.json')),
        benchmark_reports=sorted(benchmark.glob('rank_*.json')),
        test_reports={kind: tests / (kind + '.json')
                      for kind in ('deterministic', 'reference', 'mutation')})
    print('receipt', json.dumps(descriptor))
    run_id = 'issue252_formal'
    suffix = 0
    while True:
        workspace = ROOT / 'results/stage2_RQ-VAE/_runs' / run_id / ARM / 'workspace.json'
        if not workspace.exists():
            break
        suffix += 1
        run_id = 'issue252_formal_r%d' % suffix
    run = run_workspace.freeze_run(ROOT, TREE, run_id, ARM)
    print('run', json.dumps({key: run[key] for key in ('run_id', 'arm', 'entry', 'result')}))
    code = launcher.launch(Path(run['entry']), Path(run['result']), python=PYTHON,
                           log_name='train.log')
    print('launcher_exit', code)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
