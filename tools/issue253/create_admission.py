"""Write the sole Stage2 training admission receipt from actual evidence.

The four validation rank reports, the four benchmark rank reports and the three
source-bound test reports must all come from the CURRENT bound source. Nothing
is selected by timestamp: the caller passes the exact run directories.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'stage2_RQ-VAE').is_dir())
ARM = 'TIGER_three_expert_attention_rq'
TREE = ROOT/'stage2_RQ-VAE'/ARM


def _ranks(directory, label):
    directory = Path(directory)
    paths = [directory/('rank_%d.json' % index) for index in range(4)]
    for path in paths:
        if not path.is_file():
            raise SystemExit('missing %s rank report: %s' % (label, path))
    return paths


def main():
    if len(sys.argv) != 4:
        raise SystemExit('usage: create_admission.py <validation-dir> <benchmark-dir> <contract-test-dir>')
    validation, benchmark, contracts = (_ranks(sys.argv[1], 'validation'),
                                        _ranks(sys.argv[2], 'benchmark'), Path(sys.argv[3]))
    tests = {}
    for kind in ('deterministic', 'reference', 'mutation'):
        path = contracts/(kind+'.json')
        if not path.is_file():
            raise SystemExit('missing source-bound test report: ' + str(path))
        tests[kind] = path
    from tools.issue252.training_admission import create_receipt
    descriptor = create_receipt(ROOT, TREE, validation_reports=validation,
                                benchmark_reports=benchmark, test_reports=tests)
    receipt = json.loads(Path(descriptor['path']).read_text(encoding='utf-8'))
    print(json.dumps({'receipt': descriptor,
                      'performance': receipt['performance'],
                      'validation_reports': receipt['validation_reports'],
                      'benchmark_reports': receipt['benchmark_reports'],
                      'test_reports': receipt['test_reports']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
