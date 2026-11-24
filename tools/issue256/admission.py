"""Run the machine admission for the Issue256 candidate and persist the receipt.

The two live gates are the frozen CLAUDE.md §19f checks: the real entry point
(one epoch with the exact checkpoint SID export) and the behaviour-graph
contract.  The diagnostic checks stay standalone and never block.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'tools/research/stage2/contract.py').is_file())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.issue256.identity import ARM, SOURCE as TREE, EVIDENCE as OUTPUT


def main() -> int:
    from tools.research.stage2 import checks
    sys.path.insert(0, str(TREE))
    import benchmark_entry
    before = benchmark_entry.source_binding('admission')
    report = checks.run_all(TREE)
    if before != benchmark_entry.source_binding('admission'):
        raise ValueError('candidate/controller sources changed during machine admission')
    report['source_hashes'] = before
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / 'ADMISSION.json'
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(path.relative_to(ROOT)),
                      'passed': report['passed'],
                      'checks': [{'name': check['name'], 'status': check['status'],
                                  'detail': check['detail'][:400]} for check in report['checks']]},
                     indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
