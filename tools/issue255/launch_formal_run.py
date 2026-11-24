"""Start the sanctioned four-card Stage2 run for the Issue255 candidate.

The run is bound to an immutable source snapshot and an independent result
directory under one run id.  It may only start after the machine admission
checks, the candidate test suites and the inclusive four-card performance
window have passed and been recorded.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'tools/research/stage2/contract.py').is_file())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.issue252 import run_workspace
from tools.research.stage2 import launcher

from tools.issue255.identity import ARM, RUN_ID, SOURCE as TREE, EVIDENCE
PYTHON = '/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9'


def main() -> int:
    from tools.issue255.dispatch_evidence import sha256, verify
    sys.path.insert(0, str(TREE))
    import benchmark_entry
    credentials = verify(ROOT, benchmark_entry.source_binding('dispatch')['files'])
    approval_path = EVIDENCE / 'OWNER_APPROVAL.json'
    approval = json.loads(approval_path.read_text())
    if approval.get('choice') != 'repair_rerun' or approval.get('run_id') != RUN_ID:
        raise ValueError('the corrected formal generation has no matching owner approval')
    workspace = ROOT / 'results/stage2_RQ-VAE/_runs' / RUN_ID / ARM / 'workspace.json'
    if workspace.exists():
        raise ValueError('the one approved corrected run is already reserved; never allocate another id')
    credentials_path = EVIDENCE / 'DISPATCH_CREDENTIALS.json'
    credentials_path.write_text(json.dumps(credentials, indent=2, sort_keys=True) + '\n')
    run = run_workspace.freeze_run(ROOT, TREE, RUN_ID, ARM,
                                   checks_fingerprint=sha256(credentials_path))
    receipt = {'schema': 'genrec.issue255.formal_run/1', **run,
               'credentials': {'path': str(credentials_path), 'sha256': sha256(credentials_path)},
               'owner_approval': {'path': str(approval_path), 'sha256': sha256(approval_path)}}
    (EVIDENCE / 'FORMAL_RUN.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print('run', json.dumps({key: run[key] for key in ('run_id', 'arm', 'entry', 'result')}), flush=True)
    code = launcher.launch(Path(run['entry']), Path(run['result']), python=PYTHON,
                           log_name='train.log')
    print('launcher_exit', code, flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
