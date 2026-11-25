"""Start the one preregistered four-card Stage2 run for the Issue256 candidate.

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

from tools.issue256.r03.identity import ARM, RUN_ID, SOURCE as TREE, EVIDENCE
PYTHON = '/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9'


def main() -> int:
    from tools.issue256.r03.dispatch_evidence import sha256, verify
    sys.path.insert(0, str(TREE))
    import benchmark_entry
    credentials = verify(ROOT, benchmark_entry.source_binding('dispatch')['files'])
    registration_path = ROOT / 'results/research/issue256/ROUNDS.json'
    registration = json.loads(registration_path.read_text())
    matches = [row for row in registration['rounds'] if row['run_id'] == RUN_ID and row['arm'] == ARM]
    if len(matches) != 1 or matches[0]['contract_sha256'] != sha256(EVIDENCE / 'CONTRACT.md'):
        raise ValueError('the new hypothesis has no matching immutable preregistration')
    workspace = ROOT / 'results/stage2_RQ-VAE/_runs' / RUN_ID / ARM / 'workspace.json'
    if workspace.exists():
        raise ValueError('this candidate run is already reserved; no same-mechanism score rerun')
    credentials_path = EVIDENCE / 'DISPATCH_CREDENTIALS.json'
    credentials_path.write_text(json.dumps(credentials, indent=2, sort_keys=True) + '\n')
    run = run_workspace.freeze_run(ROOT, TREE, RUN_ID, ARM,
                                   checks_fingerprint=sha256(credentials_path))
    receipt = {'schema': 'genrec.issue256.formal_run/1', **run,
               'credentials': {'path': str(credentials_path), 'sha256': sha256(credentials_path)},
               'registration': {'path': str(registration_path), 'sha256': sha256(registration_path)}}
    (EVIDENCE / 'FORMAL_RUN.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print('run', json.dumps({key: run[key] for key in ('run_id', 'arm', 'entry', 'result')}), flush=True)
    code = launcher.launch(Path(run['entry']), Path(run['result']), python=PYTHON,
                           log_name='train.log')
    print('launcher_exit', code, flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
