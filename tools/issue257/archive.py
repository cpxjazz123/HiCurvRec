"""Verify (and optionally trigger) the four-piece R2 archive; never a second uploader.

The uploader itself stays `results/research/issue139/upload_stage3.py`.  This
module only *checks* that the four pieces exist locally and that the ledger entry
for the arm is bound to the same SID digest, so a round cannot be reported as
archived while its pieces or their hashes are missing.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LEDGER = REPO / 'results/research/issue139/r2_ledger.json'
UPLOADER = REPO / 'results/research/issue139/upload_stage3.py'


def ledger_entry(arm, *, ledger=LEDGER):
    payload = json.loads(Path(ledger).read_text(encoding='utf-8'))
    matches = [(number, entry) for number, entry in payload.items()
               if isinstance(entry, dict) and entry.get('arm') == str(arm)]
    if not matches:
        return None, payload
    number, entry = sorted(matches, key=lambda row: row[0])[-1]
    return {'number': number, **entry}, payload


def verify(identity, *, ledger=LEDGER):
    arm = identity.get('arm')
    sid_sha = identity['sid']['sha256']
    entry, _ = ledger_entry(arm, ledger=ledger)
    checks = []
    if entry is None:
        checks.append({'check': 'ledger_entry', 'status': 'FAIL', 'reason': 'no entry for this arm'})
    else:
        bound = str(entry.get('key', '')).split('/')[-1]
        checks.append({'check': 'ledger_entry', 'status': 'PASS' if bound == sid_sha[:12] else 'FAIL',
                       'number': entry['number'], 'key': entry.get('key'),
                       'sid_prefix': sid_sha[:12]})
        checks.append({'check': 'ledger_files', 'status': 'PASS' if int(entry.get('files') and len(entry.get('files', {})) or 0) > 0 else 'FAIL',
                       'files': len(entry.get('files', {})),
                       'recall_keys': len(entry.get('recall_keys', []))})
    pieces = {'sid': identity.get('sid'),
              'checkpoint': identity.get('stage2_checkpoint'),
              'test_final': identity.get('test_final')}
    for name, item in pieces.items():
        if not item or not Path(item['path']).is_file():
            checks.append({'check': 'piece:' + name, 'status': 'FAIL', 'reason': 'missing'})
        else:
            checks.append({'check': 'piece:' + name, 'status': 'PASS', 'path': item['path']})
    source = None
    binding = identity.get('archive_binding')
    if binding and Path(binding['path']).is_file():
        payload = json.loads(Path(binding['path']).read_text(encoding='utf-8'))
        source = payload.get('source_tree')
    checks.append({'check': 'piece:source', 'status': 'PASS' if source and Path(source).is_dir() else 'FAIL',
                   'path': source})
    failed = [row for row in checks if row['status'] == 'FAIL']
    return {'schema': 'genrec.pipeline.archive_check/1', 'arm': arm,
            'checks': checks, 'status': 'PASS' if not failed else 'FAIL', 'failed': failed}


def upload(arm):
    """Invoke the existing uploader (single archive implementation)."""
    done = subprocess.run([sys.executable, str(UPLOADER), str(arm)],
                          cwd=str(REPO), capture_output=True, text=True)
    return {'exit_code': done.returncode, 'stdout': done.stdout[-4000:], 'stderr': done.stderr[-4000:]}
