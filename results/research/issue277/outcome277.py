"""Issue #277 — record the prospective outcome for a frozen prediction.

The prediction file (PROSPECTIVE.json) is immutable; this script only reads it and writes
PROSPECTIVE_OUTCOME.json with the prediction, the actual Stage3 label, the timing evidence and the
H11 decision. Usage:

    python3 results/research/issue277/outcome277.py <arm> [<arm> ...]
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

P = Path('results/research/issue277')


def sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def latest_test_final(arm: str) -> tuple[Path | None, dict | None]:
    hits = sorted(glob.glob(f'results/stage3_T5Train/{arm}/logs/*/*/test_final.json'),
                  key=os.path.getmtime)
    if not hits:
        return None, None
    f = Path(hits[-1])
    return f, json.loads(f.read_text())


def main(arms: list[str]) -> None:
    frozen = json.loads((P / 'PROSPECTIVE.json').read_text())
    frozen_sha = sha_file(P / 'PROSPECTIVE.json')
    by_sid = {r['sid_sha256']: r for r in frozen['rows']}
    commit_time = subprocess.run(['git', 'show', '-s', '--format=%cI', '03c8e1df'],
                                 capture_output=True, text=True).stdout.strip()
    out = {'schema': 'genrec.issue277.prospective_outcome/1',
           'frozen_predictions_sha256': frozen_sha,
           'frozen_commit': '03c8e1df', 'frozen_commit_time': commit_time,
           'checked_utc': datetime.now(timezone.utc).isoformat(),
           'rows': []}
    for arm in arms:
        f, rec = latest_test_final(arm)
        if f is None:
            out['rows'].append({'arm': arm, 'status': 'no_test_final'})
            continue
        sid = rec.get('sid_sha256')
        pred = by_sid.get(sid)
        mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc).isoformat()
        after = datetime.fromisoformat(mtime) > datetime.fromisoformat(commit_time)
        row = {'arm': arm, 'test_final': str(f), 'test_final_sha256': sha_file(f),
               'test_final_mtime_utc': mtime, 'sid_sha256': sid,
               'recall@10': rec.get('test_recall@10'),
               'label_after_freeze': after}
        if pred is None:
            row['status'] = 'sid_not_in_frozen_predictions'
        else:
            row.update({'status': 'matched',
                        'predicted_recall@10': pred['predicted_recall@10'],
                        'predicted_above_0.06': pred['predicted_recall@10'] > 0.06,
                        'actual_above_0.06': (rec.get('test_recall@10') or 0) > 0.06,
                        'abs_error': abs(pred['predicted_recall@10'] - (rec.get('test_recall@10') or 0)),
                        'signed_error': pred['predicted_recall@10'] - (rec.get('test_recall@10') or 0)})
        out['rows'].append(row)
    (P / 'PROSPECTIVE_OUTCOME.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main(sys.argv[1:])
