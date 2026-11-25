"""R06 observation: which graphlet roles move at level 2.

This is not an ablation.  It compares the frozen R05 SID with the R06 SID after
the one formal R06 run and reports, per graphlet orbit role, the fraction of
items whose hard code changed at each level.  Level 2 is the only level the
token residual targets; levels 1 and 3 are controls.  The script never trains
and never mutates an artifact.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ROLE_TABLE = ROOT / 'stage2_RQ-VAE' / 'TIGER_topology_residual_l2_rq' / 'role_table.json'


def load_sids(path: Path) -> dict[str, list[int]]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f'not a non-empty item SID mapping: {path}')
    return {str(k): list(v) for k, v in payload.items()}


def main() -> int:
    if len(sys.argv) != 4:
        print('usage: role_divergence.py <r05_item_sids.json> <r06_item_sids.json> <out.json>')
        return 2
    r05 = load_sids(Path(sys.argv[1]))
    r06 = load_sids(Path(sys.argv[2]))
    roles_payload = json.loads(ROLE_TABLE.read_text(encoding='utf-8'))
    table = np.asarray(roles_payload['table'], dtype=np.int64)
    role_names = {int(v): k for k, v in roles_payload['roles'].items()}
    role_names[-1] = 'NO_ROLE'

    ids = sorted(set(map(int, r05)) & set(map(int, r06)))
    if len(ids) != len(r05) or len(ids) != len(r06):
        raise ValueError('R05 and R06 must cover the same item identities')
    changed = defaultdict(lambda: np.zeros(3, dtype=np.int64))
    total = Counter()
    for item in ids:
        role = int(table[item]) if 0 <= item < len(table) else -1
        total[role] += 1
        for level in range(3):
            a = r05[str(item)]
            b = r06[str(item)]
            if len(a) <= level or len(b) <= level:
                raise ValueError(f'SID shorter than three levels at item {item}')
            if a[level] != b[level]:
                changed[role][level] += 1

    report = {
        'measure': 'fraction of hard codes changed from R05 to R06, by role',
        'caveat': 'observational full-run comparison; not a switched-off ablation',
        'n_items': len(ids),
        'levels': ['L1', 'L2', 'L3'],
        'overall': {
            'changed': [int(sum(changed[r][i] for r in total)) for i in range(3)],
            'fraction': [float(sum(changed[r][i] for r in total) / len(ids)) for i in range(3)],
        },
        'roles': {},
    }
    for role in sorted(total):
        report['roles'][role_names.get(role, str(role))] = {
            'role_id': role,
            'n_items': int(total[role]),
            'changed': [int(x) for x in changed[role]],
            'fraction': [float(x / total[role]) for x in changed[role]],
        }
    out = Path(sys.argv[3])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
