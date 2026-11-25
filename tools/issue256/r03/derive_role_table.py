"""Derive one graphlet-orbit role per item from the frozen training instances.

The role vocabulary is the one the role arms used: every (graphlet type, orbit)
pair is its own label, so a path leaf and a claw centre are different roles
while the four cycle nodes share one.  Only the training instances of the frozen
asset are read, so the label is a training-only behaviour signal.

The result is written into the candidate tree as a compact run-length encoded
JSON file, because the tree's JSON files are part of the hashed source
inventory; a per-item list would add 24k lines to every diff.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'stage2_RQ-VAE').is_dir())
INSTANCES = ROOT / 'results' / 'research' / 'graphlet_roles' / 'instances.npz'
OUTPUT = ROOT / 'stage2_RQ-VAE' / 'TIGER_role_conditioned_l1_rq' / 'role_table.json'

TYPES = ('K3', 'P3', 'C4', 'P4', 'K1_3')
ROLE_NAMES = {
    'K3': ('all',),
    'P3': ('leaf', 'centre'),
    'C4': ('all',),
    'P4': ('end', 'internal'),
    'K1_3': ('leaf', 'centre'),
}
ROLE_INDEX: dict[tuple[str, int], int] = {}
for _type in TYPES:
    for _orbit in range(len(ROLE_NAMES[_type])):
        ROLE_INDEX[(_type, _orbit)] = len(ROLE_INDEX)
N_ROLES = len(ROLE_INDEX)
ITEM_COUNT = 24587


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    payload = np.load(INSTANCES, allow_pickle=False)
    role_counts: list[Counter[int]] = [Counter() for _ in range(ITEM_COUNT)]
    occurrences: Counter[int] = Counter()
    for name in TYPES:
        nodes = payload[f'{name}_nodes']
        roles = payload[f'{name}_roles']
        split = payload[f'{name}_split']
        keep = split == 0
        nodes, roles = nodes[keep], roles[keep]
        for row_nodes, row_roles in zip(nodes, roles):
            for node, orbit in zip(row_nodes, row_roles):
                role = ROLE_INDEX[(name, int(orbit))]
                role_counts[int(node)][role] += 1
                occurrences[role] += 1
    table = np.full(ITEM_COUNT, -1, dtype=np.int64)
    for item, counter in enumerate(role_counts):
        if counter:
            # The most frequent (type, orbit) the item takes in the training
            # instances; ties break towards the smaller role index.
            table[item] = min(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    covered = int((table >= 0).sum())
    report = {
        'schema': 'genrec.issue256.r03.role_table/1',
        'item_count': ITEM_COUNT,
        'roles': {f'{name}:{orbit}': ROLE_INDEX[(name, orbit)]
                  for name in TYPES for orbit in range(len(ROLE_NAMES[name]))},
        'role_count': N_ROLES,
        'source': {'path': str(INSTANCES), 'sha256': sha256(INSTANCES)},
        'split_rule': 'instance split flag == 0 (training instances only)',
        'coverage': covered,
        'coverage_fraction': covered / ITEM_COUNT,
        'role_distribution': {str(role): int((table == role).sum()) for role in range(N_ROLES)},
        'instance_occurrences': {str(role): int(value) for role, value in sorted(occurrences.items())},
        'derivation': 'tools/issue256/r03/derive_role_table.py',
    }
    # The per-item labels are one compact line, so the hashed source file stays
    # small enough for every diff.
    body = json.dumps(report, indent=2)
    body = body[:-2].rstrip() + ',\n  "table": ' + json.dumps(table.tolist(), separators=(',', ':')) + '\n}\n'
    OUTPUT.write_text(body)
    print(json.dumps({key: report[key] for key in
                      ('item_count', 'role_count', 'coverage', 'coverage_fraction',
                       'role_distribution', 'source')}, indent=2))
    print('written', OUTPUT)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
