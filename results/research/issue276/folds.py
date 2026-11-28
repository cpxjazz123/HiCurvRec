"""Issue #276 — freeze the 5-fold SID-group assignment (recall-band stratified, seed 42).

Rule frozen in ANALYSIS_CONTRACT.md: groups sorted inside recall bands `<0.045`, `[0.045,0.06]`,
`>0.06`; each band shuffled with seed 42 and assigned round-robin to folds 0..4. Writes FOLDS.json
with the table hash so the assignment cannot drift after the table is rebuilt.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

P = Path('results/research/issue276')
BANDS = [('<0.045', lambda v: v < 0.045),
         ('0.045-0.06', lambda v: 0.045 <= v <= 0.06),
         ('>0.06', lambda v: v > 0.06)]


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> None:
    table = json.load(open(P / 'SID_TABLE.json'))
    groups = []
    for g in table['groups']:
        lab = [r['label_recall@10'] for r in g['rows'] if isinstance(r['label_recall@10'], (int, float))]
        if not lab:
            continue
        groups.append({'group': g['group'], 'label': sum(lab) / len(lab), 'n_rows': len(g['rows'])})
    rng = random.Random(42)
    assignment = {}
    band_counts = {name: 0 for name, _ in BANDS}
    per_fold = {i: {name: 0 for name, _ in BANDS} for i in range(5)}
    for name, test in BANDS:
        members = [g for g in groups if test(g['label'])]
        band_counts[name] = len(members)
        rng.shuffle(members)
        for i, g in enumerate(members):
            f = i % 5
            assignment[g['group']] = f
            per_fold[f][name] += 1
    out = {
        'schema': 'genrec.issue276.folds/1',
        'seed': 42,
        'rule': 'bands <0.045 / [0.045,0.06] / >0.06, shuffled per band with seed 42, round-robin',
        'table_sha256': sha(P / 'SID_TABLE.json'),
        'bands': band_counts,
        'per_fold_bands': per_fold,
        'assignment': assignment,
    }
    (P / 'FOLDS.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k: out[k] for k in ('bands', 'per_fold_bands', 'table_sha256')}, indent=1))


if __name__ == '__main__':
    main()
