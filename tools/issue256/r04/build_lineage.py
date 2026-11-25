"""Write the R03 lineage record: every executed file against its parents.

The record names the frozen baseline, the archived R03 tree the candidate was
copied from, and the one mechanism change the candidate makes, so a reader can
see exactly which bytes are inherited, which are new and which are the
candidate's own change.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'stage2_RQ-VAE').is_dir())
TREE = ROOT / 'stage2_RQ-VAE' / 'TIGER_role_conditioned_l1_rq'
PARENT = ROOT / 'stage2_RQ-VAE' / '_archive' / 'TIGER_periodic_budget_prefix_rq'
BASELINE = ROOT / 'stage2_RQ-VAE' / 'baseline' / 'TIGER_RQ-VAE'
OUTPUT = ROOT / 'results/research/issue256/rounds/r04/LINEAGE.json'


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(root: Path) -> dict[str, str]:
    files = {}
    for path in sorted(root.rglob('*')):
        if (not path.is_file() or path.suffix not in ('.py', '.json')
                or '__pycache__' in path.parts or 'logs' in path.parts):
            continue
        files[str(path.relative_to(root))] = sha256(path)
    return files


def main() -> int:
    tree, parent, baseline = inventory(TREE), inventory(PARENT), inventory(BASELINE)
    changed = sorted(name for name in tree
                     if parent.get(name) != tree[name])
    added = sorted(name for name in tree if name not in parent)
    removed = sorted(name for name in parent if name not in tree)
    role_of = {
        'model/mechanism.py': 'changed: the role mixture drives the level-3 cost; levels 1 and 2 keep the item gate',
        'tests/test_role_mixture.py': 'changed: the R04 acceptance tests (the role drives the refinement level)',
        'validation_entry.py': 'changed: the switch and gate-intervention wiring check follows level 3',
        'model/layer_gate.py': 'changed: documentation of the two learned levels',
        'model/role_mix.py': 'changed: documentation of the level the table drives',
        'role_table.json': 'unchanged: the same frozen training-only graphlet orbit role',
        'train_rqvae.py': 'changed: the arm name only',
        'benchmark_entry.py': 'changed: the controller namespace and the round paths',
    }
    entries = []
    rewrites = (('tools.issue256.r03', 'tools.issue256.r04'),
                ('tools/issue256/r03', 'tools/issue256/r04'),
                ('results/research/issue256/rounds/r03', 'results/research/issue256/rounds/r04'),
                ('issue256_r03_', 'issue256_r04_'),
                ('TIGER_role_conditioned_l1_rq', 'TIGER_role_conditioned_l3_rq'))
    for name in sorted(tree):
        entry = {'path': name, 'sha256': tree[name],
                 'baseline_sha256': baseline.get(name),
                 'inherited_sha256': parent.get(name),
                 'role': role_of.get(name)}
        if entry['role'] is None:
            if name not in parent:
                entry['role'] = 'new file of the R04 candidate'
            else:
                inherited = (PARENT / name).read_text(encoding='utf-8')
                rewritten = inherited
                for old, new in rewrites:
                    rewritten = rewritten.replace(old, new)
                if rewritten == (TREE / name).read_text(encoding='utf-8'):
                    entry['role'] = ('changed: the controller namespace, the round paths and '
                                     'the arm name only; the mechanism bytes are inherited')
                elif tree[name] == parent[name]:
                    entry['role'] = 'inherited mechanism, diagnostics or frozen configuration'
                else:
                    entry['role'] = 'changed: the R03 mechanism bytes differ from the parent'
        entries.append(entry)
    record = {
        'schema': 'genrec.issue256.r03.lineage/1',
        'round_id': 'r04',
        'candidate': 'stage2_RQ-VAE/TIGER_role_conditioned_l1_rq',
        'copy_source': 'stage2_RQ-VAE/_archive/TIGER_periodic_budget_prefix_rq',
        'copy_source_reason': ('the R04 mechanism is the R03 mechanism family with one '
                                'change (the role table drives the level-3 cost); the copy is '
                                'the archived R03 tree, which itself '
                                'carries the #255 three-curvature lineage'),
        'frozen_tiger_source': 'stage2_RQ-VAE/baseline/TIGER_RQ-VAE',
        'baseline_source_hashes': baseline,
        'files': entries,
        'main_change': ('the level-1 curvature mixture comes from the item graphlet orbit '
                         'role instead of the item own free gate; levels 2 and 3 keep the '
                         'learned mixture, the Sinkhorn balanced assignment, the assignment '
                         'floor, the losses and the recipe are unchanged'),
        'new_assets': [{'path': 'stage2_RQ-VAE/TIGER_role_conditioned_l1_rq/role_table.json',
                        'sha256': tree['role_table.json'],
                        'derivation': 'tools/issue256/r03/derive_role_table.py',
                        'source': {'path': 'results/research/graphlet_roles/instances.npz',
                                   'sha256': json.loads((TREE / 'role_table.json').read_text())['source']['sha256']},
                        'split_rule': 'training instances only'}],
        'changed_files': changed,
        'added_files': added,
        'removed_files': removed,
        'trained_weight_initialization': False,
        'recipe': {'world_size': 4, 'global_batch': 4096, 'seed': 42, 'epochs': 3000,
                   'stage3_epochs': 150, 'beam': 20},
        'engineering_dependency_repairs': [
            {'path': 'results/research/issue256/rounds/r03/LINEAGE.json',
             'sha256': sha256(ROOT / 'results/research/issue256/rounds/r03/LINEAGE.json')},
        ],
        'budget_scope_disclosure': ('the role mixture is 10x3 parameters outside the calibrated '
                                     'module groups; the module budget machinery and its '
                                     'calibration are unchanged'),
        'contract_amendment': ('rounds/r04/CONTRACT.md section 2: the role table moves from the '
                               'level-1 cost to the level-3 cost because the R03 export measured '
                               'the item-unique information in H(L3|L1L2) = 2.0193 bits over 23247 '
                               'distinct triples; no other mechanism component changes'),
    }
    OUTPUT.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'files': len(entries), 'changed': changed, 'added': added,
                      'removed': removed}, indent=2))
    print('written', OUTPUT)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
