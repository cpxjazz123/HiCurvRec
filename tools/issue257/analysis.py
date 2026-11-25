"""Run-identity-parameterised scientific analysis; no issue number or arm is hardcoded.

Everything is computed from the persisted artefacts of one run: the exported SID,
the shared layer-metrics record (when present), the frozen role table of the
executed source tree (when present) and the Stage3 final record with its
predictions.  Missing inputs are reported as null, never filled in from a
different run.
"""
from __future__ import annotations

import collections
import json
import math
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents
            if (p / 'tools' / 'research' / 'stage2' / 'contract.py').is_file())


def _entropy(counts, total):
    return -sum(k / total * math.log2(k / total) for k in counts.values())


def sid_chain(rows):
    """Exact conditional entropies of the three semantic codes."""
    three = [tuple(int(v) for v in row[:3]) for row in rows]
    total = len(three)
    h1 = _entropy(collections.Counter(r[0] for r in three), total)
    h12 = _entropy(collections.Counter(r[:2] for r in three), total)
    h123 = _entropy(collections.Counter(three), total)
    # Full precision: the replay compares these against stored records whose
    # values are not rounded; the report rounds for display only.
    return {'items': total,
            'H_L1': h1,
            'H_L2_given_L1': h12 - h1,
            'H_L3_given_L1L2': h123 - h12,
            'H_triple': h123,
            'distinct_L1': len({r[0] for r in three}),
            'distinct_L2': len({r[:2] for r in three}),
            'distinct_triples': len(set(three)),
            'items_per_L2': total / len({r[:2] for r in three}),
            'items_per_triple': total / len(set(three)),
            'collision': 1.0 - len(set(three)) / total}


def role_conditioning(rows, table):
    """How the frozen role is distributed inside the prefixes (null when absent)."""
    if table is None:
        return None
    three = [tuple(int(v) for v in row[:3]) for row in rows]
    if len(table) != len(three):
        return {'available': False, 'reason': 'role table length differs from the SID'}
    prefixes = collections.defaultdict(list)
    cells = collections.defaultdict(list)
    for index, row in enumerate(three):
        prefixes[row[:2]].append(index)
        cells[(row[:2], int(table[index]))].append(row[2])
    def stats(values):
        values = sorted(values)
        n = len(values)
        return {'n': n, 'mean': round(sum(values) / n, 4), 'median': values[n // 2],
                'p90': values[int(n * 0.9)], 'max': values[-1]}
    return {'available': True,
            'prefixes': len(prefixes),
            'roles_per_prefix': stats([len({int(table[i]) for i in members}) for members in prefixes.values()]),
            'items_per_prefix': stats([len(members) for members in prefixes.values()]),
            'prefix_role_cells': len(cells),
            'items_per_cell': stats([len(members) for members in cells.values()]),
            'distinct_L3_per_cell': stats([len(set(members)) for members in cells.values()])}


def _role_table(source_tree):
    if not source_tree:
        return None
    path = Path(source_tree) / 'role_table.json'
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding='utf-8'))
    table = payload.get('table')
    return table if isinstance(table, list) else None


def layer_record(identity, arm, repo=REPO):
    digest = identity['sid']['sha256']
    path = Path(repo) / 'results' / 'stage2_RQ-VAE' / str(arm) / 'stage2_metrics' / (digest + '.json')
    if not path.is_file():
        return None, path
    payload = json.loads(path.read_text(encoding='utf-8'))
    return payload, path


def reduce_predictions(paths, expected_users):
    """Independent per-user top-10 reduction of the persisted Stage3 predictions."""
    shards = []
    for path in paths:
        rows = []
        for line in Path(path).read_text(encoding='utf-8').splitlines():
            if line.strip():
                rows.append(json.loads(line))
        shards.append(rows)
    users = [row for shard in shards for row in shard]
    if len(users) != int(expected_users):
        raise ValueError('prediction rows %d differ from expected %d' % (len(users), expected_users))
    hit5 = hit10 = 0
    ndcg5 = ndcg10 = 0.0
    for row in users:
        target = row.get('target_item')
        if target is None:
            raise ValueError('prediction row has no target_item')
        rank = row.get('target_rank')
        if rank is None:
            ranked = [entry['item'] if isinstance(entry, dict) else int(entry)
                      for entry in row.get('topk', [])]
            position = ranked.index(target) if target in ranked else None
        else:
            position = int(rank)
        if position is not None and position < 5:
            hit5 += 1
            ndcg5 += 1.0 / math.log2(position + 2)
        if position is not None and position < 10:
            hit10 += 1
            ndcg10 += 1.0 / math.log2(position + 2)
    total = len(users)
    return {'n_eval': total,
            'test_recall@5': hit5 / total, 'test_recall@10': hit10 / total,
            'test_ndcg@5': ndcg5 / total, 'test_ndcg@10': ndcg10 / total}


def analyse(identity, *, repo=REPO, role_table=None):
    sid_path = identity['sid']['path']
    rows = [row for row in json.loads(Path(sid_path).read_text(encoding='utf-8')).values()]
    chain = sid_chain(rows)
    record, record_path = layer_record(identity, identity.get('arm'), repo=repo)
    table = role_table if role_table is not None else _role_table(_source_tree(identity))
    out = {'schema': 'genrec.pipeline.analysis/1',
           'identity': identity,
           'sid_chain': chain,
           'role_conditioning': role_conditioning(rows, table),
           'layer_record': ({'path': str(record_path), 'catalogue': record.get('catalogue'),
                             'validation': _selected(record.get('validation'))}
                            if record else None),
           'stage3': identity.get('metrics')}
    return out


def _selected(validation):
    if not isinstance(validation, dict):
        return None
    keep = ('rows', 'l1_positive_hit_rate', 'l1_negative_hit_probability',
            'l1_hard_er', 'conditional_g3', 'conditional_log_enrichment_bits')
    return {key: validation.get(key) for key in keep}


def _source_tree(identity):
    binding = identity.get('archive_binding')
    if not binding:
        return None
    payload = json.loads(Path(binding['path']).read_text(encoding='utf-8'))
    tree = payload.get('source_tree')
    return tree if tree and Path(tree).is_dir() else None
