"""Stage 1 of Issue254: the full paired per-user ledger and its aggregation.

Read-only over the two finished generations.  It joins the arms by the global
test row (and checks the user id and the target item on every row), forms the
four paired groups per cutoff, recomputes both arms' recall/NDCG from the raw
prediction files, and aggregates the result to one row per target item.
"""
from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tools.issue254 import contract, reference

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT/'results/research/issue254'

ARMS = {
    'issue252': ROOT/'results/stage3_T5Train/TIGER_gnn_attention_sectional_rq/logs/'
                   'Amazon_2023_Instruments/Nov-24-2026_15-34-06',
    'issue253': ROOT/'results/stage3_T5Train/TIGER_three_expert_attention_rq/logs/'
                   'Amazon_2023_Instruments/Nov-24-2026_21-16-28',
}
TEST_PARQUET = ROOT/'results/stage0_build_parquet/test.parquet'


def sample_identity(row: dict) -> str:
    payload = [row['user'], row['history'], row['seen_history'], row['target_item']]
    return hashlib.sha256(json.dumps(payload, separators=(',', ':')).encode()).hexdigest()


def read_test_rows() -> List[dict]:
    """Preserve full and model-truncated input, including the exclusion history."""
    import pyarrow.parquet as pq
    rows = []
    for index, row in enumerate(pq.read_table(TEST_PARQUET).to_pylist()):
        sample = {'global_row': index, 'user': int(row['user']),
                  'target_item': int(row['target']),
                  'history': [int(v) for v in row['history']],
                  'seen_history': [int(v) for v in row['seen_history']]}
        sample['sample_sha256'] = sample_identity(sample)
        rows.append(sample)
    if len({r['user'] for r in rows}) != len(rows):
        raise ValueError('duplicate test user')
    return rows


def load_arm(run_dir: Path, test_rows: List[dict]) -> dict:
    """Recover identities from the original split and independently check all dumps."""
    final = json.loads((run_dir/'test_final.json').read_text())
    world = int(final['stage3_world_size'])
    if world != 4 or final['n_eval'] != len(test_rows):
        raise ValueError('unexpected population or world size')
    sid_payload = json.loads(Path(final['sid_path']).read_text())
    sids = {int(item): [int(v) + 1 for v in sid] for item, sid in sid_payload.items()}
    if set(sids) != set(range(len(sids))):
        raise ValueError('catalogue IDs not contiguous')
    decode = {tuple(sid): item for item, sid in sids.items()}
    if len(decode) != len(sids):
        raise ValueError('duplicate four-token catalogue SID')
    rows = reference.load_prediction_rows(run_dir)
    expected_shards = {'rank%d' % r for r in range(world)}
    if {r['shard'] for r in rows} != expected_shards:
        raise ValueError('missing prediction shard')
    shard_payloads = {key: json.loads((run_dir/'_test_shards'/(key+'.json')).read_text())
                      for key in expected_shards}
    ranks = {}
    for row in rows:
        index = row['global_row']
        shard = int(row['shard'][4:])
        local = row['local_row_index']
        if index != shard + local * world or not 0 <= index < len(test_rows):
            raise ValueError('prediction sharding identity mismatch')
        if index in ranks:
            raise ValueError('duplicate global row')
        sample = test_rows[index]
        if row['target_item'] != sample['target_item']:
            raise ValueError('target identity mismatch')
        if row['target_sid'] != sids[row['target_item']]:
            raise ValueError('target SID identity mismatch')
        candidates = row['candidates']
        if len(candidates) != contract.SAVED_CANDIDATES or len(set(candidates)) != len(candidates):
            raise ValueError('duplicate or incomplete candidate sequence')
        seen = set(sample['seen_history'])
        for item, canonical, generated in zip(candidates, row['candidate_sids'], row['generated_sids']):
            if item not in sids or canonical != sids[item] or item in seen:
                raise ValueError('illegal catalogue/SID/history decode')
            if generated is not None and decode.get(tuple(generated)) != item:
                raise ValueError('generated SID decode mismatch')
        rank = reference.recompute_rank(candidates, row['target_item'])
        payload = shard_payloads[row['shard']]
        shard_user = payload['per_user'][local]
        if shard_user['local_row_index'] != local:
            raise ValueError('shard local row mismatch')
        if rank != row['dumped_rank'] or rank != shard_user['target_rank']:
            raise ValueError('dumped rank disagrees with actual candidate sequence')
        row.update(user=sample['user'], sample_sha256=sample['sample_sha256'])
        ranks[index] = rank
    if set(ranks) != set(range(len(test_rows))):
        raise ValueError('missing prediction population')
    shard_counts = {}
    for key, payload in shard_payloads.items():
        shard_rows = [r for r in rows if r['shard'] == key]
        recall_sums, ndcg_sums, n = reference.metric_sums([ranks[r['global_row']] for r in shard_rows])
        if n != payload['n_eval'] or n != len(payload['per_user']):
            raise ValueError('shard population mismatch')
        for k in contract.CUTOFFS:
            if recall_sums[k] != payload['recall_sums'][str(k)]:
                raise ValueError('shard recall sum mismatch')
            if not math.isclose(ndcg_sums[k], payload['ndcg_sums'][str(k)], abs_tol=1e-9):
                raise ValueError('shard NDCG sum mismatch')
        shard_counts[key] = n
    return {'rows': rows, 'ranks': ranks, 'final': final,
            'shard_counts': shard_counts, 'run_dir': str(run_dir)}


def build_ledger(arm_252: dict, arm_253: dict, test_rows: List[dict]) -> Tuple[List[dict], dict]:
    """Join validated composite identities; no row-only or partial-population joins."""
    expected = {r['global_row']: r for r in test_rows}
    if len(expected) != len(test_rows) or len({r['user'] for r in test_rows}) != len(test_rows):
        raise ValueError('duplicate test identity')
    keyed_arms = []
    for arm in (arm_252, arm_253):
        keyed = {}
        seen_rows = set()
        for row in arm['rows']:
            index = row['global_row']
            if index in seen_rows:
                raise ValueError('duplicate prediction row')
            seen_rows.add(index)
            if index not in expected:
                raise ValueError('extra sample identity')
            sample = expected[index]
            key = (row['user'], row['sample_sha256'], row['target_item'])
            original = (sample['user'], sample_identity(sample), sample['target_item'])
            if key != original:
                raise ValueError('user/sample/target identity mismatch')
            if key in keyed:
                raise ValueError('duplicate sample identity')
            if index not in arm['ranks']:
                raise ValueError('missing rank record, not an unhit target')
            keyed[key] = row
        if seen_rows != set(expected):
            raise ValueError('missing prediction population')
        keyed_arms.append(keyed)
    ledger = []
    for sample in test_rows:
        key = (sample['user'], sample_identity(sample), sample['target_item'])
        left, right = [table[key] for table in keyed_arms]
        r252 = arm_252['ranks'][left['global_row']]
        r253 = arm_253['ranks'][right['global_row']]
        row = dict(sample)
        row.update(history_len=len(sample['history']), input_history=sample['history'][-20:],
                   last_history_item=sample['history'][-1] if sample['history'] else -1,
                   target_seen=sample['target_item'] in sample['seen_history'])
        for name, rank, prediction in [('issue252', r252, left), ('issue253', r253, right)]:
            row['rank_'+name] = rank
            row['truncated_rank_'+name] = contract.truncated_rank(rank)
            row['rank_censored_'+name] = rank is None
            row['target_sid_'+name] = prediction['target_sid']
            row['topk_'+name] = prediction['candidates']
            row['canonical_sids_'+name] = prediction['candidate_sids']
            row['generated_sids_'+name] = prediction['generated_sids']
            for k in contract.CUTOFFS:
                row['hit%d_%s' % (k, name)] = contract.hit(rank, k)
                row['ndcg%d_%s' % (k, name)] = contract.ndcg_contribution(rank, k)
        row['censored_rank_score_delta'] = contract.truncated_rank(r253) - contract.truncated_rank(r252)
        for k in contract.CUTOFFS:
            row['group%d' % k] = contract.group(r252, r253, k)
            row['delta_hit%d' % k] = int(row['hit%d_issue253' % k]) - int(row['hit%d_issue252' % k])
            row['delta_ndcg%d' % k] = row['ndcg%d_issue253' % k] - row['ndcg%d_issue252' % k]
        ledger.append(row)
    return ledger, {}


def summarise(ledger: List[dict], arms: Dict[str, dict]) -> dict:
    pairs = [(row['rank_issue252'], row['rank_issue253']) for row in ledger]
    summary = {'population': len(ledger), 'cutoffs': {}}
    for k in contract.CUTOFFS:
        counts = contract.group_counts(pairs, k)
        cell = {'groups': counts, 'net_change': contract.net_change(counts, len(ledger))}
        for a, name in enumerate(contract.ARMS):
            cell['recall_'+name] = contract.recall(pairs, k, a)
            cell['ndcg_'+name] = contract.ndcg(pairs, k, a)
            final = arms[name]['final']
            for metric, final_key in [('recall', 'test_recall@'), ('ndcg', 'test_ndcg@')]:
                if not math.isclose(cell[metric+'_'+name], final[final_key+str(k)],
                                    rel_tol=0, abs_tol=1e-12):
                    raise ValueError('independent final metric mismatch')
        if not math.isclose(cell['recall_issue253'] - cell['recall_issue252'],
                            cell['net_change'], abs_tol=1e-12):
            raise ValueError('paired conservation failure')
        summary['cutoffs'][str(k)] = cell
    return summary


def write_csv(path: Path, rows: List[dict], columns: List[str]) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: json.dumps(row[column], separators=(',', ':'))
                             if isinstance(row.get(column), (list, dict)) else row.get(column)
                             for column in columns})


def main() -> int:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    tables = EVIDENCE/'tables'
    test_rows = read_test_rows()
    arms = {name: load_arm(path, test_rows) for name, path in ARMS.items()}
    ledger, gaps = build_ledger(arms['issue252'], arms['issue253'], test_rows)
    summary = summarise(ledger, arms)
    summary['schema'] = 'genrec.issue254.paired/2'
    summary['purpose'] = 'evaluation'
    summary['join'] = {'gaps': gaps, 'matched': len(ledger),
                       'unique_users': len({r['user'] for r in ledger}),
                       'rule': '(user, SHA256(full history,seen history,target,user),target); verified split/shard/target/SID reconstruction',
                       'raw_user_ids_dumped': False}
    summary['reference_cross_check'] = {
        name: {'candidate_vs_dumped_vs_shard_ranks': 'exact', 'shard_counts': arm['shard_counts'],
               'final_metric_atol': 1e-12} for name, arm in arms.items()}
    summary['target_seen_count'] = sum(r['target_seen'] for r in ledger)
    write_csv(tables/'paired_user_ledger.csv', ledger, list(ledger[0]))
    summary['per_item_tables'] = {}
    for k in contract.CUTOFFS:
        rows = contract.per_item_contributions(ledger, k)
        extra = {}
        for sample in ledger:
            bucket = extra.setdefault(sample['target_item'], [0, 0.0])
            bucket[0] += sample['censored_rank_score_delta']
            bucket[1] += sample['delta_ndcg%d' % k]
        for item in rows:
            count_rank, sum_ndcg = extra[item['target_item']]
            item['mean_censored_rank_score_delta'] = count_rank / item['users']
            item['net_ndcg_contribution'] = sum_ndcg / len(ledger)
            item['macro_recall_delta'] = item['hit_rate_issue253'] - item['hit_rate_issue252']
        total = math.fsum(row['net_contribution'] for row in rows)
        if not math.isclose(total, summary['cutoffs'][str(k)]['net_change'], abs_tol=1e-12):
            raise ValueError('item contributions do not conserve global delta')
        write_csv(tables/('per_item_contribution_k%d.csv' % k), rows, list(rows[0]))
        summary['per_item_tables'][str(k)] = {
            'rows': len(rows), 'sum_net_contribution': total,
            'macro_recall_delta': math.fsum(r['macro_recall_delta'] for r in rows) / len(rows),
            'largest_net_regressions': sorted(rows, key=lambda r: (r['net_contribution'], r['target_item']))[:20],
            'largest_net_improvements': sorted(rows, key=lambda r: (-r['net_contribution'], r['target_item']))[:20]}
    (EVIDENCE/'PAIRED_SUMMARY.json').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'population': len(ledger), 'cutoffs': summary['cutoffs'],
                      'join': summary['join'], 'summary': 'PAIRED_SUMMARY.json'}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
