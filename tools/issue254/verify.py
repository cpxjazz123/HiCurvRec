"""Independent raw-prediction/table reconciliation and reload/SHA inventory.

Does not import the pairing, metric, candidate, mechanism or statistics code.
"""
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'stage2_RQ-VAE').is_dir())
OUTPUT = ROOT / 'results/research/issue254'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def demand(condition, message):
    if not condition:
        raise AssertionError(message)


def prediction_digest(target, target_sid, candidates, canonical, generated):
    payload = [target, target_sid, candidates, canonical, generated]
    return hashlib.sha256(json.dumps(payload, separators=(',', ':')).encode()).hexdigest()


def main():
    import numpy as np
    import pyarrow.parquet as pq
    evidence = json.loads((OUTPUT / 'EVIDENCE.json').read_text())
    paired = json.loads((OUTPUT / 'PAIRED_SUMMARY.json').read_text())
    statistics = json.loads((OUTPUT / 'STATISTICS_SUMMARY.json').read_text())
    structure = json.loads((OUTPUT / 'STRUCTURE_SUMMARY.json').read_text())
    mechanism = json.loads((OUTPUT / 'MECHANISM_SUMMARY.json').read_text())
    candidate = json.loads((OUTPUT / 'CANDIDATE_SUMMARY.json').read_text())
    connections = json.loads((OUTPUT / 'CONNECTIONS_SUMMARY.json').read_text())
    tests = json.loads((OUTPUT / 'TEST_RESULTS.json').read_text())
    demand(tests['exit_code'] == tests['errors'] == tests['failures'] == tests['skipped'] == 0,
           'tests failed/errored/skipped')
    demand(tests['tests'] >= 37, 'missing diagnostic tests')
    for path, expected in tests['sources'].items():
        demand(digest(ROOT / path) == expected, 'test evidence has stale source: ' + path)
    checked_inputs = {}
    def check(path, expected):
        path = Path(path)
        if not path.is_absolute(): path = ROOT / path
        key = str(path.relative_to(ROOT))
        actual = checked_inputs.get(key) or digest(path)
        demand(actual == expected, 'stale input: ' + key)
        checked_inputs[key] = actual
    def records(value):
        if isinstance(value, dict):
            if 'path' in value and 'sha256' in value and value.get('present', True):
                check(value['path'], value['sha256'])
            for v in value.values(): records(v)
        elif isinstance(value, list):
            for v in value: records(v)
    records(evidence)
    for mapping in [structure['input_sha256'], mechanism['inputs_sha256'], connections['inputs_sha256']]:
        for path, expected in mapping.items(): check(path, expected)
    split = pq.read_table(evidence['test_parquet']['path']).to_pylist()
    originals = {}
    for name, generation in evidence['generations'].items():
        originals[name] = {}
        for record in generation['stage3_predictions']:
            with Path(record['path']).open() as stream:
                for line in stream:
                    row = json.loads(line)
                    index = row['global_row']
                    demand(index not in originals[name], 'duplicate original sample')
                    items = [r['item'] for r in row['topk']]
                    canonical = [r['sid_canonical'] for r in row['topk']]
                    generated = [r['sid_generated'] for r in row['topk']]
                    originals[name][index] = prediction_digest(row['target_item'], row['target_sid'], items, canonical, generated)
        demand(set(originals[name]) == set(range(len(split))), 'missing original population')
    counts = {k: Counter() for k in (5,10)}
    items = {k: defaultdict(Counter) for k in (5,10)}
    discounts = {k: {name: [] for name in evidence['generations']} for k in (5,10)}
    user_ids, sample_ids = set(), set()
    with (OUTPUT / 'tables/paired_user_ledger.csv').open(newline='') as stream:
        ledger = csv.DictReader(stream)
        for row in ledger:
            index, user, target = int(row['global_row']), int(row['user']), int(row['target_item'])
            actual = split[index]
            history, seen = json.loads(row['history']), json.loads(row['seen_history'])
            identity = hashlib.sha256(json.dumps([user,history,seen,target],separators=(',', ':')).encode()).hexdigest()
            demand(user == actual['user'] and target == actual['target'] and history == actual['history']
                   and seen == actual['seen_history'] and identity == row['sample_sha256'], 'split identity mismatch')
            demand(user not in user_ids and identity not in sample_ids, 'duplicate joined identity')
            user_ids.add(user); sample_ids.add(identity)
            ranks = {}
            for name in evidence['generations']:
                candidates = json.loads(row['topk_'+name])
                observed = prediction_digest(target, json.loads(row['target_sid_'+name]), candidates,
                                            json.loads(row['canonical_sids_'+name]), json.loads(row['generated_sids_'+name]))
                demand(observed == originals[name][index], 'saved candidate table changed')
                ranks[name] = candidates.index(target) if target in candidates else None
                stored = None if row['rank_'+name] == '' else int(row['rank_'+name])
                demand(ranks[name] == stored, 'candidate/rank mismatch')
            for k in (5,10):
                left, right = [ranks[n] is not None and ranks[n] < k for n in ('issue252','issue253')]
                label = ('both_hit' if left and right else 'only_issue252_hit' if left else
                         'only_issue253_hit' if right else 'neither_hit')
                counts[k][label] += 1
                items[k][target][label] += 1
                demand(row['group%d' % k] == label, 'paired group mismatch')
                for name, rank in ranks.items():
                    value = 0. if rank is None or rank >= k else 1 / math.log2(rank+2)
                    discounts[k][name].append(value)
    n = len(user_ids)
    demand(n == len(split) == 57439, 'population not conserved')
    independent = {}
    for k in (5,10):
        net = counts[k]['only_issue253_hit'] - counts[k]['only_issue252_hit']
        item_nets = []
        with (OUTPUT / ('tables/per_item_contribution_k%d.csv' % k)).open(newline='') as stream:
            seen_items = set()
            for row in csv.DictReader(stream):
                item = int(row['target_item'])
                demand(item not in seen_items and item in items[k], 'duplicate/unknown target summary')
                seen_items.add(item)
                tally = items[k][item]
                demand(int(row['users']) == sum(tally.values()), 'target contexts collapsed')
                for label in ('both_hit','only_issue252_hit','only_issue253_hit','neither_hit'):
                    demand(int(row[label]) == tally[label], 'per-item paired count mismatch')
                item_net = tally['only_issue253_hit'] - tally['only_issue252_hit']
                demand(abs(float(row['net_contribution']) - item_net/n) < 1e-12, 'wrong global denominator')
                item_nets.append(item_net)
        demand(seen_items == set(items[k]), 'missing target item summary')
        demand(sum(item_nets) == net, 'net item/global count mismatch')
        demand(dict(counts[k]) == paired['cutoffs'][str(k)]['groups'], 'paired groups differ from raw recheck')
        independent[str(k)] = {'groups': dict(counts[k]), 'net_hits': net, 'recall_delta': net/n,
                              'sum_per_item_net_hits': sum(item_nets), 'targets': len(items[k])}
        for name in evidence['generations']:
            hit_count = counts[k]['both_hit'] + counts[k]['only_'+name+'_hit']
            final_path = evidence['generations'][name]['stage3_test_final']['path']
            final = json.loads(Path(final_path).read_text())
            metrics = {'recall':hit_count/n,'ndcg':math.fsum(discounts[k][name])/n}
            for metric,value in metrics.items():
                demand(abs(value-final['test_'+metric+'@'+str(k)]) < 1e-12, 'original metric mismatch')
            independent[str(k)][name] = metrics
        demand(abs(net/n-statistics['overall'][str(k)]['recall_delta']) < 1e-12, 'statistical endpoint changed')
    descriptors = []
    for path in sorted((OUTPUT / 'tables').iterdir()):
        if not path.is_file(): continue
        description = {'path': str(path.relative_to(ROOT)), 'sha256': digest(path), 'bytes': path.stat().st_size,
                       'purpose': 'evaluation', 'committed': False, 'reload_verified': True}
        if path.suffix == '.csv':
            with path.open(newline='') as stream:
                reader = csv.reader(stream)
                columns = next(reader)
                rows = 0
                for values in reader:
                    demand(len(values) == len(columns), 'ragged CSV: '+path.name)
                    rows += 1
            description.update(format='csv', rows=rows, columns=columns)
        elif path.suffix == '.npz':
            with np.load(path, allow_pickle=False) as blob:
                description.update(format='npz', arrays={key:{'shape':list(blob[key].shape),'dtype':str(blob[key].dtype)} for key in blob.files})
        elif path.suffix == '.json':
            value = json.loads(path.read_text())
            description.update(format='json', rows=len(value) if isinstance(value,list) else None)
        else:
            raise ValueError('unregistered table format: '+path.name)
        demand(digest(path) == description['sha256'], 'table changed during reload')
        descriptors.append(description)
    demand(structure['context']['computed'] and structure['context']['rows'] == n, 'context join missing')
    demand(structure['full_stored_triples'] == 28391948 and mechanism['items'] == 24587, 'incomplete structural/mechanism coverage')
    demand(all(v['hard_q_sid_codebook_exact'] for v in mechanism['arms'].values()), 'hard q not checkpoint bound')
    demand(connections['population'] == n and connections['candidate_rows'] == candidate['candidate_error_rows'],
           'candidate/history/target mechanism joins incomplete')
    reconciliation = {'schema':'genrec.issue254.independent_reconciliation/1','population':n,'atol':1e-12,
                      'raw_candidate_tables_exact':True,'endpoints':independent,'all_checks_passed':True}
    (OUTPUT / 'RECONCILIATION.json').write_text(json.dumps(reconciliation,indent=2)+'\n')
    checks = [{'module':'identity/source/R2','status':'PASS','evidence':'EVIDENCE.json','checked_input_files':len(checked_inputs)},
              {'module':'paired metrics/item contribution conservation','status':'PASS','evidence':'RECONCILIATION.json'},
              {'module':'graph/triples/masks/context','status':'PASS','evidence':'STRUCTURE_SUMMARY.json'},
              {'module':'CPU checkpoint gate/attention/q/SID','status':'PASS','evidence':'MECHANISM_SUMMARY.json'},
              {'module':'actual candidate coverage and cases','status':'PASS','evidence':'CANDIDATE_SUMMARY.json'},
              {'module':'candidate/history/target topology and mechanism joins','status':'PASS','evidence':'CONNECTIONS_SUMMARY.json'},
              {'module':'paired/cluster statistics and controls','status':'PASS','evidence':'STATISTICS_SUMMARY.json'},
              {'module':'known answers/reference/designated mutations','status':'PASS','evidence':'TEST_RESULTS.json'},
              {'module':'new gradients/optimizer/DDP/training','status':'NA','reason':'read-only analysis; new training/mechanism changes forbidden'},
              {'module':'T5 pruned beam trajectory/probability attribution','status':'NA','reason':'not saved in the original runs; no fabricated trace'}]
    (OUTPUT / 'MODULE_CHECKS.json').write_text(json.dumps({'schema':'genrec.issue254.module_checks/1','modules':checks},indent=2)+'\n')
    source_files = sorted((ROOT / 'tools/issue254').rglob('*.py')) + [OUTPUT / 'freeze_evidence.py']
    summaries = sorted(OUTPUT.glob('*SUMMARY.json')) + [OUTPUT / 'EVIDENCE.json',OUTPUT / 'RECONCILIATION.json',OUTPUT / 'MODULE_CHECKS.json',OUTPUT / 'TEST_RESULTS.json']
    manifest = {'schema':'genrec.issue254.analysis_manifest/1','purpose':'evaluation','population':n,
                'training_view_readable':False,'static_assets_rebuilt':False,'data_directory':'results/research/issue254/tables',
                'inputs_sha256':checked_inputs, 'sources_sha256':{str(p.relative_to(ROOT)):digest(p) for p in source_files},
                'summaries_sha256':{str(p.relative_to(ROOT)):digest(p) for p in summaries},'outputs':descriptors,
                'commands':['python3 results/research/issue254/freeze_evidence.py',
                            'genrec_env/bin/python3.10 -m tools.issue254.paired',
                            'genrec_env_v2/bin/python3.9 -m tools.issue254.structure',
                            'genrec_env_v2/bin/python3.9 -m tools.issue254.mechanism',
                            'genrec_env/bin/python3.10 -m tools.issue254.candidates',
                            'genrec_env/bin/python3.10 -m tools.issue254.statistics',
                            'genrec_env/bin/python3.10 -m tools.issue254.connections',
                            'genrec_env/bin/python3.10 -m tools.issue254.verify'],
                'reload_verified':True,'privacy':'raw users and full contexts stay in ignored shared-storage tables; commits contain only summaries/schema/SHA/source'}
    (OUTPUT / 'MANIFEST.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'population':n,'endpoints':independent,'checked_input_files':len(checked_inputs),
                      'reloaded_tables':len(descriptors),'all_checks_passed':True},indent=2))


if __name__ == '__main__':
    main()
