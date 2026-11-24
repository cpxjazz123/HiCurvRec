"""Exact saved-output diagnostics, not a reconstruction of missing T5 beam traces."""
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

from tools.issue254 import contract
from tools.issue254.paired import write_csv
from tools.issue254.structure import load_graphs, relation_features

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'stage2_RQ-VAE').is_dir())
OUTPUT = ROOT / 'results/research/issue254'


def coverage_category(target_sid, candidate_sids, target_present):
    if target_present:
        return 'target_present'
    shared = max((contract.prefix_levels(target_sid, sid) for sid in candidate_sids), default=0)
    return ('saved_topk_lacks_target_l1', 'target_l1_covered_l2_absent',
            'target_l2_covered_l3_absent', 'target_l3_covered_target_absent')[shared]


def candidate_overlap(left, right):
    a, b = set(left), set(right)
    return {'shared': len(a & b), 'new_issue253': len(b - a), 'removed_issue252': len(a - b)}


def read_ledger():
    with (OUTPUT / 'tables/paired_user_ledger.csv').open(newline='') as stream:
        for raw in csv.DictReader(stream):
            for key in ('global_row', 'user', 'target_item', 'history_len', 'last_history_item'):
                raw[key] = int(raw[key])
            for key in ('history', 'seen_history', 'input_history'):
                raw[key] = json.loads(raw[key])
            for arm in contract.ARMS:
                key = 'rank_' + arm
                raw[key] = None if raw[key] == '' else int(raw[key])
                for prefix in ('target_sid_', 'topk_', 'canonical_sids_', 'generated_sids_'):
                    raw[prefix + arm] = json.loads(raw[prefix + arm])
            yield raw


def main():
    tables = OUTPUT / 'tables'
    graphs = load_graphs()
    evidence = json.loads((OUTPUT / 'EVIDENCE.json').read_text())
    sids, collisions = {}, {}
    for arm in contract.ARMS:
        source = evidence['generations'][arm]['stage2_sid']['path']
        sids[arm] = {int(item): [v + 1 for v in sid] for item, sid in json.loads(Path(source).read_text()).items()}
        collisions[arm] = Counter(tuple(sid[:3]) for sid in sids[arm].values())
    paired = json.loads((OUTPUT / 'PAIRED_SUMMARY.json').read_text())
    extremes = paired['per_item_tables']['10']
    loss_items = [r['target_item'] for r in extremes['largest_net_regressions'][:3]]
    gain_items = [r['target_item'] for r in extremes['largest_net_improvements'][:3]]
    case_targets = {contract.ONLY_252: loss_items, contract.ONLY_253: gain_items,
                    contract.BOTH: loss_items, contract.NEITHER: loss_items}
    cases, used_cases, summary_rows = [], set(), []
    coverage = defaultdict(Counter)
    details_count = 0
    fallback = Counter()
    detail_path = tables / 'candidate_errors.csv'
    detail_columns = ['global_row', 'user', 'sample_sha256', 'group5', 'group10', 'target_item',
                      'last_history_item', 'arm', 'candidate_display_rank', 'candidate_item',
                      'canonical_sid', 'generated_sid', 'fallback', 'shared_with_other_arm_top10',
                      'target_prefix_levels', 'candidate_suffix', 'target_suffix']
    # Relation schema is independent of the catalogue: all three observed item pairs.
    example_relation = relation_features(graphs['forward'], graphs['undirected'], 0, 1)
    relation_columns = [label + '_' + key for label in ('last_target', 'last_candidate', 'target_candidate')
                        for key in example_relation]
    with detail_path.open('w', newline='') as detail_stream:
        writer = csv.DictWriter(detail_stream, fieldnames=detail_columns + relation_columns)
        writer.writeheader()
        for raw in read_ledger():
            target = raw['target_item']
            context = {k: raw[k] for k in ('global_row', 'user', 'sample_sha256', 'target_item', 'group5', 'group10')}
            for cutoff in contract.CUTOFFS:
                overlap = candidate_overlap(raw['topk_issue252'][:cutoff], raw['topk_issue253'][:cutoff])
                context.update({key + '_k' + str(cutoff): value for key, value in overlap.items()})
                for arm in contract.ARMS:
                    canonical = raw['canonical_sids_' + arm][:cutoff]
                    prefix = [contract.prefix_levels(raw['target_sid_' + arm], sid) for sid in canonical]
                    category = coverage_category(raw['target_sid_' + arm], canonical,
                                                 contract.hit(raw['rank_' + arm], cutoff))
                    context['coverage_k%d_%s' % (cutoff, arm)] = category
                    coverage['k%d/%s/%s' % (cutoff, raw['group%d' % cutoff], arm)][category] += 1
                    for level in range(1, 4):
                        positions = [i + 1 for i, shared in enumerate(prefix) if shared >= level]
                        context['nearest_l%d_k%d_%s' % (level, cutoff, arm)] = positions[0] if positions else None
                        context['count_l%d_k%d_%s' % (level, cutoff, arm)] = len(positions)
            for arm in contract.ARMS:
                context['collision_group_size_' + arm] = collisions[arm][tuple(sids[arm][target][:3])]
                context['fallback_count_' + arm] = sum(v is None for v in raw['generated_sids_' + arm])
                fallback[arm] += context['fallback_count_' + arm]
            summary_rows.append(context)
            discordant = any(raw['group%d' % k] in (contract.ONLY_252, contract.ONLY_253) for k in contract.CUTOFFS)
            last = raw['last_history_item']
            last_target = relation_features(graphs['forward'], graphs['undirected'], last, target) if last >= 0 else {}
            if discordant:
                for arm, other in [('issue252', 'issue253'), ('issue253', 'issue252')]:
                    other_items = set(raw['topk_' + other])
                    for position, (item, canonical, generated) in enumerate(zip(raw['topk_' + arm],
                              raw['canonical_sids_' + arm], raw['generated_sids_' + arm])):
                        if item == target:
                            continue
                        detail = {k: raw[k] for k in ('global_row', 'user', 'sample_sha256', 'group5', 'group10',
                                                      'target_item', 'last_history_item')}
                        detail.update(arm=arm, candidate_display_rank=position + 1, candidate_item=item,
                                      canonical_sid=json.dumps(canonical), generated_sid=json.dumps(generated),
                                      fallback=generated is None, shared_with_other_arm_top10=item in other_items,
                                      target_prefix_levels=contract.prefix_levels(raw['target_sid_' + arm], canonical),
                                      candidate_suffix=canonical[3], target_suffix=raw['target_sid_' + arm][3])
                        relations = {'last_target': last_target,
                                     'last_candidate': relation_features(graphs['forward'], graphs['undirected'], last, item) if last >= 0 else {},
                                     'target_candidate': relation_features(graphs['forward'], graphs['undirected'], target, item)}
                        for label, values in relations.items():
                            detail.update({label + '_' + k: v for k, v in values.items()})
                        writer.writerow(detail)
                        details_count += 1
            group = raw['group10']
            case_key = (group, target)
            if target in case_targets[group] and case_key not in used_cases:
                used_cases.add(case_key)
                wrong = next((i for i in raw['topk_issue253'] if i != target), None)
                case = {'case': 'C%02d' % (len(cases) + 1), 'group10': group, 'target_item': target,
                        'last_history_item': last, 'wrong_item_issue253': wrong,
                        'rank_issue252': '>10' if raw['rank_issue252'] is None else raw['rank_issue252'] + 1,
                        'rank_issue253': '>10' if raw['rank_issue253'] is None else raw['rank_issue253'] + 1,
                        'coverage_issue253': context['coverage_k10_issue253'],
                        'last_target_relation': last_target,
                        'target_wrong_relation': relation_features(graphs['forward'], graphs['undirected'], target, wrong),
                        'last_wrong_relation': relation_features(graphs['forward'], graphs['undirected'], last, wrong) if last >= 0 else {}}
                cases.append(dict(public=case, private_identity={k: raw[k] for k in ('global_row', 'user', 'sample_sha256', 'history')}))
    write_csv(tables / 'candidate_context.csv', summary_rows, list(summary_rows[0]))
    (tables / 'cases.json').write_text(json.dumps(cases, indent=2, allow_nan=True) + '\n')
    # JSON null for undefined graph statistics, never impute an absent relation.
    def sanitize(value):
        import math
        import numpy as np
        if isinstance(value, dict): return {k: sanitize(v) for k, v in value.items()}
        if isinstance(value, list): return [sanitize(v) for v in value]
        if isinstance(value, (bool, np.bool_)): return bool(value)
        if isinstance(value, (int, np.integer)): return int(value)
        if isinstance(value, (float, np.floating)): return float(value) if math.isfinite(value) else None
        return value
    summary = {'schema': 'genrec.issue254.candidates/1', 'purpose': 'evaluation',
               'population': len(summary_rows), 'candidate_error_rows': details_count,
               'candidate_error_scope': 'all wrong selected candidates for the union of discordant@5/@10 samples, both arms',
               'coverage_by_cutoff_group_arm': {k: dict(v) for k, v in coverage.items()},
               'fallback_selected_candidates': dict(fallback),
               'saved_candidates_per_arm_per_sample': 10, 'search_beam': 20,
               'exact_target_ranks_beyond10': 'NA; no full beam trajectories/probabilities were saved',
               'illegal_selected_decode': 0, 'rank_validation_source': 'PAIRED_SUMMARY.json exact loader checks',
               'case_selection': 'first sample per top3 net-loss/net-gain target; both/neither controls reuse loss targets',
               'cases': [sanitize(c['public']) for c in cases],
               'graph_input_sha256': graphs['hashes'],
               'interpretation': 'saved output branch coverage is observable; T5 prefix pruning/probability/ranking cause is unidentifiable'}
    (OUTPUT / 'CANDIDATE_SUMMARY.json').write_text(json.dumps(sanitize(summary), indent=2, allow_nan=False) + '\n')
    print(json.dumps({'population': len(summary_rows), 'candidate_error_rows': details_count,
                      'cases': len(cases), 'fallback': dict(fallback)}, sort_keys=True))


if __name__ == '__main__':
    main()
