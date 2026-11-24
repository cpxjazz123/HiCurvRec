"""Paired/user and target-cluster inference for one post-hoc comparison."""
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, t as student_t

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'stage2_RQ-VAE').is_dir())
OUTPUT = ROOT / 'results/research/issue254'
SEED = 254
REPETITIONS = 10000


def exact_mcnemar(regressions, improvements):
    n = int(regressions + improvements)
    return float(binomtest(int(improvements), n, .5, alternative='two-sided').pvalue) if n else 1.0


def paired_interval(values, repetitions=REPETITIONS, seed=SEED):
    """Exact empirical bootstrap law via histogram multinomial, no NxB copying."""
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError('paired interval requires finite nonempty values')
    distinct, counts = np.unique(values, return_counts=True)
    draws = np.random.default_rng(seed).multinomial(len(values), counts / len(values), size=repetitions)
    means = draws @ distinct / len(values)
    return np.quantile(means, [.025, .975]).tolist()


def cluster_interval(numerators, denominators, repetitions=REPETITIONS, seed=SEED):
    """Resample complete target clusters and re-form the ratio in every replicate."""
    numerators, denominators = np.asarray(numerators, float), np.asarray(denominators, float)
    if not len(numerators) or len(numerators) != len(denominators) or (denominators <= 0).any():
        raise ValueError('invalid cluster populations')
    rng, n = np.random.default_rng(seed), len(numerators)
    results = np.empty(repetitions)
    for start in range(0, repetitions, 64):
        stop = min(start + 64, repetitions)
        indices = rng.integers(0, n, size=(stop - start, n))
        results[start:stop] = numerators[indices].sum(1) / denominators[indices].sum(1)
    return np.quantile(results, [.025, .975]).tolist()


def holm_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    corrected = np.minimum(1., np.maximum.accumulate(p[order] * (len(p) - np.arange(len(p)))))
    output = np.empty_like(corrected)
    output[order] = corrected
    return output.tolist()


def adjusted_association(features, outcome, clusters):
    """Complete-case OLS with item-cluster sandwich SE; conditional association only."""
    names = list(features)
    x = np.column_stack([np.asarray(features[k], float) for k in names])
    y, cluster = np.asarray(outcome, float), np.asarray(clusters)
    keep = np.isfinite(x).all(1) & np.isfinite(y)
    excluded = int((~keep).sum())
    x, y, cluster = x[keep], y[keep], cluster[keep]
    if len(y) < 3:
        raise ValueError('insufficient complete cases')
    std = x.std(0)
    variable = std > 1e-15
    omitted = [name for name, valid in zip(names, variable) if not valid]
    names = [name for name, valid in zip(names, variable) if valid]
    x = (x[:, variable] - x[:, variable].mean(0)) / std[variable]
    design = np.column_stack([np.ones(len(x)), x])
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError('adjusted model design is rank deficient')
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ beta
    keys, index = np.unique(cluster, return_inverse=True)
    g, n, p = len(keys), len(y), design.shape[1]
    if g <= 1 or n <= p:
        raise ValueError('insufficient independent item clusters')
    scores = np.zeros((g, p))
    np.add.at(scores, index, design * residual[:, None])
    bread = np.linalg.inv(design.T @ design)
    covariance = bread @ (scores.T @ scores) @ bread * g / (g - 1) * (n - 1) / (n - p)
    se = np.sqrt(np.maximum(0., np.diag(covariance)))
    critical = student_t.ppf(.975, g - 1)
    coefficients = []
    for i, name in enumerate(names, 1):
        pv = float(2 * student_t.sf(abs(beta[i] / se[i]), g - 1)) if se[i] > 0 else (0. if beta[i] else 1.)
        coefficients.append({'feature': name, 'coefficient_per_sd': float(beta[i]),
                             'cluster_se': float(se[i]), 'ci95': [float(beta[i] - critical * se[i]), float(beta[i] + critical * se[i])],
                             'p': pv})
    for row, corrected in zip(coefficients, holm_adjust([r['p'] for r in coefficients])):
        row['p_holm_model_family'] = corrected
    return {'population': n, 'excluded_incomplete': excluded, 'clusters': g,
            'intercept': float(beta[0]), 'omitted_constant_features': omitted,
            'coefficients': coefficients, 'interpretation': 'post-hoc complete-case conditional association, not causal mediation'}


def endpoint(frame, k):
    delta = frame['delta_hit%d' % k].to_numpy(float)
    regressions, improvements = int((delta < 0).sum()), int((delta > 0).sum())
    clusters = frame.groupby('target_item')['delta_hit%d' % k].agg(['sum', 'size'])
    return {'population': len(frame), 'targets': len(clusters), 'regressions': regressions, 'improvements': improvements,
            'recall_delta': float(delta.mean()), 'net_hit_count': improvements - regressions,
            'mcnemar_exact_two_sided_p': exact_mcnemar(regressions, improvements),
            'paired_user_ci95': paired_interval(delta),
            'target_cluster_ci95': cluster_interval(clusters['sum'], clusters['size']),
            'item_macro_delta': float((clusters['sum'] / clusters['size']).mean()),
            'item_macro_ci95': cluster_interval(clusters['sum'] / clusters['size'], np.ones(len(clusters))),
            'ndcg_delta': float(frame['delta_ndcg%d' % k].mean()),
            'ndcg_paired_ci95': paired_interval(frame['delta_ndcg%d' % k])}


def strata(frame):
    import pandas as pd
    frequency = frame['test_target_frequency']
    degree = frame['degree_undirected']
    length = frame['history_len']
    baseline = frame['rank_issue252']
    return {'target_test_frequency': pd.cut(frequency, [0, 1, 4, 9, np.inf], labels=['1', '2-4', '5-9', '>=10']),
            'target_train_frequency': pd.cut(frame['train_target_frequency'], [-1, 0, 4, 19, 99, np.inf], labels=['0', '1-4', '5-19', '20-99', '>=100']),
            'behavior_degree': pd.cut(degree, [-1, 4, 19, 99, np.inf], labels=['0-4', '5-19', '20-99', '>=100']),
            'history_length': pd.cut(length, [-1, 4, 9, 19, np.inf], labels=['0-4', '5-9', '10-19', '>=20']),
            'baseline_rank': pd.Series(np.where(baseline.isna(), 'not_in10', np.where(baseline < 5, '0-4', '5-9')), index=frame.index),
            'curvature': frame['curvature_class'], 'dominant_family_exploratory': frame['dominant_family'].fillna('missing')}


def training_frequencies(root, manifest):
    """Train split is an upstream manifest input, not an entry in generated assets."""
    import pandas as pd
    from tools.issue254.structure import sha256
    path = Path(root) / 'results/stage0_build_parquet/train.parquet'
    expected = manifest['inputs']['train']
    if sha256(path) != expected:
        raise ValueError('training frequency source changed')
    return pd.read_parquet(path, columns=['target'])['target'].value_counts(), expected


def main():
    import pandas as pd
    tables = OUTPUT / 'tables'
    ledger_columns = ['global_row', 'user', 'sample_sha256', 'target_item', 'history_len', 'last_history_item',
                      'rank_issue252', 'rank_issue253', 'group5', 'group10', 'delta_hit5', 'delta_hit10',
                      'ndcg5_issue252', 'ndcg5_issue253', 'ndcg10_issue252', 'ndcg10_issue253',
                      'delta_ndcg5', 'delta_ndcg10', 'censored_rank_score_delta', 'target_seen']
    users = pd.read_csv(tables / 'paired_user_ledger.csv', usecols=ledger_columns)
    if len(users) != 57439 or users['user'].duplicated().any() or users['sample_sha256'].duplicated().any():
        raise ValueError('invalid paired population')
    structure = pd.read_csv(tables / 'structure_items.csv')
    mechanism = pd.read_csv(tables / 'mechanism_items.csv').rename(columns={'item_id': 'item'})
    items = structure.merge(mechanism, on='item', how='outer', validate='one_to_one', indicator=True)
    if len(items) != 24587 or not (items['_merge'] == 'both').all():
        raise ValueError('mechanism/structure catalogue mismatch')
    items = items.drop(columns='_merge')
    manifest = json.loads((ROOT / 'assets/manifest.json').read_text())
    train_frequency, train_digest = training_frequencies(ROOT, manifest)
    test_frequency = users['target_item'].value_counts()
    items['test_target_frequency'] = items['item'].map(test_frequency).fillna(0).astype(int)
    items['train_target_frequency'] = items['item'].map(train_frequency).fillna(0).astype(int)
    contributions = pd.read_csv(tables / 'per_item_contribution_k10.csv')
    items = items.merge(contributions.rename(columns={'target_item': 'item'}), on='item', how='left', validate='one_to_one')
    # These are differences of within-model diagnostics; no coordinate/code-number comparison.
    for stem in ['gate_negative', 'gate_zero', 'gate_positive', 'gate_entropy', 'attention_1hop', 'attention_2hop',
                 'reconstruction_mse_q', 'reconstruction_mse_z', 'quantization_mse',
                 'curvature_mae_q', 'curvature_mae_z', 'curvature_sign_accuracy_q',
                 'edge_prefix_l1_out', 'edge_prefix_l2_out', 'edge_prefix_l3_out',
                 'sid_l1_occupancy', 'sid_l2_occupancy', 'collision_group_size']:
        items[stem + '_delta'] = items[stem + '_issue253'] - items[stem + '_issue252']
    items.to_csv(tables / 'joined_item_diagnostics.csv', index=False)
    # Avoid copying the complete per-item family/orbit matrices into every user context.
    selected = ['item', 'degree_in', 'degree_out', 'degree_undirected', 'reciprocal_ratio',
                'induced_neighbor_density', 'two_hop_size', 'curvature_class', 'curvature_missing',
                'curvature_negative_ratio', 'curvature_positive_ratio', 'curvature_mean', 'curvature_variance',
                'dominant_family', 'dominant_family_ratio', 'test_target_frequency', 'train_target_frequency']
    selected += [c for c in items if c.endswith('_delta') and c != 'macro_recall_delta']
    selected += [c for c in items if c.startswith(('gate_', 'attention_')) and c.endswith(('issue252', 'issue253'))]
    selected = list(dict.fromkeys(selected))
    users = users.merge(items[selected], left_on='target_item', right_on='item', how='left', validate='many_to_one')
    keys = ['global_row', 'user', 'sample_sha256', 'target_item']
    context = pd.read_csv(tables / 'context_graph.csv')
    if not context[keys].equals(users[keys]):
        raise ValueError('context graph sample identity/order mismatch')
    users = users.merge(context, on=keys, how='inner', validate='one_to_one')
    candidate = pd.read_csv(tables / 'candidate_context.csv')
    if not candidate[keys].equals(users[keys]):
        raise ValueError('candidate sample identity/order mismatch')
    users = users.merge(candidate.drop(columns=['group5', 'group10']), on=keys, how='inner', validate='one_to_one')
    if len(users) != 57439:
        raise ValueError('joined population changed')
    users.to_csv(tables / 'joined_user_diagnostics.csv', index=False)
    overall = {str(k): endpoint(users, k) for k in (5, 10)}
    subgroup_rows = []
    for name, values in strata(users).items():
        if values.isna().any():
            raise ValueError('unassigned stratum: ' + name)
        for label in sorted(values.unique().tolist()):
            subset = users[values == label]
            for k in (5, 10):
                delta = subset['delta_hit%d' % k].to_numpy(float)
                b, c = int((delta < 0).sum()), int((delta > 0).sum())
                subgroup_rows.append({'stratum': name, 'level': str(label), 'cutoff': k,
                                      'population': len(subset), 'targets': int(subset['target_item'].nunique()),
                                      'global_population': len(users), 'coverage': len(subset) / len(users),
                                      'regressions': b, 'improvements': c, 'net_hit_count': c - b,
                                      'conditional_recall_delta': float(delta.mean()),
                                      'global_recall_contribution': float(delta.sum() / len(users)),
                                      'paired_ci95': paired_interval(delta), 'mcnemar_p': exact_mcnemar(b, c)})
    for row, corrected in zip(subgroup_rows, holm_adjust([r['mcnemar_p'] for r in subgroup_rows])):
        row['p_holm_all_subgroups_both_cutoffs'] = corrected
    pd.DataFrame(subgroup_rows).to_csv(tables / 'stratified_report.csv', index=False)
    feature_columns = ['gate_zero_delta', 'gate_entropy_delta', 'attention_2hop_delta',
                       'curvature_mae_q_delta', 'reconstruction_mse_q_delta', 'quantization_mse_delta',
                       'edge_prefix_l1_out_delta', 'edge_prefix_l2_out_delta', 'collision_group_size_delta',
                       'curvature_negative_ratio', 'degree_undirected', 'history_len', 'test_target_frequency']
    groups = {}
    for k in (5, 10):
        groups[str(k)] = {}
        for label, subset in users.groupby('group%d' % k):
            groups[str(k)][label] = {'samples': len(subset), 'targets': int(subset['target_item'].nunique()),
                                    'mean': {col: float(subset[col].mean()) if subset[col].notna().any() else None for col in feature_columns},
                                    'valid_counts': {col: int(subset[col].notna().sum()) for col in feature_columns}}
    features = {'log_train_target_frequency': np.log1p(users['train_target_frequency']),
                'log_test_target_frequency': np.log1p(users['test_target_frequency']),
                'log_behavior_degree': np.log1p(users['degree_undirected']),
                'log_history_length': np.log1p(users['history_len']),
                'negative_curvature_fraction': users['curvature_negative_ratio'],
                'euclidean_gate_delta': users['gate_zero_delta'],
                'curvature_mae_delta': users['curvature_mae_q_delta'],
                'reconstruction_mse_delta': users['reconstruction_mse_q_delta'],
                'edge_prefix_l1_out_delta': users['edge_prefix_l1_out_delta']}
    adjusted = adjusted_association(features, users['delta_hit10'], users['target_item'])
    summary = {'schema': 'genrec.issue254.statistics/1', 'purpose': 'evaluation', 'seed': SEED,
               'bootstrap_repetitions': REPETITIONS, 'ci': 'percentile 95%', 'delta_direction': 'issue253 - issue252',
               'primary_endpoint': 'paired hit@10 change', 'overall': overall, 'groups': groups,
               'strata': subgroup_rows, 'holm_family_size': len(subgroup_rows),
               'adjusted_exploratory_association': adjusted,
               'additional_strata': ['target_train_frequency', 'dominant_family_exploratory'],
               'additional_strata_reason': 'training popularity controls exposure; exact motif family is the requested structural attribution, both post-hoc',
               'train_frequency_source_sha256': train_digest,
               'limitations': ['Single completed run per arm; model/expert count, fixed supervision, execution and trajectories changed together.',
                               'Catalogue-derived T5 vocab/EOS also differ; not a single-factor or identical-vocabulary causal comparison.',
                               'Baseline-rank subgrouping conditions on an outcome and is descriptive.',
                               'Complete-case adjusted associations are neither causal effects nor mediation identification.',
                               'No model multi-seed training, new baseline or experiment.']}
    (OUTPUT / 'STATISTICS_SUMMARY.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'overall': overall, 'strata': len(subgroup_rows),
                      'adjusted_population': adjusted['population'], 'adjusted_clusters': adjusted['clusters']}, indent=2))


if __name__ == '__main__':
    main()
