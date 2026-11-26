import json, statistics as st
from pathlib import Path

base = Path('results/research/issue267')
d = json.loads((base / 'SCORE_PREDICTION.json').read_text())
hist = [(x['index'], x['arm'], x['S'], x['recall@10']) for x in d['historical']]
new_rec = 0.05078431031180905
new_s = d['metrics']['S']
pts = hist + [(110, 'TIGER_relative_curvature_ranking_rq', new_s, new_rec)]


def rank(v):
    s = sorted(range(len(v)), key=lambda i: v[i])
    r = [0] * len(v)
    for pos, i in enumerate(s):
        r[i] = pos + 1
    return r


def pearson(a, b):
    ma, mb = st.mean(a), st.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    return num / ((sum((x - ma) ** 2 for x in a) ** .5) * (sum((y - mb) ** 2 for y in b) ** .5))


def kend(a, b):
    c = dsc = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            s = (a[i] - a[j]) * (b[i] - b[j])
            if s > 0:
                c += 1
            elif s < 0:
                dsc += 1
    return c, dsc


S = [t[2] for t in pts]
R = [t[3] for t in pts]
order = [t[0] for t in sorted(pts, key=lambda t: t[3])]
c, dsc = kend(S, R)
tau = (c - dsc) / (c + dsc)
below = [t[0] for t in pts if t[3] < new_rec]
out = {
    'schema': 'genrec.issue267.stage3_recall_11/1',
    'new': {'index': 110, 'arm': 'TIGER_relative_curvature_ranking_rq',
            'recall@10': new_rec, 'S': new_s},
    'stage3': {'n_eval': 57439,
               'test_recall@5': 0.03300893121398353,
               'test_recall@10': new_rec,
               'test_ndcg@5': 0.021512508774359167,
               'test_ndcg@10': 0.027244785026023777,
               'stage3_run': 'results/stage3_T5Train/TIGER_relative_curvature_ranking_rq/logs/Amazon_2023_Instruments/Nov-26-2026_17-58-19'},
    'parent_109_recall@10': 0.052594926791900976,
    'delta_vs_parent': new_rec - 0.052594926791900976,
    'relative_delta_vs_parent': (new_rec - 0.052594926791900976) / 0.052594926791900976,
    'recall_ascending_order': order,
    'new_rank_ascending': order.index(110) + 1,
    'n_arms': len(pts),
    'correlations': {'pearson_all11': pearson(S, R),
                     'spearman_all11': pearson(rank(S), rank(R)),
                     'kendall_all11': {'concordant': c, 'discordant': dsc, 'tau': tau},
                     'pearson_hist10': pearson([t[2] for t in hist], [t[3] for t in hist]),
                     'kendall_hist10': {'concordant': 45, 'discordant': 0}},
    'prediction': {'predicted_rank': 11, 'actual_rank': order.index(110) + 1,
                   'predicted_above_range': True, 'correct_side': False,
                   'arms_below_new_recall': below,
                   'inversions': len(below)},
    'score_contract_sha256': 'c39e01ef076f22d4b799cc3ca7eb19ad910e1f9d2cb133004417c3190d16c8ca',
}
(base / 'analysis' / 'STAGE3_RECALL_11.json').write_text(
    json.dumps(out, indent=2, sort_keys=True) + '\n')
print(json.dumps(out, indent=1))
