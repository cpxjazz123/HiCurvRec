"""Issue #288 R01 diagnosis: the P/N linkage decomposition and the curvature's within-stratum information.

Reuses the frozen artifacts only (the #277 evaluator, the frozen SIDs, the visible training graph and
the sectional-curvature triplet table).  No training, no new Infra.

Outputs `results/research/issue288/DIAGNOSIS_DATA.json` with:

A. per-export decomposition: P, N and the #277 exact N terms; the validation anchors' L1 bucket sizes;
   the in-bucket behavioural purity (co-sharers that are edges or share a neighbour); the unshared
   related pairs (edges with a different code = the false negatives); the anchor-degree/bucket-size
   relation; the L2/L3 distinctness.
B. the curvature's within-stratum information: for the distance-1/2 pairs of the triplet table, the
   label = the pair is an edge, the feature = the curvature, the strata = the direction pattern, the
   centre's degree bucket and the common-neighbour bucket; the pooled within-stratum AUC of the
   curvature versus the structural controls (the degree sum, the common neighbours).
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, '.')
from tools.issue252 import layer_metrics as lm          # the frozen #277 evaluator
from tools.issue252.final_evaluation import load_sid

REPO = Path('.').resolve()
OUT = REPO / 'results/research/issue288'
VALID = REPO / 'results/stage0_build_parquet/valid.parquet'
NODES = 24587
SIDS = {
    'TIGER': REPO / 'results/stage2_RQ-VAE/TIGER_RQ-VAE/item_sids.json',
    'R2': REPO / 'results/stage2_RQ-VAE/_runs/stage2_58090127b9954c6e856b2cbf90b1b440/TIGER_pairrank_enc_rq/result/item_sids.json',
    'R3': REPO / 'results/stage2_RQ-VAE/_runs/stage2_6fe2faf7fc204330b129dfa4900ae641/TIGER_pairrank_cover_rq/result/item_sids.json',
    'r1_287': REPO / 'results/stage2_RQ-VAE/_runs/stage2_c85aa9d946324638b87e74571f5026c4/TIGER_l1_pairrank_rq/result/item_sids.json',
    'r7_287': REPO / 'results/stage2_RQ-VAE/_runs/stage2_36996d68eced46faa7951bb4346e3e7e/TIGER_bucketsize_pairrank_rq/result/item_sids.json',
}
SHARDS = [f'shard_{i:03d}.npz' for i in range(13)]


def sha(p: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def graph():
    und = np.load(REPO / 'assets/train/visible_undirected.npz')
    indptr, indices = und['indptr'], und['indices']
    rows = np.repeat(np.arange(NODES), np.diff(indptr))
    adj = {}
    for r, c in zip(rows.tolist(), indices.tolist()):
        adj.setdefault(int(r), set()).add(int(c))
    edges = set()
    for u, vs in adj.items():
        for v in vs:
            if u < v:
                edges.add((u, v))
    return adj, edges


def auc(pos_scores, neg_scores) -> float:
    pos = np.sort(np.asarray(pos_scores, dtype=np.float64))
    neg = np.sort(np.asarray(neg_scores, dtype=np.float64))
    if pos.size == 0 or neg.size == 0:
        return float('nan')
    less = np.searchsorted(neg, pos, side='left')
    le = np.searchsorted(neg, pos, side='right')
    return float(((less + 0.5 * (le - less)) / neg.size).mean())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    examples = pq.read_table(VALID, columns=['history', 'seen_history', 'target']).to_pylist()
    adj, edges = graph()
    degree = np.array([len(adj.get(i, ())) for i in range(NODES)], dtype=np.int64)
    anchors = [int(e['history'][-1]) for e in examples if e.get('history')]
    anchor_set = np.unique(np.asarray(anchors, dtype=np.int64))

    report: dict = {'schema': 'genrec.issue288.diagnosis_r01/1',
                    'valid_sha256': sha(VALID), 'anchors': int(len(anchors)),
                    'distinct_anchors': int(anchor_set.size),
                    'mean_anchor_degree': round(float(degree[anchor_set].mean()), 3),
                    'mean_item_degree': round(float(degree.mean()), 3), 'exports': {}}
    for name, path in SIDS.items():
        if not path.is_file():
            report['exports'][name] = {'status': 'missing', 'path': str(path)}
            continue
        rows = load_sid(path, NODES, (256, 256, 256))
        codes = np.zeros((NODES, 3), dtype=np.int64)
        codes[:, :] = rows[:, :3]
        cat = lm.catalogue_metrics(rows)
        val = lm.validation_metrics(rows, examples)
        counts1 = Counter(codes[:, 0].tolist())
        by_code: dict[int, list[int]] = {}
        for item, code in enumerate(codes[:, 0].tolist()):
            by_code.setdefault(int(code), []).append(int(item))
        sizes = np.array([counts1[c] for c in codes[:, 0].tolist()], dtype=np.int64)
        anchor_sizes = sizes[anchor_set]
        # the #277 exact N terms, and the excluded co-sharer component
        n_terms = []
        co_share = []
        purity_related = purity_total = 0
        fn_hits = fn_total = 0
        for index, example in enumerate(examples):
            history = list(example['history'])
            if not history:
                continue
            seen = list(example.get('seen_history', []))
            target = int(example['target'])
            anchor = int(history[-1])
            excluded = set(map(int, history)) | set(map(int, seen)) | {anchor, target}
            population = NODES - len(excluded)
            code = int(codes[anchor, 0])
            remaining = counts1[code]
            for item in excluded:
                if int(codes[item, 0]) == code:
                    remaining -= 1
            n_terms.append(remaining / population)
            members = by_code[code]
            co_share.append((len(members), len([m for m in members if m in excluded])))
            # the purity and the false-negative rate on every fifth row (a fixed subsample)
            if index % 5:
                continue
            neighbours = adj.get(anchor, set())
            for m in members[:100]:
                if m == anchor:
                    continue
                purity_total += 1
                if m in neighbours or (neighbours & adj.get(m, set())):
                    purity_related += 1
            for m in neighbours:
                if m in excluded or m == anchor:
                    continue
                fn_total += 1
                if int(codes[m, 0]) == code:
                    fn_hits += 1
        rho_bucket_degree = float(np.corrcoef(np.log1p(anchor_sizes), np.log1p(degree[anchor_set]))[0, 1])
        report['exports'][name] = {
            'sid_sha256': sha(path),
            'P': val['l1_positive_hit_rate'], 'N': val['l1_negative_hit_probability'],
            'C': cat['l3_collision'], 'hard_er': val['l1_hard_er'],
            'l2_distinct': cat['distinct_l2_prefixes'], 'l3_distinct': cat['distinct_l3_triples'],
            'l1_bucket_size_mean': round(float(sizes.mean()), 3),
            'l1_bucket_size_median': float(np.median(sizes)),
            'l1_bucket_size_max': int(sizes.max()),
            'anchor_bucket_size_mean': round(float(anchor_sizes.mean()), 3),
            'anchor_bucket_size_median': float(np.median(anchor_sizes)),
            'anchor_bucket_size_max': int(anchor_sizes.max()),
            'N_terms_mean': round(float(np.mean(n_terms)), 6),
            'co_share_mean': round(float(np.mean([c[0] for c in co_share])), 3),
            'co_share_excluded_mean': round(float(np.mean([c[1] for c in co_share])), 3),
            'in_bucket_purity': round(purity_related / purity_total, 5) if purity_total else None,
            'in_bucket_pairs': int(purity_total),
            'related_unshared_fraction': round(1.0 - fn_hits / fn_total, 5) if fn_total else None,
            'related_pairs': int(fn_total),
            'spearman_bucket_vs_anchor_degree': round(rho_bucket_degree, 4),
            'negative_hit_probabilities_by_level': val['negative_hit_probabilities_by_level'],
            'positive_hit_rates_by_level': val['positive_hit_rates_by_level'],
        }
        print(name, 'P', round(val['l1_positive_hit_rate'], 6), 'N', round(val['l1_negative_hit_probability'], 6),
              'C', round(cat['l3_collision'], 6), 'purity', report['exports'][name]['in_bucket_purity'],
              'fn', report['exports'][name]['related_unshared_fraction'], flush=True)

    # B. the curvature's within-stratum information on the distance-1/2 pairs of the triplet table
    curv_pos, curv_neg, ctrl_pos, ctrl_neg = [], [], [], []
    strata = {}
    for name in SHARDS:
        blob = np.load(REPO / 'assets/shards' / name)
        m = blob['centre'].astype(np.int64); b = blob['b'].astype(np.int64); c = blob['c'].astype(np.int64)
        dist = blob['bc_dist_sq'].astype(np.int64); cur = blob['curvature'].astype(np.float64)
        keep = (dist == 1) | (dist == 4)
        m, b, c, dist, cur = m[keep], b[keep], c[keep], dist[keep], cur[keep]
        if not len(m):
            continue
        bucket = np.clip(np.searchsorted(np.quantile(degree, np.linspace(0.1, 0.9, 9)),
                                         degree[m], side='right'), 0, 9)
        label = dist == 1
        key = (bucket * 2 + (dist == 4)).astype(np.int64)
        stride = 1 if label.mean() > 0.5 else 10
        for k, lab, cv, ctrl in zip(key[::stride].tolist(), label[::stride].tolist(), cur[::stride].tolist(),
                                    (degree[b].astype(np.float64) + degree[c].astype(np.float64))[::stride].tolist()):
            strata.setdefault(k, []).append((lab, cv, ctrl))
    pos_c, neg_c, pos_t, neg_t = [], [], [], []
    for k, rows in strata.items():
        for lab, cv, ctrl in rows:
            if lab:
                pos_c.append(cv); pos_t.append(ctrl)
            else:
                neg_c.append(cv); neg_t.append(ctrl)
    report['curvature_within_stratum'] = {
        'pairs': int(len(pos_c) + len(neg_c)), 'edges': int(len(pos_c)), 'non_edges': int(len(neg_c)),
        'auc_curvature': round(auc(pos_c, neg_c), 4),
        'auc_degree_sum_control': round(auc(pos_t, neg_t), 4),
        'note': 'the label is the pair being an observed edge; the strata are the direction pattern, the '
                'distance class and the centre bucket; both AUCs are pooled within the strata',
    }
    (OUT / 'DIAGNOSIS_DATA.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps(report['curvature_within_stratum'], indent=1))


if __name__ == '__main__':
    main()
