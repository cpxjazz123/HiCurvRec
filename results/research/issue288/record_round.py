"""Issue #288: write one round's record (the frozen #277 evaluation plus the diagnosis metrics).

Usage: python3.10 results/research/issue288/record_round.py <tag> <run_dir> <sid_path>
The evaluation is the frozen #277 evaluator on the 57,439 validation rows; the diagnosis metrics
(the in-bucket purity, the unshared related pairs, the bucket sizes, the per-level distinctness)
are computed with the same definitions as DIAGNOSIS.md.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, '.')
from tools.issue252 import layer_metrics as lm
from tools.issue252.final_evaluation import load_sid

REPO = Path('.').resolve()
OUT = REPO / 'results/research/issue288'
VALID = REPO / 'results/stage0_build_parquet/valid.parquet'
NODES = 24587
THRESHOLDS = {'P_gt': 0.08097285816257246, 'N_lt': 0.008752316503966327, 'C_lt': 0.1511367795989751}


def sha(p):
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
    return adj


def diagnosis(codes, examples, adj) -> dict:
    counts1 = Counter(codes[:, 0].tolist())
    by_code: dict[int, list[int]] = {}
    for item, code in enumerate(codes[:, 0].tolist()):
        by_code.setdefault(int(code), []).append(int(item))
    sizes = np.array([counts1[c] for c in codes[:, 0].tolist()], dtype=np.int64)
    purity_related = purity_total = fn_hits = fn_total = 0
    for index, example in enumerate(examples):
        history = list(example['history'])
        if not history or index % 5:
            continue
        seen = list(example.get('seen_history', []))
        anchor = int(history[-1])
        excluded = set(map(int, history)) | set(map(int, seen)) | {anchor, int(example['target'])}
        neighbours = adj.get(anchor, set())
        for m in by_code[int(codes[anchor, 0])][:100]:
            if m == anchor:
                continue
            purity_total += 1
            if m in neighbours or (neighbours & adj.get(m, set())):
                purity_related += 1
        for m in neighbours:
            if m in excluded or m == anchor:
                continue
            fn_total += 1
            if int(codes[m, 0]) == int(codes[anchor, 0]):
                fn_hits += 1
    return {'l1_bucket_size_mean': round(float(sizes.mean()), 3),
            'l1_bucket_size_median': float(np.median(sizes)),
            'l1_bucket_size_max': int(sizes.max()),
            'in_bucket_purity': round(purity_related / purity_total, 5) if purity_total else None,
            'in_bucket_pairs': int(purity_total),
            'related_unshared_fraction': round(1.0 - fn_hits / fn_total, 5) if fn_total else None,
            'related_pairs': int(fn_total)}


def main() -> None:
    tag, run_dir, sid_path = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    examples = pq.read_table(VALID, columns=['history', 'seen_history', 'target']).to_pylist()
    rows = load_sid(sid_path, NODES, (256, 256, 256))
    codes = np.zeros((NODES, 3), dtype=np.int64)
    codes[:, :] = rows[:, :3]
    cat = lm.catalogue_metrics(rows)
    val = lm.validation_metrics(rows, examples)
    P = float(val['l1_positive_hit_rate']); N = float(val['l1_negative_hit_probability'])
    C = float(cat['l3_collision'])
    rec = {'round': tag, 'run_dir': str(run_dir), 'sid_path': str(sid_path), 'sid_sha256': sha(sid_path),
           'P': P, 'N': N, 'C': C,
           'passes_P': P > THRESHOLDS['P_gt'], 'passes_N': N < THRESHOLDS['N_lt'],
           'passes_C': C < THRESHOLDS['C_lt'],
           'passes_all': P > THRESHOLDS['P_gt'] and N < THRESHOLDS['N_lt'] and C < THRESHOLDS['C_lt'],
           'hard_er': float(val['l1_hard_er']),
           'diagnostics': {'distinct_l2_prefixes': cat['distinct_l2_prefixes'],
                           'distinct_l3_triples': cat['distinct_l3_triples'],
                           'l3_refinement_gain_bits': cat['l3_refinement_gain_bits'],
                           'positive_hit_rates_by_level': val['positive_hit_rates_by_level'],
                           'negative_hit_probabilities_by_level': val['negative_hit_probabilities_by_level']},
           'diagnosis': diagnosis(codes, examples, graph()),
           'thresholds': THRESHOLDS}
    path = OUT / 'results' / f'{tag}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=1) + '\n')
    print(json.dumps({'round': tag, 'P': P, 'N': N, 'C': C, 'passes_all': rec['passes_all'],
                      'purity': rec['diagnosis']['in_bucket_purity'],
                      'fn': rec['diagnosis']['related_unshared_fraction']}, indent=1))


if __name__ == '__main__':
    main()
