"""Issue #287: the unified #277 evaluation of a checkpoint-exported SID (P, N, C) and the thresholds.

Usage:  python3.10 results/research/issue287/eval287.py <sid_path> [<sid_path> ...]
Every SID is evaluated with the frozen #275/#277 evaluator (tools.issue252.layer_metrics) on the
57,439 validation rows; the three frozen thresholds of issue #287 are applied and reported.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, '.')
from tools.issue252 import layer_metrics as lm
from tools.issue252.final_evaluation import load_sid

REPO = Path('.').resolve()
OUT = REPO / 'results/research/issue287'
THRESHOLDS = {'P_gt': 0.08097285816257246, 'N_lt': 0.008752316503966327, 'C_lt': 0.1511367795989751}
STRETCH = {'P_ge': 0.09253295, 'C_band': [0.0303, 0.0871]}
VALID = REPO / 'results/stage0_build_parquet/valid.parquet'


def sha(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def evaluate(sid_path: Path, examples) -> dict:
    rows = load_sid(sid_path, 24587, (256, 256, 256))
    cat = lm.catalogue_metrics(rows)
    val = lm.validation_metrics(rows, examples)
    P = float(val['l1_positive_hit_rate'])
    N = float(val['l1_negative_hit_probability'])
    C = float(cat['l3_collision'])
    return {
        'sid_path': str(sid_path), 'sid_sha256': sha(sid_path),
        'P': P, 'N': N, 'C': C, 'hard_er': float(val['l1_hard_er']),
        'passes_P': bool(P > THRESHOLDS['P_gt']),
        'passes_N': bool(N < THRESHOLDS['N_lt']),
        'passes_C': bool(C < THRESHOLDS['C_lt']),
        'passes_all': bool(P > THRESHOLDS['P_gt'] and N < THRESHOLDS['N_lt'] and C < THRESHOLDS['C_lt']),
        'stretch_P': bool(P >= STRETCH['P_ge']),
        'stretch_C_band': bool(STRETCH['C_band'][0] <= C <= STRETCH['C_band'][1]),
        'diagnostics': {
            'distinct_l2_prefixes': cat['distinct_l2_prefixes'],
            'distinct_l3_triples': cat['distinct_l3_triples'],
            'l3_refinement_gain_bits': cat['l3_refinement_gain_bits'],
            'prefix_entropy_bits': cat['prefix_entropy_bits'],
            'codebook_occupancy': cat['codebook_occupancy'],
            'effective_rows': val['effective_rows'],
            'positive_hit_rates_by_level': val['positive_hit_rates_by_level'],
            'negative_hit_probabilities_by_level': val['negative_hit_probabilities_by_level'],
            'conditional_prefix_survival_difference': val['conditional_prefix_survival_difference'],
            'conditional_log_enrichment_bits': val['conditional_log_enrichment_bits']}}


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]]
    examples = pq.read_table(VALID, columns=['history', 'seen_history', 'target']).to_pylist()
    target = OUT / 'EVAL_287.json'
    out = {'schema': 'genrec.issue287.eval/1', 'thresholds': THRESHOLDS, 'stretch': STRETCH,
           'valid_sha256': sha(VALID), 'evaluator': {'path': str(Path(lm.__file__)), 'sha256': sha(Path(lm.__file__))},
           'rows': []}
    if target.is_file():
        previous = json.loads(target.read_text())
        out['rows'] = [r for r in previous.get('rows', [])]
    for p in paths:
        if not p.is_file():
            out['rows'].append({'sid_path': str(p), 'status': 'missing'})
            continue
        rec = evaluate(p, examples)
        out['rows'] = [r for r in out['rows'] if r.get('sid_path') != rec['sid_path']]
        out['rows'].append(rec)
        print(json.dumps({k: rec[k] for k in ('sid_path', 'P', 'N', 'C', 'passes_all')}, indent=1), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
