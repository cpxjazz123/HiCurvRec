"""Issue #287: the run trajectory and the frozen #277 prediction for the exported SID."""
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


def sha(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    run = Path(sys.argv[1])
    sid_path = Path(sys.argv[2])
    metrics = run / 'result' / 'logs' / 'training_metrics.jsonl'
    if not metrics.exists():
        metrics = run / 'logs' / 'training_metrics.jsonl'
    traj = []
    events = []
    for line in open(metrics):
        rec = json.loads(line)
        events.append(rec.get('event'))
        if rec.get('event') == 'train':
            traj.append({k: rec.get(k) for k in
                          ('epoch', 'loss', 'recon', 'epoch_time_s', 'cumulative_time_s', 'collision',
                           'raw_unique', 'pair_loss', 'pair_ranking_accuracy',
                           'pair_margin_satisfaction_rate', 'pair_mean_distance_gap', 'pair_weight')})
    evals = [t for t in traj if t.get('collision') is not None]
    # the frozen #277 predictor (diagnostic only)
    examples = pq.read_table(REPO / 'results/stage0_build_parquet/valid.parquet',
                             columns=['history', 'seen_history', 'target']).to_pylist()
    rows = load_sid(sid_path, 24587, (256, 256, 256))
    cat = lm.catalogue_metrics(rows)
    val = lm.validation_metrics(rows, examples)
    features = {
        'l1_pos_hit': float(val['l1_positive_hit_rate']),
        'l1_neg_hit': float(val['l1_negative_hit_probability']),
        'l1_hard_er': float(val['l1_hard_er']),
        'l2_distinct': float(cat['distinct_l2_prefixes']),
        'l3_collision': float(cat['l3_collision']),
        'l3_refine_bits': float(cat['l3_refinement_gain_bits'])}
    predictor = json.load(open(REPO / 'results/research/issue277/PREDICTOR.json'))
    order = predictor['features']
    x = np.array([features[k] for k in order], dtype=float)
    mean = np.array(predictor['scaler']['mean']); scale = np.array(predictor['scaler']['scale'])
    coef = np.array(predictor['coef_standardised']); intercept = predictor['fit_procedure']['ridge_intercept_standardised']
    pred277 = float(intercept + coef @ ((x - mean) / scale))
    out = {
        'schema': 'genrec.issue287.summary/1',
        'run_dir': str(run), 'metrics_sha256': sha(metrics),
        'events': sorted(set(events)),
        'n_epochs_recorded': len(traj), 'n_eval_epochs': len(evals),
        'first_eval': evals[0] if evals else None,
        'last_eval': evals[-1] if evals else None,
        'epoch_time_s_mean': round(float(np.mean([t['epoch_time_s'] for t in traj if t.get('epoch_time_s')])), 4),
        'total_wall_time_s': traj[-1].get('cumulative_time_s') if traj else None,
        'final_collision': evals[-1].get('collision') if evals else None,
        'collision_trajectory': [{'epoch': t['epoch'], 'collision': t['collision']} for t in evals],
        'pair_ranking_accuracy_last': evals[-1].get('pair_ranking_accuracy') if evals else None,
        'pair_weight': evals[-1].get('pair_weight') if evals else None,
        'features': features,
        'frozen_277_prediction': pred277,
        'frozen_277_predictor_sha256': sha(REPO / 'results/research/issue277/PREDICTOR.json'),
        'sid_path': str(sid_path), 'sid_sha256': sha(sid_path),
        'checkpoint': str(run / 'rqvae_best.pth'),
        'checkpoint_sha256': sha(run / 'rqvae_best.pth') if (run / 'rqvae_best.pth').is_file() else None,
    }
    (OUT / 'SUMMARY_287.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k: out[k] for k in ('n_epochs_recorded', 'n_eval_epochs', 'final_collision',
                                          'epoch_time_s_mean', 'total_wall_time_s', 'features',
                                          'frozen_277_prediction', 'sid_sha256')}, indent=1))


if __name__ == '__main__':
    main()
