"""Issue #287: write the per-round record (the evaluated SID plus the training summary).

Usage: python3.10 results/research/issue287/record_round.py <tag> <run_dir> <sid_path>
The record is the union of the frozen #277 evaluation (P, N, C) and the run's own training
summary (the final collision, the epochs and the wall time), so every round is comparable.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, '.')
sys.path.insert(0, 'results/research/issue287')
from eval287 import evaluate, sha, VALID

REPO = Path('.').resolve()
OUT = REPO / 'results/research/issue287'


def training_summary(run_dir: Path) -> dict:
    metrics = run_dir / 'result' / 'logs' / 'training_metrics.jsonl'
    if not metrics.exists():
        metrics = run_dir / 'logs' / 'training_metrics.jsonl'
    epochs = 0; traj = []; events = []; final_collision = None
    for line in metrics.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        event = row.get('event')
        if event in ('epoch', 'train'):
            epochs = max(epochs, int(row.get('epoch', 0)))
            traj.append(row)
        elif event == 'ckpt_saved' and row.get('reason') == 'final_epoch':
            final_collision = row.get('collision')
        elif event:
            events.append(row)
    times = [float(r['epoch_time_s']) for r in traj if r.get('epoch_time_s')]
    return {'n_epochs_recorded': epochs,
            'final_collision': final_collision,
            'epoch_time_s_mean': round(sum(times) / len(times), 4) if times else None,
            'total_wall_time_s': round(sum(times), 3),
            'events': [e.get('event') for e in events],
            'metrics_sha256': sha(metrics)}


def main() -> None:
    tag, run_dir, sid_path = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    examples = pq.read_table(VALID, columns=['history', 'seen_history', 'target']).to_pylist()
    rec = evaluate(sid_path, examples)
    out = {'round': tag, 'run_dir': str(run_dir), 'sid_path': str(sid_path),
           'sid_sha256': rec['sid_sha256'],
           'P': rec['P'], 'N': rec['N'], 'C': rec['C'],
           'passes_P': rec['passes_P'], 'passes_N': rec['passes_N'], 'passes_C': rec['passes_C'],
           'passes_all': rec['passes_all'],
           'hard_er': rec['hard_er'], 'diagnostics': rec['diagnostics'],
           'training': training_summary(run_dir),
           'thresholds': {'P_gt': 0.08097285816257246, 'N_lt': 0.008752316503966327,
                          'C_lt': 0.1511367795989751}}
    path = OUT / 'results' / f'{tag}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({'round': tag, 'P': rec['P'], 'N': rec['N'], 'C': rec['C'],
                      'passes_all': rec['passes_all']}, indent=1))


if __name__ == '__main__':
    main()
