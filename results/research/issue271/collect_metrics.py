"""Issue #271 post-run SID and structure diagnostics (reuse, no new public tool)."""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
SID_QUALITY = REPO / 'tools/research/stage2/sid_quality.py'
PYTHON = '/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9'
K269 = REPO / 'results/stage2_RQ-VAE/_runs/stage2_bba78cf7c160467da1c8ae7de9b6e26f/TIGER_calibrated_kappa_rq/result/item_sids.json'


def entropy(codes):
    _, counts = np.unique(codes, axis=0, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def load(path):
    d = json.loads(Path(path).read_text())
    return np.asarray([d[str(i)] for i in range(24587)], dtype=np.int64)


def main() -> None:
    result = Path(sys.argv[1])
    sid_path = result / 'item_sids.json'
    codes = load(sid_path)
    three = codes[:, :3]
    unique3 = len(np.unique(three, axis=0))
    unique2 = len(np.unique(three[:, :2], axis=0))
    unique1 = len(np.unique(three[:, :1], axis=0))
    payload = {
        'schema': 'genrec.issue271.sid_metrics/1',
        'sid': str(sid_path),
        'items': int(len(codes)),
        'collision': 1.0 - unique3 / len(codes),
        'unique_three': int(unique3), 'unique_two': int(unique2), 'unique_one': int(unique1),
        'l2_per_l1': unique2 / unique1, 'three_per_two': unique3 / unique2,
        'l3_refinement_gain_bits': entropy(three) - entropy(three[:, :2]),
        'l3_group_size_mean': len(codes) / unique3,
        'l2_group_size_mean': len(codes) / unique2,
        'code_usage': [int(len(np.unique(codes[:, level]))) for level in range(4)],
    }
    proc = subprocess.run([str(PYTHON), str(SID_QUALITY), str(sid_path)],
                          capture_output=True, text=True, timeout=1800)
    try:
        payload['sid_quality'] = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        payload['sid_quality'] = {'exit': proc.returncode, 'stdout': proc.stdout[-500:],
                                  'stderr': proc.stderr[-500:]}
    if K269.is_file():
        old = load(K269)
        old_three = old[:, :3]
        old_two = old_three[:, :2]
        new_two = three[:, :2]
        old_groups = {tuple(row): i for i, row in enumerate(np.unique(old_two, axis=0))}
        new_groups = {tuple(row): i for i, row in enumerate(np.unique(new_two, axis=0))}
        pairs = [(old_groups[tuple(a)], new_groups[tuple(b)]) for a, b in zip(old_two, new_two)]
        old_to_new = {}
        for a, b in pairs:
            old_to_new.setdefault(a, set()).add(b)
        payload['vs_k269'] = {
            'k269_sid': str(K269),
            'l2_groups_k269': len(old_groups), 'l2_groups_candidate': len(new_groups),
            'k269_groups_split': int(sum(1 for v in old_to_new.values() if len(v) > 1)),
            'candidate_groups_merging': int(len(new_groups) - len({b for v in old_to_new.values() for b in v})),
        }
    out = result / 'issue271_sid_metrics.json'
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: payload[k] for k in ('collision', 'unique_three', 'unique_two', 'unique_one',
                                               'l3_refinement_gain_bits', 'code_usage')}, indent=1))


if __name__ == '__main__':
    main()
