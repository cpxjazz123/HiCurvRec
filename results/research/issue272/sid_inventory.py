"""Issue #272: SID capacity metrics for the frozen reference arms and the candidate."""
from __future__ import annotations
import hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
PY2 = '/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9'
SIDS = {
    'TIGER_RQ-VAE': REPO / 'results/stage2_RQ-VAE/TIGER_RQ-VAE/item_sids.json',
    'TIGER_euclid_local_curve_supervision_rq': REPO / 'results/stage2_RQ-VAE/_runs/stage2_4891fa8df3e843839de770e52e156892/TIGER_euclid_local_curve_supervision_rq/result/item_sids.json',
    'TIGER_calibrated_kappa_rq': REPO / 'results/stage2_RQ-VAE/_runs/stage2_bba78cf7c160467da1c8ae7de9b6e26f/TIGER_calibrated_kappa_rq/result/item_sids.json',
    'TIGER_global_j_rq': REPO / 'results/stage2_RQ-VAE/_runs/stage2_cc9a31a9325b439f98c8b1779aaa32aa/TIGER_global_j_rq/result/item_sids.json',
}


def entropy(codes):
    _, counts = np.unique(codes, axis=0, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def sha(p: Path) -> str:
    h = hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()


def main():
    out = {'schema': 'genrec.issue272.sid_inventory/1', 'arms': {}}
    for arm, sid in SIDS.items():
        d = json.loads(sid.read_text())
        codes = np.asarray([d[str(i)] for i in range(24587)], dtype=np.int64)
        three = codes[:, :3]
        row = {
            'sid': str(sid), 'sid_sha256': sha(sid), 'items': int(len(codes)),
            'collision': 1 - len(np.unique(three, axis=0)) / len(codes),
            'unique_L1': int(len(np.unique(three[:, :1], axis=0))),
            'unique_L2': int(len(np.unique(three[:, :2], axis=0))),
            'unique_L3': int(len(np.unique(three, axis=0))),
            'L3_gain_bits': entropy(three) - entropy(three[:, :2]),
            'code_usage': [int(len(np.unique(codes[:, k]))) for k in range(4)],
            'l3_group_size_mean': len(codes) / len(np.unique(three, axis=0)),
        }
        proc = subprocess.run([PY2, '-m', 'tools.research.stage2.sid_quality', str(sid)],
                              capture_output=True, text=True, timeout=1800, cwd=str(REPO))
        try:
            row['sid_quality'] = json.loads(proc.stdout.strip().splitlines()[-1])
        except Exception:
            row['sid_quality'] = {'exit': proc.returncode, 'stdout': proc.stdout[-400:],
                                  'stderr': proc.stderr[-400:]}
        out['arms'][arm] = row
        print(arm, 'L2', row['unique_L2'], 'L3', row['unique_L3'], 'rank',
              row['sid_quality'].get('effective_rank'), 'quant', row['sid_quality'].get('quant_error'), flush=True)
    (REPO / 'results/research/issue272/sid_inventory.json').write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n")


if __name__ == '__main__':
    main()
