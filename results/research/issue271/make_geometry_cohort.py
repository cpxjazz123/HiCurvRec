"""Issue #271: build the two-row cohort for the frozen local-geometry evaluator."""
from __future__ import annotations
import hashlib, json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / 'results/research/issue271'
CAND = 'TIGER_global_j_rq'


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def candidate_result() -> Path:
    runs = sorted((REPO / 'results/stage2_RQ-VAE/_runs').glob(f'*/{CAND}/result'))
    for run in reversed(runs):
        if (run / 'item_sids.json').is_file() and (run / 'rqvae_best.pth').is_file():
            return run
    top = REPO / 'results/stage2_RQ-VAE' / CAND
    if (top / 'item_sids.json').is_file() and (top / 'rqvae_best.pth').is_file():
        return top
    raise SystemExit('candidate result not found')


def row(arm: str, result: Path, verified: bool, binding: str):
    sid = result / 'item_sids.json'
    ckpt = result / 'rqvae_best.pth'
    return {'arm': arm, 'sid': str(sid), 'sid_sha256': sha(sid),
            'checkpoint': str(ckpt), 'checkpoint_sha256': sha(ckpt),
            'binding': binding, 'record_verified': verified, 'usable_geometry': True}


tiger = REPO / 'results/stage2_RQ-VAE/TIGER_RQ-VAE'
target = candidate_result()
rows = [row('TIGER_RQ-VAE', tiger, True, 'canonical'),
        row(CAND, target, '_runs' in target.parts, 'immutable_run' if '_runs' in target.parts else 'arm_dir_legacy')]
(OUT / 'geometry_cohort.json').write_text(json.dumps({'rows': rows}, indent=2, sort_keys=True) + '\n')
print(json.dumps({'candidate_result': str(target), 'rows': [r['arm'] for r in rows]}, indent=1))
