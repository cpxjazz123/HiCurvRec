"""Finalize one finished Stage2 run: checkpoint-bound SID export, layered SID
metrics record (before any Stage3 score is read) and the archive binding.

Usage: finalize_stage2.py <run_id>
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.issue252.archive_audit import create_binding, sha256  # noqa: E402
from tools.issue252.layer_metrics import record_layers  # noqa: E402
from tools.issue252.run_workspace import load_run  # noqa: E402

ARM = 'TIGER_three_expert_attention_rq'
PYTHON = '/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9'


def main():
    run_id = sys.argv[1]
    run = load_run(ROOT, run_id, ARM)
    result = Path(run['result'])
    tree = Path(run['source'])
    sid = result/'item_sids.json'
    checkpoint = result/'rqvae_best.pth'
    manifest = result/'export_manifest.json'
    binding = result/'archive_binding.json'
    completed = subprocess.run(
        [PYTHON, '-m', 'tools.research.stage2.export_check', '--sid', str(sid),
         '--tree', str(tree), '--rewrite', '--print-json'],
        cwd=str(ROOT), capture_output=True, text=True, timeout=3600,
        env={**__import__('os').environ, 'PYTHONPATH': str(ROOT)})
    if completed.returncode:
        raise SystemExit('export check failed: '+(completed.stderr or completed.stdout)[-2000:])
    print(completed.stdout[-1200:], flush=True)
    create_binding(tree, checkpoint, sid, manifest, output=binding, repo=ROOT)
    layer = record_layers(sid)
    print('layer record', layer['path'], layer['sha256'], flush=True)
    receipt = json.loads(binding.read_text())
    payload = json.loads(sid.read_text())
    triples = {tuple(value[:3]) for value in payload.values()}
    print(json.dumps({'run_id': run_id, 'sid_sha256': sha256(sid),
                      'checkpoint_sha256': sha256(checkpoint),
                      'binding_sha256': sha256(binding),
                      'manifest_sha256': sha256(manifest),
                      'items': len(payload), 'distinct_l3': len(triples),
                      'l3_collision': 1.-len(triples)/len(payload),
                      'layer': layer['path']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
