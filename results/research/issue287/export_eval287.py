"""Issue #287: the checkpoint-bound export and the unified evaluation of the final SID."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, '.')
from tools.research.stage2 import export_check

import sys
RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    'results/stage2_RQ-VAE/_runs/stage2_c85aa9d946324638b87e74571f5026c4/TIGER_l1_pairrank_rq')
SRC = RUN / 'src'
RESULT = RUN / 'result'
manifest = export_check.check(SRC, RESULT / 'rqvae_best.pth', RESULT / 'item_sids.json', rewrite=True)
export_check.write_manifest(manifest, RESULT / 'export_manifest.json')
print(json.dumps(manifest, indent=1))
