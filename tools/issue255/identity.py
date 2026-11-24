"""One approved corrected Issue255 generation; the first run remains invalid evidence."""
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'tools/research/stage2/contract.py').is_file())
ARM = 'TIGER_prefix_curvature_gate_rq'
RUN_ID = 'issue255_formal_r1'
SOURCE = ROOT / 'stage2_RQ-VAE' / ARM
RUN = ROOT / 'results/stage2_RQ-VAE/_runs' / RUN_ID / ARM
RESULT = RUN / 'result'
EVIDENCE = ROOT / 'results/research/issue255'
INVALID_RUN_ID = 'issue255_formal'
