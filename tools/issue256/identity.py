"""One preregistered hypothesis; later rounds receive new names and workspaces."""
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'tools/research/stage2/contract.py').is_file())
ROUND = 'r01'
ARM = 'TIGER_module_budget_prefix_rq'
RUN_ID = 'issue256_r01_formal'
SOURCE = ROOT / 'stage2_RQ-VAE' / ARM
EVIDENCE = ROOT / 'results/research/issue256/rounds' / ROUND
RUN = ROOT / 'results/stage2_RQ-VAE/_runs' / RUN_ID / ARM
RESULT = RUN / 'result'
PYTHON = '/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9'
