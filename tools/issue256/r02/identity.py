"""Immutable identity of the preregistered native-block refresh candidate."""
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p/'stage2_RQ-VAE').is_dir() and (p/'tools').is_dir())
ROUND = 'r02'
ARM = 'TIGER_periodic_budget_prefix_rq'
RUN_ID = 'issue256_r02_formal'
SOURCE = ROOT/'stage2_RQ-VAE'/ARM
EVIDENCE = ROOT/'results/research/issue256/rounds/r02'
RUN = ROOT/'results/stage2_RQ-VAE/_runs'/RUN_ID/ARM
RESULT = RUN/'result'
PYTHON = '/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9'
