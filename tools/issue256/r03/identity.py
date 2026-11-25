"""Immutable identity of the preregistered role-conditioned level-1 candidate."""
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p/'stage2_RQ-VAE').is_dir() and (p/'tools').is_dir())
ROUND = 'r03'
ARM = 'TIGER_role_conditioned_l1_rq'
RUN_ID = 'issue256_r03_formal'
SOURCE = ROOT/'stage2_RQ-VAE'/ARM
EVIDENCE = ROOT/'results/research/issue256/rounds/r03'
RUN = ROOT/'results/stage2_RQ-VAE/_runs'/RUN_ID/ARM
RESULT = RUN/'result'
PYTHON = '/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9'
