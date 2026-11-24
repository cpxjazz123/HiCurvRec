"""Source-fixed issue253 steady_profile; only the sanctioned four-card launcher."""
import os
from pathlib import Path
from tools.research.stage2.launcher import launch

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p/'tools/research/stage2/contract.py').is_file())
PYTHON = '/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9'
ENTRY = ROOT/'stage2_RQ-VAE/TIGER_three_expert_attention_rq/steady_profile_entry.py'
OUTPUT = ROOT/'results/research/issue253/steady_profile'

def main():
    return launch(ENTRY, OUTPUT/('launcher_'+str(os.getpid())), python=PYTHON,
                  log_name='steady_profile.log')

if __name__ == '__main__':
    raise SystemExit(main())
