"""All diagnostic GPU work goes through the existing frozen four-card launcher."""
from pathlib import Path
from tools.research.stage2.launcher import launch

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'stage2_RQ-VAE').is_dir())
PYTHON='/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9'
ENTRY=ROOT/'stage2_RQ-VAE/_archive/TIGER_gnn_attention_sectional_rq/profile_entry.py'
OUTPUT=ROOT/'results/research/issue252/profiling_baseline'


def main():
    return launch(ENTRY,OUTPUT,python=PYTHON,log_name='baseline_profile.log')


if __name__=='__main__':raise SystemExit(main())
