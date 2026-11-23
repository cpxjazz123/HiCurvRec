"""Only the controlled four-rank validation; never dispatch formal epochs."""
from pathlib import Path
import sys

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'stage2_RQ-VAE').is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from tools.research.stage2.launcher import launch

PYTHON='/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9'
ENTRY=ROOT/'stage2_RQ-VAE/TIGER_gnn_attention_sectional_rq/validation_entry.py'
OUTPUT=ROOT/'results/research/issue251/runtime'


def main():
    return launch(ENTRY,OUTPUT,python=PYTHON,log_name='controlled_validation.log')

if __name__=='__main__':raise SystemExit(main())
