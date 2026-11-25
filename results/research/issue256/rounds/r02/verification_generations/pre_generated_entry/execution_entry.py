"""Framework path adapter for one immutable formal Stage2 generation."""
from pathlib import Path
import sys

SOURCE = Path(__file__).resolve().parent
REPO = next(p for p in SOURCE.parents
            if (p / "tools/research/stage2/contract.py").is_file())
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if sys.path[0] != str(SOURCE):
    sys.path.insert(0, str(SOURCE))
from tools.issue252.run_workspace import load_run
from tools.research.stage2 import launcher

run = load_run(REPO, 'issue256_r02_formal', 'TIGER_periodic_budget_prefix_rq')
RESULT = Path(run["result"])
if __name__ == "__main__":
    if not launcher.is_child_rank():
        raise SystemExit(launcher.launch(Path(__file__), RESULT, python=sys.executable))
    import train_rqvae as t
    t.RUN_ID = run["run_id"]
    t.RESULT_DIR = RESULT
    t.LOG_DIR = RESULT / "logs"
    t.METRICS_PATH = t.LOG_DIR / "training_metrics.jsonl"
    t.OUTPUT_SID = RESULT / "sids_for_hgrec.npy"
    t.OUTPUT_JSON = RESULT / "item_sids.json"
    t.CHECKPOINT = RESULT / "rqvae_best.pth"
    t.main()
