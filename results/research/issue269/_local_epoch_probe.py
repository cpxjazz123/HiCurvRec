"""Single-card epoch probe on the real training path (diagnostic, no public API)."""
import os, sys
from pathlib import Path

os.environ.setdefault("GENREC_SINGLE_GPU", "1")
REPO = Path(__file__).resolve().parents[3]
TREE = REPO / sys.argv[3] if len(sys.argv) > 3 else REPO / "stage2_RQ-VAE/TIGER_calibrated_kappa_rq"
sys.path.insert(0, str(TREE))
import train_rqvae as t

t.EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 50
if len(sys.argv) > 2 and int(sys.argv[2]) > 0:
    t.SINGLE_CARD_BATCH = int(sys.argv[2])
out = REPO / "results/research/issue269/_scratch_epoch_probe"
out.mkdir(parents=True, exist_ok=True)
t.RESULT_DIR = out
t.LOG_DIR = out / "logs"
t.LOG_DIR.mkdir(parents=True, exist_ok=True)
t.METRICS_PATH = t.LOG_DIR / "training_metrics.jsonl"
t.OUTPUT_SID = out / "sids_for_hgrec.npy"
t.OUTPUT_JSON = out / "item_sids.json"
t.CHECKPOINT = out / "rqvae_best.pth"
t.main()
