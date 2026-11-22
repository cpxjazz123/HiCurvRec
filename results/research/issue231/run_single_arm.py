"""Issue #231, owner-scoped: train the REAL Graphlet arm only, four-card DDP, serial.

The shuffle control answers a different question (whether the gain comes from the right roles)
and is deferred until the first comparison is settled. This driver therefore launches exactly
one arm through the framework's only launch path, which itself enforces four-card DDP and the
one-at-a-time lock.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tools.research.stage2 import launcher  # noqa: E402

ARM = "TIGER_sid_graphlet_rq"
ENTRY = Path(__file__).resolve().parents[3] / "stage2_RQ-VAE" / ARM / "train_rqvae.py"
RESULT = Path(__file__).resolve().parents[3] / "results" / "stage2_RQ-VAE" / ARM

if __name__ == "__main__":
    print("== launching", ARM, flush=True)
    rc = launcher.launch(ENTRY, RESULT, env_extra={"CUDA_VISIBLE_DEVICES": "0,1,2,3"})
    print("== finished", ARM, "rc=", rc, flush=True)
    sys.exit(int(rc))
