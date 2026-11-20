"""Launch one Stage3 run for a SID whose Stage2 source tree no longer exists.

The MCP ``stage3_start`` gate re-exports the SID from its Stage 2 tree.  Arms trained
before the trees were archived cannot satisfy that gate, and the owner's 2026-11-20
instruction is to enter Stage 3 unconditionally, so this harness mirrors the MCP's own
harness (same frozen protocol, same paths, same launcher contract) without the gate.

    python run_stage3_direct.py <item_sids.json> <variant>
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
STAGE3_DIR = REPO / "stage3_T5Train"
RESULT_ROOT = REPO / "results" / "stage3_T5Train"

# torchrun re-executes this file without arguments for every rank, so the run is
# identified through the environment (inherited by all workers).
if os.environ.get("GENREC_STAGE3_SID") and os.environ.get("GENREC_STAGE3_VARIANT"):
    sid = Path(os.environ["GENREC_STAGE3_SID"]).resolve()
    variant = os.environ["GENREC_STAGE3_VARIANT"]
else:
    sid, variant = Path(sys.argv[1]).resolve(), sys.argv[2]
if not sid.is_file():
    raise SystemExit(f"no SID at {sid}")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(STAGE3_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE3_DIR))

spec = importlib.util.spec_from_file_location("genrec_stage3_direct", STAGE3_DIR / "train_HG-Rec.py")
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)
trainer.CODE_PATH = str(sid)
trainer.RQVAE_VARIANT = variant
log_dir = RESULT_ROOT / variant / "logs"
trainer.LOG_PATH = str(log_dir) + os.sep
trainer.SAVE_PATH = str(RESULT_ROOT / variant / "ckpt") + os.sep
trainer.PER_USER_DUMP = True
trainer._LAUNCHER["script"] = os.path.abspath(__file__)
trainer._LAUNCHER["log"] = str(log_dir / "_stage3_launcher.log")
log_dir.mkdir(parents=True, exist_ok=True)
os.environ.update({
    "NCCL_IB_DISABLE": "1",
    "NCCL_P2P_DISABLE": "1",
    "NCCL_SHM_DISABLE": "1",
    "NCCL_TIMEOUT": "3600",
    "TORCH_NCCL_BLOCKING_WAIT": "1",
})

if "RANK" in os.environ or os.environ.get(trainer.LAUNCHED_ENV):
    try:
        trainer.main()
    finally:
        trainer.cleanup_distributed()
else:
    trainer._launch_via_torchrun()
