"""Run Stage3 with iter7 SIDs through its hard-coded four-GPU launcher."""
import importlib.util
import os
import sys

STAGE3_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/stage3_T5Train"
sys.path.insert(0, STAGE3_DIR)
spec = importlib.util.spec_from_file_location(
    "genrec_stage3_iter7", f"{STAGE3_DIR}/train_HG-Rec.py"
)
if spec is None or spec.loader is None:
    raise RuntimeError("Could not load Stage3 trainer")
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)
trainer.CODE_PATH = (
    "/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter7/"
    "results/item_sids.json"
)
trainer.RQVAE_VARIANT = "iter7_wide_curriculum_warmstart"
trainer.LOG_PATH = (
    "/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/"
    "curvature_RQ-VAE_iter7/logs/"
)
trainer.SAVE_PATH = (
    "/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/"
    "curvature_RQ-VAE_iter7/ckpt/"
)
trainer._LAUNCHER["script"] = os.path.abspath(__file__)
trainer._LAUNCHER["log"] = os.path.join(
    trainer.LOG_PATH, "_stage3_launcher.log"
)
if "RANK" in os.environ:
    trainer.main()
else:
    trainer._launch_via_torchrun()