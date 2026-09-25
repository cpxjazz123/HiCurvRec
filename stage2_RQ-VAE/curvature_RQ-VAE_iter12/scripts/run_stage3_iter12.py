"""Run Stage3 with iter12 SIDs through its hard-coded four-GPU launcher."""
import importlib.util
import os
import sys

# The installed torchvision build predates the runtime torch build and tries
# to register an absent nms operator while Transformers imports T5.  Stage3
# is text-only, so define the schema up front; this keeps every torchrun
# worker from failing during module import without changing model behavior.
try:
    import torch

    _TORCHVISION_NMS_LIB = torch.library.Library("torchvision", "DEF")
    _TORCHVISION_NMS_LIB.define(
        "nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor"
    )
except (ImportError, RuntimeError):
    _TORCHVISION_NMS_LIB = None

STAGE3_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/stage3_T5Train"
sys.path.insert(0, STAGE3_DIR)
spec = importlib.util.spec_from_file_location(
    "genrec_stage3_iter12", f"{STAGE3_DIR}/train_HG-Rec.py"
)
if spec is None or spec.loader is None:
    raise RuntimeError("Could not load Stage3 trainer")
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)
trainer.CODE_PATH = (
    "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter12/"
    "item_sids.json"
)
trainer.RQVAE_VARIANT = "iter12_behavior_branching_per_layer_curvature"
trainer.LOG_PATH = (
    "/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/"
    "curvature_RQ-VAE_iter12/logs/"
)
trainer.SAVE_PATH = (
    "/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/"
    "curvature_RQ-VAE_iter12/ckpt/"
)
trainer._LAUNCHER["script"] = os.path.abspath(__file__)
trainer._LAUNCHER["log"] = os.path.join(
    trainer.LOG_PATH, "_stage3_launcher.log"
)
if "RANK" in os.environ:
    trainer.main()
else:
    trainer._launch_via_torchrun()
