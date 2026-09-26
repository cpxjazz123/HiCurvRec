"""Run Stage3 with iter19 SIDs through its hard-coded four-GPU launcher."""
import importlib.util
import os
import sys

try:
    import torch
    _TORCHVISION_NMS_LIB = torch.library.Library("torchvision", "DEF")
    _TORCHVISION_NMS_LIB.define(
        "nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor"
    )
except (ImportError, RuntimeError):
    pass

STAGE3_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/stage3_T5Train"
sys.path.insert(0, STAGE3_DIR)
spec = importlib.util.spec_from_file_location(
    "genrec_stage3_iter19", f"{STAGE3_DIR}/train_HG-Rec.py"
)
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)
trainer.CODE_PATH = (
    "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/"
    "curvature_RQ-VAE_iter19/item_sids.json"
)
trainer.RQVAE_VARIANT = "iter19_reciprocal_sinkhorn_eps"
trainer.LOG_PATH = (
    "/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/"
    "curvature_RQ-VAE_iter19/logs/"
)
trainer.SAVE_PATH = (
    "/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/"
    "curvature_RQ-VAE_iter19/ckpt/"
)
trainer._LAUNCHER["script"] = os.path.abspath(__file__)
trainer._LAUNCHER["log"] = os.path.join(trainer.LOG_PATH, "_stage3_launcher.log")
# Keep the stage3 launcher's setdefault calls aligned with CLAUDE.md.
os.environ.update(
    {
        "NCCL_IB_DISABLE": "1",
        "NCCL_P2P_DISABLE": "1",
        "NCCL_SHM_DISABLE": "1",
        "NCCL_TIMEOUT": "3600",
        "TORCH_NCCL_BLOCKING_WAIT": "1",
    }
)

if "RANK" in os.environ:
    trainer.main()
else:
    trainer._launch_via_torchrun()
