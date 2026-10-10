"""Run frozen Stage3 against the TIGER + iter48 residual-contrastive SIDs.

Ablation arm C: TIGER + the historical iter48 behaviour-contrastive
mechanism, ported onto the current TIGER recipe. Only CODE_PATH, RQVAE_VARIANT,
LOG_PATH and SAVE_PATH are wired; every frozen Stage3 field (150 epochs,
early stop disabled, no_eval, skip_test, seed=42, beam=20) stays as the shared
trainer defines it, so arm C and the TIGER baseline are protocol-comparable.
"""
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
    "genrec_stage3_tiger_behavior", f"{STAGE3_DIR}/train_HG-Rec.py"
)
if spec is None or spec.loader is None:
    raise RuntimeError("Could not load Stage3 trainer")
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)
trainer.CODE_PATH = (
    "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/"
    "TIGER_BEHAVIOR_RQ-VAE/item_sids.json"
)
trainer.RQVAE_VARIANT = "tiger_behavior"
trainer.LOG_PATH = (
    "/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/"
    "tiger_behavior/logs/"
)
trainer.SAVE_PATH = (
    "/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/"
    "tiger_behavior/ckpt/"
)
trainer._LAUNCHER["script"] = os.path.abspath(__file__)
trainer._LAUNCHER["log"] = os.path.join(
    trainer.LOG_PATH, "_stage3_launcher.log"
)
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
    print("[tiger_behavior] launching frozen Stage3 torchrun", flush=True)
    trainer._launch_via_torchrun()
