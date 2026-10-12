"""Stage3 launcher for the directional-contrastive Stage2 arms."""
import importlib.util
import os
import shutil
import sys

ARM = "resample_proj_dir"
ARM_CODE_PATHS = {
    "resample": "resample/item_sids.json",
    "resample_proj_sym": "resample_proj_sym/item_sids.json",
    "resample_proj_dir": "resample_proj_dir/item_sids.json",
}
if ARM not in ARM_CODE_PATHS:
    raise ValueError(f"Unsupported directional Stage2 arm: {ARM}")
STAGE3_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/stage3_T5Train"
sys.path.insert(0, STAGE3_DIR)
spec = importlib.util.spec_from_file_location(
    f"genrec_stage3_directional_{ARM}", f"{STAGE3_DIR}/train_HG-Rec.py"
)
if spec is None or spec.loader is None:
    raise RuntimeError("Could not load Stage3 trainer")
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)
trainer.CODE_PATH = (
    "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/"
    f"TIGER_BEHAVIOR_SAMPLE_RQ-VAE/{ARM_CODE_PATHS[ARM]}"
)
trainer.RQVAE_VARIANT = f"tiger_sample_{ARM}"
trainer.LOG_PATH = (
    "/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/"
    f"{trainer.RQVAE_VARIANT}/logs/"
)
trainer.SAVE_PATH = (
    "/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/"
    f"{trainer.RQVAE_VARIANT}/ckpt/"
)
trainer.PER_USER_DUMP = True
trainer._LAUNCHER["script"] = os.path.abspath(__file__)
trainer._LAUNCHER["log"] = os.path.join(trainer.LOG_PATH, "_stage3_launcher.log")
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
    sync_dir = os.path.join(trainer.LOG_PATH, "_ddp_sync")
    if os.path.exists(sync_dir):
        shutil.rmtree(sync_dir)
    print(f"[{trainer.RQVAE_VARIANT}] cleared DDP sync directory: {sync_dir}", flush=True)
    trainer._launch_via_torchrun()
