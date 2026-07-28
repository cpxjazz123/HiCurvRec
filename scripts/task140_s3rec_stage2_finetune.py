"""Task #140 — S³Rec Stage 2 finetune launcher.
2026-07-24

Per s3rec.py line 109-122:
- train_stage='finetune' → uses CE/BPR loss on actual recommendation task
- Load pretrained state_dict from pre_model_path at line 120
- Loss type CE for paper-aligned reproducibility (per FDSA/S3Rec original)

R12: trainer.fit auto-saves best valid ckpt (RecBole built-in best_metric tracking).
"""
import argparse
import glob
import os
import sys

os.environ["TRITON_CACHE_DIR"] = "/home/wlia0047/.triton/cache_task140"
os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/RecBole')

from recbole.quick_start import run


def find_latest_pretrain_ckpt():
    """Auto-discover latest S3Rec pretrain ckpt."""
    candidates = []
    for pattern in [
        "/home/wlia0047/ar57/wenyu/GeneRec/products/task140/train/pretrain/*.pth",
        "/home/wlia0047/ar57/wenyu/GeneRec/RecBole/saved/*S3Rec*.pth",
    ]:
        candidates.extend(glob.glob(pattern))
    candidates.sort(key=os.path.getmtime)
    return candidates[-1] if candidates else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre_model_path", type=str, default=None,
                        help="Path to S3Rec pretrain .pth; auto-discovery if None")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--stopping_step", type=int, default=20)
    parser.add_argument("--loss_type", type=str, default="CE", choices=["CE", "BPR"])
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    pre_model_path = args.pre_model_path or find_latest_pretrain_ckpt()
    if not pre_model_path:
        print("❌ No pretrain ckpt found. Run stage 1 first.")
        sys.exit(1)
    print(f"[Task #140 Stage 2] Loading pretrain: {pre_model_path}")

    YAML_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/RecBole/musical_instruments_sequential_paper.yaml"
    config_file_list = [YAML_PATH]
    config_dict = {
        "train_stage": "finetune",
        "pre_model_path": pre_model_path,
        "loss_type": args.loss_type,
        "epochs": args.epochs,
        "stopping_step": args.stopping_step,
        "seed": args.seed,
        "gpu_id": args.gpu_id,
        "show_progress": False,
        "checkpoint_dir": "/home/wlia0047/ar57/wenyu/GeneRec/products/task140/train/finetune/",
    }

    run(
        model="S3Rec",
        dataset="Musical_Instruments",
        config_file_list=config_file_list,
        config_dict=config_dict,
    )


if __name__ == "__main__":
    main()
