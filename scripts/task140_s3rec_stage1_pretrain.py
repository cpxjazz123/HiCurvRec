"""Task #140 — S³Rec Stage 1 pretrain launcher.
2026-07-24

Per s3rec.py line 408-417: train_stage='pretrain' runs 4 self-supervised losses (aap+mip+map+sp)
With loss_type=CE defined, but pretrain doesn't use loss_type (uses self.loss_fct=BCEWithLogitsLoss).

Output: products/task140/train/pretrain/S3Rec-Musical_Instruments-{TIMESTAMP}-{HASH}.pth
  → every save_step (10 epochs) + final, for stage 2 finetune handoff (per s3rec.py line 122 load_state_dict).

Per R12: ckpt auto-saved, no manual save needed.
"""
import argparse
import os
import sys

# Triton cache isolation (per CLAUDE.md GPU table)
os.environ["TRITON_CACHE_DIR"] = "/home/wlia0047/.triton/cache_task140"
os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/RecBole')

from recbole.quick_start import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--save_step", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    # yaml lives at RecBole/musical_instruments_sequential_paper.yaml (not in recbole/properties/)
    YAML_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/RecBole/musical_instruments_sequential_paper.yaml"
    config_file_list = [YAML_PATH]
    config_dict = {
        "train_stage": "pretrain",
        "pretrain_epochs": args.epochs,
        "save_step": args.save_step,
        "seed": args.seed,
        "gpu_id": args.gpu_id,
        "show_progress": False,
        "checkpoint_dir": "/home/wlia0047/ar57/wenyu/GeneRec/products/task140/train/pretrain/",
    }

    run(
        model="S3Rec",
        dataset="Musical_Instruments",
        config_file_list=config_file_list,
        config_dict=config_dict,
    )


if __name__ == "__main__":
    main()
