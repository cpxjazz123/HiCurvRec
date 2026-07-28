"""Task #141 — Caser paper-aligned fix launcher (lr=0.001, wd=0.0).
2026-07-24.

修复 Task #87 Caser paper-aligned 配置混用 bug:
- 原 paper_aligned yaml 用 lr=0.003 + wd=0.05 (sequential transformer default)
- Caser CNN 不能收敛 (loss 13416 不下降, valid R@10 ≈ 0.0003)
- 正确 paper Caser 设置: lr=0.001 + wd=0.0 (RecBole default 也这样)

Programmatic launch via recbole.quick_start.run() 配 config_dict override:
config_dict priority > yaml file (RecBole Config 行为).

R12: RecBole built-in best valid ckpt saving (stopping_step=10 best_metric tracking).
"""
import argparse
import os
import sys

# Triton cache isolation (per CLAUDE.md GPU table)
os.environ["TRITON_CACHE_DIR"] = "/home/wlia0047/.triton/cache_task141"
os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/RecBole')

from recbole.quick_start import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--stopping_step", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.001)  # paper Caser
    parser.add_argument("--weight_decay", type=float, default=0.0)     # paper Caser
    args = parser.parse_args()

    # yaml lives at RecBole top-level, NOT in recbole/properties/
    YAML_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/RecBole/musical_instruments_sequential_paper.yaml"
    config_file_list = [YAML_PATH]

    config_dict = {
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "stopping_step": args.stopping_step,
        "epochs": args.epochs,
        "seed": args.seed,
        "gpu_id": args.gpu_id,
        "show_progress": False,
        "checkpoint_dir": "/home/wlia0047/ar57/wenyu/GeneRec/products/task141/train/",
    }

    print(f"[Task #141 Caser paper-aligned fix] lr={args.learning_rate}, wd={args.weight_decay}, "
          f"stopping_step={args.stopping_step}, seed={args.seed}, gpu_id={args.gpu_id}")

    run(
        model="Caser",
        dataset="Musical_Instruments",
        config_file_list=config_file_list,
        config_dict=config_dict,
    )


if __name__ == "__main__":
    main()
