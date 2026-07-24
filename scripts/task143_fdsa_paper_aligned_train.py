"""Task #143 — FDSA paper-aligned fix launcher (config_dict override selected_features=[]).

2026-07-24.

修复 Task #85 FDSA +51.9% 异常正 outlier 根因:
- RecBole recbole/properties/model/FDSA.yaml:10 默认启用 selected_features=['class']
  (使用 item class token 做 feature embedding)
- paper FDSA 实际只用 item_id, 没启 class feature
- 当前复现"成功"是不公平比较

config_dict override (RecBole Config priority: config_dict > yaml 文件 > model default):
- selected_features=[] 关掉 FDSA class token
- 沿用 yaml paper-aligned 其它设置 (lr=0.003, wd=0.05, MAX_ITEM_LIST_LENGTH=20, etc.)

R12: RecBole built-in best valid ckpt saving (stopping_step=20 best_metric tracking).
R7: GPU 1 free.
R141 兼容: RecBole trainer.py torch.load weights_only=False patch 已生效.
"""
import argparse
import os
import sys

# Triton cache isolation (per CLAUDE.md GPU table)
os.environ["TRITON_CACHE_DIR"] = "/home/wlia0047/.triton/cache_task143"
os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/RecBole')

from recbole.quick_start import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--stopping_step", type=int, default=20)  # paper patience
    parser.add_argument("--learning_rate", type=float, default=0.003)  # paper FDSA
    parser.add_argument("--weight_decay", type=float, default=0.05)     # paper FDSA
    parser.add_argument("--selected_features", type=str, nargs="+", default=[])  # 关掉 class token
    args = parser.parse_args()

    # yaml lives at RecBole top-level, NOT in recbole/properties/
    # Task #143: use paper_no_class variant (load_col.item: [item_id], no class column)
    # so FDSA default selected_features=['class'] gracefully skips.
    YAML_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/RecBole/musical_instruments_sequential_paper_no_class.yaml"
    config_file_list = [YAML_PATH]

    config_dict = {
        "selected_features": args.selected_features,  # 关掉 FDSA class token (paper 没启)
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "stopping_step": args.stopping_step,
        "epochs": args.epochs,
        "seed": args.seed,
        "gpu_id": args.gpu_id,
        "show_progress": False,
        "checkpoint_dir": "/home/wlia0047/ar57/wenyu/GeneRec/products/task143/train/",
    }

    print(f"[Task #143 FDSA paper-aligned fix] selected_features={args.selected_features}, "
          f"lr={args.learning_rate}, wd={args.weight_decay}, "
          f"stopping_step={args.stopping_step}, seed={args.seed}, gpu_id={args.gpu_id}")

    run(
        model="FDSA",
        dataset="Musical_Instruments",
        config_file_list=config_file_list,
        config_dict=config_dict,
    )


if __name__ == "__main__":
    main()