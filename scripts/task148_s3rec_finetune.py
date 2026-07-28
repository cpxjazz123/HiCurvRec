"""Task #148 — S³Rec paper-aligned finetune (Stage 2).

承接 Task #140 pretrain ckpt → finetune 推荐任务 (BPR loss).
DECOR paper Table 2 S³Rec Instruments R@10=0.0538 (paper-aligned target).

Config:
- train_stage='finetune' (跟 Task #140 pretrain 区分)
- pre_model_path=Task #140 pretrain ckpt
- learning_rate=0.001 (paper S³Rec)
- stopping_step=20 (paper patience)
- epochs=200, seed=2025

GPU 1 (R7 验证空闲).
"""
import os
import sys
os.environ["TRITON_CACHE_DIR"] = "/home/wlia0047/.triton/cache_task148_s3rec_finetune"
os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/RecBole')
from recbole.quick_start import run

PRETRAIN_CKPT = "/home/wlia0047/ar57/wenyu/GeneRec/products/task140/train/pretrain/S3Rec-Musical_Instruments-10.pth"

YAML_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/RecBole/musical_instruments_sequential_paper.yaml"

config_dict = {
    "train_stage": "finetune",          # Stage 2 (vs Task #140 pretrain)
    "pre_model_path": PRETRAIN_CKPT,    # init from Task #140 pretrain
    "learning_rate": 0.001,             # paper S3Rec (跟 yaml 一致)
    "weight_decay": 0.0,                # paper S3Rec
    "stopping_step": 20,                # paper patience
    "epochs": 200, "seed": 2025, "gpu_id": 0,  # gpu_id=0 因为 CUDA_VISIBLE_DEVICES=1 已重映射
    "checkpoint_dir": "/home/wlia0047/ar57/wenyu/GeneRec/products/task148/train/",
    "show_progress": False,
    "valid_metric": "NDCG@10",          # paper-aligned valid metric
    "metrics": ["Recall", "NDCG"],
    "topk": [5, 10],
}

run(
    model="S3Rec",
    dataset="Musical_Instruments",
    config_file_list=[YAML_PATH],
    config_dict=config_dict,
)
