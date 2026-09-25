"""v318 baseline migration to Amazon 2023 Instruments (5core, 24K items).

R53 v3.8 + 2026-09-05 migration: 与 baseline curvature_base/curvature_config.py 路径不同,
指向 Amazon 2023 dataset (24,587 items + 57,439 users), 其余 v318 cyclic c(t) 机制保持.

所有路径相对于本文件所在目录, 不依赖 cwd.
使用 torchrun 时 cwd 可以是任意目录, 路径始终正确.
"""
import os

# === 路径推导锚点: 本文件所在目录 (tasks/v318_amazon2023/) ===
_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))

# === 机制标识 (R52 + R53 联动) — 与 v318 baseline 一致 ===
MECHANISM_NAME = "v318_amazon2023"  # ← 2026-09-05 NEW: v318 baseline 迁移到 Amazon 2023 Instruments 5core
SAVE_DIR_ROOT = os.path.join(_CONFIG_DIR, "out/decoder/instruments_hgrec_configs/hgrec_{}/".format(MECHANISM_NAME))
CONFIG_PATH = os.path.join(_CONFIG_DIR, "configs/decoder_instruments_hgrec_{}.gin".format(MECHANISM_NAME))
BEST_CKPT_PATH = os.path.join(SAVE_DIR_ROOT, "best_ckpt.pt")
USE_HGREC_ARCH = True

# === Stage 0 / Stage 1 / Stage 2 路径 (Amazon 2023 Instruments 数据集, 绝对路径) ===
ITEM_JSON_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/dataset/Amazon_2023_Instruments/Instruments.item.json"
ITEM_EMB_NPY = "/home/wlia0047/ar57/wenyu/GeneRec/dataset/Amazon_2023_Instruments/item_emb.npy"
ITEM_IDS_JSON = "/home/wlia0047/ar57/wenyu/GeneRec/dataset/Amazon_2023_Instruments/item_ids.json"
RQVAE_OUT_DIR = os.path.join(_CONFIG_DIR, "out/rqvae/amazon2023_instruments")
RQVAE_CKPT_PATH = os.path.join(RQVAE_OUT_DIR, "rqvae_final.pt")
RAW_SIDS_NPY = os.path.join(RQVAE_OUT_DIR, "sids_raw.npy")
SIDS_NPY = "/home/wlia0047/ar57/wenyu/GeneRec/dataset/Amazon_2023_Instruments/sids_for_hgrec.npy"
PARQUET_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/dataset/Amazon_2023_Instruments"

# === R51+ 6 项确定性约束 (硬编码, 无 setdefault) ===
os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# === R47 HF cache 强制路径 (硬编码, 写到 hj82_scratch2) ===
os.environ["HF_HOME"] = "/home/wlia0047/hj82_scratch2/wenyu/.cache/huggingface"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/home/wlia0047/hj82_scratch2/wenyu/hf_models/hub"
os.environ["TRANSFORMERS_CACHE"] = "/home/wlia0047/hj82_scratch2/wenyu/hf_models/hub"

# === R7 GPU 选择 (硬编码 4 卡) ===
CUDA_VISIBLE_DEVICES = "0,1,2,3"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

# === TF / transformers 警告抑制 (硬编码) ===
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
