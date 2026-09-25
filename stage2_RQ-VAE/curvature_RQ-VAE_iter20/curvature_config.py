"""iter20 curvature-scaled behavior contrastive temperature experiment."""
import os

# === 路径推导锚点: 本文件所在目录 ===
_CONFIG_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter20"

# === 机制标识 (R52 + R53 联动) ===
MECHANISM_NAME = "iter20_curvature_behavior_temperature"
SAVE_DIR_ROOT = os.path.join(_CONFIG_DIR, "out/decoder/instruments_hgrec_configs/hgrec_{}/".format(MECHANISM_NAME))
CONFIG_PATH = os.path.join(_CONFIG_DIR, "configs/decoder_instruments_hgrec_{}.gin".format(MECHANISM_NAME))
BEST_CKPT_PATH = os.path.join(SAVE_DIR_ROOT, "best_ckpt.pt")
USE_HGREC_ARCH = True  # R36i: 强制 HG-Rec 路径

# === Stage 0 / Stage 1 / Stage 2 路径 (post-Refactor 全部从上游 stage 读) ===
DATASET_ROOT    = "/home/wlia0047/ar57/wenyu/GeneRec/dataset"
STAGE0_DIR      = "/home/wlia0047/ar57/wenyu/GeneRec/results/stage0_build_parquet"
STAGE1_OUT_DIR  = "/home/wlia0047/ar57/wenyu/GeneRec/stage1_GeneEmbedding/output"
ITEM_JSON_PATH  = os.path.join(DATASET_ROOT, "Amazon_2023_Instruments/Instruments.item.json")  # 仅 docs 参考
ITEM_EMB_NPY    = os.path.join(STAGE1_OUT_DIR, "sentence_t5.npy")  # stage1 输出
ITEM_IDS_JSON   = os.path.join(STAGE1_OUT_DIR, "item_ids.json")    # stage1 sidecar
TRAIN_PARQUET   = os.path.join(STAGE0_DIR, "train.parquet")
VALID_PARQUET   = os.path.join(STAGE0_DIR, "valid.parquet")
TEST_PARQUET    = os.path.join(STAGE0_DIR, "test.parquet")
ITEMS_PARQUET   = os.path.join(STAGE0_DIR, "items.parquet")
RQVAE_OUT_DIR   = "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter20/out/rqvae/instruments"
RQVAE_CKPT_PATH = os.path.join(RQVAE_OUT_DIR, "rqvae_best.pth")
RAW_SIDS_NPY    = os.path.join(RQVAE_OUT_DIR, "sids_raw.npy")
SIDS_NPY        = "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter20/dataset/Instruments/sids_for_hgrec.npy"
ITEM_SIDS_JSON  = "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter20/item_sids.json"
PARQUET_DIR     = STAGE0_DIR

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
