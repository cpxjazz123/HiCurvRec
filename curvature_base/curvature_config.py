"""R53 v3.8: 所有机制变量 + R51+ 约束 + HF cache 路径 集中硬编码.
修改 MECHANISM_NAME 即可切换实验 (cp -r baseline 后编辑此文件即可).
"""
import os

# === 机制标识 (R52 + R53 联动) ===
MECHANISM_NAME = "codebook_manifold_contrastive"  # ← baseline 硬编码为 v69 G5
SAVE_DIR_ROOT = f"out/decoder/instruments_hgrec_configs/hgrec_{MECHANISM_NAME}/"
CONFIG_PATH = f"configs/decoder_instruments_hgrec_{MECHANISM_NAME}.gin"
BEST_CKPT_PATH = f"{SAVE_DIR_ROOT}best_ckpt.pt"
USE_HGREC_ARCH = True  # R36i: 强制 HG-Rec 路径

# === Stage 0 / Stage 1 / Stage 2 路径 (R44 + R47 + R52 相对化) ===
ITEM_JSON_PATH = "./dataset/Instruments.item.json"  # R44 数据集硬编码
ITEM_EMB_NPY = "./dataset/Instruments/item_emb.npy"  # Stage 0 输出
RQVAE_CKPT_PATH = "./out/rqvae/instruments/rqvae_final.pt"  # Stage 1 ckpt
SIDS_NPY = "./dataset/Instruments/sids_for_hgrec.npy"  # Stage 2 输出

# === R51+ 6 项确定性约束 (硬编码, 无 setdefault) ===
os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# === R47 HF cache 强制路径 (硬编码, 写到 hj82_scratch2) ===
os.environ["HF_HOME"] = "/home/wlia0047/hj82_scratch2/wenyu/.cache/huggingface"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/home/wlia0047/hj82_scratch2/wenyu/hf_models/hub"
os.environ["TRANSFORMERS_CACHE"] = "/home/wlia0047/hj82_scratch2/wenyu/hf_models/hub"

# === R7 GPU 选择 (硬编码 4 卡, shell env 可覆盖但脚本不依赖) ===
CUDA_VISIBLE_DEVICES = "0,1,2,3"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

# === TF / transformers 警告抑制 (硬编码) ===
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"