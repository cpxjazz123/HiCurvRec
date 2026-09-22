"""R53 v3.8: 所有机制变量 + R51+ 约束 + HF cache 路径 集中硬编码.
修改 MECHANISM_NAME 即可切换实验 (cp -r baseline 后编辑此文件即可).

所有路径相对于本文件所在目录 (curvature_base/), 不依赖 cwd.
使用 torchrun 时 cwd 可以是任意目录, 路径始终正确.
"""
import os

# === 路径推导锚点: 本文件所在目录 ===
_CONFIG_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter9"

# === 机制标识 (R52 + R53 联动) ===
MECHANISM_NAME = "iter9_sid_dropout_aug"  # iter9 P2: SID Dropout Augmentation (training-only L1/L2 mask, L0 不 mask). 在 Stage 1 RQ-VAE 训练时, 对量化 embedding 的 L1/L2 层随机 mask (p=0.15), 让 RQ-VAE 学会"残差路径"; L0 始终保留 (coarse semantic 完整); 不修改 c(t)/codebook/commitment/几何 → 理论上不被 v321 R36h ceiling lock 锁定. baseline: iter8 (migrated_from_curvature_base) USE_GEODESIC_MIDPOINT_COMMIT=True/USE_M2_INTRINSIC=True/USE_M3_TRANSPORT=True/USE_SID_DROPOUT_AUG=False.
SAVE_DIR_ROOT = os.path.join(_CONFIG_DIR, "out/decoder/instruments_hgrec_configs/hgrec_{}/".format(MECHANISM_NAME))
CONFIG_PATH = os.path.join(_CONFIG_DIR, "configs/decoder_instruments_hgrec_{}.gin".format(MECHANISM_NAME))
BEST_CKPT_PATH = os.path.join(SAVE_DIR_ROOT, "best_ckpt.pt")
USE_HGREC_ARCH = True  # R36i: 强制 HG-Rec 路径

# === Stage 0 / Stage 1 / Stage 2 路径 (相对于 curvature_base/) ===
ITEM_JSON_PATH = os.path.join(_CONFIG_DIR, "dataset/Instruments.item.json")  # R44 数据集
ITEM_EMB_NPY = os.path.join(_CONFIG_DIR, "dataset/Instruments/item_emb.npy")  # Stage 0 输出
ITEM_IDS_JSON = os.path.join(_CONFIG_DIR, "dataset/Instruments/item_ids.json")  # Stage 0 输出
RQVAE_OUT_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter9/out/rqvae/instruments"  # iter9 独立 checkpoint 目录  # Stage 1 ckpt 目录 (migrated curvature_base baseline, R36+R42+R53)
RQVAE_CKPT_PATH = os.path.join(RQVAE_OUT_DIR, "rqvae_final.pt")  # Stage 1 ckpt (MD5: 3de83cefdd4894e41a2a7a8b31b2b385)
RAW_SIDS_NPY = os.path.join(RQVAE_OUT_DIR, "sids_raw.npy")  # Stage 2.1 推理输出
SIDS_NPY = os.path.join(_CONFIG_DIR, "dataset/Instruments/sids_for_hgrec.npy")  # Stage 2.2 输出 (MD5: 59406c2501fe27e829007065cbc45726), Stage 3 输入
PARQUET_DIR = os.path.join(_CONFIG_DIR, "dataset/Instruments")  # HG-Rec train/valid/test.parquet

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