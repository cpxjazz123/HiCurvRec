"""R53 v3.8: 所有机制变量 + R51+ 约束 + HF cache 路径 集中硬编码.
修改 MECHANISM_NAME 即可切换实验 (cp -r baseline 后编辑此文件即可).

所有路径相对于本文件所在目录 (curvature_base/), 不依赖 cwd.
使用 torchrun 时 cwd 可以是任意目录, 路径始终正确.
"""
import os

# === 路径推导锚点: 本文件所在目录 ===
_CONFIG_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter28"

# === 机制标识 (R52 + R53 联动) ===
MECHANISM_NAME = "iter28_density_radial_alpha025_v1"  # ← 2026-09-19 iter28 (R36m / Stage 0 Pre-RQ Geometry Adapter). 用户设计: Curvature-aware radial preconditioning. 在 v318 baseline (cyclic c(t)∈[0.3,1.0], T=50k) 基础上, 在 Stage 1 embedding 进入 RQ-VAE 之前做一个纯几何变换: tilde_e_i = (ρ_i / ρ_bar)^α · e_i, 其中 ρ_i 是 kNN-50 Euclidean 距离均值, ρ_bar 全局均值, α=0.25 第一版. 单变量 vs iter11 baseline: RQ-VAE 主体 (curriculum / midpoint / commit / Sinkhorn) 完全不动, 只改输入 embedding 的几何形状 (density-aware radial scaling, 改 radius 不改 direction). 论文支撑: 几何预调节 (Kar & Jain 2011 metric learning) + density-aware manifold (Goldberger et al. 2005). **关键 invariant**: α=0 严格等于 baseline (验证 isolation 干净); direction preservation cos(e, tilde_e) = 1.0 (radial scaling 只改 radius 不改 direction); RQ-VAE.py 一字不动. **单变量 sweep 计划**: iter28 = α=0.25, 后续 iter29+ sweep α ∈ {-0.5, -0.25, 0, 0.5} 各一遍.
SAVE_DIR_ROOT = os.path.join(_CONFIG_DIR, "out/decoder/instruments_hgrec_configs/hgrec_{}/".format(MECHANISM_NAME))
CONFIG_PATH = os.path.join(_CONFIG_DIR, "configs/decoder_instruments_hgrec_{}.gin".format(MECHANISM_NAME))
BEST_CKPT_PATH = os.path.join(SAVE_DIR_ROOT, "best_ckpt.pt")
USE_HGREC_ARCH = True  # R36i: 强制 HG-Rec 路径

# === Stage 0 / Stage 1 / Stage 2 路径 (相对于 curvature_base/) ===
ITEM_JSON_PATH = os.path.join(_CONFIG_DIR, "dataset/Instruments.item.json")  # R44 数据集
ITEM_EMB_NPY = os.path.join(_CONFIG_DIR, "dataset/Instruments/item_emb.npy")  # Stage 0 输出
ITEM_IDS_JSON = os.path.join(_CONFIG_DIR, "dataset/Instruments/item_ids.json")  # Stage 0 输出
RQVAE_OUT_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter28/out/rqvae/instruments"  # Stage 1 ckpt 目录 (iter28 独立子目录, 不覆盖 baseline)
RQVAE_CKPT_PATH = os.path.join(RQVAE_OUT_DIR, "rqvae_final.pt")  # Stage 1 ckpt (iter28 训练产出)
RAW_SIDS_NPY = os.path.join(RQVAE_OUT_DIR, "sids_raw.npy")  # Stage 2.1 推理输出 (iter28 独立)
SIDS_NPY = os.path.join(_CONFIG_DIR, "dataset/Instruments/sids_for_hgrec.npy")  # Stage 2.2 输出 (iter28 独立, 不覆盖 baseline sids_for_hgrec.npy)
# iter28 (R36m): ITEM_EMB_NPY override 到 transformed embedding (density-aware radial α=0.25)
# 原始 item_emb.npy 在 curvature_RQ-VAE/dataset/Instruments/ (mean_norm=1.0, 已 normalized)
# transformed 文件在 iter28/dataset/Instruments/item_emb_transformed_alpha0.25.npy
ITEM_EMB_NPY = os.path.join(_CONFIG_DIR, "dataset/Instruments/item_emb_transformed_alpha0.25.npy")
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