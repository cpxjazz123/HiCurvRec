"""R53 v3.8: 所有机制变量 + R51+ 约束 + HF cache 路径 集中硬编码.
修改 MECHANISM_NAME 即可切换实验 (cp -r baseline 后编辑此文件即可).

所有路径相对于本文件所在目录 (curvature_base/), 不依赖 cwd.
使用 torchrun 时 cwd 可以是任意目录, 路径始终正确.
"""
import os

# === 路径推导锚点: 本文件所在目录 ===
_CONFIG_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter30"

# === 机制标识 (R52 + R53 联动) ===
MECHANISM_NAME = "iter30_sphere_to_hyperbolic_c10_v1"  # ← 2026-09-19 iter30 (R36m-S2 / Stage 0 Pre-RQ Manifold Adapter Sphere→Hyperbolic 显式映射 c=1.0). 用户 2026-09-19 决策: iter28 R36m radial α=+0.25 oracle 0.0970 (-63.7%) + iter29 R36m radial α=-0.5 oracle 0.0983 (-63.2%) 两个反向实验**几乎精确同值** (-63.7% ± 0.5%), **R36m radial manipulation 框架证伪**: 任何方向的 radius scaling 都破坏 iter11 有效结构. **跳出 radial 框架**: Sphere→Hyperbolic 显式映射 h_i = exp_0^c(e_i) = tanh(√c · ||e_i||) · e_i / (√c · ||e_i||). 与 R36m radial 单调 pow **根本不同**: tanh 软截断让 norm 大的 items 饱和压缩到接近 1/√c, norm 小的保留. 单变量 vs iter11 baseline: RQ-VAE 主体一字不动, 只改输入 embedding 几何 (manifold 显式变换, **非线性** magnitude). 单变量 sweep 计划: iter30 = c=1.0 第一版, iter31 = c=0.5 低曲率, iter32 = c=2.0 高曲率. 论文支撑: Ungar 2008 "Hyperbolic Geometry" + Ganea 2018 "Hyperbolic Neural Networks" exp_0^c 标准公式. 关键 invariant: direction preservation cos(e, h) = 1.0 (沿 v 方向映射); Poincaré ball 约束 ||h|| < 1/√c (永远内).
SAVE_DIR_ROOT = os.path.join(_CONFIG_DIR, "out/decoder/instruments_hgrec_configs/hgrec_{}/".format(MECHANISM_NAME))
CONFIG_PATH = os.path.join(_CONFIG_DIR, "configs/decoder_instruments_hgrec_{}.gin".format(MECHANISM_NAME))
BEST_CKPT_PATH = os.path.join(SAVE_DIR_ROOT, "best_ckpt.pt")
USE_HGREC_ARCH = True  # R36i: 强制 HG-Rec 路径

# === Stage 0 / Stage 1 / Stage 2 路径 (相对于 curvature_base/) ===
ITEM_JSON_PATH = os.path.join(_CONFIG_DIR, "dataset/Instruments.item.json")  # R44 数据集
ITEM_EMB_NPY = os.path.join(_CONFIG_DIR, "dataset/Instruments/item_emb.npy")  # Stage 0 输出
ITEM_IDS_JSON = os.path.join(_CONFIG_DIR, "dataset/Instruments/item_ids.json")  # Stage 0 输出
RQVAE_OUT_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/out/rqvae/instruments"  # Stage 1 ckpt 目录 (iter30 独立子目录, 不覆盖 iter29/28/baseline)
RQVAE_CKPT_PATH = os.path.join(RQVAE_OUT_DIR, "rqvae_final.pt")  # Stage 1 ckpt (iter28 训练产出)
RAW_SIDS_NPY = os.path.join(RQVAE_OUT_DIR, "sids_raw.npy")  # Stage 2.1 推理输出 (iter30 独立)
SIDS_NPY = os.path.join(_CONFIG_DIR, "dataset/Instruments/sids_for_hgrec.npy")  # Stage 2.2 输出 (iter28 独立, 不覆盖 baseline sids_for_hgrec.npy)
# iter28 (R36m): ITEM_EMB_NPY override 到 transformed embedding (density-aware radial α=0.25)
# 原始 item_emb.npy 在 curvature_RQ-VAE/dataset/Instruments/ (mean_norm=1.0, 已 normalized)
# transformed 文件在 iter28/dataset/Instruments/item_emb_transformed_alpha0.25.npy
ITEM_EMB_NPY = os.path.join(_CONFIG_DIR, "dataset/Instruments/item_emb_hyperbolic_c1.0.npy")
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