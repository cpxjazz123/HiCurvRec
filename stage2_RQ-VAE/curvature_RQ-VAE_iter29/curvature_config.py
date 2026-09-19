"""R53 v3.8: 所有机制变量 + R51+ 约束 + HF cache 路径 集中硬编码.
修改 MECHANISM_NAME 即可切换实验 (cp -r baseline 后编辑此文件即可).

所有路径相对于本文件所在目录 (curvature_base/), 不依赖 cwd.
使用 torchrun 时 cwd 可以是任意目录, 路径始终正确.
"""
import os

# === 路径推导锚点: 本文件所在目录 ===
_CONFIG_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter29"

# === 机制标识 (R52 + R53 联动) ===
MECHANISM_NAME = "iter29_density_radial_alpha_neg05_v1"  # ← 2026-09-19 iter29 (R36m / Stage 0 Pre-RQ Geometry Adapter 反向 α=-0.5 单点实验). 用户 2026-09-19 决策: iter28 oracle top10=0.0970 (-63.7% vs baseline 0.2672) R37 FAIL 后, **不转向正交/子空间等新方向**, 而是先做一次反向实验 iter29 = α=-0.5 回答"是不是任何 radial manipulation 都破坏 iter11 的有效结构". 若 iter29 同样 -63.7% 退化, 则 R36m 整个 radial 假设证伪 (改 radius scaling 必然失败), Stage 0 后续设计需要完全跳出 radial 框架; 若 iter29 oracle 显著不同 (-63.7% 之外), 则 radial 是单调破坏假设证伪, 后续可考虑混合方向 (rotation + scale) 或 conditional radial. 单变量 vs iter11 baseline: RQ-VAE 主体一字不动, 只改输入 embedding 几何形状 (density-aware radial scaling, α=-0.5 反向: 放大稀疏 item 半径, 缩小密集 item 半径). 严格 vs iter28 (α=+0.25 单调方向) 形成反向对照, cost 1 次 Stage2 实验 (~16 min 训练 + 30s oracle).
SAVE_DIR_ROOT = os.path.join(_CONFIG_DIR, "out/decoder/instruments_hgrec_configs/hgrec_{}/".format(MECHANISM_NAME))
CONFIG_PATH = os.path.join(_CONFIG_DIR, "configs/decoder_instruments_hgrec_{}.gin".format(MECHANISM_NAME))
BEST_CKPT_PATH = os.path.join(SAVE_DIR_ROOT, "best_ckpt.pt")
USE_HGREC_ARCH = True  # R36i: 强制 HG-Rec 路径

# === Stage 0 / Stage 1 / Stage 2 路径 (相对于 curvature_base/) ===
ITEM_JSON_PATH = os.path.join(_CONFIG_DIR, "dataset/Instruments.item.json")  # R44 数据集
ITEM_EMB_NPY = os.path.join(_CONFIG_DIR, "dataset/Instruments/item_emb.npy")  # Stage 0 输出
ITEM_IDS_JSON = os.path.join(_CONFIG_DIR, "dataset/Instruments/item_ids.json")  # Stage 0 输出
RQVAE_OUT_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter29/out/rqvae/instruments"  # Stage 1 ckpt 目录 (iter29 独立子目录, 不覆盖 iter28 / baseline)
RQVAE_CKPT_PATH = os.path.join(RQVAE_OUT_DIR, "rqvae_final.pt")  # Stage 1 ckpt (iter28 训练产出)
RAW_SIDS_NPY = os.path.join(RQVAE_OUT_DIR, "sids_raw.npy")  # Stage 2.1 推理输出 (iter29 独立)
SIDS_NPY = os.path.join(_CONFIG_DIR, "dataset/Instruments/sids_for_hgrec.npy")  # Stage 2.2 输出 (iter28 独立, 不覆盖 baseline sids_for_hgrec.npy)
# iter28 (R36m): ITEM_EMB_NPY override 到 transformed embedding (density-aware radial α=0.25)
# 原始 item_emb.npy 在 curvature_RQ-VAE/dataset/Instruments/ (mean_norm=1.0, 已 normalized)
# transformed 文件在 iter28/dataset/Instruments/item_emb_transformed_alpha0.25.npy
ITEM_EMB_NPY = os.path.join(_CONFIG_DIR, "dataset/Instruments/item_emb_transformed_alpha-0.5.npy")
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