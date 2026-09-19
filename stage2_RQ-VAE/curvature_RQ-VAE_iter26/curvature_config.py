"""R53 v3.8: 所有机制变量 + R51+ 约束 + HF cache 路径 集中硬编码.
修改 MECHANISM_NAME 即可切换实验 (cp -r baseline 后编辑此文件即可).

所有路径相对于本文件所在目录 (curvature_base/), 不依赖 cwd.
使用 torchrun 时 cwd 可以是任意目录, 路径始终正确.
"""
import os

# === 路径推导锚点: 本文件所在目录 ===
_CONFIG_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter26"

# === 机制标识 (R52 + R53 联动) ===
MECHANISM_NAME = "iter26_riemannian_adam_v1"  # ← 2026-09-19 iter26: R36n d Riemannian Adam (R36n d 单变量). 在 v318 baseline (cyclic c(t)) 基础上, 把 optimizer 从 AdamW 换成 Riemannian Adam: 对所有参数 (尤其是 Poincaré ball 上的 codebook / encoder output embedding) 的梯度, 在 optimizer.step() 之前做 Riemannian metric rescale g_r = (1 - c||x||²)² / (4c), 然后走标准 AdamW 更新. 历史 v320 R37 FAIL (-6.49%) + v361 R37 FAIL (-71.5% utility), 但 v361 是叠 Lorentz, v320 是 (无 cyclic + Riemannian Adam). iter26 是 (cyclic c(t) + Riemannian Adam) 单变量, 与历史失败条件不同. 论文支撑: Nickel & Kiela 2017 "Poincaré Embeddings" + Becigneul & Ganea 2019 "Riemannian Adaptive Optimization". **必须保持单变量**: 只换 optimizer, 不改 curriculum / midpoint / commit / Sinkhorn.
SAVE_DIR_ROOT = os.path.join(_CONFIG_DIR, "out/decoder/instruments_hgrec_configs/hgrec_{}/".format(MECHANISM_NAME))
CONFIG_PATH = os.path.join(_CONFIG_DIR, "configs/decoder_instruments_hgrec_{}.gin".format(MECHANISM_NAME))
BEST_CKPT_PATH = os.path.join(SAVE_DIR_ROOT, "best_ckpt.pt")
USE_HGREC_ARCH = True  # R36i: 强制 HG-Rec 路径

# === Stage 0 / Stage 1 / Stage 2 路径 (相对于 curvature_base/) ===
ITEM_JSON_PATH = os.path.join(_CONFIG_DIR, "dataset/Instruments.item.json")  # R44 数据集
ITEM_EMB_NPY = os.path.join(_CONFIG_DIR, "dataset/Instruments/item_emb.npy")  # Stage 0 输出
ITEM_IDS_JSON = os.path.join(_CONFIG_DIR, "dataset/Instruments/item_ids.json")  # Stage 0 输出
RQVAE_OUT_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter26/out/rqvae/instruments"  # Stage 1 ckpt 目录 (iter26 独立子目录, 不覆盖 baseline)
RQVAE_CKPT_PATH = os.path.join(RQVAE_OUT_DIR, "rqvae_final.pt")  # Stage 1 ckpt (iter26 训练产出)
RAW_SIDS_NPY = os.path.join(RQVAE_OUT_DIR, "sids_raw.npy")  # Stage 2.1 推理输出 (iter26 独立)
SIDS_NPY = os.path.join(_CONFIG_DIR, "dataset/Instruments/sids_for_hgrec.npy")  # Stage 2.2 输出 (iter26 独立, 不覆盖 baseline sids_for_hgrec.npy)
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

# === iter26 Riemannian Adam 超参 (R36n d) ===
USE_RIEMANNIAN_ADAM = True  # ← 2026-09-19 iter26 单变量开关; True 时 optimizer 走 Riemannian Adam (Poincaré metric rescale + 标准 AdamW 更新), False 时走 AdamW (baseline 行为).
# Riemannian Adam 超参 (硬编码, 见 modules/rqvae.py:_poincare_grad_rescale)
# 公式: grad_r = grad_euclid * ((1 - c * ||x||²)² / (4c)), 然后走 AdamW (m/v 缓存 + bias correction).
# c 是 cyclic c(t) (与 v318 baseline 共享, 单变量: 只换 optimizer, 不动 curriculum).
# ||x||² clamp 到 [0, 1/c - 1e-3] 防止 c·||x||² ≥ 1 让 rescale → 0 卡死.
RIEMANNIAN_ADAM_LR = 1e-3            # 与 baseline AdamW lr 相同 (单变量)
RIEMANNIAN_ADAM_BETAS = (0.9, 0.999) # AdamW 标准 beta1/beta2
RIEMANNIAN_ADAM_EPS = 1e-8           # AdamW 标准 eps
RIEMANNIAN_ADAM_WEIGHT_DECAY = 1e-4  # 与 baseline 相同
RIEMANNIAN_ADAM_NORM_CLAMP = 0.9999  # ||x||² / (1/c) 上限, 防 ball boundary 数值发散