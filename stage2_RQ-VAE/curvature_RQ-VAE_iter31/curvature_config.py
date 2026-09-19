"""R53 v3.8: 所有机制变量 + R51+ 约束 + HF cache 路径 集中硬编码.
修改 MECHANISM_NAME 即可切换实验 (cp -r baseline 后编辑此文件即可).

所有路径相对于本文件所在目录 (curvature_base/), 不依赖 cwd.
使用 torchrun 时 cwd 可以是任意目录, 路径始终正确.
"""
import os

# === 路径推导锚点: 本文件所在目录 ===
_CONFIG_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter31"

# === 机制标识 (R52 + R53 联动) ===
MECHANISM_NAME = "iter31_per_layer_hetero_c_v1"  # ← 2026-09-19 iter31 (R36n b per-layer 异质 c). iter26 R36n d Riemannian Adam + iter28 R36m radial α=+0.25 + iter29 R36m radial α=-0.5 + iter30 R36m-S2 Sphere→Hyperbolic 4 个 Stage 0/Stage 1 机制 oracle_top10 都 ≈ baseline 0.0338 (±1.2%). Stage 0 magnitude/manifold 改动路线**整体证伪**. 用户 2026-09-19 决策: Stage 1 端真改 RQ-VAE 内部机制 (per-layer 异质 c). iter31 方案: 3 个 RQ 层各自独立 c_min/c_max, e.g. L0=[0.2, 0.8], L1=[0.4, 1.2], L2=[0.3, 1.5], 让每一层都有独立的"曲率预算". RQ-VAE 主体一字不动, 只改 RqVae.__init__ 接受 per-layer c_cyclic_min/max 数组, 传给每个 Quantize 层. 关键 invariant: cyclic 公式 c(t)=c_min+(c_max-c_min)·|sin(πt/T)| 完全保留, 仅 min/max per-layer 异质. 单变量 vs iter11 baseline: iter11 用全局 c ∈ [0.3, 1.0] T=50k, iter31 每层独立 [c_min_l, c_max_l] 数组. 论文支撑: Chami et al. 2019 "Hyperbolic Graph Convolutional Neural Networks" (per-layer curvature is established practice for hyperbolic GCN). 期望 oracle 提升: 让 L0 (coarsest) 用低曲率 (大 norm radius), L1/L2 (finer) 用较高曲率 (小 norm radius), 形成"由粗到细"的几何梯度, 避免 iter11 全局 c 导致的"中间层曲率冲突".
SAVE_DIR_ROOT = os.path.join(_CONFIG_DIR, "out/decoder/instruments_hgrec_configs/hgrec_{}/".format(MECHANISM_NAME))
CONFIG_PATH = os.path.join(_CONFIG_DIR, "configs/decoder_instruments_hgrec_{}.gin".format(MECHANISM_NAME))
BEST_CKPT_PATH = os.path.join(SAVE_DIR_ROOT, "best_ckpt.pt")
USE_HGREC_ARCH = True  # R36i: 强制 HG-Rec 路径

# === Stage 0 / Stage 1 / Stage 2 路径 (相对于 curvature_base/) ===
ITEM_JSON_PATH = os.path.join(_CONFIG_DIR, "dataset/Instruments.item.json")  # R44 数据集
ITEM_EMB_NPY = "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE/dataset/Instruments/item_emb.npy"  # 用 baseline 原始 item_emb.npy (mean_norm=1.0, 已 normalized) — iter31 取消 Stage 0 变换, 直接吃 Stage 1 输出
ITEM_IDS_JSON = os.path.join(_CONFIG_DIR, "dataset/Instruments/item_ids.json")  # Stage 0 输出
RQVAE_OUT_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/out/rqvae/instruments"  # Stage 1 ckpt 目录 (iter31 独立子目录)
RQVAE_CKPT_PATH = os.path.join(RQVAE_OUT_DIR, "rqvae_final.pt")  # Stage 1 ckpt (iter31 训练产出)
RAW_SIDS_NPY = os.path.join(RQVAE_OUT_DIR, "sids_raw.npy")  # Stage 2.1 推理输出 (iter31 独立)
SIDS_NPY = os.path.join(_CONFIG_DIR, "dataset/Instruments/sids_for_hgrec.npy")  # Stage 2.2 输出 (iter31 独立, 不覆盖 baseline sids_for_hgrec.npy)
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