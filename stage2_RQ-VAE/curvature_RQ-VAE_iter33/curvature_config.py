"""R53 v3.8: 所有机制变量 + R51+ 约束 + HF cache 路径 集中硬编码.
修改 MECHANISM_NAME 即可切换实验 (cp -r baseline 后编辑此文件即可).

所有路径相对于本文件所在目录 (curvature_base/), 不依赖 cwd.
使用 torchrun 时 cwd 可以是任意目录, 路径始终正确.
"""
import os

# === 路径推导锚点: 本文件所在目录 ===
_CONFIG_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter33"

# === 机制标识 (R52 + R53 联动) ===
MECHANISM_NAME = "iter33_per_item_curvature_v1"  # ← 2026-09-19 iter33 (R36-S0 per-item curvature). iter32 R36n b+f stack (per-layer c + sk_eps) R37 FAIL: valid +9.4% 但 test_R@10=-1.65%, valid→test drift 0.886 (vs iter28 0.985) 放大 10x. 根因: per-layer 异质让 SID 过拟合 valid 集共现分布, 不泛化 test. iter33 转向 Stage 0 端新 axis: **per-item curvature curriculum**. 思路: 按 item 频率调 c, 高频 item (popular, train 共现多) 用低 c (紧凑空间, 易区分), 低频 item (long-tail, 训练样本少) 用高 c (扩展空间, 稀疏表示). c_per_item = c_base + alpha * (1 - normalized_freq). Stage 0 precompute: 从 train.parquet 统计 item frequency → 写 c_per_item.npy. Stage 2 RQ-VAE: 接收 c_per_item, 在 quantize 时按 batch item_idx 查 c_per_item. 与 iter32 per-layer 异质 (R36n b+f) 完全正交: per-layer 控层间异质, per-item 控 item 间异质. 论文支撑: "Frequency-aware representation learning" (long-tail 问题的标准做法, e.g. BBN, Logit Adjustment). 期望: 在 iter32 valid 信号基础上, 让 test 集长尾 item 也获得合适 curvature 表征, 减小 valid→test drift.
SAVE_DIR_ROOT = os.path.join(_CONFIG_DIR, "out/decoder/instruments_hgrec_configs/hgrec_{}/".format(MECHANISM_NAME))
CONFIG_PATH = os.path.join(_CONFIG_DIR, "configs/decoder_instruments_hgrec_{}.gin".format(MECHANISM_NAME))
BEST_CKPT_PATH = os.path.join(SAVE_DIR_ROOT, "best_ckpt.pt")
USE_HGREC_ARCH = True  # R36i: 强制 HG-Rec 路径

# === Stage 0 / Stage 1 / Stage 2 路径 (相对于 curvature_base/) ===
ITEM_JSON_PATH = os.path.join(_CONFIG_DIR, "dataset/Instruments.item.json")  # R44 数据集
ITEM_EMB_NPY = "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter33/dataset/Instruments/item_emb_iter33.npy"  # iter33 per-item curvature projected item_emb (Stage 0 precompute 2 输出)
ITEM_IDS_JSON = os.path.join(_CONFIG_DIR, "dataset/Instruments/item_ids.json")  # Stage 0 输出
RQVAE_OUT_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter33/out/rqvae/instruments"  # Stage 1 ckpt 目录 (iter33 独立子目录)
RQVAE_CKPT_PATH = os.path.join(RQVAE_OUT_DIR, "rqvae_final.pt")  # Stage 1 ckpt (iter33 训练产出)
RAW_SIDS_NPY = os.path.join(RQVAE_OUT_DIR, "sids_raw.npy")  # Stage 2.1 推理输出 (iter33 独立)
SIDS_NPY = os.path.join(_CONFIG_DIR, "dataset/Instruments/sids_for_hgrec.npy")  # Stage 2.2 输出 (iter33 独立, 不覆盖 baseline sids_for_hgrec.npy)
PARQUET_DIR = os.path.join(_CONFIG_DIR, "dataset/Instruments")  # HG-Rec train/valid/test.parquet
C_PER_ITEM_NPY = os.path.join(_CONFIG_DIR, "dataset/Instruments/c_per_item.npy")  # iter33 per-item curvature 预计算 (Stage 0 precompute 输出)

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