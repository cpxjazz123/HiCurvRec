"""R53 v3.8: 所有机制变量 + R51+ 约束 + HF cache 路径 集中硬编码.
修改 MECHANISM_NAME 即可切换实验 (cp -r baseline 后编辑此文件即可).

所有路径相对于本文件所在目录 (curvature_base/), 不依赖 cwd.
使用 torchrun 时 cwd 可以是任意目录, 路径始终正确.
"""
import os

# === 路径推导锚点: 本文件所在目录 ===
_CONFIG_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter27"

# === 机制标识 (R52 + R53 联动) ===
MECHANISM_NAME = "iter27_per_layer_hetero_c_v1"  # ← 2026-09-19 iter27: R36n b per-layer hetero c_l 单变量. 在 v318 baseline (cyclic c(t)∈[0.3,1.0], T=50k) 基础上, 给每层 RQ 量化器加 phase shift 让每层 c_l(t) 异质: layer 0 用 phase=global_step, layer 1 用 phase=global_step + T/3, layer 2 用 phase=global_step + 2T/3. 这样 3 层在同一个 step 时 c 不同 (一个在 c_min, 一个在 c_max, 一个在中间), 形成 per-layer hetero c_l. 单变量: 只改 phase shift, 不改 c_min/max/period/commit/midpoint. 历史 v329/v359/v372 (per-layer hetero c_l) 多次 R37 FAIL, 但都是 (异质 c_min/c_max 不同 + cyclic), iter27 是 (共享 c_min/c_max + 异质 phase) 单变量, 与历史失败条件不同. 论文支撑: SGDR (Loshchilov & Hutter 2017) cyclical LR + RQ-VAE multi-phase curriculum (Nickel & Kiela 2018 multi-curvature extension). **必须保持单变量**: 只加 phase shift, 不动 c_min/c_max/commit/midpoint/Sinkhorn.
SAVE_DIR_ROOT = os.path.join(_CONFIG_DIR, "out/decoder/instruments_hgrec_configs/hgrec_{}/".format(MECHANISM_NAME))
CONFIG_PATH = os.path.join(_CONFIG_DIR, "configs/decoder_instruments_hgrec_{}.gin".format(MECHANISM_NAME))
BEST_CKPT_PATH = os.path.join(SAVE_DIR_ROOT, "best_ckpt.pt")
USE_HGREC_ARCH = True  # R36i: 强制 HG-Rec 路径

# === Stage 0 / Stage 1 / Stage 2 路径 (相对于 curvature_base/) ===
ITEM_JSON_PATH = os.path.join(_CONFIG_DIR, "dataset/Instruments.item.json")  # R44 数据集
ITEM_EMB_NPY = os.path.join(_CONFIG_DIR, "dataset/Instruments/item_emb.npy")  # Stage 0 输出
ITEM_IDS_JSON = os.path.join(_CONFIG_DIR, "dataset/Instruments/item_ids.json")  # Stage 0 输出
RQVAE_OUT_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter27/out/rqvae/instruments"  # Stage 1 ckpt 目录 (iter27 独立子目录, 不覆盖 baseline)
RQVAE_CKPT_PATH = os.path.join(RQVAE_OUT_DIR, "rqvae_final.pt")  # Stage 1 ckpt (iter27 训练产出)
RAW_SIDS_NPY = os.path.join(RQVAE_OUT_DIR, "sids_raw.npy")  # Stage 2.1 推理输出 (iter27 独立)
SIDS_NPY = os.path.join(_CONFIG_DIR, "dataset/Instruments/sids_for_hgrec.npy")  # Stage 2.2 输出 (iter27 独立, 不覆盖 baseline sids_for_hgrec.npy)
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

# === iter27 per-layer hetero c_l 超参 (R36n b) ===
USE_PER_LAYER_HETERO_C = True  # ← 2026-09-19 iter27 单变量开关; True 时每层 RQ 量化器 phase shift 异质, 形成 per-layer hetero c_l(t).
# Phase shift 是相对于 global_step 的额外 step offset, 不改 c_min/c_max/period.
# 设计: 3 层 phase 错开 1/3 周期 (T/3 ≈ 16667 步), 让任何 step 时 3 层处于 c 周期不同相位:
#   layer 0: phase = global_step
#   layer 1: phase = global_step + T/3
#   layer 2: phase = global_step + 2T/3
# 这样同一 global_step 时 3 层的 c_l(t) 不同 (一个在 c_min, 一个在 c_max, 一个在中间), 形成 per-layer hetero.
# 单变量 vs baseline: 只加 phase offset, 不动 c_min/c_max/period/commit/midpoint/Sinkhorn.
# 与历史 v329/v359/v372 失败条件不同: 它们是 (异质 c_min/c_max) 或 (异质 c_range), iter27 是 (共享 c 范围 + 异质 phase).
PER_LAYER_PHASE_OFFSET_STEPS = [0, 16667, 33333]  # layer 0/1/2 偏移, T=50_000 周期内 1/3 等分
PER_LAYER_PHASE_OFFSET_RATIONALE = "1/3 phase 错开让 3 层 c 永远不同相位, 单 global_step 下 3 个不同 c 值"