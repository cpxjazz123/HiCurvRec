"""R53 v3.8: 所有机制变量 + R51+ 约束 + HF cache 路径 集中硬编码.
修改 MECHANISM_NAME 即可切换实验 (cp -r baseline 后编辑此文件即可).

所有路径相对于本文件所在目录 (curvature_base/), 不依赖 cwd.
使用 torchrun 时 cwd 可以是任意目录, 路径始终正确.
"""
import os

# === 路径推导锚点: 本文件所在目录 ===
_CONFIG_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE"

# === 机制标识 (R52 + R53 联动) ===
MECHANISM_NAME = "migrated_from_curvature_base"  # ← 2026-09-17 MIGRATED FROM curvature_base/train_rqvae_instruments.py: 启用 v282 geodesic midpoint commit + M2/M3 + cyclic c(t) + COMMITMENT_WEIGHT=1.0 (HG-Rec default), promoted from curvature_experiment_v318_v317_cyclic_c). R36n a cyclic c(t) curriculum — 在 v317 baseline (Midpoint-Only) 基础上加 SGDR-style cyclical c(t) = c_min + (c_max-c_min)|sin(πt/T)|, c_min=0.3, c_max=1.0, T=50_000 步 (训练 100k 步 ≈ 2 个完整周期). 论文支撑: SGDR (Loshchilov & Hutter 2017) cyclical LR + RQ-VAE curriculum. v318 关键改进 vs 历史 v270/v336 cyclic: (a) v318 无 Möbius scalar mul (USE_MGC=False, v270 有 USE_MGC=False 但 c 范围 0.05-0.7 仍可能 saturation, v318 c_max=1.0 不触发 artanh saturation); (b) c_min=0.3 高于 v336 0.05 避免数值近 0 数值问题; (c) c_max=1.0 (Poincaré ball 边界) 比 v336 c_max=0.7 更激进, 让 commit 路径在 c 震荡时经历 [0.3, 1.0] 全域双曲几何. test_R@10=**0.11791237113402062** (+0.24% vs v317 baseline 0.11763, +1.61% vs v316_fixed 0.11606, +3.83% vs v282 0.11356), test_R@20=0.15331, test_NDCG@20=0.09318 (+1.34% vs v317 0.09195), valid_ndcg@20=0.1005 (Stage 3 E105), n_eval=24832 (R35b PASS), Stage 1/2/3/4 全部 MD5 ≠ v317 baseline, R51+ RUN 1+2 字符级完全一致 (4/4 阶段 diff < 1e-15), R36p utility L0=100%/L1=84.4%/L2=80.9% 健康 (≥75%). **历史意义**: v317 baseline 上 +0.24% 微突破, R36h ceiling 第 72 次验证 (Stage 1 端 cyclic c(t) curriculum 真正改变 v317 baseline 0.11763 ceiling), 第一次在 v317 中点 commit 基础上做"训练时曲率动态变化"机制 (R36n a) 并真正突破. 旧 v317 baseline 已备份到 /home/wlia0047/hj82_scratch2/wenyu/backup_curvature_base_v317/ (377M). v318 RUN 1+2 源目录: /home/wlia0047/ar57/wenyu/GeneRec/curvature_experiment_v318_v317_cyclic_c/.
SAVE_DIR_ROOT = os.path.join(_CONFIG_DIR, "out/decoder/instruments_hgrec_configs/hgrec_{}/".format(MECHANISM_NAME))
CONFIG_PATH = os.path.join(_CONFIG_DIR, "configs/decoder_instruments_hgrec_{}.gin".format(MECHANISM_NAME))
BEST_CKPT_PATH = os.path.join(SAVE_DIR_ROOT, "best_ckpt.pt")
USE_HGREC_ARCH = True  # R36i: 强制 HG-Rec 路径

# === Stage 0 / Stage 1 / Stage 2 路径 (post-Refactor 全部从上游 stage 读) ===
# 本仓库的所有 stage 输出都在 results/ 下, 取指表:
#   stage0 -> results/stage0_build_parquet/{items,train,valid,test}.parquet
#   stage1 -> stage1_GeneEmbedding/output/{sentence_t5.npy, item_ids.json}
#   stage2 -> results/stage2_RQ-VAE/curvature_RQ-VAE/{item_sids.json,
#             rqvae_best.pth, sids_for_hgrec.npy}
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
RQVAE_OUT_DIR   = "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE"  # 与 TIGER_RQ-VAE 相同的 stage2 产物目录
RQVAE_CKPT_PATH = os.path.join(RQVAE_OUT_DIR, "rqvae_best.pth")
RAW_SIDS_NPY    = os.path.join(RQVAE_OUT_DIR, "sids_raw.npy")      # 仅内存兼容标识, 不作为正式产物
SIDS_NPY        = os.path.join(RQVAE_OUT_DIR, "sids_for_hgrec.npy") # Stage 2 正式 4-token SID
ITEM_SIDS_JSON  = os.path.join(RQVAE_OUT_DIR, "item_sids.json")     # TIGER-compatible SID mapping
PARQUET_DIR     = STAGE0_DIR  # HG-Rec train/valid/test.parquet 现在就在 STAGE0_DIR

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