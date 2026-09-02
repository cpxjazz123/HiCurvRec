"""R53 v3.8: 所有机制变量 + R51+ 约束 + HF cache 路径 集中硬编码.
修改 MECHANISM_NAME 即可切换实验 (cp -r baseline 后编辑此文件即可).

所有路径相对于本文件所在目录 (curvature_base/), 不依赖 cwd.
使用 torchrun 时 cwd 可以是任意目录, 路径始终正确.
"""
import os

# === 路径推导锚点: 本文件所在目录 ===
_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))

# === 机制标识 (R52 + R53 联动) ===
MECHANISM_NAME = "v317_midpoint_only"  # ← v317 NEW PROMOTED BASELINE (2026-09-03, promoted from curvature_experiment_v317_midpoint_only). R36n e 几何变换: 在 v316_fixed 基础上 commit 路径走 Geodesic Midpoint mid = exp_0(0.5*log_0(res) + 0.5*log_0(emb), c=1.0), 但 **关闭 USE_MGC / USE_ANISOTROPY_REG / USE_SPREAD_LOSS 三组件**, 只保留 Geodesic Midpoint Commit + 静态 c=1.0 + 静态 c_end 阶梯. 论文支撑: Ungar 2008 Gyrogroup 群论 / Ungar 2009 "Hyperbolic Geometry" Ch.4 (geodesic midpoint). test_R@10=0.11763047680412371 (+1.36% vs v316_fixed 0.11606), test_R@20=0.15327 (+5.16%), test_NDCG@20=0.09195 (-0.71% vs v316_fixed 0.09261, 微弱降), valid_ndcg@20=0.0997 (Stage 3 E91), n_eval=24832 (R35b PASS), Stage 1/2/3/4 全部 MD5 ≠ v316_fixed baseline, R51+ RUN 1+2 字符级完全一致 (4/4 阶段 diff < 1e-15), R36p utility L0=100%/L1=82.4%/L2=80.1% 健康 (≥75%). **历史意义**: v316_fixed 5 个组件消融实验 (A1-A5) 中唯一突破 baseline 的实验, R36h ceiling 第 71 次验证 (Stage 1 端纯曲率变更真正第一次打破 v316_fixed 0.11606 ceiling, +1.36% 突破). 旧 v316_fixed baseline 已备份到 /home/wlia0047/hj82_scratch2/wenyu/backup_curvature_base_v316_fixed/ (227M). A5 (midpoint/) 源 + RUN 2 备份: /home/wlia0047/hj82_scratch2/wenyu/backup_ablation_A5_midpoint_run12_r51_pass/ (RUN1: 377M + RUN 2: 187M).
SAVE_DIR_ROOT = os.path.join(_CONFIG_DIR, "out/decoder/instruments_hgrec_configs/hgrec_{}/".format(MECHANISM_NAME))
CONFIG_PATH = os.path.join(_CONFIG_DIR, "configs/decoder_instruments_hgrec_{}.gin".format(MECHANISM_NAME))
BEST_CKPT_PATH = os.path.join(SAVE_DIR_ROOT, "best_ckpt.pt")
USE_HGREC_ARCH = True  # R36i: 强制 HG-Rec 路径

# === Stage 0 / Stage 1 / Stage 2 路径 (相对于 curvature_base/) ===
ITEM_JSON_PATH = os.path.join(_CONFIG_DIR, "dataset/Instruments.item.json")  # R44 数据集
ITEM_EMB_NPY = os.path.join(_CONFIG_DIR, "dataset/Instruments/item_emb.npy")  # Stage 0 输出
ITEM_IDS_JSON = os.path.join(_CONFIG_DIR, "dataset/Instruments/item_ids.json")  # Stage 0 输出
RQVAE_OUT_DIR = os.path.join(_CONFIG_DIR, "out/rqvae/instruments")  # Stage 1 ckpt 目录 (v317 RQ-VAE)
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