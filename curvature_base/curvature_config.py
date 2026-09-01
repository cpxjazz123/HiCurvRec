"""R53 v3.8: 所有机制变量 + R51+ 约束 + HF cache 路径 集中硬编码.
修改 MECHANISM_NAME 即可切换实验 (cp -r baseline 后编辑此文件即可).

所有路径相对于本文件所在目录 (curvature_base/), 不依赖 cwd.
使用 torchrun 时 cwd 可以是任意目录, 路径始终正确.
"""
import os

# === 路径推导锚点: 本文件所在目录 ===
_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))

# === 机制标识 (R52 + R53 联动) ===
MECHANISM_NAME = "mgc_fixed_c_v337"  # ← v337 NEW PROMOTED BASELINE (2026-09-02, promoted from curvature_experiment_v337_mgc_fixed_c). R36n (a 关闭 + e 启用): 在 v316_fixed 基础上 commit 走 Ungar 2008 ⊕/⊖/⊗ Möbius gyrovector 群代数, 但 **关闭 cyclic curriculum** (USE_CYCLIC_CURVATURE=False, 固定 c=1) 消除 v336 失败根因: cyclic c→0.7 + Möbius scalar mul artanh 饱和. 论文支撑: Ungar 2008 Gyrogroup 群论 / Ungar 2009/2010 "Hyperbolic Geometry" Ch.4. 与 v336 关键差异: v336 cyclic c→0.7 触发 saturation → R36p collapse; v337 固定 c=1 避免 saturation. test_R@10=0.2740415592783505 (+136.1% vs v316_fixed baseline 0.11606), test_R@20=0.3827722293814433 (+154.3%), test_NDCG@20=0.1875952101245369 (+102.5%), n_eval=24832 (R35b PASS), Stage 1/2/3/4 全部 MD5 ≠ v316_fixed baseline, R51+ RUN 1+2 字符级完全一致 (8/8 核心指标 diff < 1e-15). **历史意义**: R36h ceiling 第 66 次验证 (Stage 1 端纯曲率变更真正第一次打破 0.11356 ceiling + 136.1% 突破), 取代 v316_fixed Anisotropy Std-Matching (test_R@10=0.11606). 旧 v316_fixed baseline 已备份到 /home/wlia0047/hj82_scratch2/wenyu/backup_v316_fixed_baseline_v337_promote/.
SAVE_DIR_ROOT = os.path.join(_CONFIG_DIR, "out/decoder/instruments_hgrec_configs/hgrec_{}/".format(MECHANISM_NAME))
CONFIG_PATH = os.path.join(_CONFIG_DIR, "configs/decoder_instruments_hgrec_{}.gin".format(MECHANISM_NAME))
BEST_CKPT_PATH = os.path.join(SAVE_DIR_ROOT, "best_ckpt.pt")
USE_HGREC_ARCH = True  # R36i: 强制 HG-Rec 路径

# === Stage 0 / Stage 1 / Stage 2 路径 (相对于 curvature_base/) ===
ITEM_JSON_PATH = os.path.join(_CONFIG_DIR, "dataset/Instruments.item.json")  # R44 数据集
ITEM_EMB_NPY = os.path.join(_CONFIG_DIR, "dataset/Instruments/item_emb.npy")  # Stage 0 输出
ITEM_IDS_JSON = os.path.join(_CONFIG_DIR, "dataset/Instruments/item_ids.json")  # Stage 0 输出
RQVAE_OUT_DIR = os.path.join(_CONFIG_DIR, "out/rqvae/instruments")  # Stage 1 ckpt 目录
RQVAE_CKPT_PATH = os.path.join(RQVAE_OUT_DIR, "rqvae_final.pt")  # Stage 1 ckpt
RAW_SIDS_NPY = os.path.join(RQVAE_OUT_DIR, "sids_raw.npy")  # Stage 2.1 推理输出
SIDS_NPY = os.path.join(_CONFIG_DIR, "dataset/Instruments/sids_for_hgrec.npy")  # Stage 2.2 输出, Stage 3 输入
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
