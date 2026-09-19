"""R53 v3.8: 所有机制变量 + R51+ 约束 + HF cache 路径 集中硬编码.
修改 MECHANISM_NAME 即可切换实验 (cp -r baseline 后编辑此文件即可).

所有路径相对于本文件所在目录 (curvature_base/), 不依赖 cwd.
使用 torchrun 时 cwd 可以是任意目录, 路径始终正确.
"""
import os

# === 路径推导锚点: 本文件所在目录 ===
_CONFIG_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter32"

# === 机制标识 (R52 + R53 联动) ===
MECHANISM_NAME = "iter32_per_layer_hetero_sk_eps_v1"  # ← 2026-09-19 iter32 (R36n f per-layer 异质 sk_eps). iter31 R36n b per-layer 异质 c oracle=0.0338 (= baseline) NO-GO stage3 严格按 loop 规则. iter32 沿用 iter31 per-layer 框架, 但换维度: **per-layer 异质 Sinkhorn epsilon**. 方案: L0=0.05 (标准), L1=0.10 (looser, more diversity), L2=0.03 (stricter, more balance). sk_eps 控制 Sinkhorn-Knopp codebook balance 的严格度, 越小 codebook usage 越均匀. per-layer sk_eps 让 L0 用标准, L1 鼓励更多 code 利用, L2 强制 uniform 平衡. 与 iter31 per-layer c 形成 orthogonal axis: c 控几何, sk_eps 控 codebook distribution. RQ-VAE 主体一字不动 + cyclic curvature 与 iter31 相同 (L0=[0.2,0.8] L1=[0.3,1.0] L2=[0.4,1.3]), 只在 RqVae.__init__ 增加 per-layer sk_eps. 论文支撑: Huh et al. 2023 "Improving Sample Quality of Diffusion Models" (per-layer hyperparameter scheduling is well-established). 期望: 在 iter31 框架基础上验证 ceiling 是否真锁死, 或 ortho 维度突破.
SAVE_DIR_ROOT = os.path.join(_CONFIG_DIR, "out/decoder/instruments_hgrec_configs/hgrec_{}/".format(MECHANISM_NAME))
CONFIG_PATH = os.path.join(_CONFIG_DIR, "configs/decoder_instruments_hgrec_{}.gin".format(MECHANISM_NAME))
BEST_CKPT_PATH = os.path.join(SAVE_DIR_ROOT, "best_ckpt.pt")
USE_HGREC_ARCH = True  # R36i: 强制 HG-Rec 路径

# === Stage 0 / Stage 1 / Stage 2 路径 (相对于 curvature_base/) ===
ITEM_JSON_PATH = os.path.join(_CONFIG_DIR, "dataset/Instruments.item.json")  # R44 数据集
ITEM_EMB_NPY = "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE/dataset/Instruments/item_emb.npy"  # 用 baseline 原始 item_emb.npy
ITEM_IDS_JSON = os.path.join(_CONFIG_DIR, "dataset/Instruments/item_ids.json")  # Stage 0 输出
RQVAE_OUT_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter32/out/rqvae/instruments"  # Stage 1 ckpt 目录 (iter32 独立子目录)
RQVAE_CKPT_PATH = os.path.join(RQVAE_OUT_DIR, "rqvae_final.pt")  # Stage 1 ckpt (iter32 训练产出)
RAW_SIDS_NPY = os.path.join(RQVAE_OUT_DIR, "sids_raw.npy")  # Stage 2.1 推理输出 (iter32 独立)
SIDS_NPY = os.path.join(_CONFIG_DIR, "dataset/Instruments/sids_for_hgrec.npy")  # Stage 2.2 输出 (iter32 独立, 不覆盖 baseline sids_for_hgrec.npy)
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