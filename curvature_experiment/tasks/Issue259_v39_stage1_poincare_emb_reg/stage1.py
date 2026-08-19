"""v39 Stage 1 wrapper — RQ-VAE 训练 (DDP 4 卡, 启用 Poincaré Embedding Reg).

R40 强制 stage n+1 输入唯一来自 stage n 实时运行产物.
Stage 1 在主目录 train_rqvae_instruments.py 基础上启用 v39 新曲率正则项:
- USE_HYP_EMB_REG = True (启动 hyp_emb_reg_weight=0.1)
- OUT_DIR → rqvae_out_v39_hyp_emb_reg/ (与 v19/v38 路径隔离)
其他超参与 v19 完全一致 (c_end=0.7 curriculum, USE_CURRICULUM=True).
"""
import os
import subprocess
import sys

MAIN_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/curvature_experiment"
LOG_DIR = "/home/wlia0047/hj82_scratch2/wenyu/claude_tmp"
LOG_PATH = os.path.join(LOG_DIR, "curvature_exp_v39_stage1.log")
os.makedirs(LOG_DIR, exist_ok=True)

# 创建 v39 专用输出目录 (RQ-VAE ckpt 不影响 v19 主目录 baseline)
V39_OUT_DIR = "/home/wlia0047/hj82_scratch2/wenyu/rqvae_dataset/instruments/rqvae_out_v39_hyp_emb_reg"
os.makedirs(V39_OUT_DIR, exist_ok=True)

# === 主目录脚本 patch: USE_HYP_EMB_REG=True + OUT_DIR=v39 ===
# R30/R43 不允许 env 读超参, 所以用临时 patch 替换主目录脚本的常量定义
SCRIPT_PATH = os.path.join(MAIN_DIR, "scripts/train_rqvae_instruments.py")
with open(SCRIPT_PATH, "r") as f:
    src = f.read()

patched = src
# 替换 OUT_DIR 默认值
patched = patched.replace(
    'OUT_DIR = "/home/wlia0047/hj82_scratch2/wenyu/rqvae_dataset/instruments/rqvae_out_v19_cend_07"',
    f'OUT_DIR = "{V39_OUT_DIR}"',
)
# 替换 USE_HYP_EMB_REG 默认值
patched = patched.replace(
    "USE_HYP_EMB_REG = False",
    "USE_HYP_EMB_REG = True",
)

# 写到 v39 任务目录临时副本 (不污染主目录 git status)
V39_SCRIPT = os.path.join(os.path.dirname(__file__), "_v39_train_rqvae_instruments.py")
with open(V39_SCRIPT, "w") as f:
    f.write(patched)
print(f"[stage1] patched script → {V39_SCRIPT}")

cmd = [
    "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/torchrun",
    "--nproc_per_node=4",
    "--master_port=29506",
    V39_SCRIPT,
]

print(f"[stage1] launching: {' '.join(cmd)}")
print(f"[stage1] log → {LOG_PATH}")
print(f"[stage1] USE_HYP_EMB_REG=True, OUT_DIR={V39_OUT_DIR}")

with open(LOG_PATH, "w") as f:
    proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)

print(f"[stage1] exit={proc.returncode}")
sys.exit(proc.returncode)