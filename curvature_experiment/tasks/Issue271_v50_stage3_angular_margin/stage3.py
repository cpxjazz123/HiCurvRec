"""v50 Stage 3 wrapper — DDP 4 卡 HG-Rec T5 训练 + Inter-Layer Angular Margin loss.

v50 创新 (R36 严格化 v2 几何变换):
  - HALC v2 + v16 diff 默认 (v19 baseline)
  - + InterLayerAngularMarginLoss (R36 严格化 v2 几何变换)
  - 强制 encoder 6 层 hidden states 之间的 Poincaré 距离 > margin
  - radius=0.3, c=1.0, margin=0.1, reg_weight=0.01 (硬编码, R30/R43)
  - 不引入 schedule / weight decay (v40-v49 全部 R37 FAIL)

Stage 1+2 复用 v19 baseline (R40 复用 ckpt + numpy).
"""
import os
import subprocess
import sys

MAIN_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/curvature_experiment"
LOG_PATH = "/home/wlia0047/hj82_scratch2/wenyu/claude_tmp/curvature_exp_v50_stage3.log"
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

code_npy = os.path.join(MAIN_DIR, "dataset/Instruments/Instruments_v19_sids_for_hgrec.npy")
if not os.path.exists(code_npy):
    print(f"[stage3] FATAL: {code_npy} not found (R40)")
    sys.exit(1)
print(f"[stage3] v19 SID OK: {code_npy}")

cmd = [
    "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/torchrun",
    "--nproc_per_node=4",
    "--master_port=29593",
    os.path.join(MAIN_DIR, "scripts/train_decoder.py"),
    os.path.join(MAIN_DIR, "configs/decoder_instruments_hgrec_v50.gin"),
]

print(f"[stage3] launching: {' '.join(cmd)}")
print(f"[stage3] log → {LOG_PATH}")

env = os.environ.copy()
env["FORCE_HGREC"] = "1"
env["USE_ANGULAR_MARGIN_V50"] = "1"  # Issue271 v50: Inter-Layer Angular Margin (R36 严格化 v2)
env["SAVE_DIR_ROOT"] = "/home/wlia0047/ar57/wenyu/GeneRec/curvature_experiment/out/decoder/instruments_hgrec_v50/"
env["HF_HOME"] = "/home/wlia0047/hj82_scratch2/wenyu/.cache/huggingface"
env["HUGGINGFACE_HUB_CACHE"] = "/home/wlia0047/hj82_scratch2/wenyu/hf_models/hub"
env["TRANSFORMERS_CACHE"] = "/home/wlia0047/hj82_scratch2/wenyu/hf_models/hub"
env["TMPDIR"] = "/home/wlia0047/hj82_scratch2/wenyu/tmp"

with open(LOG_PATH, "w") as f:
    proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False, env=env)

print(f"[stage3] exit={proc.returncode}")
sys.exit(proc.returncode)
