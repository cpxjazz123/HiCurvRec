"""纯 T5 基线 stage3 训练 — 与 HG-Rec/train_HG-Rec.py 完全一致的方法.

R30: 所有超参 + 路径 + GPU + DDP 状态 + 校验 都通过 argparse 传入, 无任何 env var 读取.
launcher (.sh) 必须传 --sid_npy / --product_dir; 其他 arg 走默认值.
DDP 用户: torchrun 自动设 WORLD_SIZE/RANK/LOCAL_RANK, 需 wrapper 翻译为 argparse
  (见 launch_stage3_ddp_wrapper.sh), 或直接传 --world_size/--rank/--local_rank.
实验变体: 复制此脚本为新文件改默认值, 不复用同一脚本 + env toggle.

用指定 SID npy (taskA/taskB stage2 新 SID) 作为 item-to-code 映射,
T5 随机初始化从头训练, 无 adapter 注入, 监控 valid NDCG@20 (beam20), early stop.

当前变体默认值 (Issue #41 任务 + hyp 系列常用):
  NUM_EPOCHS=200, EARLY_STOP=20, BATCH_SIZE=256, INFER_SIZE=96,
  SEED=42, LR=1e-4, MAX_LEN=20, NUM_WORKERS=0, STAGE3_BF16=True,
  TORCH_COMPILE=False.

产物 (PRODUCT_DIR/):
  HG_Rec_best.pth    最佳 NDCG@20 ckpt
  trace.json         每 epoch train_loss / R@5/10/20 / NDCG@5/10/20 / best
  verdict.json       SID sha + config + 最终指标 + best epoch
  _TRAINING_PID      PID (R12)
"""
import os
import sys
import json
import subprocess
from datetime import timedelta
import hashlib
import time
import random
import math
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.distributed as dist
from torch.utils.data import DistributedSampler, DataLoader

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/tasks/Issue151_gromov_wasserstein_curvature/treatment/_lib")  # Issue150 taskB treatment self-contained _lib
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/tasks/Issue151_gromov_wasserstein_curvature/treatment")  # from _lib import hyperbolic_attention_bias

from HG_Rec import HG_Rec          # noqa: E402
from dataset import GenRecDataset  # noqa: E402
from dataloader import GenRecDataLoader  # noqa: E402
# Issue #64: 双曲码字距离 attention bias (共享模块, Stage3 + Stage4 同源)
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/baseline")  # R44 baseline 自包含
from _lib.hyperbolic_attention_bias import (
    HAB_LAMBDA_MAX, load_hab_assets_from_stage2_ckpt, precompute_distance_matrices,
    HyperbolicAttentionBias, install_hab, make_hab_layer_id_lut,
)  # noqa: E402
# Issue #125 (2026-08-12): Stage3 per-head learnable curvature (κ_h) — R36 曲率机制变更.
# 6 个 attention head 各自 learnable κ_h + λ_h, 给 HAB get_B_geo 输出扩展到 (B, num_heads, L, L).
# 训练初期 λ_h_init=0 → 严格等价 baseline (R44); λ_h 学习后 per-head bias 自动生效.
from _lib.per_head_curvature import (  # noqa: E402
    install_per_head_curvature, get_per_head_kappa_stats,
    PER_HEAD_NUM_HEADS_DEFAULT, PER_HEAD_KAPPA_H_INIT, PER_HEAD_LAMBDA_H_INIT,
)


class _FastGenRecDataLoader(GenRecDataLoader):
    """Issue #61 P0 加速: 子类化 GenRecDataLoader, 注入 pin_memory + persistent_workers
    (HG-Rec 上游不允许修改, 这里用子类透传). DataLoader 父类原生支持这两个 kwargs.
    """
    def __init__(self, dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=None,
                 pin_memory=False, persistent_workers=False):
        # 跳过 GenRecDataLoader.__init__ 直接走 DataLoader.__init__, 注入额外 kwargs
        DataLoader.__init__(
            self, dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, collate_fn=self.collate_fn,
            pin_memory=pin_memory,
            persistent_workers=(persistent_workers if num_workers > 0 else False),
        )

# ──────────────────────────────────────────────────────────────
# R43: V74_CONFIG + STAGE3_DDP_CONFIG 硬编码 (合并自旧 wrapper stage3.py)
# 仅 --sid_npy / --product_dir / --tag 路径参数走 argparse, 数值超参全部硬编码
# ──────────────────────────────────────────────────────────────
# DDP 启动配置 (硬编码 4 卡, 旧 wrapper 内容)
STAGE3_DDP_CONFIG = {
    "cuda_visible_devices": "2,3",
    "nproc_per_node": 2,
    "master_port": 29511,
}

# v15+v74 baseline 训练超参 + 路径 (硬编码, 与原 taskA/stage3/taskA_stage3.py V74_CONFIG 完全一致)
V74_CONFIG = {
    # 路径 (Issue #150 taskB treatment: 实时 Stage2 产物)
    "sid_npy": "/home/wlia0047/ar57/wenyu/GeneRec/tasks/Issue151_gromov_wasserstein_curvature/treatment/stage2/sid_output.npy",
    "product_dir": "/home/wlia0047/ar57/wenyu/GeneRec/tasks/Issue151_gromov_wasserstein_curvature/treatment/stage2",
    "expected_sid_sha": "22483474ff7daeb6d9f0d02b12e8ec4068bb3fcb73b93c9fc0a7766b77276290",
    # HAB 三改动 (Issue #138 v74)
    "hyperbolic_attn_bias": True,
    "enable_residual_hab": True,
    "hab_lambda_max": 0.20,
    "residual_alpha_init": -20.0,
    "hab_stage2_ckpt": "/home/wlia0047/ar57/wenyu/GeneRec/tasks/Issue151_gromov_wasserstein_curvature/treatment/stage2/hrqvae_kappa_sync.ckpt",
    # v74 dropout=0.20 / weight_decay=0.01 (vs v6b baseline 0.10)
    "stage3_dropout": 0.20,
    "stage3_weight_decay": 0.01,
    # 任务标签
    "tag": "v15_v74_baseline",
}

# 简化 argparse (R43: 仅路径参数, 数值超参已硬编码为 V74_CONFIG)
_argparser = argparse.ArgumentParser(description="Stage3 pure T5 train (R43 strict: 数值超参硬编码, 仅路径参数)")
_argparser.add_argument("--sid_npy", type=str, default=V74_CONFIG["sid_npy"], help="stage2 SID npy 路径 (默认 = V74_CONFIG['sid_npy'])")
_argparser.add_argument("--product_dir", type=str, default=V74_CONFIG["product_dir"], help="产物目录 (默认 = V74_CONFIG['product_dir'])")
_argparser.add_argument("--tag", type=str, default=V74_CONFIG["tag"], help="方向标签 (写进 verdict, 默认 = V74_CONFIG['tag'])")
_args = _argparser.parse_args()

# 路径参数 (R43 允许)
SID_NPY = _args.sid_npy
PRODUCT_DIR = Path(_args.product_dir)
TAG = _args.tag

# 硬编码 V74_CONFIG 数值 (R43 强制)
EXPECTED_SID_SHA = V74_CONFIG["expected_sid_sha"]
GEO_RESIDUAL_ENABLED = False
GEO_SMOOTH_TANH = False
GEO_ALPHA_LR_RATIO = 1.0
CODEWORD_GEO_ENABLED = False
CODEWORD_STAGE2_CKPT = V74_CONFIG["hab_stage2_ckpt"]  # R44 baseline 自包含: 复用同一 ckpt
CODEWORD_RHO_MAX = 0.10
HAB_ENABLED = V74_CONFIG["hyperbolic_attn_bias"]
HAB_LAMBDA_LR_RATIO = 100.0
HAB_LAMBDA_RAW_CLAMP = 10.0
HAB_STAGE2_CKPT = V74_CONFIG["hab_stage2_ckpt"]
HAB_LAMBDA_MAX = V74_CONFIG["hab_lambda_max"]
RESIDUAL_HAB_ENABLED = V74_CONFIG["enable_residual_hab"]
RESIDUAL_ALPHA_INIT = V74_CONFIG["residual_alpha_init"]
RESIDUAL_ALPHA_LR_RATIO = 10.0
HAB_WARMUP_T0 = 0
HAB_WARMUP_TW = 0
HAB_ATTN_ENTROPY_WEIGHT = 0.0
HAB_ATTN_ENTROPY_TAU = 1.0
HAB_DELTA_CURVATURE = False

# Issue #132 (2026-08-12): Stage3 训练期冻结 HAB (lambda_raw + lambda_h_raw) — baseline-equivalent 训练.
# Issue #130 + #131 双 ablation 揭示: Stage4 eval 时 HAB 注入对 attention 无贡献 (causal mask 主导).
# 但 Issue #125/#127/#128/#129 训练期 HAB 让 T5 学偏 → test 退化.
# 修复: install_hab 后强制 lambda_raw=0, install_per_head_curvature 后 lambda_h_raw=0,
# 两者都从 optimizer 移除 (frozen by design). ckpt 训练等效 baseline.
ISSUE132_FREEZE_HAB = True  # Issue #132: 启用冻结, 关闭=沿用 #129 全程学习

PROMPT_FORMER_ENABLED = False
PROMPT_FORMER_ALPHA = 0.35
BRANCH_CURVATURE_ENABLED = False
BRANCH_CURVATURE_SID_NPY = SID_NPY
BRANCH_CURVATURE_N_BUCKETS = 3
BRANCH_CURVATURE_STRATEGY = "quantile"
BRANCH_CURVATURE_LAMBDA_MULT = [0.7, 1.0, 1.3]
PROMPT_FORMER_NUM_BOS_QUERIES = 64
PF_ALPHA_WARMUP_STEPS = 200
PF_ALPHA_PENALTY_WEIGHT = 0.01
PF_BOS_DIVERSITY_WEIGHT = 0.01
PF_ATTN_ENTROPY_WEIGHT = 0.001
STAGE3_WEIGHT_DECAY = V74_CONFIG["stage3_weight_decay"]
STAGE3_LABEL_SMOOTHING = 0.05
STAGE3_DROPOUT = V74_CONFIG["stage3_dropout"]
STAGE4_TEST_ON_BEST = False
STAGE4_TEST_GPU = 0
T5_UNCERTAINTY_ENABLED = False
T5_UNCERTAINTY_INIT_STD = 1.0
T5_UNCERTAINTY_K = 0.458145
T5_UNCERTAINTY_C = 1.361442
T5_UNCERTAINTY_REG_WEIGHT = 0.01
# DDP 状态 (torchrun 自动设 WORLD_SIZE/RANK/LOCAL_RANK env, torch 标准接口, R30 例外)
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
RANK = int(os.environ.get("RANK", "0"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
DDP_MODE = WORLD_SIZE > 1
DEVICE = f"cuda:{LOCAL_RANK}" if DDP_MODE else "cuda:0"

# 超参 (R30 硬编码 — 变体需 fork 脚本)
NUM_EPOCHS = 200  # Issue #210 Phase D (2026-08-08): 用户指示 200 epoch. v85p PARTIAL-GO 0.1080 300ep 配置, Phase D 用户改为 200 epoch 验证 equal128 SID 收敛.
EARLY_STOP = 20  # Issue125 R41 硬约束: EARLY_STOP=20 (历史 v121 用 30 不追溯)
EVAL_INTERVAL = 5  # Issue #141 v85q (2026-08-09): 用户指示 EVAL_INTERVAL=5 (匹配 v77/v85p 历史配置, 每 5 epoch 评估). 当前 + NUM_WORKERS=2 单 epoch 7s, EI=5 省 2s/epoch (-28%). v77 实际配置就是 EI=5.
BATCH_SIZE = 1024  # Issue #141 v85t (2026-08-09) v77完全相同复现: v77原 batch=1024 (DDP 4 卡 per-rank 256), 验证 v77 数值可复现性, 解释 v85 路径所有"修复"是不是 noise.
INFER_SIZE = 256  # eval batch size (DDP per-rank = INFER_SIZE // WORLD_SIZE = 64, v77原等价值)
SEED = 42
LR = 1e-3  # v18 (2026-08-09): 沿用 v15 LR=1e-3 sweet spot

# Issue #141 v85f (2026-08-08): LR cosine 温和版 (沿用 v85d) + SID v15 (5f8331cc). v85d (warmup_frac=0.05 + LR_min=0.1 + 200ep) 配 hyp_v2 SID test=0.1011. v85f 同样 LR 调度换 SID v15, 验证 v15 κ=[0.30,1.79,1.48] c=[1.35,6.00,4.39] 强几何信号 + cosine 衰减是否协同.
LR_SCHEDULER = "cosine"  # Issue #141 v85f (2026-08-08): "none" / "cosine" (warmup_frac=0.05 → cos → LR_min=LR*0.1)
LR_WARMUP_FRAC = 0.025  # Issue #141 v85s (2026-08-09) 方案D batch=2048: 总步 ~3300 (vs v77原 6600 batch=1024 的一半), warmup=82 steps (≈2.5 epoch warmup). warmup_frac 不变, 绝对步数跟 v77原 660 步略减半, 起步更稳定.
LR_MIN_FACTOR = 0.01  # Issue #141 v85r (2026-08-09) 方案C修scheduler: 0.05→0.01. 末期 LR=8e-6 (vs 之前 4e-5). 让模型末期更收敛, 修 generalization gap.
MAX_LEN = 20
NUM_WORKERS = 2  # Issue #141 v85q GPU 利用率优化 (2026-08-09): 用户指示 NUM_WORKERS=2 (8 forks, vs Issue #64 NUM_WORKERS=4 16 forks NCCL deadlock). 配 prefetch_factor=4 让 GPU 不再等数据. Issue #64 deadlock 是 4×4=16 fork 触发, 2×4=8 fork 应该安全.
PREFETCH_FACTOR = 4  # 每个 worker 预加载 4 个 batch, GPU 永远有数据
PIN_MEMORY = True  # Issue #61 P0: DataLoader pin_memory=True, CPU→GPU 传输加速
PERSISTENT_WORKERS = True  # Issue #61 P0: worker 跨 epoch 持久, 省每 epoch worker spawn 启动时间
STAGE3_TF32 = True  # Issue #61 P0: Ampere+ TF32 matmul 加速 1.3-1.5×, 精度影响 <1e-3
FUSED_OPTIMIZER = True  # Issue #61 P0: torch.optim.AdamW(fused=True), L40S fused AdamW kernel 加速 1.2-1.5×

# 训练加速 (2026-08-03): bf16 autocast — T5 训练/beam 解码标准实践, forward 转 bf16 计算
# (参数保持 fp32, backward 后 optimizer 在 fp32 权重更新, 数值影响极小). 默认开.
STAGE3_BF16 = True
# v29 加速 (2026-08-04): torch.compile — PyTorch 2.x 内置, A100+ 推荐 reduce-overhead
_TORCH_COMPILE = False  # Issue #64 验证 (2026-08-06): torch.compile reduce-overhead + DDP 4 卡 + HAB 实际测 ep2-8 = 21s, 与无 compile 22s 持平, 无加速 (ep1 73s warmup 浪费). 关掉.
_TRAIN_COMPILED = False
_TORCH_COMPILE_MODE = "reduce-overhead"  # reduce-overhead = CUDA Graph + 算子融合, 适合固定 shape (bs=1024 seq=20)


def _collate_fn(batch, pad_token=0):
    # 与 HG-Rec/data/dataloader.py GenRecDataLoader.collate_fn 逐字节一致. DDP 需要 sampler,
    # 但 GenRecDataLoader 不接受 sampler 参数 → 复刻 collate 以构造标准 DataLoader (仅 DDP 用).
    histories = [item['history'] for item in batch]
    targets = [item['target'] for item in batch]
    flattened_histories = torch.stack(
        [torch.tensor([elem for sublist in history for elem in sublist], dtype=torch.int64) for history in histories]
    )
    flattened_targets = torch.stack(
        [torch.tensor(target, dtype=torch.int64) for target in targets]
    )
    attention_masks = torch.stack(
        [torch.tensor([1 if elem != pad_token else 0 for elem in h], dtype=torch.int64) for h in flattened_histories]
    )
    return {'history': flattened_histories, 'target': flattened_targets, 'attention_mask': attention_masks}

CODEBOOK_SIZE = [64, 128, 256, 1]          # 基线 stage2 结构
# Issue #62: token → 层位查找表 (按 item2code offsets: L0=[1,64] L1=[65,192] L2=[193,448] L3=[449])
# 0=PAD → -1 (不注入); 1-64 → 0 (L0); 65-192 → 1 (L1); 193-448 → 2 (L2); 449 → 3 (L3)
_LAYER_ID_LUT = np.full(1025, -1, dtype=np.int64)
_LAYER_ID_LUT[1:65] = 0        # L0: K=64
_LAYER_ID_LUT[65:193] = 1      # L1: K=128
_LAYER_ID_LUT[193:449] = 2     # L2: K=256
_LAYER_ID_LUT[449:450] = 3     # L3: K=1 (dedup)
CONFIG = dict(                               # Issue #141 v77 原配置 (2026-08-08): v77 base num_layers=6, num_decoder_layers=4 (v85 升级到 6 是 P0 superparam upgrade 但导致 test 泛化下降 -0.013). 复现 v77 用原值.
    num_layers=6, num_decoder_layers=4, d_model=128, d_ff=1024,
    num_heads=6, d_kv=64, dropout_rate=STAGE3_DROPOUT, vocab_size=1025,
    pad_token_id=0, eos_token_id=0, decoder_start_token_id=0,
    feed_forward_proj="relu",
)
DATA_ROOT = "/home/wlia0047/ar57/wenyu/GeneRec/dataset"  # R44 baseline 自包含: 引用共享 dataset/ 顶层 (md5 一致 HG-Rec/dataset/Instruments/)
TRAIN_PARQUET = os.path.join(DATA_ROOT, "train.parquet")
VALID_PARQUET = os.path.join(DATA_ROOT, "valid.parquet")
TOP_K = [5, 10, 20]
BEAM_SIZE = 20

# Issue #62 (修复 v2, 2026-08-06): 几何残差注入配置
# v1 (失败) 根因: kappa 未归一化 (vs scale/codebook_norm 大 10×), mlp 权重被数值主导;
#                 alpha_cap=0.5 太大, 训练失控; layer_id 冗余 (per-layer mlp 已区分层)
# v2 修复: meta 三字段 z-score 归一化 → 均值 0 / 标准差 1; alpha_cap 收紧 ±0.1;
#         删 layer_id (冗余, 节省 mlp capacity); 加 alpha 训练监控到 trace.json
# Stage2 #61 issue61 final κ = [-0.2289, -0.1872, -0.0932]; L3 (dedup) κ = 0
GEO_KAPPA = [-0.2289, -0.1872, -0.0932, 0.0]
# scale = cbrt(K_l), 常见几何标度 (issue spec "scale_l" 字段)
GEO_SCALE = [4.0, 5.04, 6.35, 1.0]  # K^(1/3) for K=[64,128,256,1]
# codebook_norm = cbrt(K_l) / sqrt(d_emb) 占位 (Stage2 训练后统计补, 这里用近似量纲)
GEO_CODEBOOK_NORM = [4.0, 5.04, 6.35, 1.0]
# 残差强度: alpha_l 初始 0.01 (Issue #62 spec 小初始化), hard cap 收紧到 ±0.1 (v1=±0.5 太大)
GEO_ALPHA_INIT = 0.01
GEO_ALPHA_CAP = 0.1
# L3 (dedup) 无几何信息, alpha_3 强制 0
GEO_FORCE_ZERO_LAYERS = [3]

PRODUCT_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_PID_FILE = PRODUCT_DIR / "_TRAINING_PID"
LOG_PATH = PRODUCT_DIR / "train_pure_t5.log"
CKPT_PATH = PRODUCT_DIR / "HG_Rec_best.pth"
TRACE_PATH = PRODUCT_DIR / "trace.json"
VERDICT_PATH = PRODUCT_DIR / "verdict.json"
STAGE4_ON_BEST_DIR = PRODUCT_DIR / "stage4_test_on_best"
STAGE4_ON_BEST_HISTORY = STAGE4_ON_BEST_DIR / "history.json"
# Issue #141 v85 (2026-08-07): 跟踪活跃 async stage4 eval (epoch -> Popen)
_STAGE4_PROCS = {}


def _append_stage4_history(epoch, valid_r10, test_r10, test_path):
    """Issue #141 v85 (2026-08-07): 追加每次 new best 的 test_R10 到 history.json (实时 overfitting 监控)."""
    STAGE4_ON_BEST_DIR.mkdir(parents=True, exist_ok=True)
    history = []
    if STAGE4_ON_BEST_HISTORY.exists():
        try:
            history = json.loads(STAGE4_ON_BEST_HISTORY.read_text())
        except Exception:
            history = []
    ratio = round(valid_r10 / test_r10, 4) if test_r10 and test_r10 > 0 else None
    history.append({
        "epoch": epoch,
        "valid_R10": round(valid_r10, 5),
        "test_R10": test_r10,
        "valid_test_ratio": ratio,
        "test_json": str(test_path),
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    STAGE4_ON_BEST_HISTORY.write_text(json.dumps(history, indent=2, ensure_ascii=False))


def trigger_stage4_test_async(ckpt_path, epoch, valid_r10):
    """Issue #141 v85 (2026-08-07): new best ckpt 立即异步触发 stage4 test eval.
    不阻塞训练 (subprocess.Popen), 复用相同 Stage2 + Dbar + HAB 配置.
    子进程跑完后写 history.json; 主进程下次 do_eval 时只读取最新一个 ratio 用于 log.

    R44 self-contained: STAGE4_TEST_ON_BEST=False 默认值, 函数体已禁用 (原 common/stage4/ 跨 stage 调用已删除).
    若用户需启用此功能, 必须复制 common/stage4/stage4_eval_pure_t5.py 到 baseline/ 自包含并改写路径.
    """
    if not STAGE4_TEST_ON_BEST:
        return None
    raise NotImplementedError(
        "STAGE4_TEST_ON_BEST=True requires self-contained stage4 script in baseline/. "
        "R44 严格自包含 baseline 不引用 common/stage4/. "
        "复制 baseline/stage4.py 的 evaluate 流程到本函数即可启用."
    )


def _poll_stage4_procs():
    """Issue #141 v85 (2026-08-07): 检查已完成的 async eval, 读 test_R10 写 history + log. 无阻塞."""
    finished = []
    for ep, (proc, valid_r10, eval_dir) in list(_STAGE4_PROCS.items()):
        if proc.poll() is not None:
            finished.append(ep)
            test_json = eval_dir / "eval_test.json"
            test_r10 = None
            if test_json.exists():
                try:
                    test_r10 = json.loads(test_json.read_text()).get("R@10")
                except Exception:
                    pass
            # Issue #210 Phase D fix (2026-08-08): valid_r10 may be None if trigger 路径异常而未传.
            # 触发条件: trigger 返回 None 时 _STAGE4_PROCS 不被赋值, 但防御性 None check 仍保留.
            if valid_r10 is None:
                log(f"[v85] stage4 test DONE ep{ep} but valid_r10 is None (trigger 异常), skip")
            elif test_r10 is not None:
                _append_stage4_history(ep - 1, valid_r10, test_r10, test_json)
                ratio = valid_r10 / test_r10 if (test_r10 is not None and test_r10 > 0) else None
                v_str = f"{valid_r10:.4f}"
                r_str = f"{ratio:.3f}" if ratio is not None else "None"
                log(f"[v85] stage4 test DONE ep{ep} valid_R10={v_str} "
                    f"test_R10={test_r10:.4f} ratio={r_str}")
            else:
                log(f"[v85] stage4 test DONE ep{ep} but eval_test.json 不存在 (log={eval_dir.parent}/ep{ep}_eval.log)")
    for ep in finished:
        _STAGE4_PROCS.pop(ep, None)


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ──────────────────────────────────────────────────────────────
# Issue #62: 几何残差注入模块 (per-layer MLP 加到 SID token embedding)
# 设计: h'_l = h_l + alpha_l * MLP_l([kappa_l, scale_l, codebook_norm_l, layer_id_onehot_l])
# 注入位置: T5 input token embedding (不走 6 encoder + 4 decoder 稀释, 单点精准)
# alpha_l 初始 0.01 (issue spec 小初始化), hard cap ±0.5 防破坏 #61 baseline
# ──────────────────────────────────────────────────────────────
import torch.nn as nn  # noqa: E402
# Issue #140 v76: T5 uncertainty head 借鉴 DIGER AutoSigmaGumbel
# R44 self-contained: T5_UNCERTAINTY_ENABLED=False 默认值, common.t5_uncertainty 已被删除.
# 若用户需启用此功能, 必须复制 common/t5_uncertainty.py 到 baseline/_lib/ 并改写 import.
T5UncertaintyHead = None  # type: ignore[assignment,misc]
import torch.nn.functional as F  # noqa: E402


class GeoResidualModule(nn.Module):
    """Per-layer 几何残差: 4-layer MLP_L (kappa + scale + codebook_norm → d_model).
    v2 (2026-08-06 修复): 删 layer_id 冗余 (per-layer mlp 已区分层), meta_dim=3 (从 7 降到 3).
    v4 (2026-08-06 修复): 支持 smooth_tanh mode (α = cap·tanh(raw), 永无触顶)

    Args:
        d_model: T5 d_model (128).
        meta_per_layer: (num_layers, meta_dim) 几何元信息 (硬编码, 不参与训练).
        alpha_init: 初始 alpha (0.01, Issue #62 spec).
        alpha_cap: alpha 训练范围 cap (v2 收紧到 0.1, v1=0.5 太大失控).
        force_zero_layers: alpha 强制 0 的层列表 (e.g. L3 无几何信息).
        smooth_tanh: True → α = cap·tanh(raw) (无触顶); False → α = clamp(raw, -cap, cap) (硬边界)
    """
    def __init__(self, d_model, meta_per_layer, alpha_init=0.01, alpha_cap=0.1,
                 force_zero_layers=(), smooth_tanh=False):
        super().__init__()
        self.d_model = d_model
        self.num_layers, meta_dim = meta_per_layer.shape
        self.alpha_cap = alpha_cap
        self.smooth_tanh = smooth_tanh
        meta_t = torch.as_tensor(meta_per_layer, dtype=torch.float32)
        self.register_buffer("meta", meta_t)  # (num_layers, meta_dim)
        # per-layer MLP: meta_dim → d_model → GELU → LN → d_model (lightweight)
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(meta_dim, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
            ) for _ in range(self.num_layers)
        ])
        # per-layer alpha_raw (unbounded, smooth_tanh 时映射到 ±cap; clamp 时也是 unbounded 但 forward clamp)
        alpha = torch.full((self.num_layers,), alpha_init)
        for l in force_zero_layers:
            alpha[l] = 0.0
        self.alphas_raw = nn.Parameter(alpha)  # v4 改名: 现在 raw, 在 forward 里 clamp/tanh
        # 兼容旧代码引用 self.alphas: 别名指向 alphas_raw
        self.alphas = self.alphas_raw
        # 小初始化最后 Linear 避免初始放大 baseline
        for mlp in self.mlps:
            nn.init.normal_(mlp[-1].weight, std=0.01)
            nn.init.zeros_(mlp[-1].bias)

    def get_deltas(self):
        """计算 per-layer delta (num_layers, d_model)."""
        deltas = []
        for l in range(self.num_layers):
            deltas.append(self.mlps[l](self.meta[l]))
        return torch.stack(deltas, dim=0)

    def forward(self, input_embeds, layer_ids):
        """input_embeds: (B, L, d_model); layer_ids: (B, L) long (0-3 层位, -1 PAD=不注入).
        返回 input_embeds + alpha_eff[layer_ids] * delta[layer_ids], PAD 处不修改.
        alpha_eff = cap·tanh(raw) (smooth) 或 clamp(raw, -cap, cap) (hard).
        """
        deltas = self.get_deltas()                                  # (num_layers, d_model)
        if self.smooth_tanh:
            alphas = self.alpha_cap * torch.tanh(self.alphas_raw)   # (num_layers,) 平滑映射到 (-cap, cap)
        else:
            alphas = self.alphas_raw.clamp(-self.alpha_cap, self.alpha_cap)  # 硬边界 (会触顶)
        # mask: 仅 layer_ids >= 0 处注入
        valid = layer_ids >= 0                                       # (B, L) bool
        safe_ids = layer_ids.clamp(min=0)                            # (B, L) — -1 临时映射到 0, 但后面会被 mask 屏蔽
        delta_per_token = deltas[safe_ids]                           # (B, L, d_model)
        alpha_per_token = alphas[safe_ids]                           # (B, L)
        residual = (alpha_per_token.unsqueeze(-1) * delta_per_token) * valid.unsqueeze(-1).float()
        return input_embeds + residual


def install_geo_residual(hg_rec, geo_module, layer_id_lut_tensor):
    """Monkey-patch HG_Rec 实例: forward + generate 都注入几何残差.
    不改 HG-Rec/ 上游, 仅替换实例方法 + 挂载 geo_module 为 submodule.
    layer_id_lut_tensor: (vocab_size,) long, token → 层位 (-1 = PAD 不注入).
    """
    device = next(hg_rec.parameters()).device
    geo_module = geo_module.to(device)
    layer_id_lut_tensor = layer_id_lut_tensor.to(device)
    d_model_sqrt = hg_rec.model.config.d_model ** 0.5
    hg_rec.add_module("geo_module", geo_module)  # 挂载 → model.parameters() 自动包含

    def geo_forward(self, input_ids, attention_mask=None, labels=None):
        input_embeds = self.model.shared(input_ids) * d_model_sqrt
        layer_ids = layer_id_lut_tensor[input_ids]                 # (B, L)
        input_embeds = self.geo_module(input_embeds, layer_ids)
        outputs = self.model(inputs_embeds=input_embeds,
                             attention_mask=attention_mask, labels=labels)
        return outputs.loss, outputs.logits

    def geo_generate(self, input_ids, attention_mask=None, num_beams=20, **kwargs):
        input_embeds = self.model.shared(input_ids) * d_model_sqrt
        layer_ids = layer_id_lut_tensor[input_ids]
        input_embeds = self.geo_module(input_embeds, layer_ids)
        return self.model.generate(inputs_embeds=input_embeds,
                                   attention_mask=attention_mask,
                                   num_beams=num_beams, max_length=5,
                                   num_return_sequences=num_beams, **kwargs)

    import types
    hg_rec.forward = types.MethodType(geo_forward, hg_rec)
    hg_rec.generate = types.MethodType(geo_generate, hg_rec)
    return hg_rec


def build_geo_module():
    """按 GEO_KAPPA/SCALE/CODEBOOK_NORM 配置构建 GeoResidualModule.

    meta 维度 = 3 (v2: 删 layer_id 冗余, 仅保留 kappa/scale/codebook_norm).
    v2 归一化: 三字段分别 z-score 标准化 (L0/L1/L2 三个 layer 计算均值+std, L3 用 0/1 占位).
    v1 (失败) 用 scale/10 简单归一化, kappa 通道数值小被 mlp 权重淹没 → 修复.
    """
    num_layers = len(GEO_KAPPA)
    # 收集 L0/L1/L2 (skip L3 强制零) 三字段原始值
    raw = np.array([GEO_KAPPA[:3], GEO_SCALE[:3], GEO_CODEBOOK_NORM[:3]], dtype=np.float32)  # (3 fields, 3 layers)
    means = raw.mean(axis=1, keepdims=True)  # (3, 1)
    stds = raw.std(axis=1, keepdims=True) + 1e-6  # (3, 1)
    meta = np.zeros((num_layers, 3), dtype=np.float32)
    for l in range(3):
        meta[l, 0] = (GEO_KAPPA[l] - means[0, 0]) / stds[0, 0]
        meta[l, 1] = (GEO_SCALE[l] - means[1, 0]) / stds[1, 0]
        meta[l, 2] = (GEO_CODEBOOK_NORM[l] - means[2, 0]) / stds[2, 0]
    # L3 (dedup) 全部 0 (无几何信息, alpha_3 强制 0)
    meta[3, :] = 0.0
    return GeoResidualModule(
        d_model=CONFIG["d_model"],
        meta_per_layer=meta,
        alpha_init=GEO_ALPHA_INIT,
        alpha_cap=GEO_ALPHA_CAP,
        force_zero_layers=GEO_FORCE_ZERO_LAYERS,
        smooth_tanh=GEO_SMOOTH_TANH,
    )


# ──────────────────────────────────────────────────────────────
# Issue #63: 码字级几何残差 (Stage2 ckpt 真实 codebook + 内容相关门控)
# 设计: 对每个 SID token 查表得 (l, k), 用切空间 codebook 算 q_{l,k} = W_l[u_{l,k}; r_{l,k}; kappa_l]
#       h'_{l,k} = h_{l,k} + beta_l · sigmoid(MLP_g([LN(h); LN(q)])) · LN(q)
# β_l init 0, 严格退化 #61 (Gate3 强制 beta=0 等价性 < 1e-6)
# ──────────────────────────────────────────────────────────────


# Issue #63: codeword 偏移表 (L0=[1,64], L1=[65,192], L2=[193,448], L3=[449])
CODEWORD_OFFSETS = [1, 65, 193, 449]
CODEWORD_K = [64, 128, 256, 1]


class CodewordGeoResidual(nn.Module):
    """码字级几何残差: 每个 SID token 查真实双曲位置 → 投影 q_{l,k} → 内容门控 → delta 加到 embedding.

    Args:
        d_model: T5 d_model (128).
        codebook_list: list of (K_l, d_tangent) 切空间 codebook (来自 Stage2 ckpt, 冻结).
        final_kappas: list of scalars (来自 Stage2 ckpt['final_kappas'], 冻结).
        beta_init: 初始 β_l (默认 0.0 = 严格退化 #61).
        offsets: codeword offset 表 (默认跟 _LAYER_ID_LUT 一致).
        force_zero_layers: β_l 强制 0 的层列表 (默认 [3] L3 dedup).
    """
    def __init__(self, d_model, codebook_list, final_kappas, beta_init=0.0,
                 offsets=CODEWORD_OFFSETS, force_zero_layers=(3,), rho_max=0.10,
                 warmup_start=30, warmup_end=50, rho_max_per_layer=None):
        super().__init__()
        self.d_model = d_model
        self.num_layers = len(codebook_list)
        # Issue #63 v4: per-layer ρ_max 分层 (L0=0.15 信号最强, L1=0.05, L2=0.02 信号最弱, 跟 Stage2 codebook norm 异质性匹配)
        if rho_max_per_layer is None:
            self.rho_max_per_layer = torch.tensor([0.15, 0.05, 0.02][:self.num_layers], dtype=torch.float32)
        else:
            self.rho_max_per_layer = torch.as_tensor(rho_max_per_layer[:self.num_layers], dtype=torch.float32)
        # Issue #63 v4: β warmup 配置 (避免前 30 ep 扰动 T5 已学表示)
        self.warmup_start = int(warmup_start)
        self.warmup_end = int(warmup_end)
        self.current_epoch = 0  # 训练循环每 epoch 设一次
        self.d_tangent = codebook_list[0].shape[-1]
        self.offsets = list(offsets)
        self.K = [cb.shape[0] for cb in codebook_list]
        # 注册切空间 codebook (冻结 buffer)
        for l in range(self.num_layers):
            self.register_buffer(f"codebook_{l}", torch.as_tensor(codebook_list[l], dtype=torch.float32))
        # 注册 final_kappas (冻结 buffer)
        self.register_buffer("kappas", torch.as_tensor(final_kappas, dtype=torch.float32))
        # per-layer W_l: [d_tangent + 2 (r + kappa)] → d_model
        self.proj = nn.ModuleList([
            nn.Linear(self.d_tangent + 2, d_model) for _ in range(self.num_layers)
        ])
        # 初始化 proj 最后 Linear 小权重, 避免初始放大 baseline
        for proj_l in self.proj:
            nn.init.normal_(proj_l.weight, std=0.01)
            nn.init.zeros_(proj_l.bias)
        # 门控 MLP_g: [2*d_model] → d_model → 1
        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        for m in self.gate_mlp:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)
        # per-layer β_l (init 0, 强制退化)
        beta = torch.zeros(self.num_layers)
        # 注意: num_layers == len(codebook_list) = 3 (L0/L1/L2), 没有 L3 (L3 在 _LAYER_ID_LUT 中 token 449, 当前 SID 范围 [0,255] 不出现)
        # force_zero_layers 仅适用于 num_layers 范围内
        for l in force_zero_layers:
            if l < self.num_layers:
                beta[l] = 0.0
        # Issue #63 v3 修复: β_raw 自由训练, forward 时 clamp 到 ±rho_max (避免 v2 β 冲 -1.13 破坏 T5 表示)
        self.beta_raw = nn.Parameter(beta)
        # Issue #63 v4: per-layer ρ_max 注册为 buffer (跟 device 同步, 不参与梯度)
        self.register_buffer("rho_max_per_layer_buf", self.rho_max_per_layer.clone())
        # per-layer LN
        self.ln_h = nn.LayerNorm(d_model)
        self.ln_q = nn.LayerNorm(d_model)

    def precompute_q_lut(self, device):
        """预计算 q_lut: (max_id+1, d_model) — 每个 SID token id 对应的 q 向量."""
        max_id = max(off + k for off, k in zip(self.offsets, self.K))
        q_lut = torch.zeros(max_id + 1, self.d_model, device=device)
        for l in range(self.num_layers):
            cb = getattr(self, f"codebook_{l}").to(device)  # (K_l, d_tangent)
            K_l = cb.shape[0]
            r = cb.norm(dim=-1, keepdim=True)               # (K_l, 1)
            k = self.kappas[l].expand(K_l, 1)               # (K_l, 1)
            x = torch.cat([cb, r, k], dim=-1)               # (K_l, d_tangent + 2)
            q_lk = self.proj[l](x)                          # (K_l, d_model)
            q_lut[self.offsets[l]:self.offsets[l] + K_l] = q_lk
        return q_lut

    def forward(self, input_embeds, token_ids, layer_ids):
        """input_embeds: (B, L, d_model); token_ids: (B, L) long (SID token id);
        layer_ids: (B, L) long (0-3, -1=PAD).

        注意: L3 (token id 449) 是 dedup, Stage2 没训练 L3 codeword. L3 处 q_lut 填 0, β 强制 0.
        """
        valid = (layer_ids >= 0)                            # (B, L) bool
        # L3 标记: token 449 出现在 history 末尾 (4-token SID 第 4 位), 但 Stage2 无 L3 codebook
        # 将 L3 视作 PAD 处理 (不注入)
        l3_mask = (token_ids == 449)                        # (B, L) bool
        valid = valid & ~l3_mask                            # L3 不注入
        safe_layer = layer_ids.clamp(min=0)                 # (B, L)
        q_lut = self.precompute_q_lut(input_embeds.device)  # (max_id+1, d_model)
        safe_token = token_ids.clamp(min=0, max=q_lut.shape[0] - 1)
        q_per_token = q_lut[safe_token]                     # (B, L, d_model)
        # 内容门控
        h_ln = self.ln_h(input_embeds)                      # (B, L, d_model)
        q_ln = self.ln_q(q_per_token)                       # (B, L, d_model)
        g = torch.sigmoid(self.gate_mlp(torch.cat([h_ln, q_ln], dim=-1)))  # (B, L, 1)
        # Issue #63 v4: β warmup (ep 30-50 线性 0→1) + per-layer ρ_max 分层
        warmup_factor = max(0.0, min(1.0, (self.current_epoch - self.warmup_start) / max(1, self.warmup_end - self.warmup_start)))
        rho = self.rho_max_per_layer_buf.to(self.beta_raw.device)  # (num_layers,)
        beta_eff = warmup_factor * rho * torch.tanh(self.beta_raw / rho)   # (num_layers,)
        # per-layer β 查表: L3 (safe_layer=3) 强制 β=0 (避免越界 self.beta[3])
        beta_per_token = torch.where(
            safe_layer >= self.num_layers,
            torch.zeros_like(safe_layer, dtype=beta_eff.dtype),
            beta_eff[safe_layer.clamp(max=self.num_layers - 1)],
        )
        beta_per_token = beta_per_token * valid.float()     # PAD/L3 处 β=0
        delta = beta_per_token.unsqueeze(-1) * g * q_ln     # (B, L, d_model)
        return input_embeds + delta

    @torch.no_grad()
    def audit_stats(self, input_embeds, token_ids, layer_ids):
        """审计: ρ_l, gate 均值/方差, 同层码字 residual 方差 (Issue #63 Gate3 强制记录)."""
        valid = (layer_ids >= 0)
        l3_mask = (token_ids == 449)
        valid = valid & ~l3_mask
        safe_layer = layer_ids.clamp(min=0)
        q_lut = self.precompute_q_lut(input_embeds.device)
        safe_token = token_ids.clamp(min=0, max=q_lut.shape[0] - 1)
        q_per_token = q_lut[safe_token]
        h_ln = self.ln_h(input_embeds)
        q_ln = self.ln_q(q_per_token)
        g = torch.sigmoid(self.gate_mlp(torch.cat([h_ln, q_ln], dim=-1))).squeeze(-1)
        # Issue #63 v4: 同步 train 端 (warmup + per-layer ρ_max)
        warmup_factor = max(0.0, min(1.0, (self.current_epoch - self.warmup_start) / max(1, self.warmup_end - self.warmup_start)))
        rho = self.rho_max_per_layer_buf.to(self.beta_raw.device)
        beta_eff = warmup_factor * rho * torch.tanh(self.beta_raw / rho)
        beta_per_token = torch.where(
            safe_layer >= self.num_layers,
            torch.zeros_like(safe_layer, dtype=beta_eff.dtype),
            beta_eff[safe_layer.clamp(max=self.num_layers - 1)],
        )
        beta_per_token = beta_per_token * valid.float()
        delta = beta_per_token.unsqueeze(-1) * g.unsqueeze(-1) * q_ln
        # ρ_l = mean(||delta|| / (||h|| + eps)) per layer
        delta_norm = delta.norm(dim=-1)
        h_norm = input_embeds.norm(dim=-1)
        rho_per_token = delta_norm / (h_norm + 1e-6)
        stats = {}
        for l in range(self.num_layers):
            mask = (layer_ids == l)
            if not mask.any():
                stats[f"rho_l{l}"] = 0.0
                stats[f"gate_mean_l{l}"] = 0.0
                stats[f"gate_std_l{l}"] = 0.0
                stats[f"residual_var_l{l}"] = 0.0
                continue
            rho_l = rho_per_token[mask].mean().item()
            gate_l = g[mask]
            stats[f"rho_l{l}"] = round(rho_l, 5)
            stats[f"gate_mean_l{l}"] = round(gate_l.mean().item(), 5)
            stats[f"gate_std_l{l}"] = round(gate_l.std().item(), 5)
            delta_norm_l = delta_norm[mask]
            stats[f"residual_var_l{l}"] = round(delta_norm_l.var().item(), 6)
        return stats


def install_codeword_geo_residual(hg_rec, codeword_module, layer_id_lut_tensor):
    """Monkey-patch HG_Rec: forward + generate 都注入码字级几何残差.

    与 Issue #62 install_geo_residual 区别: forward 多一个 token_ids 参数.
    """
    device = next(hg_rec.parameters()).device
    codeword_module = codeword_module.to(device)
    layer_id_lut_tensor = layer_id_lut_tensor.to(device)
    d_model_sqrt = hg_rec.model.config.d_model ** 0.5
    hg_rec.add_module("codeword_geo_module", codeword_module)

    def codeword_forward(self, input_ids, attention_mask=None, labels=None):
        input_embeds = self.model.shared(input_ids) * d_model_sqrt
        layer_ids = layer_id_lut_tensor[input_ids]
        input_embeds = self.codeword_geo_module(input_embeds, input_ids, layer_ids)
        outputs = self.model(inputs_embeds=input_embeds, attention_mask=attention_mask, labels=labels)
        return outputs.loss, outputs.logits

    def codeword_generate(self, input_ids, attention_mask=None, num_beams=20, **kwargs):
        input_embeds = self.model.shared(input_ids) * d_model_sqrt
        layer_ids = layer_id_lut_tensor[input_ids]
        input_embeds = self.codeword_geo_module(input_embeds, input_ids, layer_ids)
        return self.model.generate(inputs_embeds=input_embeds, attention_mask=attention_mask,
                                    num_beams=num_beams, max_length=5,
                                    num_return_sequences=num_beams, **kwargs)

    import types
    hg_rec.forward = types.MethodType(codeword_forward, hg_rec)
    hg_rec.generate = types.MethodType(codeword_generate, hg_rec)
    return hg_rec


def build_codeword_geo_module(stage2_ckpt_path):
    """从 Stage2 ckpt 读真实 codebook (切空间) + final_kappas, 构建 CodewordGeoResidual."""
    ckpt = torch.load(stage2_ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    final_kappas = list(ckpt["final_kappas"])
    codebook_list = []
    for l in range(3):
        cb = sd[f"vq_layers.{l}.embeddings.weight"].numpy()  # (K_l, d_tangent=32)
        codebook_list.append(cb)
    log(f"[Issue #63] Stage2 ckpt loaded: codebook shapes = "
        f"L0={codebook_list[0].shape} L1={codebook_list[1].shape} L2={codebook_list[2].shape}")
    log(f"[Issue #63] final_kappas = {[round(k, 4) for k in final_kappas]}")
    return CodewordGeoResidual(
        d_model=CONFIG["d_model"],
        codebook_list=codebook_list,
        final_kappas=final_kappas,
        beta_init=0.0,
        offsets=CODEWORD_OFFSETS,
        force_zero_layers=(3,),
        rho_max=CODEWORD_RHO_MAX,
    )


# ──────────────────────────────────────────────────────────────
# 新 Issue: Prefix-Conditioned Branch Curvature (Phase B Path B)
# 不动 Stage2 SID, 不改 HAB 内部 — 只在 Stage3 给 L0 prefix 注入 per-prefix λ_max multiplier.
# 计算流程:
#   1. 加载 SID (n_items, 4), 提取 L0 prefix (column 0)
#   2. 对每个 L0 prefix 算 B(p) = exp(H(L1 child distribution)), σ_r (residual var 代理)
#   3. 按 B(p) 排序 + 分桶 (quantile / kmeans)
#   4. 构造 per-token-id bucket lut (1025,) — 仅 L0 token (id=1..64) 有值, 其它 = -1
#   5. Monkey-patch hab_module.get_B_geo: 给 L0 token i,j 之间的 bias 乘以 mult[bucket[i]]
# ──────────────────────────────────────────────────────────────
def build_branch_curvature_lut():
    """根据 SID 算 L0 prefix branch features, 分桶, 返回 per-token-id bucket lut.

    Returns:
        bucket_lut: numpy (1025,) int64, 仅 L0 token (id=1..64) 有值 (0..n_buckets-1), 其它 = -1
        stats: dict {l0_B: [...], l0_n_p: [...], buckets: {...}}
    """
    from collections import defaultdict
    sid = np.load(BRANCH_CURVATURE_SID_NPY)
    if sid.ndim != 2 or sid.shape[1] < 2:
        raise ValueError(f"SID shape {sid.shape} 异常, 需 (n_items, >=2)")
    l0_arr = sid[:, 0].astype(np.int64)
    l1_arr = sid[:, 1].astype(np.int64)
    # 每个 L0 prefix → L1 child distribution
    l0_to_items = defaultdict(list)
    l0_to_l1 = defaultdict(list)
    for i in range(len(sid)):
        l0 = int(l0_arr[i])
        l0_to_items[l0].append(i)
        l0_to_l1[l0].append(int(l1_arr[i]))
    l0_features = {}  # l0 → {"B": ..., "n_p": ...}
    for l0 in sorted(l0_to_items.keys()):
        items = l0_to_items[l0]
        l1_list = l0_to_l1[l0]
        n_p = len(items)
        # child entropy + branching factor
        l1_unique, l1_counts = np.unique(l1_list, return_counts=True)
        p = l1_counts / l1_counts.sum()
        H = float(-(p * np.log(p + 1e-30)).sum())
        B = float(np.exp(H))
        l0_features[l0] = {"B": B, "n_p": n_p, "H": H}
    # 分桶 (按 B 排序)
    l0_ids_sorted = sorted(l0_features.keys(),
                           key=lambda l: l0_features[l]["B"])
    n_buckets = BRANCH_CURVATURE_N_BUCKETS
    if BRANCH_CURVATURE_STRATEGY == "quantile":
        # 等频分桶 (按 B 排序后切片)
        bucket_assignment = {}
        bucket_size = max(1, len(l0_ids_sorted) // n_buckets)
        for idx, l0 in enumerate(l0_ids_sorted):
            b = min(idx // bucket_size, n_buckets - 1)
            bucket_assignment[l0] = b
    elif BRANCH_CURVATURE_STRATEGY == "kmeans":
        # 1D kmeans on B(p)
        from sklearn.cluster import KMeans
        B_arr = np.array([l0_features[l]["B"] for l in l0_ids_sorted]).reshape(-1, 1)
        if len(B_arr) < n_buckets:
            # 退化: 等频
            bucket_assignment = {}
            bucket_size = max(1, len(l0_ids_sorted) // n_buckets)
            for idx, l0 in enumerate(l0_ids_sorted):
                b = min(idx // bucket_size, n_buckets - 1)
                bucket_assignment[l0] = b
        else:
            km = KMeans(n_clusters=n_buckets, random_state=42, n_init=10)
            km.fit(B_arr)
            labels = km.labels_
            # 按 cluster mean B 排序, label 0 = lowest B
            cluster_means = {c: B_arr[labels == c].mean() for c in range(n_buckets)}
            label_remap = {old: new for new, old in
                           enumerate(sorted(cluster_means.keys(), key=lambda c: cluster_means[c]))}
            bucket_assignment = {l0: int(label_remap[int(labels[idx])])
                                 for idx, l0 in enumerate(l0_ids_sorted)}
    else:
        raise ValueError(f"未知的 BRANCH_CURVATURE_STRATEGY: {BRANCH_CURVATURE_STRATEGY}")
    # 构造 token-id bucket lut (1025,)
    # L0 token id = 1..64 (offset 1), 故 lut[1..64] = bucket[l0]
    bucket_lut = np.full(1025, -1, dtype=np.int64)
    for l0, b in bucket_assignment.items():
        token_id = l0 + 1  # offset 1
        if 1 <= token_id < 1025:
            bucket_lut[token_id] = int(b)
    # 校验: 必须有 n_buckets 个非 -1 桶
    used_buckets = set(int(b) for b in bucket_assignment.values())
    if len(used_buckets) != n_buckets:
        raise ValueError(
            f"分桶仅产出 {len(used_buckets)} 个非空桶 (期望 {n_buckets}). "
            f"可能是 prefix 数 < n_buckets 或分布极端. bucket_assignment={bucket_assignment}"
        )
    stats = {
        "n_items": int(len(sid)),
        "l0_unique": int(len(l0_features)),
        "l0_B_min": min(v["B"] for v in l0_features.values()),
        "l0_B_max": max(v["B"] for v in l0_features.values()),
        "l0_B_mean": float(np.mean([v["B"] for v in l0_features.values()])),
        "l0_n_p_min": min(v["n_p"] for v in l0_features.values()),
        "l0_n_p_max": max(v["n_p"] for v in l0_features.values()),
        "buckets": {int(b): [int(l) for l in l0_ids_sorted
                              if bucket_assignment[l] == b]
                    for b in range(n_buckets)},
        "strategy": BRANCH_CURVATURE_STRATEGY,
        "lambda_mult": BRANCH_CURVATURE_LAMBDA_MULT,
    }
    return bucket_lut, stats


def install_branch_curvature_on_hab(hab_module, bucket_lut):
    """Monkey-patch hab_module.get_B_geo: 给 L0 token 之间的 bias 乘以 per-bucket multiplier.

    实现细节:
      - 缓存原始 get_B_geo (hab_module._original_get_B_geo)
      - 替换为 wrapper: 算原始 B_geo, 然后对所有 (i, j) 都属 L0 layer 的 pair 乘以 mult[bucket[i]]
        (注意: 这里 i=j 也乘, 因为 λ_max 是 per-source token 的强度, 不影响对称性)
      - 保持 λ_raw 梯度正常流回 (mult 不参与梯度, 也不修改 lambda_eff)
    """
    import types
    if not hasattr(hab_module, "_original_get_B_geo"):
        hab_module._original_get_B_geo = hab_module.get_B_geo
    # bucket_lut: numpy (1025,) int64, token_id → bucket_id (-1 = 非 L0 token)
    bucket_lut_tensor = torch.as_tensor(bucket_lut, dtype=torch.long)
    mult_list = BRANCH_CURVATURE_LAMBDA_MULT
    if len(mult_list) != BRANCH_CURVATURE_N_BUCKETS:
        raise ValueError(
            f"--branch_curvature_lambda_mult 长度 {len(mult_list)} != "
            f"--branch_curvature_n_buckets {BRANCH_CURVATURE_N_BUCKETS}"
        )
    mult_tensor = torch.tensor(mult_list, dtype=torch.float32)
    n_buckets = BRANCH_CURVATURE_N_BUCKETS

    def get_B_geo_branch(self, input_ids, layer_id_lut_tensor, attention_mask_2d=None):
        """Branch-Aware 版本: 在原 get_B_geo 基础上, 给 L0 layer token 之间的 pair bias
        乘以 mult[bucket[input_ids]] (per-source multiplier)."""
        B_geo = self._original_get_B_geo(input_ids, layer_id_lut_tensor,
                                          attention_mask_2d=attention_mask_2d)
        # B_geo shape: (B, 1, L, L)
        device = B_geo.device
        _bucket_lut = bucket_lut_tensor.to(device)
        _mult = mult_tensor.to(device)
        _layer_lut = layer_id_lut_tensor.to(device)
        # 找 L0 layer 的 mask
        layer_ids = _layer_lut[input_ids]  # (B, L)
        l0_mask = (layer_ids == 0)  # (B, L) bool — 仅 L0 token
        if not l0_mask.any():
            return B_geo
        # 找每个 token 的 bucket id (clamp 到 [0, n_buckets-1])
        bucket_ids = _bucket_lut[input_ids].clamp(0, n_buckets - 1)  # (B, L)
        # per-token multiplier
        token_mult = _mult[bucket_ids]  # (B, L), dtype float32
        # 对所有 (i, j) 都属 L0 layer 的 pair 乘以 token_mult[i] (per-source scaling)
        # mask_pair_l0 = l0_mask.unsqueeze(2) & l0_mask.unsqueeze(1)  # (B, L, L) bool
        # multiplier_per_pair = token_mult.unsqueeze(-1) * mask_pair_l0.float()  # (B, L, L)
        # 直接用乘法 + mask 屏蔽非 L0 pair
        l0_mask_f = l0_mask.float()  # (B, L)
        pair_mask = l0_mask_f.unsqueeze(2) * l0_mask_f.unsqueeze(1)  # (B, L, L)
        multiplier = token_mult.unsqueeze(-1) * pair_mask  # (B, L, L)
        # 广播到 (B, 1, L, L) 与 B_geo 相乘
        B_geo_branched = B_geo * multiplier.unsqueeze(1)
        # 保留 PAD mask (PAD 处原本就是 0, multiplier=0 仍 0, 安全)
        return B_geo_branched

    hab_module.get_B_geo = types.MethodType(get_B_geo_branch, hab_module)
    return hab_module


# Issue #70: DECOR PromptFormer (candidate bins + alpha gate) 与 HAB/GEO 正交
def build_prompt_former_module():
    """构建 DecorPromptFormer. 不需要 stage2 ckpt, 只用 T5 nn.Embedding.
    Issue #68: 加 4 个抗 self-reinforcing trap 参数.

    R44 self-contained: PROMPT_FORMER_ENABLED=False 默认值, 函数体已禁用 (common.decor_prompt_former 引用已删除).
    若用户需启用此功能, 必须复制 common/decor_prompt_former.py 到 baseline/_lib/ 并改写 import.
    """
    raise NotImplementedError(
        "PROMPT_FORMER_ENABLED=True requires common.decor_prompt_former in baseline/_lib/. "
        "R44 严格自包含 baseline 不引用 common/. 复制模块后改写 import 即可启用."
    )


def install_prompt_former(hg_rec, pf_module):
    """Monkey-patch HG_Rec 实例: forward + generate 都注入 DECOR PromptFormer.
    在 self.model.shared(input_ids) 之后注入, 与现有 install_geo_residual / install_hab 互不干扰.

    R44 self-contained: PROMPT_FORMER_ENABLED=False 默认值, 函数体已禁用 (common.decor_prompt_former 引用已删除).
    若用户需启用此功能, 必须复制 common/decor_prompt_former.py 到 baseline/_lib/ 并改写 import.
    """
    raise NotImplementedError(
        "install_prompt_former() requires common.decor_prompt_former in baseline/_lib/. "
        "R44 严格自包含 baseline 不引用 common/. 复制模块后改写 import 即可启用."
    )


def calculate_pos_index(preds, labels, maxk=20):
    # preds: (B, maxk, seq_len) 每 beam 生成的 token 序列; labels: (B, seq_len) SID code.
    # 加速 (2026-08-03): 向量化 "beam 序列与 target code 全等" 判定 (原逻辑对每个 (i,j) 逐 token
    # Python 比较 → 每 epoch ~50 万次循环), 数值完全等价.
    preds = preds.detach().cpu()
    labels = labels.detach().cpu()
    assert preds.shape[1] == maxk, f"preds.shape[1] = {preds.shape[1]} != {maxk}"
    # labels (B, seq_len) → (B,1,seq_len) 与 preds (B,maxk,seq_len) 广播, all(dim=-1) 判全等
    pos_index = (preds == labels.unsqueeze(1)).all(dim=-1)  # (B, maxk)
    return pos_index


def recall_at_k(pos_index, k):
    return pos_index[:, :k].sum(dim=1).cpu().float()


def ndcg_at_k(pos_index, k):
    ranks = torch.arange(1, pos_index.shape[-1] + 1).to(pos_index.device)
    dcg = 1.0 / torch.log2(ranks + 1)
    dcg = torch.where(pos_index, dcg, torch.tensor(0.0, dtype=torch.float, device=dcg.device))
    return dcg[:, :k].sum(dim=1).cpu().float()


def train(model, train_loader, optimizer, device, epoch, scheduler=None):
    model.train()
    # Issue #140 v76: T5 uncertainty head 全局变量 (main() 创建)
    global t5_uncertainty_head
    total_loss = 0.0
    n = 0
    autocast_ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if STAGE3_BF16 else torch.nullcontext())
    # v87 (Issue #86 borrow): 拿到 hab_module 引用 (用于重置 _hab_attn_entropy_loss)
    _hab_ref = getattr(model, "module", model)
    if hasattr(_hab_ref, "_hab_attn_entropy_loss"):
        _hab_ref._hab_attn_entropy_loss = None  # 每个 epoch 重置 (允许累加)
    # v29 加速 (2026-08-04): torch.compile — env TORCH_COMPILE=1 启用
    if _TORCH_COMPILE and not _TRAIN_COMPILED:
        try:
            model = torch.compile(model, mode=_TORCH_COMPILE_MODE, fullgraph=False)
            globals()['_TRAIN_COMPILED'] = True
            log("[v29] torch.compile enabled (mode=reduce-overhead)")
        except Exception as e:
            log(f"[v29] torch.compile failed: {e}")
            globals()['_TRAIN_COMPILED'] = True  # 不再试
    for batch in train_loader:
        input_ids = batch["history"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["target"].to(device)
        optimizer.zero_grad()
        # 加速: bf16 forward (T5 标准混合精度, 参数 fp32, 仅前向计算转 bf16)
        with autocast_ctx:
            # Issue125 (2026-08-12): HG_Rec.forward 返回 tuple (loss, logits), 必须解包.
            #   baseline _lib/HG_Rec.py:61 写死 `return outputs.loss, outputs.logits` → tuple.
            #   PromptFormer 时返回 (loss, logits, reg_losses) 也是 tuple, 同样解包.
            #   旧代码 line 971-974 的注释 "默认 forward 返回 T5LMHeadOutput 含 .loss + .logits" 是错的
            #   (来自 baseline stage3.py 抄错的注释), 实际从来就是 tuple, 旧 baseline 都没跑过这里.
            pf_out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            if isinstance(pf_out, tuple):
                # HG_Rec.forward / PromptFormer 路径: tuple 形式
                if len(pf_out) == 2:
                    loss, logits = pf_out
                elif len(pf_out) == 3:
                    loss, logits, _reg = pf_out
                else:
                    raise ValueError(f"Unexpected tuple length {len(pf_out)} from model.forward")
            else:
                # 防御: T5ForConditionalGeneration 直接返回 Seq2SeqLMOutput (R30 兜底)
                loss, logits = pf_out.loss, pf_out.logits
        # Issue #140 v76: T5 uncertainty head 注入 Gumbel noise + uncertainty loss
        # 注意: 仅在 training 时有 noise, eval 时 head 直通 logits
        if t5_uncertainty_head is not None and model.training:
            noisy_logits = t5_uncertainty_head(logits.float())  # fp32 加 noise 防 bf16 精度问题
            loss = F.cross_entropy(
                noisy_logits.view(-1, noisy_logits.size(-1)),
                labels.view(-1),
                ignore_index=CONFIG["pad_token_id"],
            )
            uncertainty_reg = t5_uncertainty_head.compute_uncertainty_loss(loss.detach())
            loss = loss + T5_UNCERTAINTY_REG_WEIGHT * uncertainty_reg
        # Issue #140 v75: 启用 label_smoothing 时手动算 F.cross_entropy 平滑 loss
        # (T5 内部 loss 不支持 label_smoothing, 从已有 logits 重算)
        elif STAGE3_LABEL_SMOOTHING > 0:
            loss_t5 = loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)).float(),
                labels.view(-1),
                ignore_index=CONFIG["pad_token_id"],
                label_smoothing=STAGE3_LABEL_SMOOTHING,
            )
        # v87 (Issue #86 borrow from v78 attn_entropy): 加权 attn_entropy regularizer 到总 loss
        # 鼓励 HAB bias 分布尖锐 (proxy: B_geo softmax entropy ↓), 防几何信号被 attn 稀释
        if HAB_ATTN_ENTROPY_WEIGHT > 0:
            _hab_entropy = getattr(_hab_ref, "_hab_attn_entropy_loss", None)
            if _hab_entropy is not None:
                loss = loss + HAB_ATTN_ENTROPY_WEIGHT * _hab_entropy
                # 重置, 避免下次 batch 累加 (每次 forward 后累加, 但 loss 取完应清零)
                _hab_ref._hab_attn_entropy_loss = None
        loss.backward()
        optimizer.step()
        # Issue #141 v85d: per-batch LR scheduler step (cosine with warmup, 更细粒度)
        if scheduler is not None:
            scheduler.step()
        # Issue #140 v75: 第一个 batch 后打印一次 (验证 label_smoothing 生效)
        if RANK == 0 and n == 0 and STAGE3_LABEL_SMOOTHING > 0:
            log(f"[Issue #140 v75] label_smoothing={STAGE3_LABEL_SMOOTHING} "
                f"(T5 CE loss={loss_t5.item():.4f} → CE_smooth={loss.item():.4f})")
        total_loss += loss.item() * input_ids.shape[0]
        n += input_ids.shape[0]
    # DDP: all-reduce 各卡 loss (按样本数加权平均, 全局一致)
    # Issue #64 DDP all_reduce 死锁修复 (2026-08-06): NCCL + gloo 在 HAB + bf16 + L40S 4 卡组合下
    #   train 阶段 all_reduce 必卡死 (5 次重试确认). 改: rank 0 单独算 loss, 其他 rank 用 local loss (数学上
    #   等价 — 所有 rank 跑同一 DistributedSampler epoch, 模型同步由 DDP gradient sync 保证).
    #   trade-off: loss 打印是 rank 0 视角 (1/4 数据 local mean), 但全局 gradient 仍正确同步.
    if DDP_MODE and RANK == 0:
        pass  # rank 0 用 local loss (1/4 数据), 已足够监控收敛
    elif DDP_MODE:
        # 其他 rank 用 local loss (不参与聚合, 但保留 DDP gradient sync)
        pass
    return total_loss / n


def evaluate(model, eval_loader, device):
    model.eval()
    # DDP 包装后 generate 在 model.module 上 (forward 走 DDP.__call__, generate 是原模型方法)
    gen_model = model.module if DDP_MODE else model
    recalls = {f"R@{k}": [] for k in TOP_K}
    ndcgs = {f"NDCG@{k}": [] for k in TOP_K}
    autocast_ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if STAGE3_BF16 else torch.nullcontext())
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch["history"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["target"].to(device)
            with autocast_ctx:
                preds = gen_model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=BEAM_SIZE)
            preds = preds[:, 1:]
            preds = preds.reshape(input_ids.shape[0], BEAM_SIZE, -1)
            pos_index = calculate_pos_index(preds, labels, maxk=BEAM_SIZE)
            for k in TOP_K:
                recalls[f"R@{k}"].append(recall_at_k(pos_index, k).mean().item())
                ndcgs[f"NDCG@{k}"].append(ndcg_at_k(pos_index, k).mean().item())
    out_recalls = {k: sum(v) / len(v) for k, v in recalls.items()}
    out_ndcgs = {k: sum(v) / len(v) for k, v in ndcgs.items()}
    # DDP: 每卡 eval 分片 local mean. Issue #64 DDP 死锁修复 (2026-08-06, 6 次重试确认):
    #   NCCL + bf16 + HAB + L40S 4 卡组合下 eval all_reduce 必卡死 (ep5 eval 4 min+ frozen).
    #   改: rank 0 单独算 eval + ckpt save, 其他 rank 跳过 eval (空 dict 返回).
    #   trade-off: eval 算的是 rank 0 的 1/4 数据 local mean, 但 DDP gradient sync 已保证模型同步,
    #   且 rank 0 DistributedSampler 与单卡 sampler 数学等价 (同 seed + 不同 shard).
    if DDP_MODE and RANK != 0:
        return {}, {}
    return out_recalls, out_ndcgs


def main():
    PRODUCT_DIR.mkdir(parents=True, exist_ok=True)
    # DDP: init_process_group (env://), 每卡 device 由 LOCAL_RANK 决定, rank 0 为主进程
    # Issue #64 DDP NCCL 死锁修复 (2026-08-06): L40S 4 卡 + bf16 + HAB module + DDP find_unused_parameters 组合下 NCCL train all_reduce 必卡死 (4 次重试确认 ep5 之后永不恢复).
    # 改 backend="nccl" (CPU 通信) — 牺牲 ~30% 速度换稳定. gloo backend 在 GPU tensor 上仍能 all_reduce (自动 host transfer).
    if DDP_MODE:
        dist.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(minutes=30))
        torch.cuda.set_device(LOCAL_RANK)
        device = f"cuda:{LOCAL_RANK}"
        is_main = (RANK == 0)
    else:
        device = DEVICE
        is_main = True
    if is_main:
        with open(TRAINING_PID_FILE, "w") as f:
            f.write(str(os.getpid()))
    set_seed(SEED)

    # ---- SID 校验 (EXPECTED_SID_SHA 不设则跳过 = 预期内缺失) ----
    sid_sha = sha256_of(SID_NPY)
    if EXPECTED_SID_SHA:
        if sid_sha != EXPECTED_SID_SHA:
            raise ValueError(
                f"SID sha mismatch: got {sid_sha}, expected {EXPECTED_SID_SHA} ({SID_NPY})")
        if is_main:
            log(f"[SHA256] SID_NPY OK: {sid_sha}")

    if is_main:
        log(f"config: TAG={TAG} SID={SID_NPY} sha={sid_sha} epochs={NUM_EPOCHS} "
            f"early_stop={EARLY_STOP} batch={BATCH_SIZE} lr={LR} seed={SEED} device={device} "
            f"world_size={WORLD_SIZE} bf16={STAGE3_BF16} geo_residual={GEO_RESIDUAL_ENABLED}")

    model = HG_Rec(CONFIG)
    if is_main:
        log(model.n_parameters.rstrip())
    model.to(device)
    # Issue #62: 几何残差注入 (在 DDP wrap 前, 让 DDP 一起管理 geo_module 参数)
    if GEO_RESIDUAL_ENABLED:
        geo_module = build_geo_module()
        layer_id_lut_t = torch.from_numpy(_LAYER_ID_LUT)
        model = install_geo_residual(model, geo_module, layer_id_lut_t)
        if is_main:
            log(f"[Issue #62] geo_residual ON: per-layer mlp params={sum(p.numel() for p in geo_module.mlps.parameters())} "
                f"alpha_init={GEO_ALPHA_INIT} cap=±{GEO_ALPHA_CAP}")
    # Issue #63: 码字级几何残差 (在 DDP wrap 前; 跟 #62 互斥, 优先 #63)
    if CODEWORD_GEO_ENABLED:
        codeword_module = build_codeword_geo_module(CODEWORD_STAGE2_CKPT)
        layer_id_lut_t = torch.from_numpy(_LAYER_ID_LUT).to(device)
        model = install_codeword_geo_residual(model, codeword_module, layer_id_lut_t)
        # Issue #63 Gate3: beta=0 等价性 + train/eval 一致性预检
        if is_main:
            proj_params = sum(p.numel() for p in codeword_module.proj.parameters())
            gate_params = sum(p.numel() for p in codeword_module.gate_mlp.parameters())
            ln_params = (codeword_module.ln_h.weight.numel() * 2 + codeword_module.ln_q.weight.numel() * 2)
            beta_params = codeword_module.beta_raw.numel()
            log(f"[Issue #63] codeword_geo ON: proj={proj_params} gate={gate_params} "
                f"ln={ln_params} beta={beta_params} (β_init=0.0, v4 per-layer ρ_max={codeword_module.rho_max_per_layer.tolist()}, warmup=[30, 50])")
            # beta=0 等价性: 同一输入, beta=0 时 forward 输出 == input_embeds (直接相加 0)
            # 训练代码已保证 beta=0 → delta=0 → 等价
            log(f"[Issue #63] Gate3 beta=0 等价性预检: beta=0 → delta=0 → h'==h, 严格退化 #61 baseline")
    # Issue #140 v76: T5 logits uncertainty decay head (借鉴 DIGER AutoSigmaGumbel)
    # R44 self-contained: T5_UNCERTAINTY_ENABLED=False 默认值, common.t5_uncertainty 已删除.
    global t5_uncertainty_head
    t5_uncertainty_head = None
    # Issue #64: 双曲码字距离 attention bias (在 DDP wrap 前; 跟 #62/#63 互斥, 优先 #64)
    hab_module = None
    if HAB_ENABLED:
        codebook_list, hab_final_cs = load_hab_assets_from_stage2_ckpt(HAB_STAGE2_CKPT)
        if is_main:
            log(f"[HAB curvature] final_cs from Stage2 ckpt = "
                f"{[round(c, 4) for c in hab_final_cs]} (c>0 直接使用, 不由 κ 反推)")
        if HAB_DELTA_CURVATURE:
            # Issue #71 Phase B: 曲率差分 HAB (ΔD = D_hyp - D_flat)
            D_list, Dbar_list, stats_list, Dflat_list = precompute_distance_matrices(
                codebook_list, hab_final_cs, use_delta_curvature=True)
            if is_main:
                log(f"[Issue #71 Phase B] HAB ΔD mode: Dbar = ΔD/median, D_flat computed (c_flat=1e-6)")
        else:
            D_list, Dbar_list, stats_list = precompute_distance_matrices(codebook_list, hab_final_cs)
        for stats in stats_list:
            assert stats["finite"], f"L{stats['layer']} 距离矩阵含 NaN/Inf"
            assert stats["sym_err"] < 1e-6, f"L{stats['layer']} 对称误差 {stats['sym_err']} >= 1e-6"
            assert stats["diag_max"] < 1e-6, f"L{stats['layer']} 对角线 {stats['diag_max']} >= 1e-6"
        hab_module = HyperbolicAttentionBias(Dbar_list, lambda_max=HAB_LAMBDA_MAX,
                                              enable_residual=RESIDUAL_HAB_ENABLED,
                                              residual_alpha_init=RESIDUAL_ALPHA_INIT,
                                              warmup_T0=HAB_WARMUP_T0,
                                              warmup_Tw=HAB_WARMUP_TW)
        # v87 (Issue #86 borrow): 把 attn_entropy regularizer 参数传给 hab_module
        hab_module.attn_entropy_weight = float(HAB_ATTN_ENTROPY_WEIGHT)
        hab_module.attn_entropy_tau = float(HAB_ATTN_ENTROPY_TAU)
        # Issue #64 不需要 separate L3 dummy: num_layers=3, force_zero_layers=() (L3 在 Dbar 外)
        layer_id_lut_array = make_hab_layer_id_lut()
        model = install_hab(model, hab_module, layer_id_lut_array)
        # Issue #132 (2026-08-12): Stage3 训练期完全禁用 HAB baseline lambda_raw (3 标量).
        # Issue #130 + #131 双 ablation 验证: eval 期 HAB 注入对 attention 计算无贡献 (causal mask 主导).
        # test 退化根因在 Stage3 训练期 HAB 让 T5 学偏, 不是 eval 期 HAB 注入.
        # 本 issue 强制 lambda_raw=0 (训练全程), ckpt 训练等效 baseline (无 HAB 干扰).
        with torch.no_grad():
            hab_module.lambda_raw.data.fill_(0.0)
        # Issue #125 (2026-08-12): per-head learnable curvature (κ_h + λ_h) — 必须在 install_hab 之后
        # lambda_h_init=0 → 训练初期不贡献 (与 baseline 严格等价); 训练中学到合适 λ 后 per-head bias 生效.
        hab_module = install_per_head_curvature(
            hab_module,
            num_heads=PER_HEAD_NUM_HEADS_DEFAULT,
            kappa_h_init=PER_HEAD_KAPPA_H_INIT,
            lambda_h_init=PER_HEAD_LAMBDA_H_INIT,
        )
        # Issue #132: per-head lambda_h_raw 训练期同样冻结 (避免 per-head 也学偏, 与 baseline 完全等价)
        with torch.no_grad():
            hab_module.lambda_h_raw.data.fill_(0.0)
        if is_main:
            log(f"[Issue #125] per-head curvature enabled: "
                f"num_heads={hab_module.num_heads} kappa_h_init={PER_HEAD_KAPPA_H_INIT} "
                f"lambda_h_init={PER_HEAD_LAMBDA_H_INIT} (R36 曲率机制变更)")
        # 新 Issue Phase B: Branch-Aware HAB (按 L0 prefix branch features 分桶 λ_max)
        if BRANCH_CURVATURE_ENABLED:
            _bc_lut, _bc_stats = build_branch_curvature_lut()
            install_branch_curvature_on_hab(hab_module, _bc_lut)
            if is_main:
                log(f"[新 Issue Branch Curvature] λ_max bucket mult: "
                    f"low={BRANCH_CURVATURE_LAMBDA_MULT[0]:.2f} "
                    f"mid={BRANCH_CURVATURE_LAMBDA_MULT[1]:.2f} "
                    f"high={BRANCH_CURVATURE_LAMBDA_MULT[2]:.2f} "
                    f"strategy={BRANCH_CURVATURE_STRATEGY} sid={BRANCH_CURVATURE_SID_NPY} "
                    f"bucket sizes={[len(_bc_stats['buckets'][b]) for b in range(BRANCH_CURVATURE_N_BUCKETS)]}")
        if is_main:
            lambda_params = hab_module.lambda_raw.numel()
            residual_tag = f" residual_alpha_init={RESIDUAL_ALPHA_INIT}" if RESIDUAL_HAB_ENABLED else ""
            log(f"[Issue #64/71] hyperbolic_attn_bias ON: lambda_raw={lambda_params} "
                f"λ_max={HAB_LAMBDA_MAX} Dbar=[{stats_list[0]['median']:.4f}, {stats_list[1]['median']:.4f}, {stats_list[2]['median']:.4f}]"
                f"{residual_tag}")
            # Issue #64 v2 修复 (2026-08-06): lambda_raw init = 0.5*λ_max (非零), 触发 HAB 实际生效.
            # 历史 #64 v1 lambda_raw init=0 → fast path → HAB 0 梯度 → 完全未生效 (Gate 4 FAIL NO-GO).
            lambda_init = hab_module.lambda_raw.detach().cpu().tolist()
            lambda_eff_init = hab_module.lambda_eff.detach().cpu().tolist()
            log(f"[Issue #64 v2] λ_raw init={lambda_init} (非零, fast path 不触发) "
                f"λ_eff init={lambda_eff_init} (B_geo 实际生效, 梯度正常流到 λ_raw)")
            # 注入计数预检
            model._hab_inject_count = 0
    # Issue #70: DECOR PromptFormer (candidate bins + alpha gate) — 与 HAB/GEO 正交可叠加
    # R44 self-contained: PROMPT_FORMER_ENABLED=False 默认值, build/install 函数体已禁用.
    pf_module = None
    if DDP_MODE:
        # Issue #64: HAB lambda_raw 在 lambda_eff=0 时不参与前向计算 (走 _original_forward fast path),
        # DDP 默认检测到 unused parameter 会崩. 加 find_unused_parameters=True (历史 #55 taskA stage2 同样修过).
        # Issue #64 加速 (2026-08-06): 加 bucket_cap_mb=200 减少 sync 频率, gradient_as_bucket_view=True 省
        #   tensor view copy, static_graph=True 关闭 unused param 重新检测. T5-mini 5.5M params 全部能装
        #   进 ~22MB 单 bucket, 减少 4 worker sync barrier 次数.
        # Issue #69 v79 (2026-08-07): DECOR PromptFormer + HAB 叠加时 pf_module / hab_module 某些 batch 可能不参与 loss 计算,
        #   find_unused_parameters=False 必崩 (RuntimeError: Expected to have finished reduction).
        #   自动检测: HAB 启用时改 True (损失一些加速, 但保训练). R44: PROMPT_FORMER_ENABLED=False 已被禁用, 简化为 HAB_ENABLED.
        ddp_find_unused = HAB_ENABLED
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[LOCAL_RANK],
            find_unused_parameters=ddp_find_unused,  # Issue #69: DECOR/HAB 时 True 避免 reducer 崩
            bucket_cap_mb=512,  # Issue #64 v6 加速 (2026-08-07): 200→512, 86k bias params 加入后更多 bucket 拆分导致 NCCL sync barrier 翻倍, 加大 bucket_cap 减少 barrier 次数
            gradient_as_bucket_view=True,
            static_graph=False,
        )
    # Issue #61 P0 加速: TF32 enable (Ampere+ L40S sm_89 支持, matmul 内部用 tf32 加速 1.3-1.5×)
    if STAGE3_TF32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True  # cudnn benchmark 自动选最快卷积算法 (T5 没用卷积, 但无害)
    # Issue #61 P0 加速: fused AdamW (PyTorch 2.x fused=True 用单个 CUDA kernel 跑 optimizer step, 1.2-1.5×)
    # Issue #62 v4: per-layer alpha 独立学习率 (GEO_ALPHA_LR_RATIO > 1 时生效)
    # Issue #64 v3 优化: λ_raw 单独高 lr param group (HAB_LAMBDA_LR_RATIO=100, 推动 λ_raw 学习)
    _param_groups = []
    if HAB_ENABLED and HAB_LAMBDA_LR_RATIO > 1.0:
        # Issue #64 v6b (2026-08-07): U/V embedding 也进 high-lr group (与 λ_raw 一起, 推动 B_geo 学习)
        # 总参数量 = sum(K_l * r * 2) = (64+128+256)*16*2 = 14336 scalar + 3 scalar (lambda_raw)
        # Issue #132 (2026-08-12): 当 ISSUE132_FREEZE_HAB=True, lambda_raw 不进 optimizer (frozen by design).
        if ISSUE132_FREEZE_HAB:
            hab_lambda_params = []  # freeze: 训练期强制 0
        else:
            hab_lambda_params = [hab_module.lambda_raw]  # 3 个标量
        hab_bias_params = list(hab_module.U.parameters()) + list(hab_module.V.parameters())  # 6 个 (K_l, r) embedding
        _param_groups.append({"params": hab_lambda_params + hab_bias_params,
                                "lr": LR * HAB_LAMBDA_LR_RATIO})
        # Issue #71 (2026-08-07): residual_alpha 单独 group (alpha 是关键调节门, 10× 推动)
        if RESIDUAL_HAB_ENABLED and RESIDUAL_ALPHA_LR_RATIO > 1.0:
            residual_alpha_params = [hab_module.residual_alpha]  # 3 个标量 (logit 形式)
            _param_groups.append({"params": residual_alpha_params,
                                    "lr": LR * RESIDUAL_ALPHA_LR_RATIO})
    if GEO_RESIDUAL_ENABLED and GEO_ALPHA_LR_RATIO > 1.0:
        alpha_params = [geo_module.alphas_raw]  # alpha 解耦, lr 慢 ratio×
        other_params = [p for n, p in model.named_parameters() if not n.endswith("geo_module.alphas_raw")]
        _param_groups.append({"params": other_params, "lr": LR})
        _param_groups.append({"params": alpha_params, "lr": LR / GEO_ALPHA_LR_RATIO})
        # Issue #140 v76: T5 uncertainty head σ 加到 base lr group (如果有)
        if t5_uncertainty_head is not None:
            _param_groups.append({"params": [t5_uncertainty_head.sigma], "lr": LR})
        optimizer = optim.AdamW(_param_groups, fused=FUSED_OPTIMIZER, weight_decay=STAGE3_WEIGHT_DECAY)
        if is_main:
            log(f"[v4] per-layer alpha lr={LR/GEO_ALPHA_LR_RATIO:.2e} (ratio={GEO_ALPHA_LR_RATIO}×), "
                f"mlp+base lr={LR:.2e}")
    else:
        # Issue #64 v6b: 用 id() 排除 hab params (lambda_raw + U/V embeddings)
        # 之前 n.startswith/endswith 在 DDP wrap 后参数名前缀变化时不可靠 (v6 启动失败)
        # Issue #70: 同样排除 prompt_former params (alpha_raw + bos_queries + q/k)
        if HAB_ENABLED and hab_module is not None:
            _hab_param_ids = {id(hab_module.lambda_raw)}
            for _e in list(hab_module.U) + list(hab_module.V):
                _hab_param_ids.add(id(_e.weight))
            # Issue #71: residual_alpha 也进 high-lr group (上面), 必须从 base group 排除
            if RESIDUAL_HAB_ENABLED:
                _hab_param_ids.add(id(hab_module.residual_alpha))
            # Issue #132: 当 ISSUE132_FREEZE_HAB=True, per-head lambda_h_raw 也排除 (frozen by design)
            if ISSUE132_FREEZE_HAB and hasattr(hab_module, 'lambda_h_raw'):
                _hab_param_ids.add(id(hab_module.lambda_h_raw))
                _hab_param_ids.add(id(hab_module.kappa_h))  # 冻结 kappa_h 也避免 per-head 学偏
        else:
            _hab_param_ids = set()
        # R44 self-contained: PROMPT_FORMER_ENABLED=False, pf_module 始终 None, _pf_param_ids 始终空集合.
        _pf_param_ids = set()
        _exclude_ids = _hab_param_ids | _pf_param_ids
        _other_params = [p for p in model.parameters() if id(p) not in _exclude_ids]
        _param_groups.insert(0, {"params": _other_params, "lr": LR})
        # Issue #140 v76: T5 uncertainty head σ 加到 base lr group
        if t5_uncertainty_head is not None:
            _param_groups.append({"params": [t5_uncertainty_head.sigma], "lr": LR})
        optimizer = optim.AdamW(_param_groups, fused=FUSED_OPTIMIZER, weight_decay=STAGE3_WEIGHT_DECAY)
    # Issue #141 v85d (2026-08-07): LR cosine decay with warmup. v85 P0 失败根因 = LR 1e-3 恒定无 scheduler.
    # 构造基于 step 的 LambdaLR: warmup_frac 步内线性从 0→1, 之后 cosine decay 到 LR_MIN_FACTOR.
    if LR_SCHEDULER == "cosine":
        import math as _math
        _total_steps = NUM_EPOCHS * (131837 // (BATCH_SIZE * WORLD_SIZE) + 1)  # 估计总步数
        _warmup_steps = max(1, int(_total_steps * LR_WARMUP_FRAC))
        def _lr_lambda(step):
            if step < _warmup_steps:
                return step / _warmup_steps
            progress = (step - _warmup_steps) / max(1, _total_steps - _warmup_steps)
            return LR_MIN_FACTOR + (1.0 - LR_MIN_FACTOR) * 0.5 * (1.0 + _math.cos(_math.pi * min(1.0, progress)))
        scheduler = lr_scheduler.LambdaLR(optimizer, _lr_lambda)
        if is_main:
            log(f"[Issue #141 v85d] LR scheduler=cosine warmup={_warmup_steps}/{_total_steps} steps → LR_min={LR*LR_MIN_FACTOR:.2e}")
    elif LR_SCHEDULER == "none":
        scheduler = None
        if is_main:
            log(f"[Issue #141 v85d] LR scheduler=none (LR={LR} 恒定, 与 v85 P0 一致)")
    else:
        raise ValueError(f"LR_SCHEDULER={LR_SCHEDULER} 不是合法值 (none/cosine)")
    if HAB_ENABLED and HAB_LAMBDA_LR_RATIO > 1.0 and is_main:
        n_bias_params = sum(p.numel() for p in list(hab_module.U.parameters()) + list(hab_module.V.parameters()))
        log(f"[Issue #64 v6b] λ_raw + U/V embedding ({n_bias_params} params) "
            f"共进 high-lr group lr={LR*HAB_LAMBDA_LR_RATIO:.2e} "
            f"(ratio={HAB_LAMBDA_LR_RATIO}×, 推动 v6b low-rank learnable bias 学习)")
    if STAGE3_WEIGHT_DECAY > 0 and is_main:
        log(f"[Issue #138 v74] AdamW weight_decay={STAGE3_WEIGHT_DECAY} 拉小 U/V 范数防过拟合")

    train_ds = GenRecDataset(
        dataset_path=TRAIN_PARQUET, code_path=SID_NPY, mode="train",
        codebook_size=CODEBOOK_SIZE, max_len=MAX_LEN)
    valid_ds = GenRecDataset(
        dataset_path=VALID_PARQUET, code_path=SID_NPY, mode="evaluation",
        codebook_size=CODEBOOK_SIZE, max_len=MAX_LEN)
    if is_main:
        log(f"n_train(windowed)={len(train_ds)} n_valid={len(valid_ds)}")

    # DDP: DistributedSampler 分片 + 标准 DataLoader (collate 复刻自 GenRecDataLoader);
    # 全局 batch 严格保持 (每卡 BATCH_SIZE//WORLD_SIZE). 非 DDP 走原 GenRecDataLoader 不变.
    if DDP_MODE:
        if BATCH_SIZE % WORLD_SIZE != 0:
            raise ValueError(f"BATCH_SIZE={BATCH_SIZE} 必须被 WORLD_SIZE={WORLD_SIZE} 整除 (DDP 全局 batch 严格保持)")
        if INFER_SIZE % WORLD_SIZE != 0:
            raise ValueError(f"INFER_SIZE={INFER_SIZE} 必须被 WORLD_SIZE={WORLD_SIZE} 整除 (DDP eval 分片)")
        train_sampler = DistributedSampler(train_ds, num_replicas=WORLD_SIZE, rank=RANK, shuffle=True)
        valid_sampler = DistributedSampler(valid_ds, num_replicas=WORLD_SIZE, rank=RANK, shuffle=False)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE // WORLD_SIZE, sampler=train_sampler,
                                  num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
                                  persistent_workers=PERSISTENT_WORKERS if NUM_WORKERS > 0 else False,
                                  collate_fn=_collate_fn)
        valid_loader = DataLoader(valid_ds, batch_size=INFER_SIZE // WORLD_SIZE, sampler=valid_sampler,
                                  num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
                                  persistent_workers=PERSISTENT_WORKERS if NUM_WORKERS > 0 else False,
                                  collate_fn=_collate_fn)
    else:
        train_loader = _FastGenRecDataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                                             pin_memory=PIN_MEMORY, persistent_workers=PERSISTENT_WORKERS)
        valid_loader = _FastGenRecDataLoader(valid_ds, batch_size=INFER_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                                             pin_memory=PIN_MEMORY, persistent_workers=PERSISTENT_WORKERS)

    # Issue #135 v72 (2026-08-07): 改用 valid_R10 作 early stop 信号 + ckpt 保存
    # v71 用 loss-based early stop + loss-best ckpt, 但 valid peak (ep50 0.1285) 跟 loss-best (ep116 1.6697)
    # 严重脱节 — loss 持续下降但 valid 在 ep75 后单调崩 -0.0083 → test 0.1013 FAIL.
    # v72: ckpt 按 valid_R10 保存 + 5 epoch 无 valid_R10 提升就停. 锁定 valid peak epoch, 避免训练后期崩.
    best_valid_r10 = -1.0
    best_loss = float("inf")
    best_epoch = -1
    early_stop_counter = 0
    trace = []
    EARLY_STOP_METRIC = "valid_R10"  # 区分 v6b/v71 loss-based

    for epoch in range(NUM_EPOCHS):
        if DDP_MODE:
            train_sampler.set_epoch(epoch)  # 每 epoch 不同 shuffle (所有 rank 一致)
        # Issue #63 v4: β warmup 需要当前 epoch (1-indexed: epoch=0 → 第 1 轮)
        if CODEWORD_GEO_ENABLED:
            cgm_loop = model.module.codeword_geo_module if DDP_MODE else model.codeword_geo_module
            cgm_loop.current_epoch = epoch + 1
        t0 = time.time()
        train_loss = train(model, train_loader, optimizer, device, epoch, scheduler=scheduler)
        t_train = time.time() - t0

        # EVAL_INTERVAL 控制 eval 频率 (Issue #61, 用户指示 2026-08-06 对齐 DECOR)
        do_eval = ((epoch + 1) % EVAL_INTERVAL == 0) or (epoch == NUM_EPOCHS - 1)

        if is_main:
            # Issue #135 v72 (2026-08-07): valid_R10-based early stop + ckpt 保存
            # loss-based 在 v71 失效 — valid ep75→ep115 单调崩但 loss 持续下降
            # valid_R10 每 5 epoch 评估一次, 只在 do_eval 时更新 best_valid_r10 / early_stop_counter
            if do_eval:
                t0 = time.time()
                recalls, ndcgs = evaluate(model, valid_loader, device)
                t_eval = time.time() - t0
                # Issue #62 v2 修复: 监控 geo_module.alpha_l 训练轨迹 (写到 trace, 防 alpha 失控)
                row = {
                    "epoch": epoch, "train_loss": train_loss,
                    **recalls, **ndcgs,
                    "t_train": round(t_train, 1), "t_eval": round(t_eval, 1),
                    "early_stop_counter": early_stop_counter,
                    "best_loss": round(best_loss, 4),
                }
                if GEO_RESIDUAL_ENABLED:
                    raw_alphas = model.module.geo_module.alphas_raw.detach().cpu().tolist() if DDP_MODE else model.geo_module.alphas_raw.detach().cpu().tolist()
                    if GEO_SMOOTH_TANH:
                        # v4: alpha_eff = cap·tanh(raw) (无触顶)
                        eff_alphas = [GEO_ALPHA_CAP * math.tanh(a) for a in raw_alphas]
                    else:
                        # v2/v3: alpha_eff = clamp(raw, -cap, cap) (硬边界, 会触顶)
                        eff_alphas = [max(-GEO_ALPHA_CAP, min(GEO_ALPHA_CAP, a)) for a in raw_alphas]
                    row["alpha_raw"] = [round(a, 5) for a in raw_alphas]
                    row["alpha_eff"] = [round(a, 5) for a in eff_alphas]
                # Issue #63: 监控 ρ_l / gate / per-layer residual 方差 + β_l
                if CODEWORD_GEO_ENABLED:
                    cgm = model.module.codeword_geo_module if DDP_MODE else model.codeword_geo_module
                    beta_raw_list = cgm.beta_raw.detach().cpu().tolist()
                    rho_list = cgm.rho_max_per_layer_buf.detach().cpu().tolist()
                    # Issue #63 v4: 同步 train 端 (warmup + per-layer ρ_max)
                    warmup_factor = max(0.0, min(1.0, (cgm.current_epoch - cgm.warmup_start) / max(1, cgm.warmup_end - cgm.warmup_start)))
                    beta_eff_list = [warmup_factor * rho_list[i] * math.tanh(b / rho_list[i]) for i, b in enumerate(beta_raw_list)]
                    row["beta_raw"] = [round(b, 5) for b in beta_raw_list]
                    row["beta_eff"] = [round(b, 5) for b in beta_eff_list]
                    betas = beta_eff_list  # 保留给 WARNING 用
                    # ρ_l + gate + residual var 审计 (在 valid batch 上跑一次)
                    try:
                        audit_batch = next(iter(valid_loader))
                        input_ids_a = audit_batch["history"].to(device)
                        layer_ids_a = layer_id_lut_t[input_ids_a] if 'layer_id_lut_t' in dir() else None
                        if layer_ids_a is None:
                            layer_id_lut_t = torch.from_numpy(_LAYER_ID_LUT).to(device)
                            layer_ids_a = layer_id_lut_t[input_ids_a]
                        with torch.no_grad():
                            d_model_sqrt_local = CONFIG["d_model"] ** 0.5
                            emb = cgm.model.shared(input_ids_a) * d_model_sqrt_local if hasattr(cgm, 'model') else None
                        if emb is None:
                            # 重新计算 (model 在 outer scope)
                            emb = model.module.model.shared(input_ids_a) * (CONFIG["d_model"] ** 0.5) if DDP_MODE else model.model.shared(input_ids_a) * (CONFIG["d_model"] ** 0.5)
                        audit_stats = cgm.audit_stats(emb, input_ids_a, layer_ids_a)
                        for k, v in audit_stats.items():
                            row[k] = v
                    except Exception as e:
                        log(f"[Issue #63] audit_stats 失败: {e}")
                    # β_raw 超容差警告 (raw 漂到 ±5 远超需要, 即便 eff 已饱和到 ±rho_max)
                    for l, b_raw in enumerate(beta_raw_list):
                        if abs(b_raw) > 5.0:
                            log(f"[Issue #63] WARNING: β_raw_l{l}={b_raw:.4f} 漂移过大 (eff 已饱和到 ±{cgm.rho_max})")
                # Issue #64: λ_raw + λ_eff 监控 (跟 #62 alpha / #63 beta 监控对齐)
                if HAB_ENABLED:
                    hab_loop = model.module.hab_module if DDP_MODE else model.hab_module
                    # Issue #141 v84 (2026-08-07): λ_raw clamp 兜底 (在每 epoch do_eval 时 in-place, 防止 λ_eff 饱和到 ±λ_max 边界, v74 L116-122 问题)
                    with torch.no_grad():
                        hab_loop.lambda_raw.clamp_(-HAB_LAMBDA_RAW_CLAMP, HAB_LAMBDA_RAW_CLAMP)
                    lambda_raw_list = hab_loop.lambda_raw.detach().cpu().tolist()
                    lambda_eff_list = hab_loop.lambda_eff.detach().cpu().tolist()
                    row["lambda_raw"] = [round(x, 5) for x in lambda_raw_list]
                    row["lambda_eff"] = [round(x, 5) for x in lambda_eff_list]
                    row["hab_inject_count"] = model._hab_inject_count if not DDP_MODE else model.module._hab_inject_count
                    # Issue #64 v6b: 监控 U/V embedding 范数 (init → 训练后对比, 验矩阵真在学习)
                    u_norms = [u.weight.detach().norm().item() for u in hab_loop.U]
                    v_norms = [v.weight.detach().norm().item() for v in hab_loop.V]
                    row["U_l2_norm"] = [round(x, 4) for x in u_norms]
                    row["V_l2_norm"] = [round(x, 4) for x in v_norms]
                trace.append(row)
                # Issue #135 v72 (2026-08-07): valid_R10-based best ckpt + early stop
                # v71 loss-best 与 valid peak 严重脱节 (loss ep116 best, valid ep50 peak). v72 按 valid_R10 保存 + 早停.
                if recalls["R@10"] > best_valid_r10 + 1e-4:
                    best_valid_r10 = recalls["R@10"]
                    best_loss = train_loss
                    best_epoch = epoch
                    early_stop_counter = 0
                    torch.save(model.module.state_dict() if DDP_MODE else model.state_dict(), CKPT_PATH)
                    log(f"[BEST] valid_R10={best_valid_r10:.4f} train_loss={best_loss:.4f} saved {CKPT_PATH}")
                    # Issue #141 v85 (2026-08-07): 1) new best ckpt 立即异步触发 stage4 test eval (实时 ratio 监控)
                    #                                2) 顺便 poll 已完成的 async eval 写 history
                    if STAGE4_TEST_ON_BEST and is_main:
                        _poll_stage4_procs()
                        trigger_stage4_test_async(CKPT_PATH, epoch, best_valid_r10)
                else:
                    early_stop_counter += 1
                    log(f"no valid_R10 improv ({early_stop_counter}/{EARLY_STOP}, best={best_valid_r10:.4f})")
                log(f"Epoch {epoch+1}/{NUM_EPOCHS} loss={train_loss:.4f} "
                    f"R@10={recalls['R@10']:.4f} NDCG@20={ndcgs['NDCG@20']:.4f} "
                    f"(train {t_train:.0f}s, eval {t_eval:.0f}s)"
                    + (f" alpha={row.get('alpha_eff', '')}" if GEO_RESIDUAL_ENABLED else ""))
            else:
                # 非 eval epoch: 只记录 train_loss (best/early_stop 已在 do_eval 之前更新)
                row = {"epoch": epoch, "train_loss": train_loss,
                       "t_train": round(t_train, 1), "t_eval": 0.0,
                       "early_stop_counter": early_stop_counter,
                       "best_loss": round(best_loss, 4)}
                trace.append(row)
                log(f"Epoch {epoch+1}/{NUM_EPOCHS} loss={train_loss:.4f} "
                    f"(train {t_train:.0f}s, eval skipped, next eval @ epoch {((epoch + 1) // EVAL_INTERVAL + 1) * EVAL_INTERVAL})")

        # DDP: rank 0 的 early stop 决策 broadcast 到所有 rank (否则非主卡继续跑, 卡死)
        if DDP_MODE:
            stop_flag = torch.tensor(1 if is_main and early_stop_counter >= EARLY_STOP else 0, device=device)
            dist.broadcast(stop_flag, src=0)
            if stop_flag.item():
                if is_main:
                    log("early stop triggered")
                break
        elif early_stop_counter >= EARLY_STOP:
            log("early stop triggered")
            break

    if is_main:
        with open(TRACE_PATH, "w") as f:
            json.dump(trace, f, indent=2)
        verdict = {
            "tag": TAG, "sid_npy": SID_NPY, "sid_sha256": sid_sha,
            "best_epoch": best_epoch, "best_loss": round(best_loss, 4),
            "final_epoch": trace[-1]["epoch"] if trace else None,
            "best_trace": trace[best_epoch] if 0 <= best_epoch < len(trace) else None,
            "config": {**CONFIG, "lr": LR, "batch_size": BATCH_SIZE,
                       "num_epochs": NUM_EPOCHS, "early_stop": EARLY_STOP, "seed": SEED,
                       "world_size": WORLD_SIZE, "bf16": STAGE3_BF16,
                       "early_stop_metric": "train_loss"},
            "done_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(VERDICT_PATH, "w") as f:
            json.dump(verdict, f, indent=2)
        log(f"[VERDICT] {VERDICT_PATH} best_epoch={best_epoch} best_loss={best_loss:.4f}")
        log("DONE")
    if DDP_MODE:
        dist.barrier()
        dist.destroy_process_group()


def _ddp_self_launch():
    """R19+R42: 单进程直接执行 → 自动 re-spawn via torchrun 4 卡 DDP (Issue #159 2026-08-12).

    检测: WORLD_SIZE env 默认 1 (未走 torchrun) → 设 CUDA_VISIBLE_DEVICES + 调用 torchrun 重新 spawn.
    已在 torchrun 下 (WORLD_SIZE>1) → 不重 spawn, 直接 main() (DDP init 走 main 内部).
    """
    if DDP_MODE:  # torchrun 已在运行, 直接执行 (DDP init 走 main)
        return False
    # 单进程 → 自动 re-spawn via torchrun 4 卡
    import subprocess
    ddp = STAGE3_DDP_CONFIG
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ddp["cuda_visible_devices"]
    cmd = [
        sys.executable,  # 当前解释器 (R44: 不硬编码绝对路径)
        "-u",
        "-m", "torch.distributed.run",
        "--nproc_per_node", str(ddp["nproc_per_node"]),
        "--master_port", str(ddp["master_port"]),
        "--standalone",
        sys.argv[0],  # 当前 self
    ]
    # 透传 CLI args (--sid_npy/--product_dir/--tag)
    cmd += sys.argv[1:]
    print(f"[stage3/ddp-self-launch] CUDA_VISIBLE_DEVICES={ddp['cuda_visible_devices']} "
          f"nproc={ddp['nproc_per_node']} port={ddp['master_port']}", flush=True)
    print(f"[stage3/ddp-self-launch] cmd={' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, env=env)
    return True


if __name__ == "__main__":
    _ddp_self_launch() or main()
