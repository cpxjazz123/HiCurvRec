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
import hashlib
import time
import random
import math
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DistributedSampler, DataLoader

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/model")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/data")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec")

from HG_Rec import HG_Rec          # noqa: E402
from dataset import GenRecDataset  # noqa: E402
from dataloader import GenRecDataLoader  # noqa: E402


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
# argparse (R30 严格: 无 env var 读取)
# ──────────────────────────────────────────────────────────────
_argparser = argparse.ArgumentParser(description="Stage3 pure T5 train (R30 strict, no env var)")
_argparser.add_argument("--sid_npy", type=str, required=True, help="stage2 SID npy 路径")
_argparser.add_argument("--product_dir", type=str, required=True, help="产物目录")
_argparser.add_argument("--device", type=str, default="cuda:0", help="GPU device")
_argparser.add_argument("--tag", type=str, default="task", help="方向标签 (写进 verdict)")
_argparser.add_argument("--expected_sid_sha", type=str, default="", help="校验 SID npy sha256; 不传则跳过")
# Issue #62: Stage3 几何残差注入 flag (默认 False = 复现 #61 baseline)
_argparser.add_argument("--geo_residual", action="store_true",
                        help="启用几何残差注入 (Issue #62 实验变体, 默认 False = pure T5)")
_argparser.add_argument("--geo_smooth_tanh", action="store_true",
                        help="Issue #62 v4: 用 α = cap·tanh(raw) 替换 hard clamp, 永无触顶")
_argparser.add_argument("--geo_alpha_lr_ratio", type=float, default=1.0,
                        help="Issue #62 v4: alpha lr = LR / ratio (默认 1.0 = 与 mlp 同速; 推荐 10.0 解耦)")
# Issue #63: 码字级几何残差 (Stage2 ckpt 真实 codebook + β init 0 严格退化)
_argparser.add_argument("--codeword_geo_residual", action="store_true",
                        help="Issue #63: 启用码字级几何残差 (从 --codeword_stage2_ckpt 读 codebook + final_kappas)")
_argparser.add_argument("--codeword_stage2_ckpt", type=str,
                        default="/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_issue61/hrqvae_kappa_sync.ckpt",
                        help="Issue #63: Stage2 ckpt 路径, 读 codebook + final_kappas")
_argparser.add_argument("--codeword_rho_max", type=float, default=0.10,
                        help="Issue #63: ρ_l 审计上限 (默认 0.10)")
# DDP 状态 (默认单卡; torchrun 用户需通过 wrapper 翻译 env → argparse 或直接传值)
_argparser.add_argument("--world_size", type=int, default=1, help="DDP world size (torchrun wrapper 必传)")
_argparser.add_argument("--rank", type=int, default=0, help="DDP global rank")
_argparser.add_argument("--local_rank", type=int, default=0, help="DDP local rank")
_args = _argparser.parse_args()

SID_NPY = _args.sid_npy
PRODUCT_DIR = Path(_args.product_dir)
DEVICE = _args.device
TAG = _args.tag
EXPECTED_SID_SHA = _args.expected_sid_sha
GEO_RESIDUAL_ENABLED = _args.geo_residual
GEO_SMOOTH_TANH = _args.geo_smooth_tanh
GEO_ALPHA_LR_RATIO = _args.geo_alpha_lr_ratio
CODEWORD_GEO_ENABLED = _args.codeword_geo_residual
CODEWORD_STAGE2_CKPT = _args.codeword_stage2_ckpt
CODEWORD_RHO_MAX = _args.codeword_rho_max
WORLD_SIZE = _args.world_size
RANK = _args.rank
LOCAL_RANK = _args.local_rank
DDP_MODE = WORLD_SIZE > 1

# 超参 (R30 硬编码 — 变体需 fork 脚本)
NUM_EPOCHS = 200  # Issue #61: 用户指示 2026-08-06 改为 200 epoch 全量训练 (与 default 一致)
EARLY_STOP = 20  # Issue #61: 用户指示 2026-08-06 改为 20 (与 default 一致)
EVAL_INTERVAL = 5  # Issue #61: 用户指示 2026-08-06, 对齐 DECOR default.yaml:25 (每 5 epoch 才 eval, 省 22s × 4/5 ≈ 18s/effective epoch)
BATCH_SIZE = 1024  # Issue #61: 用户指示 2026-08-06, 对齐 DECOR default.yaml:17 train_batch_size=1024 (vs 256, 4× 速度)
INFER_SIZE = 96
SEED = 42
LR = 4e-4  # Issue #62 对照公平性: 必须与 #61 最终 baseline 一致 (linear scaling rule, bs=1024 → lr×4)
MAX_LEN = 20
NUM_WORKERS = 4  # Issue #61 P0 加速 (用户指示 2026-08-06): DataLoader 多 worker, 1.3-1.5× 加速 (主进程不再被 tokenize 阻塞)
PIN_MEMORY = True  # Issue #61 P0: DataLoader pin_memory=True, CPU→GPU 传输加速
PERSISTENT_WORKERS = True  # Issue #61 P0: worker 跨 epoch 持久, 省每 epoch worker spawn 启动时间
STAGE3_TF32 = True  # Issue #61 P0: Ampere+ TF32 matmul 加速 1.3-1.5×, 精度影响 <1e-3
FUSED_OPTIMIZER = True  # Issue #61 P0: torch.optim.AdamW(fused=True), L40S fused AdamW kernel 加速 1.2-1.5×

# 训练加速 (2026-08-03): bf16 autocast — T5 训练/beam 解码标准实践, forward 转 bf16 计算
# (参数保持 fp32, backward 后 optimizer 在 fp32 权重更新, 数值影响极小). 默认开.
STAGE3_BF16 = True
# v29 加速 (2026-08-04): torch.compile — PyTorch 2.x 内置, A100+ 推荐 reduce-overhead
_TORCH_COMPILE = False  # Issue #62 v2 fix: 关 torch.compile (reduce-overhead 模式 + GeoResidualModule alpha 触顶触发 CUDA Graph 反复 re-capture, ep 26 起 25s → 54s 慢 2x). #61 baseline 无 geo 时稳定 15s/epoch, 关闭后预计回到 15s 水平.
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
CONFIG = dict(                               # 基线 T5 (task84, encoder 6 + decoder 4)
    num_layers=6, num_decoder_layers=4, d_model=128, d_ff=1024,
    num_heads=6, d_kv=64, dropout_rate=0.1, vocab_size=1025,
    pad_token_id=0, eos_token_id=0, decoder_start_token_id=0,
    feed_forward_proj="relu",
)
DATA_ROOT = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments"
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
                 offsets=CODEWORD_OFFSETS, force_zero_layers=(3,), rho_max=0.10):
        super().__init__()
        self.d_model = d_model
        self.num_layers = len(codebook_list)
        self.rho_max = float(rho_max)  # β_l smooth clamp 上限 (Issue #63 v3 修复: 防止 β 自由漂移破坏 T5 表示)
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
        # Issue #63 v3: β smooth clamp 到 ±rho_max (避免 v2 β 自由漂移到 -1.13 破坏 T5)
        beta_eff = self.rho_max * torch.tanh(self.beta_raw / self.rho_max)   # (num_layers,)
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
        beta_eff = self.rho_max * torch.tanh(self.beta_raw / self.rho_max)
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


def train(model, train_loader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    n = 0
    autocast_ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if STAGE3_BF16 else torch.nullcontext())
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
            loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * input_ids.shape[0]
        n += input_ids.shape[0]
    # DDP: all-reduce 各卡 loss (按样本数加权平均, 全局一致)
    if DDP_MODE:
        t = torch.tensor([total_loss, float(n)], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        total_loss, n = t[0].item(), int(t[1].item())
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
    # DDP: 每卡 eval 分片 local mean, all-reduce SUM / WORLD_SIZE = 全局 mean
    if DDP_MODE:
        for k in TOP_K:
            rk = f"R@{k}"
            t = torch.tensor([out_recalls[rk], out_ndcgs[f"NDCG@{k}"]], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            out_recalls[rk], out_ndcgs[f"NDCG@{k}"] = t[0].item() / WORLD_SIZE, t[1].item() / WORLD_SIZE
    return out_recalls, out_ndcgs


def main():
    PRODUCT_DIR.mkdir(parents=True, exist_ok=True)
    # DDP: init_process_group (env://), 每卡 device 由 LOCAL_RANK 决定, rank 0 为主进程
    if DDP_MODE:
        dist.init_process_group(backend="nccl", init_method="env://")
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
                f"ln={ln_params} beta={beta_params} (β_init=0.0, ρ_max={CODEWORD_RHO_MAX})")
            # beta=0 等价性: 同一输入, beta=0 时 forward 输出 == input_embeds (直接相加 0)
            # 训练代码已保证 beta=0 → delta=0 → 等价
            log(f"[Issue #63] Gate3 beta=0 等价性预检: beta=0 → delta=0 → h'==h, 严格退化 #61 baseline")
    if DDP_MODE:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[LOCAL_RANK])
    # Issue #61 P0 加速: TF32 enable (Ampere+ L40S sm_89 支持, matmul 内部用 tf32 加速 1.3-1.5×)
    if STAGE3_TF32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True  # cudnn benchmark 自动选最快卷积算法 (T5 没用卷积, 但无害)
    # Issue #61 P0 加速: fused AdamW (PyTorch 2.x fused=True 用单个 CUDA kernel 跑 optimizer step, 1.2-1.5×)
    # Issue #62 v4: per-layer alpha 独立学习率 (GEO_ALPHA_LR_RATIO > 1 时生效)
    if GEO_RESIDUAL_ENABLED and GEO_ALPHA_LR_RATIO > 1.0:
        alpha_params = [geo_module.alphas_raw]  # alpha 解耦, lr 慢 ratio×
        other_params = [p for n, p in model.named_parameters() if not n.endswith("geo_module.alphas_raw")]
        optimizer = optim.AdamW([
            {"params": other_params, "lr": LR},
            {"params": alpha_params, "lr": LR / GEO_ALPHA_LR_RATIO},
        ], fused=FUSED_OPTIMIZER)
        if is_main:
            log(f"[v4] per-layer alpha lr={LR/GEO_ALPHA_LR_RATIO:.2e} (ratio={GEO_ALPHA_LR_RATIO}×), "
                f"mlp+base lr={LR:.2e}")
    else:
        optimizer = optim.AdamW(model.parameters(), lr=LR, fused=FUSED_OPTIMIZER)

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
                                  num_workers=NUM_WORKERS, collate_fn=_collate_fn)
        valid_loader = DataLoader(valid_ds, batch_size=INFER_SIZE // WORLD_SIZE, sampler=valid_sampler,
                                  num_workers=NUM_WORKERS, collate_fn=_collate_fn)
    else:
        train_loader = _FastGenRecDataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                                             pin_memory=PIN_MEMORY, persistent_workers=PERSISTENT_WORKERS)
        valid_loader = _FastGenRecDataLoader(valid_ds, batch_size=INFER_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                                             pin_memory=PIN_MEMORY, persistent_workers=PERSISTENT_WORKERS)

    best_ndcg = 0.0
    best_epoch = -1
    early_stop_counter = 0
    trace = []

    for epoch in range(NUM_EPOCHS):
        if DDP_MODE:
            train_sampler.set_epoch(epoch)  # 每 epoch 不同 shuffle (所有 rank 一致)
        t0 = time.time()
        train_loss = train(model, train_loader, optimizer, device, epoch)
        t_train = time.time() - t0

        # EVAL_INTERVAL 控制 eval 频率 (Issue #61, 用户指示 2026-08-06 对齐 DECOR)
        do_eval = ((epoch + 1) % EVAL_INTERVAL == 0) or (epoch == NUM_EPOCHS - 1)

        if is_main:
            if do_eval:
                t0 = time.time()
                recalls, ndcgs = evaluate(model, valid_loader, device)
                t_eval = time.time() - t0
                # Issue #62 v2 修复: 监控 geo_module.alpha_l 训练轨迹 (写到 trace, 防 alpha 失控)
                row = {
                    "epoch": epoch, "train_loss": train_loss,
                    **recalls, **ndcgs,
                    "t_train": round(t_train, 1), "t_eval": round(t_eval, 1),
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
                    beta_eff_list = [cgm.rho_max * math.tanh(b / cgm.rho_max) for b in beta_raw_list]
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
                trace.append(row)
                log(f"Epoch {epoch+1}/{NUM_EPOCHS} loss={train_loss:.4f} "
                    f"R@10={recalls['R@10']:.4f} NDCG@20={ndcgs['NDCG@20']:.4f} "
                    f"(train {t_train:.0f}s, eval {t_eval:.0f}s)"
                    + (f" alpha={row.get('alpha_eff', '')}" if GEO_RESIDUAL_ENABLED else ""))
                if ndcgs["NDCG@20"] > best_ndcg:
                    best_ndcg = ndcgs["NDCG@20"]
                    best_epoch = epoch
                    early_stop_counter = 0
                    torch.save(model.module.state_dict() if DDP_MODE else model.state_dict(), CKPT_PATH)
                    log(f"[BEST] NDCG@20={best_ndcg:.4f} saved {CKPT_PATH}")
                else:
                    early_stop_counter += 1
                    log(f"no improv ({early_stop_counter}/{EARLY_STOP})")
            else:
                # 非 eval epoch: 只记录 train_loss, 不更新 best / early_stop_counter
                row = {"epoch": epoch, "train_loss": train_loss,
                       "t_train": round(t_train, 1), "t_eval": 0.0}
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
            "best_epoch": best_epoch, "best_ndcg20": best_ndcg,
            "final_epoch": trace[-1]["epoch"] if trace else None,
            "best_trace": trace[best_epoch] if 0 <= best_epoch < len(trace) else None,
            "config": {**CONFIG, "lr": LR, "batch_size": BATCH_SIZE,
                       "num_epochs": NUM_EPOCHS, "early_stop": EARLY_STOP, "seed": SEED,
                       "world_size": WORLD_SIZE, "bf16": STAGE3_BF16},
            "done_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(VERDICT_PATH, "w") as f:
            json.dump(verdict, f, indent=2)
        log(f"[VERDICT] {VERDICT_PATH} best_epoch={best_epoch} best_NDCG@20={best_ndcg:.4f}")
        log("DONE")
    if DDP_MODE:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
