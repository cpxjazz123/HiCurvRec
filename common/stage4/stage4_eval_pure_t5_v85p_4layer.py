"""纯 T5 stage3 产物评估 — 基线 beam20 协议, 对 test.parquet 全量.

与 train_HG-Rec.py evaluate 完全一致:
  HG_Rec.generate(num_beams=20, num_return_sequences=20, max_length=5)
  preds = preds[:, 1:] -> (B, 20, 4);  R@K = target 4-token 出现在 top-k beam.

Stage4 协议 = test only (held-out). valid 评估由 Stage3 training 循环覆盖
  (val_trace.json 每个 eval_interval epoch 记录 valid R@K/NDCG, 用于 best ckpt
   选择 + 早停). Stage4 不重复 valid 评估.

R30: 所有超参 + 路径 + GPU + 校验都通过 argparse 传入, 无任何 env var 读取.
launcher (.sh) 必须传 --ckpt_path / --sid_npy / --product_dir; 其他 arg 走默认值.
实验变体: 复制此脚本为新文件改默认值, 不复用同一脚本 + env toggle.

当前变体默认值 (Issue #41 + hyp 系列常用):
  BATCH_SIZE=96, SEED=42, MAX_LEN=20, BEAM_SIZE=20, TOP_K=[5,10,20],
  EVAL_PARQUET=test.parquet (硬编码, 不可覆盖).
"""
import os
import sys
import json
import hashlib
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/model")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/data")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec")

import numpy as np  # Issue #141: np 顶层导入, 不再 shadow

from HG_Rec import HG_Rec          # noqa: E402
from dataset import GenRecDataset  # noqa: E402
from dataloader import GenRecDataLoader  # noqa: E402
# Issue #64: 双曲码字距离 attention bias (跟 Stage3 train 同源共享模块)
import sys as _sys
_sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec")
from common.hyperbolic_attention_bias import (  # noqa: E402
    HAB_LAMBDA_MAX, load_hab_assets_from_stage2_ckpt, precompute_distance_matrices,
    HyperbolicAttentionBias, install_hab, make_hab_layer_id_lut,
)
# Issue #236 (2026-08-10): Stage3 Poincaré Attention Scoring eval 同步
from common.poincare_attention_scoring import install_poincare_attention_scoring  # noqa: E402
# Issue #238 (2026-08-10): Stage4 Poincaré Re-ranking (post-generation rerank)
from common.poincare_rerank import load_poincare_assets, rerank_with_poincare  # noqa: E402

# Issue #107 (2026-08-10): Stage4 SCSB 需要 module-level SCSB_LAYER_ID_LUT
# (token -> layer: 0=PAD->-1, 1-64->L0, 65-192->L1, 193-448->L2, 449->L3)
# 注: 不用 _LAYER_ID_LUT 命名避免与函数内 local 变量冲突
SCSB_LAYER_ID_LUT = np.full(1025, -1, dtype=np.int64)
SCSB_LAYER_ID_LUT[1:65] = 0
SCSB_LAYER_ID_LUT[65:193] = 1
SCSB_LAYER_ID_LUT[193:449] = 2
SCSB_LAYER_ID_LUT[449:450] = 3

# ──────────────────────────────────────────────────────────────
# argparse (R30 严格: 无 env var 读取)
# ──────────────────────────────────────────────────────────────
_argparser = argparse.ArgumentParser(description="Stage4 pure T5 eval (R30 strict, no env var)")
_argparser.add_argument("--ckpt_path", type=str, required=True, help="HG_Rec_best.pth 路径")
_argparser.add_argument("--sid_npy", type=str, required=True, help="stage2 SID npy 路径")
_argparser.add_argument("--product_dir", type=str, required=True, help="产物目录 (verdict 落这里)")
_argparser.add_argument("--device", type=str, default="cuda:0", help="GPU device")
_argparser.add_argument("--tag", type=str, default="task", help="方向标签 (taskA/taskB), 写进 verdict")
_argparser.add_argument("--expected_sid_sha", type=str, default="", help="校验 SID 文件 sha256; 不传则跳过")
_argparser.add_argument("--geo_residual", action="store_true", help="Issue #62: 加载 geo_module 子模块并 monkey-patch forward/generate, 用于 Stage3 v3 ckpt 评估")
_argparser.add_argument("--codeword_geo_residual", action="store_true", help="Issue #63: 加载 codeword_geo_module 子模块, 用于 Stage3 Issue #63 ckpt 评估")
_argparser.add_argument("--codeword_stage2_ckpt", type=str, default="/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_issue61/hrqvae_kappa_sync.ckpt", help="Issue #63: Stage2 ckpt 路径, 读 codebook + final_kappas")
# Issue #64: 双曲 attention bias 评估 (跟 Stage3 train 共享同一个 HyperbolicAttentionBias 类)
_argparser.add_argument("--hyperbolic_attn_bias", action="store_true", help="Issue #64: 加载 hab_module 子模块, 用于 Stage3 Issue #64 ckpt 评估")
_argparser.add_argument("--hab_stage2_ckpt", type=str, default="/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_issue61/hrqvae_kappa_sync.ckpt", help="Issue #64: Stage2 ckpt 路径, 读 codebook + final_kappas")
_argparser.add_argument("--hab_lambda_max", type=float, default=0.20, help="Issue #64: λ_max tanh 上限 (默认 0.20, 跟 Stage3 train 一致)")
# Issue #71: HAB 残差学习 (Dbar + α·delta) — 修复 v6b learned B 偏离 Dbar 460-660% 导致过拟合
_argparser.add_argument("--enable_residual_hab", action="store_true", help="Issue #71: HAB 残差模式, B = Dbar_frozen + sigmoid(α)·(U·V^T - Dbar_frozen), anchor 永远是 Dbar")
_argparser.add_argument("--residual_alpha_init", type=float, default=0.5, help="Issue #71: residual_alpha sigmoid init (默认 0.5)")
# Issue #71 Phase A (2026-08-07): HAB warmup (eval 时 T_0=T_w=0 让 w=1 立即生效)
_argparser.add_argument("--hab_warmup_T0", type=int, default=0, help="Issue #71 Phase A: HAB warmup T_0 (eval 默认 0)")
_argparser.add_argument("--hab_warmup_Tw", type=int, default=0, help="Issue #71 Phase A: HAB warmup T_w (eval 默认 0)")
# Issue #71 Phase B (2026-08-07): 曲率差分 HAB (eval 默认 False, 与 train 一致)
_argparser.add_argument("--hab_delta_curvature", action="store_true",
                        help="Issue #71 Phase B: HAB 用 ΔD = D_hyp - D_flat (c_flat=1e-6) 替代完整 D_hyp")
# Issue #224 (2026-08-10): Stage3 κ frozen→learnable c_perturb eval 支持 (v23 启用).
#   加载 ckpt 的 c_perturb_raw 参数 + 启用 c_perturb_enabled, 让 eval 也应用 Stage3 学到的曲率扰动.
_argparser.add_argument("--c_perturb_enabled", action="store_true",
                        help="Issue #224: 启用 Stage3 曲率扰动 eval. 若 ckpt 含 hab_module.c_perturb_raw, 加载并启用 (±scale 距离扰动)")
_argparser.add_argument("--c_perturb_scale", type=float, default=0.10,
                        help="Issue #224: c_perturb_scale (与 train 端一致, 默认 0.10 = ±10% 距离扰动)")

# Issue #71 Phase A (2026-08-07): eval 时 T_0=T_w=0 → warmup_w=1 立即生效
# HAB_WARMUP_T0_EVAL / HAB_WARMUP_TW_EVAL 在 _args = parse_args() 之后赋值 (line ~99)
# Issue #70: DECOR PromptFormer (candidate bins + alpha gate) eval-time 安装 (与 HAB/GEO 正交)
_argparser.add_argument("--enable_prompt_former", action="store_true", help="Issue #70: 安装 DecorPromptFormer 模块, 加载 pf_module.* 参数, monkey-patch forward+generate")
_argparser.add_argument("--prompt_former_alpha", type=float, default=0.35, help="Issue #70: alpha gate sigmoid 初始值 (Stage3 train 默认 0.35)")
_argparser.add_argument("--prompt_former_num_bos_queries", type=int, default=64, help="Issue #70: learnable bos_queries count (DECOR 默认 64)")
_argparser.add_argument("--beam_size", type=int, default=20, help="Issue #45 beam_size sweep: 默认 20, 尝试 10/30/50")
# Issue #95 single-ckpt diverse beam search (DBS): num_beam_groups + diversity_penalty 强制组间差异
_argparser.add_argument("--num_beam_groups", type=int, default=0, help="Issue #95 DBS: 0=标准 beam, >0=DBS (需整除 beam_size)")
_argparser.add_argument("--diversity_penalty", type=float, default=0.5, help="Issue #95 DBS: 组间 diversity penalty")
_argparser.add_argument("--length_penalty", type=float, default=1.0, help="Issue #95 single-ckpt beam=20: length penalty (HF generate kwarg, 1.0=neutral)")
# Issue #236 (2026-08-10): Stage4 同步支持 Poincaré Attention Scoring eval
_argparser.add_argument("--poincare_attn_scoring", action="store_true",
                        help="Issue #236: 启用 Poincaré Attention Scoring eval (与 Stage3 训练一致)")
_argparser.add_argument("--poincare_c_init", type=float, default=1.0,
                        help="Issue #236: eval 初始曲率 c (eval 阶段通常 c_learnable=False, 用训练末值)")
# Issue #238 (2026-08-10): Stage4 Hyperbolic Re-ranking — 绕开 #100 cross-entropy 失败路径
# 曲率不进 cross-entropy, 只进 post-generation rerank (Stage4)
_argparser.add_argument("--poincare_rerank", action="store_true",
                        help="Issue #238: 启用 Poincaré 距离 rerank (score = orig + alpha*R_geo, R_geo = -d_P(history, cand))")
_argparser.add_argument("--rerank_alpha", type=float, default=0.5,
                        help="Issue #238: R_geo 权重 alpha (默认 0.5, 0 = 不 rerank 与 v18 baseline 等价)")
_argparser.add_argument("--rerank_layer", type=int, default=0, choices=[0, 1, 2],
                        help="Issue #238: 用哪层 codebook 算 R_geo (0=L0/64 entries, 1=L1/128, 2=L2/256)")
# Issue #107 (2026-08-10): Stage3 SCSB eval 同步 — Curvature Soft-Bias on T5 logits
_argparser.add_argument("--scsb_enabled", action="store_true",
                        help="Issue #107: 启用 SCSB (eval 时 lm_head 后注入 curvature soft bias)")
_argparser.add_argument("--scsb_alpha_init", type=float, default=1.0,
                        help="Issue #107: α 初始值 (eval 从 ckpt load 训练末值, init 仅作 fallback)")
_argparser.add_argument("--scsb_beta", type=float, default=0.1,
                        help="Issue #107: β 固定缩放 (与 Stage3 train 一致, 默认 0.1)")
_argparser.add_argument("--scsb_stage2_ckpt", type=str,
                        default="/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/hrqvae_kappa_sync.ckpt",
                        help="Issue #107: Stage2 ckpt 路径 (提供 codebook + per-layer κ)")
# v23 (2026-08-11): Stage4 镜像 Stage3 curvature residual (eval 时同步注入, 保持 train/eval 一致)
_argparser.add_argument("--curvature_residual_enabled", action="store_true",
                        help="v23: 启用 Stage2 branch curvature → Stage4 L1 token residual 注入 (与 Stage3 train 同步)")
_argparser.add_argument("--curvature_residual_decoder_enabled", action="store_true",
                        help="v23 v2: decoder 端也注入 curvature (encoder+decoder L1+L2)")
_argparser.add_argument("--curvature_residual_layer", type=int, default=1,
                        choices=[1, 2],
                        help="v23: 注入层位 (默认 L1, 兼容 v1)")
_argparser.add_argument("--curvature_residual_mlp_hidden", type=int, default=64,
                        help="v23: MLP f_1 hidden dim (默认 64, 与 Stage3 train 一致)")
_argparser.add_argument("--curvature_residual_alpha_init", type=float, default=0.0,
                        help="v23: α 初始值 (eval 从 ckpt load 训练末值, init 仅作 fallback)")
# Issue #108 (2026-08-10): Stage4 HSCSB eval 同步 (Hierarchical SCSB)
_argparser.add_argument("--hscsb_enabled", action="store_true",
                        help="Issue #108: 启用 HSCSB eval (3 层累积 + cross-layer bias)")
_argparser.add_argument("--hscsb_alpha_init", type=str, default="1.0,1.0,1.0",
                        help="Issue #108: 3 个 α_ℓ 初始值 (与 train 一致)")
_argparser.add_argument("--hscsb_beta_l0", type=float, default=0.1,
                        help="Issue #108: β_0 fixed (与 train 一致)")
_argparser.add_argument("--hscsb_beta_l1", type=float, default=0.2,
                        help="Issue #108: β_1 fixed (与 train 一致)")
_argparser.add_argument("--hscsb_beta_l2", type=float, default=0.4,
                        help="Issue #108: β_2 fixed (与 train 一致)")
_argparser.add_argument("--hscsb_beta_cross", type=float, default=0.05,
                        help="Issue #108: β_cross fixed (与 train 一致)")
_argparser.add_argument("--hscsb_stage2_ckpt", type=str,
                        default="/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/hrqvae_kappa_sync.ckpt",
                        help="Issue #108: Stage2 ckpt 路径 (与 train 一致)")
# Issue #110 (2026-08-10): RDB — Raw Distance Bias (Stage3 lm_head raw -d_P bias, no log_softmax)
_argparser.add_argument("--rdb_enabled", action="store_true",
                        help="Issue #110: 启用 RDB 评估同步 (与 train 一致)")
_argparser.add_argument("--rdb_alpha_init", type=float, default=1.0,
                        help="Issue #110: RDB α 初始值 (与 train 一致)")
_argparser.add_argument("--rdb_stage2_ckpt", type=str,
                        default="/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/hrqvae_kappa_sync.ckpt",
                        help="Issue #110: Stage2 ckpt 路径 (与 train 一致)")
# Issue #111 (2026-08-10): HRes — Stage3 Decoder Hyperbolic Residual 评估同步
_argparser.add_argument("--hres_enabled", action="store_true",
                        help="Issue #111: 启用 HRes 评估同步 (与 train 一致)")
_argparser.add_argument("--hres_beta_init", type=float, default=0.01,
                        help="Issue #111: HRes β 初始值 (与 train 一致)")
_argparser.add_argument("--hres_stage2_ckpt", type=str,
                        default="/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_v15_capmatch_1000ep/hrqvae_kappa_sync.ckpt",
                        help="Issue #111: Stage2 ckpt 路径 (与 train 一致)")
_args = _argparser.parse_args()

CKPT_PATH = _args.ckpt_path
SID_NPY = _args.sid_npy
PRODUCT_DIR = Path(_args.product_dir)
DEVICE = _args.device
TAG = _args.tag
EXPECTED_SID_SHA = _args.expected_sid_sha
GEO_RESIDUAL = _args.geo_residual
GEO_RESIDUAL_ENABLED = _args.geo_residual
CODEWORD_GEO = _args.codeword_geo_residual
CODEWORD_GEO_ENABLED = _args.codeword_geo_residual
CODEWORD_STAGE2_CKPT = _args.codeword_stage2_ckpt
HAB_ENABLED = _args.hyperbolic_attn_bias
HAB_STAGE2_CKPT = _args.hab_stage2_ckpt
HAB_LAMBDA_MAX_VAL = _args.hab_lambda_max
# Issue #71: HAB 残差学习开关
RESIDUAL_HAB_ENABLED = _args.enable_residual_hab
RESIDUAL_ALPHA_INIT = _args.residual_alpha_init
# Issue #71 Phase A (2026-08-07): HAB warmup eval (T_0=T_w=0 让 w=1 立即生效, 与训练末态对齐)
HAB_WARMUP_T0_EVAL = _args.hab_warmup_T0
HAB_WARMUP_TW_EVAL = _args.hab_warmup_Tw
# Issue #71 Phase B (2026-08-07): 曲率差分 HAB eval (与 train 一致)
HAB_DELTA_CURVATURE_EVAL = _args.hab_delta_curvature
# Issue #224 (2026-08-10): c_perturb Stage3 κ frozen→learnable eval 支持 (v23 启用)
CPERTURB_ENABLED = _args.c_perturb_enabled
CPERTURB_SCALE = _args.c_perturb_scale
# Issue #236 (2026-08-10): Poincaré Attention Scoring eval
POINCARE_ATTN_SCORING = _args.poincare_attn_scoring
POINCARE_C_INIT = _args.poincare_c_init
# Issue #107 (2026-08-10): SCSB eval 常量
SCSB_ENABLED = _args.scsb_enabled
SCSB_ALPHA_INIT = _args.scsb_alpha_init
SCSB_BETA = _args.scsb_beta
SCSB_STAGE2_CKPT = _args.scsb_stage2_ckpt
# v23 (2026-08-11): Stage4 curvature residual eval 常量 (镜像 Stage3 train)
CURVATURE_RESIDUAL_ENABLED = _args.curvature_residual_enabled
CURVATURE_RESIDUAL_DECODER_ENABLED = bool(_args.curvature_residual_decoder_enabled)
CURVATURE_RESIDUAL_LAYER = int(_args.curvature_residual_layer)
CURVATURE_RESIDUAL_MLP_HIDDEN = int(_args.curvature_residual_mlp_hidden)
CURVATURE_RESIDUAL_ALPHA_INIT = float(_args.curvature_residual_alpha_init)
# Issue #108 (2026-08-10): HSCSB eval 常量
HSCSB_ENABLED = _args.hscsb_enabled
HSCSB_ALPHA_INIT = [float(x) for x in _args.hscsb_alpha_init.split(",")]
HSCSB_BETA_L0 = _args.hscsb_beta_l0
HSCSB_BETA_L1 = _args.hscsb_beta_l1
HSCSB_BETA_L2 = _args.hscsb_beta_l2
HSCSB_BETA_CROSS = _args.hscsb_beta_cross
HSCSB_STAGE2_CKPT = _args.hscsb_stage2_ckpt
# Issue #110 (2026-08-10): RDB eval 常量
RDB_ENABLED = _args.rdb_enabled
RDB_ALPHA_INIT = float(_args.rdb_alpha_init)
RDB_STAGE2_CKPT = _args.rdb_stage2_ckpt
# Issue #111 (2026-08-10): HRes eval 常量
HRES_ENABLED = _args.hres_enabled
HRES_BETA_INIT = float(_args.hres_beta_init)
HRES_STAGE2_CKPT = _args.hres_stage2_ckpt
# Issue #238 (2026-08-10): Poincaré Re-ranking (Stage4 post-generation)
POINCARE_RERANK = _args.poincare_rerank
RERANK_ALPHA = _args.rerank_alpha
RERANK_LAYER = _args.rerank_layer
# Issue #70: DECOR PromptFormer 配置
PROMPT_FORMER_ENABLED = _args.enable_prompt_former
PROMPT_FORMER_ALPHA = _args.prompt_former_alpha
PROMPT_FORMER_NUM_BOS_QUERIES = _args.prompt_former_num_bos_queries
# Stage4 = test only (held-out). 硬编码, 不允许覆盖 (valid 由 Stage3 val_trace 覆盖)
EVAL_PARQUET = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/test.parquet"

# 超参 (R30 硬编码 — 变体需 fork 脚本)
BATCH_SIZE = 96
SEED = 42

CODEBOOK_SIZE = [64, 128, 256, 1]
# Issue #63: codeword 偏移表 (跟 Stage3 训练一致)
CODEWORD_OFFSETS = [1, 65, 193, 449]
CONFIG = dict(
    num_layers=6, num_decoder_layers=4, d_model=128, d_ff=1024,  # v85p_repro 4-layer fork
    num_heads=6, d_kv=64, dropout_rate=0.1, vocab_size=1025,
    pad_token_id=0, eos_token_id=0, decoder_start_token_id=0,
    feed_forward_proj="relu",
)
TOP_K = [5, 10, 20]
BEAM_SIZE = _args.beam_size  # Issue #45: 允许 sweep beam_size
# Issue #95 DBS: num_beam_groups > 0 触发 diverse beam search
NUM_BEAM_GROUPS = _args.num_beam_groups
DIVERSITY_PENALTY = _args.diversity_penalty
# Issue #95 single-ckpt beam=20: try length_penalty to favor longer sequences
LENGTH_PENALTY = getattr(_args, "length_penalty", 1.0)
MAX_LEN = 20

PRODUCT_DIR.mkdir(parents=True, exist_ok=True)
VERDICT_PATH = PRODUCT_DIR / "eval_test.json"


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# Issue #70: DECOR PromptFormer eval-mode 安装 (跟 Stage3 train install_prompt_former 等价,
# 但只 monkey-patch forward+generate, 不动 optimizer / DDP / param group)
def install_prompt_former_eval(hg_rec, pf_module):
    """注入 DecorPromptFormer 到 HG_Rec.eval() 调用路径.

    关键点:
      1. add_module(pf_module) → state_dict() 自动包含 pf_module.bos_queries / alpha_raw /
         q_ctx.weight / k_candidates.weight, load_state_dict(strict=True) 才能匹配 ckpt
      2. forward 与 generate 都先 self.model.shared(input_ids) * sqrt(d_model) → e_fused,
         再 pf_module(e_fused, input_ids, attention_mask, e_fused_embedding=self.model.shared)
         → e_final = α*e_soft + (1-α)*e_fused
      3. e_fused_embedding 必须传 self.model.shared (与 Stage3 train 一致, 同一 nn.Embedding)
    """
    import types
    from common.decor_prompt_former import DecorPromptFormer  # noqa: F401

    device = next(hg_rec.parameters()).device
    pf_module = pf_module.to(device)
    hg_rec.add_module("pf_module", pf_module)
    d_model_sqrt = hg_rec.model.config.d_model ** 0.5

    def pf_forward(self, input_ids, attention_mask=None, labels=None):
        e_fused = self.model.shared(input_ids) * d_model_sqrt        # (B, L, D)
        e_final, _aux = self.pf_module(
            e_fused, input_ids,
            attention_mask=attention_mask,
            e_fused_embedding=self.model.shared,
        )
        outputs = self.model(inputs_embeds=e_final,
                             attention_mask=attention_mask, labels=labels)
        return outputs.loss, outputs.logits

    def pf_generate(self, input_ids, attention_mask=None, num_beams=20, **kwargs):
        e_fused = self.model.shared(input_ids) * d_model_sqrt
        # Issue #68: pf_module 现在返回 (e_final, aux, reg_losses), eval 时 reg_losses 无用
        e_final, _aux, _reg_losses = self.pf_module(
            e_fused, input_ids,
            attention_mask=attention_mask,
            e_fused_embedding=self.model.shared,
        )
        return self.model.generate(inputs_embeds=e_final,
                                   attention_mask=attention_mask,
                                   num_beams=num_beams, max_length=5,
                                   num_return_sequences=num_beams, **kwargs)

    hg_rec.forward = types.MethodType(pf_forward, hg_rec)
    hg_rec.generate = types.MethodType(pf_generate, hg_rec)
    return hg_rec


# ──────────────────────────────────────────────────────────────
# v23 (2026-08-11): Curvature Residual eval — Stage4 镜像 Stage3 train
# 核心公式 (与 Stage3 train 完全一致):
#     e'_{q_1} = e_{q_1} + alpha_1 * f_1([kappa_1, delta_kappa_{1, q_0}])
# 保持 train/eval 一致: 同一 Stage2 ckpt + 同一 MLP 结构 + 同一 alpha (eval 从 ckpt load 训练末值)
# ──────────────────────────────────────────────────────────────
class CurvatureResidualModuleEval(nn.Module):
    """v23 v1+v2 eval: 镜像 Stage3 train 的 CurvatureResidualModule.

    v23 v2: L1 + L2 (encoder + decoder), 4 个 α scalars (alpha_1_enc, alpha_2_enc,
    alpha_1_dec, alpha_2_dec), 2 个 MLPs (f1 shared, f2 shared).
    与 train 版的区别: alpha 从 ckpt load 训练末值.
    """

    def __init__(self, kappa_l1, delta_kappa_l1, kappa_l2, delta_kappa_l2,
                 d_model=128, mlp_hidden=64, alpha_init=0.0):
        super().__init__()
        self.register_buffer("kappa_l1", torch.tensor(float(kappa_l1)))
        self.register_buffer("delta_kappa_l1", delta_kappa_l1.detach().clone())  # (64, 1)
        self.register_buffer("kappa_l2", torch.tensor(float(kappa_l2)))
        self.register_buffer("delta_kappa_l2", delta_kappa_l2.detach().clone())  # (8192, 1)
        self.f1 = nn.Sequential(
            nn.Linear(2, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, d_model),
        )
        self.f2 = nn.Sequential(
            nn.Linear(2, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, d_model),
        )
        self.alpha_1_enc = nn.Parameter(torch.tensor(float(alpha_init)))
        self.alpha_2_enc = nn.Parameter(torch.tensor(float(alpha_init)))
        self.alpha_1_dec = nn.Parameter(torch.tensor(float(alpha_init)))
        self.alpha_2_dec = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, input_ids, input_embeds, side='encoder'):
        # Defensive device sync: install 时模型可能在 CPU, generate 时已迁到 CUDA
        lut = self._cached_layer_id_lut
        if lut.device != input_ids.device:
            lut = lut.to(input_ids.device)
            self._cached_layer_id_lut = lut
        layer_ids = lut[input_ids]  # (B, L)
        is_l1 = (layer_ids == 1)
        is_l2 = (layer_ids == 2)
        # L1
        q_0_id_l1 = torch.roll(input_ids, shifts=1, dims=-1) * is_l1.long()
        l0_idx_l1 = (q_0_id_l1 - 1).clamp(min=0, max=self.delta_kappa_l1.shape[0] - 1)
        delta_kappa_l1 = self.delta_kappa_l1[l0_idx_l1]
        kappa_l1_expanded = self.kappa_l1.expand_as(delta_kappa_l1)
        cur_input_l1 = torch.cat([kappa_l1_expanded, delta_kappa_l1], dim=-1)
        cur_emb_l1 = self.f1(cur_input_l1)
        mask_l1 = is_l1.unsqueeze(-1).to(input_embeds.dtype)
        # L2
        q_0_id_l2 = torch.roll(input_ids, shifts=2, dims=-1) * is_l2.long()
        q_1_id_l2 = torch.roll(input_ids, shifts=1, dims=-1) * is_l2.long()
        l0_idx_l2 = (q_0_id_l2 - 1).clamp(min=0, max=63)
        l1_idx_l2 = (q_1_id_l2 - 65).clamp(min=0, max=127)
        l2_idx_l2 = (l0_idx_l2 * 128 + l1_idx_l2).clamp(min=0, max=self.delta_kappa_l2.shape[0] - 1)
        delta_kappa_l2 = self.delta_kappa_l2[l2_idx_l2]
        kappa_l2_expanded = self.kappa_l2.expand_as(delta_kappa_l2)
        cur_input_l2 = torch.cat([kappa_l2_expanded, delta_kappa_l2], dim=-1)
        cur_emb_l2 = self.f2(cur_input_l2)
        mask_l2 = is_l2.unsqueeze(-1).to(input_embeds.dtype)
        # Select α
        if side == 'encoder':
            alpha_1, alpha_2 = self.alpha_1_enc, self.alpha_2_enc
        elif side == 'decoder':
            alpha_1, alpha_2 = self.alpha_1_dec, self.alpha_2_dec
        else:
            raise ValueError(f"side must be encoder/decoder, got {side!r}")
        return input_embeds + alpha_1 * cur_emb_l1 * mask_l1 + alpha_2 * cur_emb_l2 * mask_l2


def install_curvature_residual_eval(hg_rec, curv_module, layer_id_lut_tensor, decoder_enabled=False):
    """v23 v1+v2 eval: Monkey-patch HG_Rec.forward + .generate (+ decoder.forward if decoder_enabled)."""
    import types
    device = next(hg_rec.parameters()).device
    curv_module = curv_module.to(device)
    layer_id_lut_tensor = layer_id_lut_tensor.to(device)
    curv_module._cached_layer_id_lut = layer_id_lut_tensor
    d_model_sqrt = hg_rec.model.config.d_model ** 0.5
    hg_rec.add_module("curvature_residual_module", curv_module)

    def curv_forward(self, input_ids, attention_mask=None, labels=None):
        input_embeds = self.model.shared(input_ids) * d_model_sqrt
        input_embeds = self.curvature_residual_module(input_ids, input_embeds, side='encoder')
        outputs = self.model(inputs_embeds=input_embeds,
                             attention_mask=attention_mask, labels=labels)
        return outputs.loss, outputs.logits

    def curv_generate(self, input_ids, attention_mask=None, num_beams=20, **kwargs):
        input_embeds = self.model.shared(input_ids) * d_model_sqrt
        input_embeds = self.curvature_residual_module(input_ids, input_embeds, side='encoder')
        return self.model.generate(inputs_embeds=input_embeds,
                                   attention_mask=attention_mask,
                                   num_beams=num_beams, max_length=5,
                                   num_return_sequences=num_beams, **kwargs)

    hg_rec.forward = types.MethodType(curv_forward, hg_rec)
    hg_rec.generate = types.MethodType(curv_generate, hg_rec)

    if decoder_enabled:
        # v23 v2: decoder 端注入. Monkey-patch model.decoder.forward
        hg_rec.model.decoder.curvature_residual_module = curv_module
        original_decoder_forward = hg_rec.model.decoder.__class__.forward

        def curv_decoder_forward(self, input_ids=None, attention_mask=None, **kwargs):
            inputs_embeds = kwargs.get('inputs_embeds', None)
            if input_ids is not None and inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids) * d_model_sqrt
                inputs_embeds = self.curvature_residual_module(input_ids, inputs_embeds, side='decoder')
                kwargs['inputs_embeds'] = inputs_embeds
                kwargs['input_ids'] = None
            return original_decoder_forward(self, attention_mask=attention_mask, **kwargs)

        hg_rec.model.decoder.forward = types.MethodType(curv_decoder_forward, hg_rec.model.decoder)

    return hg_rec


# ──────────────────────────────────────────────────────────────
# Issue #107 (2026-08-10): Stage3 SCSB — Curvature Soft-Bias eval 同步
# 在 T5 lm_head 之后插入 α·β·log_softmax(-d_P(codeword, history_centroid))
# 与 Stage3 train 完全一致,保证 inference 时 curvature bias 也生效
# ──────────────────────────────────────────────────────────────


class SCSBModuleEval(nn.Module):
    """Issue #107 (2026-08-10): Stage4 SCSB — Curvature Soft-Bias (eval 同源).

    与 Stage3 train SCSBModule 行为一致:
      bias = α · β · log_softmax(-d_P(codeword_emb, history_centroid))
    """

    def __init__(self, codebooks_t, cs, kappas, layer_id_lut,
                 alpha_init=1.0, beta=0.1, vocab_size=1025):
        super().__init__()
        codebooks_ball = []
        for tan_emb, c in zip(codebooks_t, cs):
            tan_emb = tan_emb.float()
            c_t = torch.tensor(float(c), dtype=torch.float32)
            ball_emb = self._exp_map_0(tan_emb, c_t)
            ball_emb = self._proj_to_ball(ball_emb, c_t)
            codebooks_ball.append(ball_emb)
        self.register_buffer("codebook_l0", codebooks_ball[0])
        self.register_buffer("codebook_l1", codebooks_ball[1])
        self.register_buffer("codebook_l2", codebooks_ball[2])
        self.register_buffer("c_l0", torch.tensor(float(cs[0]), dtype=torch.float32))
        self.register_buffer("c_l1", torch.tensor(float(cs[1]), dtype=torch.float32))
        self.register_buffer("c_l2", torch.tensor(float(cs[2]), dtype=torch.float32))
        self.register_buffer("layer_id_lut",
                             torch.tensor(layer_id_lut, dtype=torch.long))
        self.register_buffer("token_offset_l0", torch.tensor(1, dtype=torch.long))
        self.register_buffer("token_offset_l1", torch.tensor(65, dtype=torch.long))
        self.register_buffer("token_offset_l2", torch.tensor(193, dtype=torch.long))
        self.register_buffer("K_l0", torch.tensor(64, dtype=torch.long))
        self.register_buffer("K_l1", torch.tensor(128, dtype=torch.long))
        self.register_buffer("K_l2", torch.tensor(256, dtype=torch.long))
        self.vocab_size = vocab_size
        # eval 模式: alpha_raw 仍为 Parameter (从 ckpt load), 但默认 init 与 train 一致
        self.alpha_raw = nn.Parameter(torch.tensor(float(alpha_init),
                                                  dtype=torch.float32))
        self.register_buffer("beta", torch.tensor(float(beta),
                                                  dtype=torch.float32))

    @property
    def alpha(self):
        return 2.0 * torch.sigmoid(self.alpha_raw)

    @staticmethod
    def _exp_map_0(v, c):
        sqrt_c = c ** 0.5
        norm_v = v.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        factor = torch.tanh(sqrt_c * norm_v) / (sqrt_c * norm_v)
        return factor * v

    @staticmethod
    def _proj_to_ball(x, c):
        R = (1.0 / c) ** 0.5
        norm_x = x.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        scale = torch.clamp(R / (norm_x + 1e-5), max=1.0)
        return x * scale

    def compute_bias(self, input_ids):
        device = input_ids.device
        B = input_ids.shape[0]
        layer_ids = self.layer_id_lut[input_ids]
        cb_list = [self.codebook_l0, self.codebook_l1, self.codebook_l2]
        c_list = [self.c_l0, self.c_l1, self.c_l2]
        offset_list = [self.token_offset_l0.item(),
                       self.token_offset_l1.item(),
                       self.token_offset_l2.item()]
        K_list = [self.K_l0.item(), self.K_l1.item(), self.K_l2.item()]

        centroid_per_layer = []
        for l in range(3):
            mask = (layer_ids == l)
            cb = cb_list[l]
            token_idx_l = input_ids - offset_list[l]
            safe_idx = token_idx_l.clamp(0, cb.shape[0] - 1)
            history_emb = cb[safe_idx]
            mask_f = mask.float().unsqueeze(-1)
            cnt = mask_f.sum(1).clamp_min(1.0)
            centroid = (mask_f * history_emb).sum(1) / cnt
            centroid_per_layer.append(centroid)

        bias = torch.zeros(B, self.vocab_size, device=device,
                           dtype=torch.float32)
        for l in range(3):
            cb = cb_list[l]
            centroid = centroid_per_layer[l]
            c = c_list[l]
            x_sq = (centroid * centroid).sum(-1, keepdim=True)
            y_sq = (cb * cb).sum(-1, keepdim=True).T
            diff_sq = ((centroid.unsqueeze(1) - cb.unsqueeze(0))
                       ** 2).sum(-1)
            num = 2.0 * diff_sq
            denom = ((1.0 - x_sq) * (1.0 - y_sq)).clamp_min(1e-30)
            arg = (1.0 + num / denom).clamp_min(1.0 + 1e-7)
            sqrt_term = torch.sqrt((arg ** 2 - 1.0).clamp_min(1e-30))
            dist = (torch.log(arg + sqrt_term)
                    / (c ** 0.5).clamp_min(1e-30))
            offset_l = offset_list[l]
            K_l = K_list[l]
            bias[:, offset_l:offset_l + K_l] = -dist

        bias_log = torch.log_softmax(bias, dim=-1)
        return self.alpha * self.beta * bias_log


def install_scsb_eval(hg_rec, scsb_module, device):
    """Issue #107 (2026-08-10): 包裹 T5 forward, 在 lm_head 之后注入 SCSB bias (eval)."""
    import types
    scsb_module = scsb_module.to(device)
    hg_rec.add_module("scsb_module", scsb_module)

    if not hasattr(hg_rec, "_scsb_orig_forward"):
        hg_rec._scsb_orig_forward = hg_rec.forward

    def scsb_forward(self, input_ids=None, attention_mask=None,
                     labels=None, **kwargs):
        out = self._scsb_orig_forward(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       labels=labels, **kwargs)
        if (input_ids is not None
                and hasattr(out, "logits") and out.logits is not None):
            bias = self.scsb_module.compute_bias(input_ids)
            out.logits = out.logits + bias.unsqueeze(1)
        return out

    hg_rec.forward = types.MethodType(scsb_forward, hg_rec)
    return hg_rec


class HSCSBModuleEval(nn.Module):
    """Issue #108 (2026-08-10): Stage4 HSCSB eval 同源实现 (与 train 一致)."""

    def __init__(self, codebooks_t, cs, kappas, layer_id_lut,
                 alpha_init=[1.0, 1.0, 1.0], beta_l=[0.1, 0.2, 0.4],
                 beta_cross=0.05, alpha_max=5.0, vocab_size=1025):
        super().__init__()
        codebooks_ball = []
        for tan_emb, c in zip(codebooks_t, cs):
            tan_emb = tan_emb.float()
            c_t = torch.tensor(float(c), dtype=torch.float32)
            ball_emb = self._exp_map_0(tan_emb, c_t)
            ball_emb = self._proj_to_ball(ball_emb, c_t)
            codebooks_ball.append(ball_emb)
        self.register_buffer("codebook_l0", codebooks_ball[0])
        self.register_buffer("codebook_l1", codebooks_ball[1])
        self.register_buffer("codebook_l2", codebooks_ball[2])
        self.register_buffer("c_l0", torch.tensor(float(cs[0]), dtype=torch.float32))
        self.register_buffer("c_l1", torch.tensor(float(cs[1]), dtype=torch.float32))
        self.register_buffer("c_l2", torch.tensor(float(cs[2]), dtype=torch.float32))
        self.register_buffer("layer_id_lut",
                             torch.tensor(layer_id_lut, dtype=torch.long))
        self.register_buffer("token_offset_l0", torch.tensor(1, dtype=torch.long))
        self.register_buffer("token_offset_l1", torch.tensor(65, dtype=torch.long))
        self.register_buffer("token_offset_l2", torch.tensor(193, dtype=torch.long))
        self.register_buffer("K_l0", torch.tensor(64, dtype=torch.long))
        self.register_buffer("K_l1", torch.tensor(128, dtype=torch.long))
        self.register_buffer("K_l2", torch.tensor(256, dtype=torch.long))
        self.vocab_size = vocab_size
        self.alpha_raw = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        self.alpha_max = float(alpha_max)
        self.register_buffer("beta_l", torch.tensor(beta_l, dtype=torch.float32))
        self.register_buffer("beta_cross", torch.tensor(float(beta_cross),
                                                        dtype=torch.float32))

    @property
    def alphas(self):
        return self.alpha_max * torch.sigmoid(self.alpha_raw)

    @staticmethod
    def _exp_map_0(v, c):
        sqrt_c = c ** 0.5
        norm_v = v.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        factor = torch.tanh(sqrt_c * norm_v) / (sqrt_c * norm_v)
        return factor * v

    @staticmethod
    def _proj_to_ball(x, c):
        R = (1.0 / c) ** 0.5
        norm_x = x.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        scale = torch.clamp(R / (norm_x + 1e-5), max=1.0)
        return x * scale

    def _compute_layer_dist(self, input_ids, l):
        device = input_ids.device
        layer_ids = self.layer_id_lut[input_ids]
        cb_list = [self.codebook_l0, self.codebook_l1, self.codebook_l2]
        c_list = [self.c_l0, self.c_l1, self.c_l2]
        offset_list = [self.token_offset_l0.item(),
                       self.token_offset_l1.item(),
                       self.token_offset_l2.item()]
        cb = cb_list[l]
        c = c_list[l]
        mask = (layer_ids == l)
        token_idx_l = input_ids - offset_list[l]
        safe_idx = token_idx_l.clamp(0, cb.shape[0] - 1)
        history_emb = cb[safe_idx]
        mask_f = mask.float().unsqueeze(-1)
        cnt = mask_f.sum(1).clamp_min(1.0)
        centroid = (mask_f * history_emb).sum(1) / cnt
        x_sq = (centroid * centroid).sum(-1, keepdim=True)
        y_sq = (cb * cb).sum(-1, keepdim=True).T
        diff_sq = ((centroid.unsqueeze(1) - cb.unsqueeze(0)) ** 2).sum(-1)
        num = 2.0 * diff_sq
        denom = ((1.0 - x_sq) * (1.0 - y_sq)).clamp_min(1e-30)
        arg = (1.0 + num / denom).clamp_min(1.0 + 1e-7)
        sqrt_term = torch.sqrt((arg ** 2 - 1.0).clamp_min(1e-30))
        dist = (torch.log(arg + sqrt_term) / (c ** 0.5).clamp_min(1e-30))
        return -dist

    def compute_bias(self, input_ids):
        device = input_ids.device
        B = input_ids.shape[0]
        cb_list = [self.codebook_l0, self.codebook_l1, self.codebook_l2]
        offset_list = [self.token_offset_l0.item(),
                       self.token_offset_l1.item(),
                       self.token_offset_l2.item()]
        K_list = [self.K_l0.item(), self.K_l1.item(), self.K_l2.item()]
        alphas = self.alphas
        bias_total = torch.zeros(B, self.vocab_size, device=device,
                                 dtype=torch.float32)
        all_dist = []
        for l in range(3):
            neg_dist_l = self._compute_layer_dist(input_ids, l)
            all_dist.append(neg_dist_l)
            layer_bias = torch.zeros(B, self.vocab_size, device=device,
                                      dtype=torch.float32)
            layer_bias[:, offset_list[l]:offset_list[l] + K_list[l]] = neg_dist_l
            layer_bias_log = torch.log_softmax(layer_bias, dim=-1)
            bias_total = bias_total + self.beta_l[l] * alphas[l] * layer_bias_log
        cross_bias = torch.zeros(B, self.vocab_size, device=device,
                                  dtype=torch.float32)
        for l in range(3):
            cross_bias[:, offset_list[l]:offset_list[l] + K_list[l]] = all_dist[l]
        cross_bias_log = torch.log_softmax(cross_bias, dim=-1)
        bias_total = bias_total + self.beta_cross * cross_bias_log
        return bias_total


def install_hscsb_eval(hg_rec, hscsb_module, device):
    """Issue #108 (2026-08-10): 包裹 T5 forward, 注入 HSCSB bias (eval)."""
    import types
    hscsb_module = hscsb_module.to(device)
    hg_rec.add_module("hscsb_module", hscsb_module)

    if not hasattr(hg_rec, "_hscsb_orig_forward"):
        hg_rec._hscsb_orig_forward = hg_rec.forward

    def hscsb_forward(self, input_ids=None, attention_mask=None,
                      labels=None, **kwargs):
        out = self._hscsb_orig_forward(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        labels=labels, **kwargs)
        if (input_ids is not None
                and hasattr(out, "logits") and out.logits is not None):
            bias = self.hscsb_module.compute_bias(input_ids)
            out.logits = out.logits + bias.unsqueeze(1)
        return out

    hg_rec.forward = types.MethodType(hscsb_forward, hg_rec)
    return hg_rec


# ──────────────────────────────────────────────────────────────
# Issue #110 (2026-08-10): RDB eval 同源实现 (与 train 一致)
# ──────────────────────────────────────────────────────────────
class RDBModuleEval(nn.Module):
    """Issue #110 (2026-08-10): Stage4 RDB eval - Raw Distance Bias."""

    def __init__(self, codebooks_t, cs, layer_id_lut,
                 alpha_init=1.0, alpha_max=3.0, vocab_size=1025):
        super().__init__()
        codebooks_ball = []
        for tan_emb, c in zip(codebooks_t, cs):
            tan_emb = tan_emb.float()
            c_t = torch.tensor(float(c), dtype=torch.float32)
            ball_emb = self._exp_map_0(tan_emb, c_t)
            ball_emb = self._proj_to_ball(ball_emb, c_t)
            codebooks_ball.append(ball_emb)
        self.register_buffer("codebook_l0", codebooks_ball[0])
        self.register_buffer("codebook_l1", codebooks_ball[1])
        self.register_buffer("codebook_l2", codebooks_ball[2])
        self.register_buffer("c_l0", torch.tensor(float(cs[0]), dtype=torch.float32))
        self.register_buffer("c_l1", torch.tensor(float(cs[1]), dtype=torch.float32))
        self.register_buffer("c_l2", torch.tensor(float(cs[2]), dtype=torch.float32))
        self.register_buffer("layer_id_lut",
                             torch.tensor(layer_id_lut, dtype=torch.long))
        self.register_buffer("token_offset_l0", torch.tensor(1, dtype=torch.long))
        self.register_buffer("token_offset_l1", torch.tensor(65, dtype=torch.long))
        self.register_buffer("token_offset_l2", torch.tensor(193, dtype=torch.long))
        self.register_buffer("K_l0", torch.tensor(64, dtype=torch.long))
        self.register_buffer("K_l1", torch.tensor(128, dtype=torch.long))
        self.register_buffer("K_l2", torch.tensor(256, dtype=torch.long))
        self.vocab_size = vocab_size
        self.alpha_raw = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))
        self.alpha_max = float(alpha_max)

    @property
    def alpha(self):
        return self.alpha_max * torch.sigmoid(self.alpha_raw)

    @staticmethod
    def _exp_map_0(v, c):
        sqrt_c = c ** 0.5
        norm_v = v.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        factor = torch.tanh(sqrt_c * norm_v) / (sqrt_c * norm_v)
        return factor * v

    @staticmethod
    def _proj_to_ball(x, c):
        R = (1.0 / c) ** 0.5
        norm_x = x.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        scale = torch.clamp(R / (norm_x + 1e-5), max=1.0)
        return x * scale

    def _compute_layer_dist(self, input_ids, l):
        device = input_ids.device
        layer_ids = self.layer_id_lut[input_ids]
        cb_list = [self.codebook_l0, self.codebook_l1, self.codebook_l2]
        c_list = [self.c_l0, self.c_l1, self.c_l2]
        offset_list = [self.token_offset_l0.item(),
                       self.token_offset_l1.item(),
                       self.token_offset_l2.item()]
        cb = cb_list[l]
        c = c_list[l]
        mask = (layer_ids == l)
        token_idx_l = input_ids - offset_list[l]
        safe_idx = token_idx_l.clamp(0, cb.shape[0] - 1)
        history_emb = cb[safe_idx]
        mask_f = mask.float().unsqueeze(-1)
        cnt = mask_f.sum(1).clamp_min(1.0)
        centroid = (mask_f * history_emb).sum(1) / cnt
        x_sq = (centroid * centroid).sum(-1, keepdim=True)
        y_sq = (cb * cb).sum(-1, keepdim=True).T
        diff_sq = ((centroid.unsqueeze(1) - cb.unsqueeze(0)) ** 2).sum(-1)
        num = 2.0 * diff_sq
        denom = ((1.0 - x_sq) * (1.0 - y_sq)).clamp_min(1e-30)
        arg = (1.0 + num / denom).clamp_min(1.0 + 1e-7)
        sqrt_term = torch.sqrt((arg ** 2 - 1.0).clamp_min(1e-30))
        dist = (torch.log(arg + sqrt_term) / (c ** 0.5).clamp_min(1e-30))
        return -dist

    def compute_bias(self, input_ids):
        device = input_ids.device
        B = input_ids.shape[0]
        cb_list = [self.codebook_l0, self.codebook_l1, self.codebook_l2]
        offset_list = [self.token_offset_l0.item(),
                       self.token_offset_l1.item(),
                       self.token_offset_l2.item()]
        K_list = [self.K_l0.item(), self.K_l1.item(), self.K_l2.item()]
        bias_total = torch.zeros(B, self.vocab_size, device=device,
                                  dtype=torch.float32)
        for l in range(3):
            dist_l = self._compute_layer_dist(input_ids, l)
            bias_total[:, offset_list[l]:offset_list[l] + K_list[l]] = dist_l
        alpha = self.alpha
        bias_total = alpha * bias_total
        return bias_total


def install_rdb_eval(hg_rec, rdb_module, device):
    """Issue #110 (2026-08-10): 包裹 T5 forward, 注入 RDB bias (eval)."""
    import types
    rdb_module = rdb_module.to(device)
    hg_rec.add_module("rdb_module", rdb_module)

    if not hasattr(hg_rec, "_rdb_orig_forward"):
        hg_rec._rdb_orig_forward = hg_rec.forward

    def rdb_forward(self, input_ids=None, attention_mask=None,
                    labels=None, **kwargs):
        out = self._rdb_orig_forward(input_ids=input_ids,
                                      attention_mask=attention_mask,
                                      labels=labels, **kwargs)
        if (input_ids is not None
                and hasattr(out, "logits") and out.logits is not None):
            bias = self.rdb_module.compute_bias(input_ids)
            out.logits = out.logits + bias.unsqueeze(1)
        return out

    hg_rec.forward = types.MethodType(rdb_forward, hg_rec)
    return hg_rec


# ──────────────────────────────────────────────────────────────
# Issue #111 (2026-08-10): HRes — Stage3 Decoder Hyperbolic Residual eval 同步
# ──────────────────────────────────────────────────────────────


class HResModuleEval(nn.Module):
    """Issue #111 v34 HRes eval 同步: 同 HResModule, 但只 forward 路径用 (eval 无需 optimizer 关心)."""

    def __init__(self, c_per_layer, beta_init=0.01, beta_max=0.5):
        super().__init__()
        c_avg = float(sum(c_per_layer) / len(c_per_layer))
        self.register_buffer("c", torch.tensor(c_avg, dtype=torch.float32))
        self.beta_raw = nn.Parameter(torch.tensor(float(beta_init), dtype=torch.float32))
        self.beta_max = float(beta_max)

    @property
    def beta(self):
        return self.beta_max * torch.sigmoid(self.beta_raw)

    @staticmethod
    def _exp_map_0(v, c):
        sqrt_c = c ** 0.5
        norm_v = v.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        factor = torch.tanh(sqrt_c * norm_v) / (sqrt_c * norm_v)
        return factor * v

    @staticmethod
    def _proj_to_ball(x, c):
        R = (1.0 / c) ** 0.5
        norm_x = x.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        scale = torch.clamp(R / (norm_x + 1e-5), max=1.0)
        return x * scale

    @staticmethod
    def _log_map_0(y, c):
        sqrt_c = c ** 0.5
        norm_y = y.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        R = (1.0 / c) ** 0.5
        scale = (1.0 / sqrt_c) * torch.arctanh(sqrt_c * norm_y / R) / (norm_y / R)
        return scale * y

    def compute_residual(self, hidden_states):
        c = self.c
        h_hyp = self._exp_map_0(hidden_states, c)
        h_hyp = self._proj_to_ball(h_hyp, c)
        h_tan = self._log_map_0(h_hyp, c)
        return h_tan - hidden_states


def install_hres_eval(hg_rec, hres_module, device):
    """Issue #111 v34 HRes eval 同步: 注册 forward_pre_hook 到 hg_rec.model.lm_head."""
    hres_module = hres_module.to(device)
    hg_rec.add_module("hres_module", hres_module)

    def hres_lm_head_pre_hook(module, args):
        if not args:
            return None
        h = args[0]
        residual = hres_module.compute_residual(h)
        h_new = h + hres_module.beta * residual
        return (h_new,) + args[1:]

    lm_head = hg_rec.model.lm_head
    lm_head.register_forward_pre_hook(hres_lm_head_pre_hook)
    return hg_rec


def calculate_pos_index(preds, labels, maxk=20):
    # Issue #41 v85h fix (2026-08-09): 原 BUG — preds[i,j].tolist() (4-token list) vs cur_label (4-token list)
    # 永远不全等 → R@10=0. 改用 train_pure_t5.py 的向量化正确逻辑:
    # labels (B, seq_len) → (B,1,seq_len) 与 preds (B,maxk,seq_len) 广播, all(dim=-1) 判全等
    preds = preds.detach().cpu()
    labels = labels.detach().cpu()
    assert preds.shape[1] == maxk, f"preds.shape[1] = {preds.shape[1]} != {maxk}"
    pos_index = (preds == labels.unsqueeze(1)).all(dim=-1)  # (B, maxk)
    return pos_index


def recall_at_k(pos_index, k):
    return pos_index[:, :k].sum(dim=1).cpu().float()


def ndcg_at_k(pos_index, k):
    ranks = torch.arange(1, pos_index.shape[-1] + 1).to(pos_index.device)
    dcg = 1.0 / torch.log2(ranks + 1)
    dcg = torch.where(pos_index, dcg, torch.tensor(0.0, dtype=torch.float, device=dcg.device))
    return dcg[:, :k].sum(dim=1).cpu().float()


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    sid_sha = sha256_of(SID_NPY)
    if EXPECTED_SID_SHA:
        if sid_sha != EXPECTED_SID_SHA:
            raise ValueError(
                f"SID sha mismatch: got {sid_sha}, expected {EXPECTED_SID_SHA} ({SID_NPY})")

    model = HG_Rec(CONFIG)
    if GEO_RESIDUAL:
        # Issue #62: geo_residual v2 配置 (与 Stage3 train 脚本一致)
        GEO_KAPPA = [-0.2289, -0.1872, -0.0932, 0.0]
        GEO_SCALE = [4.0, 5.04, 6.35, 1.0]
        GEO_CODEBOOK_NORM = [4.0, 5.04, 6.35, 1.0]
        GEO_ALPHA_INIT = 0.01
        GEO_ALPHA_CAP = 0.1
        GEO_FORCE_ZERO_LAYERS = [3]

        class GeoResidualModuleEval(nn.Module):
            """Stage4 eval 用的精简版 (跟 Stage3 train 完全一致, 仅去训练相关字段)."""
            def __init__(self, d_model, meta_per_layer, alpha_init=0.01, alpha_cap=0.1, force_zero_layers=()):
                super().__init__()
                self.d_model = d_model
                self.num_layers, meta_dim = meta_per_layer.shape
                self.alpha_cap = alpha_cap
                meta_t = torch.as_tensor(meta_per_layer, dtype=torch.float32)
                self.register_buffer("meta", meta_t)
                self.mlps = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(meta_dim, d_model),
                        nn.GELU(),
                        nn.LayerNorm(d_model),
                        nn.Linear(d_model, d_model),
                    ) for _ in range(self.num_layers)
                ])
                alpha = torch.full((self.num_layers,), alpha_init)
                for l in force_zero_layers:
                    alpha[l] = 0.0
                # Issue #62 v4/v5 ckpt 兼容: ckpt 存的是 geo_module.alphas_raw, 此处注册同名参数
                self.alphas_raw = nn.Parameter(alpha)
                # 兼容 v3 ckpt (key = geo_module.alphas): 如果传入 state_dict 含 alphas, 重命名加载
                for mlp in self.mlps:
                    nn.init.normal_(mlp[-1].weight, std=0.01)
                    nn.init.zeros_(mlp[-1].bias)

            def get_deltas(self):
                deltas = [self.mlps[l](self.meta[l]) for l in range(self.num_layers)]
                return torch.stack(deltas, dim=0)

            def forward(self, input_embeds, layer_ids):
                deltas = self.get_deltas()
                alphas = self.alphas_raw.clamp(-self.alpha_cap, self.alpha_cap)
                valid = layer_ids >= 0
                safe_ids = layer_ids.clamp(min=0)
                delta_per_token = deltas[safe_ids]
                alpha_per_token = alphas[safe_ids]
                residual = (alpha_per_token.unsqueeze(-1) * delta_per_token) * valid.unsqueeze(-1).float()
                return input_embeds + residual

        _LAYER_ID_LUT = np.full(1025, -1, dtype=np.int64)
        _LAYER_ID_LUT[1:65] = 0
        _LAYER_ID_LUT[65:193] = 1
        _LAYER_ID_LUT[193:449] = 2
        _LAYER_ID_LUT[449:450] = 3

        num_layers = len(GEO_KAPPA)
        raw = np.array([GEO_KAPPA[:3], GEO_SCALE[:3], GEO_CODEBOOK_NORM[:3]], dtype=np.float32)
        means = raw.mean(axis=1, keepdims=True)
        stds = raw.std(axis=1, keepdims=True) + 1e-6
        meta = np.zeros((num_layers, 3), dtype=np.float32)
        for l in range(3):
            meta[l, 0] = (GEO_KAPPA[l] - means[0, 0]) / stds[0, 0]
            meta[l, 1] = (GEO_SCALE[l] - means[1, 0]) / stds[1, 0]
            meta[l, 2] = (GEO_CODEBOOK_NORM[l] - means[2, 0]) / stds[2, 0]
        meta[3, :] = 0.0

        geo_module = GeoResidualModuleEval(
            d_model=CONFIG["d_model"],
            meta_per_layer=meta,
            alpha_init=GEO_ALPHA_INIT,
            alpha_cap=GEO_ALPHA_CAP,
            force_zero_layers=GEO_FORCE_ZERO_LAYERS,
        )
        layer_id_lut_tensor = torch.as_tensor(_LAYER_ID_LUT, dtype=torch.long)

        # Monkey-patch: 跟 Stage3 train 一致
        device0 = torch.device(DEVICE)
        geo_module = geo_module.to(device0)
        layer_id_lut_tensor = layer_id_lut_tensor.to(device0)
        d_model_sqrt = CONFIG["d_model"] ** 0.5
        model.add_module("geo_module", geo_module)

        def geo_forward(self, input_ids, attention_mask=None, labels=None):
            input_embeds = self.model.shared(input_ids) * d_model_sqrt
            layer_ids = layer_id_lut_tensor[input_ids]
            input_embeds = self.geo_module(input_embeds, layer_ids)
            outputs = self.model(inputs_embeds=input_embeds, attention_mask=attention_mask, labels=labels)
            return outputs.loss, outputs.logits

        def geo_generate(self, input_ids, attention_mask=None, num_beams=20, **kwargs):
            input_embeds = self.model.shared(input_ids) * d_model_sqrt
            layer_ids = layer_id_lut_tensor[input_ids]
            input_embeds = self.geo_module(input_embeds, layer_ids)
            return self.model.generate(inputs_embeds=input_embeds, attention_mask=attention_mask,
                                       num_beams=num_beams, max_length=5,
                                       num_return_sequences=num_beams, **kwargs)

        import types
        model.forward = types.MethodType(geo_forward, model)
        model.generate = types.MethodType(geo_generate, model)
        print("[Issue #62] geo_residual enabled for eval (forward + generate patched)", flush=True)

    if CODEWORD_GEO:
        # Issue #63: 码字级几何残差 eval (跟 Stage3 train 完全一致, 仅去训练相关字段)
        import torch.nn as nn
        import numpy as _np

        class CodewordGeoResidualEval(nn.Module):
            """Stage4 eval 用的精简版 (跟 Stage3 train 完全一致, 仅去训练相关字段)."""
            def __init__(self, d_model, codebook_list, final_kappas, beta_init=0.0,
                         offsets=CODEWORD_OFFSETS, force_zero_layers=(3,), rho_max=0.10,
                         warmup_start=30, warmup_end=50, rho_max_per_layer=None):
                super().__init__()
                self.d_model = d_model
                self.num_layers = len(codebook_list)
                # Issue #63 v4: 跟 train 对齐 (warmup + per-layer ρ_max)
                if rho_max_per_layer is None:
                    self.rho_max_per_layer = torch.tensor([0.15, 0.05, 0.02][:self.num_layers], dtype=torch.float32)
                else:
                    self.rho_max_per_layer = torch.as_tensor(rho_max_per_layer[:self.num_layers], dtype=torch.float32)
                self.warmup_start = int(warmup_start)
                self.warmup_end = int(warmup_end)
                self.current_epoch = 200  # Stage4 eval 默认 warmup_factor=1.0 (best ckpt 时训练已完成 warmup)
                self.d_tangent = codebook_list[0].shape[-1]
                self.offsets = list(offsets)
                self.K = [cb.shape[0] for cb in codebook_list]
                for l in range(self.num_layers):
                    self.register_buffer(f"codebook_{l}", torch.as_tensor(codebook_list[l], dtype=torch.float32))
                self.register_buffer("kappas", torch.as_tensor(final_kappas, dtype=torch.float32))
                self.proj = nn.ModuleList([
                    nn.Linear(self.d_tangent + 2, d_model) for _ in range(self.num_layers)
                ])
                for proj_l in self.proj:
                    nn.init.normal_(proj_l.weight, std=0.01)
                    nn.init.zeros_(proj_l.bias)
                self.gate_mlp = nn.Sequential(
                    nn.Linear(2 * d_model, d_model),
                    nn.GELU(),
                    nn.Linear(d_model, 1),
                )
                for m in self.gate_mlp:
                    if isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight, std=0.01)
                        nn.init.zeros_(m.bias)
                beta = torch.zeros(self.num_layers)
                for l in force_zero_layers:
                    beta[l] = 0.0
                self.beta_raw = nn.Parameter(beta)  # Issue #63 v3: 跟 train 对齐, forward 时 clamp 到 ±rho_max
                self.register_buffer("rho_max_per_layer_buf", self.rho_max_per_layer.clone())  # v4
                self.ln_h = nn.LayerNorm(d_model)
                self.ln_q = nn.LayerNorm(d_model)

            def precompute_q_lut(self, device):
                max_id = max(off + k for off, k in zip(self.offsets, self.K))
                q_lut = torch.zeros(max_id + 1, self.d_model, device=device)
                for l in range(self.num_layers):
                    cb = getattr(self, f"codebook_{l}").to(device)
                    K_l = cb.shape[0]
                    r = cb.norm(dim=-1, keepdim=True)
                    k = self.kappas[l].expand(K_l, 1)
                    x = torch.cat([cb, r, k], dim=-1)
                    q_lk = self.proj[l](x)
                    q_lut[self.offsets[l]:self.offsets[l] + K_l] = q_lk
                return q_lut

            def forward(self, input_embeds, token_ids, layer_ids):
                valid = (layer_ids >= 0)
                safe_layer = layer_ids.clamp(min=0)
                q_lut = self.precompute_q_lut(input_embeds.device)
                safe_token = token_ids.clamp(min=0, max=q_lut.shape[0] - 1)
                q_per_token = q_lut[safe_token]
                h_ln = self.ln_h(input_embeds)
                q_ln = self.ln_q(q_per_token)
                g = torch.sigmoid(self.gate_mlp(torch.cat([h_ln, q_ln], dim=-1)))
                # Issue #63 v4: 同步 train 端 (warmup + per-layer ρ_max)
                warmup_factor = max(0.0, min(1.0, (self.current_epoch - self.warmup_start) / max(1, self.warmup_end - self.warmup_start)))
                rho = self.rho_max_per_layer_buf.to(self.beta_raw.device)
                beta_eff = warmup_factor * rho * torch.tanh(self.beta_raw / rho)
                beta_per_token = beta_eff[safe_layer.clamp(max=self.num_layers - 1)] * valid.float()
                delta = beta_per_token.unsqueeze(-1) * g * q_ln
                return input_embeds + delta

        # 从 Stage2 ckpt 读 codebook + final_kappas
        ckpt_stage2 = torch.load(CODEWORD_STAGE2_CKPT, map_location="cpu", weights_only=False)
        sd_stage2 = ckpt_stage2["model_state_dict"]
        final_kappas = list(ckpt_stage2["final_kappas"])
        codebook_list = []
        for l in range(3):
            cb = sd_stage2[f"vq_layers.{l}.embeddings.weight"].numpy()
            codebook_list.append(cb)
        print(f"[Issue #63] Stage2 ckpt loaded: codebook shapes = L0={codebook_list[0].shape} "
              f"L1={codebook_list[1].shape} L2={codebook_list[2].shape}", flush=True)
        print(f"[Issue #63] final_kappas = {[round(k, 4) for k in final_kappas]}", flush=True)

        _LAYER_ID_LUT = _np.full(1025, -1, dtype=_np.int64)
        _LAYER_ID_LUT[1:65] = 0
        _LAYER_ID_LUT[65:193] = 1
        _LAYER_ID_LUT[193:449] = 2
        _LAYER_ID_LUT[449:450] = 3

        codeword_module = CodewordGeoResidualEval(
            d_model=CONFIG["d_model"],
            codebook_list=codebook_list,
            final_kappas=final_kappas,
            beta_init=0.0,
            offsets=CODEWORD_OFFSETS,
            force_zero_layers=(3,),
            rho_max=CODEWORD_RHO_MAX,
        )
        device0 = torch.device(DEVICE)
        codeword_module = codeword_module.to(device0)
        layer_id_lut_tensor = torch.as_tensor(_LAYER_ID_LUT, dtype=torch.long).to(device0)
        d_model_sqrt = CONFIG["d_model"] ** 0.5
        model.add_module("codeword_geo_module", codeword_module)

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
        model.forward = types.MethodType(codeword_forward, model)
        model.generate = types.MethodType(codeword_generate, model)
        print("[Issue #63] codeword_geo_residual enabled for eval (forward + generate patched)", flush=True)

    # Issue #64: 安装 HAB (跟 Stage3 train 共享同一个 HyperbolicAttentionBias 实现)
    # install_hab 通过 monkey-patch model.encoder.forward 注入 B_geo (无需 patch generate, 因为
    # generate 内部也是 encoder + decoder 路径, encoder 已被 patch)
    if HAB_ENABLED:
        codebook_list, hab_final_cs = load_hab_assets_from_stage2_ckpt(HAB_STAGE2_CKPT)
        print(f"[HAB curvature] final_cs from Stage2 ckpt = "
              f"{[round(c, 4) for c in hab_final_cs]} (c>0 直接使用, 不由 κ 反推)", flush=True)
        D_list, Dbar_list, stats_list = precompute_distance_matrices(
            codebook_list, hab_final_cs, use_delta_curvature=HAB_DELTA_CURVATURE_EVAL)
        for stats in stats_list:
            assert stats["finite"], f"L{stats['layer']} 距离矩阵含 NaN/Inf"
            assert stats["sym_err"] < 1e-5, f"L{stats['layer']} 对称误差 {stats['sym_err']} >= 1e-5"
            assert stats["diag_max"] < 1e-6, f"L{stats['layer']} 对角线 {stats['diag_max']} >= 1e-6"
        hab_module = HyperbolicAttentionBias(Dbar_list, lambda_max=HAB_LAMBDA_MAX_VAL,
                                              enable_residual=RESIDUAL_HAB_ENABLED,
                                              residual_alpha_init=RESIDUAL_ALPHA_INIT,
                                              warmup_T0=HAB_WARMUP_T0_EVAL,
                                              warmup_Tw=HAB_WARMUP_TW_EVAL)
        layer_id_lut_array = make_hab_layer_id_lut()
        install_hab(model, hab_module, layer_id_lut_array)
        print(f"[Issue #64] hyperbolic_attn_bias enabled for eval "
              f"(encoder.forward patched, λ_max={HAB_LAMBDA_MAX_VAL}, "
              f"Dbar median=[{stats_list[0]['median']:.4f}, {stats_list[1]['median']:.4f}, {stats_list[2]['median']:.4f}])",
              flush=True)

    # Issue #236 (2026-08-10): Stage4 eval 同步支持 Poincaré Attention Scoring
    # eval 时 c_learnable=False (避免改变训练末态的 c), c 直接读 ckpt 中的 log_c_param
    poincare_attn_module = None
    if POINCARE_ATTN_SCORING:
        # eval 时 c_learnable=False 但 ckpt 中保存的 log_c_param 仍可被加载
        # install_poincare_attention_scoring 会自动把 log_c_param 注册到 model state_dict
        poincare_attn_module = install_poincare_attention_scoring(
            model, c_init=POINCARE_C_INIT, c_learnable=False)
        print(f"[Issue #236] poincare_attn_scoring ON for eval (c_learnable=False, "
              f"c_init={POINCARE_C_INIT})", flush=True)

    # Issue #107 (2026-08-10): SCSB eval 同步 — Curvature Soft-Bias on T5 logits
    # 必须在 load_state_dict 之前 add_module(scsb_module), 否则 strict load 找不到 scsb_module.* keys
    scsb_module = None
    if SCSB_ENABLED:
        sd_ckpt = torch.load(SCSB_STAGE2_CKPT, map_location="cpu",
                              weights_only=False)
        sd = sd_ckpt["model_state_dict"]
        codebooks_t = [
            sd["vq_layers.0.embeddings.weight"].float(),
            sd["vq_layers.1.embeddings.weight"].float(),
            sd["vq_layers.2.embeddings.weight"].float(),
        ]
        cs = [float(c) for c in sd_ckpt["final_cs"]]
        kappas = [float(k) for k in sd_ckpt["final_kappas"]]
        scsb_module = SCSBModuleEval(
            codebooks_t=codebooks_t,
            cs=cs,
            kappas=kappas,
            layer_id_lut=SCSB_LAYER_ID_LUT,
            alpha_init=SCSB_ALPHA_INIT,
            beta=SCSB_BETA,
            vocab_size=1025,
        ).to(DEVICE)
        install_scsb_eval(model, scsb_module, DEVICE)
        print(f"[Issue #107 v30 SCSB] eval ON: c_per_layer={[f'{c:.4f}' for c in cs]} "
              f"beta={SCSB_BETA}", flush=True)
    # v23 (2026-08-11): Stage4 curvature residual 镜像 Stage3 train, 保持 train/eval 一致
    # v23 v2 (2026-08-11): decoder 端也注入, L1+L2
    if CURVATURE_RESIDUAL_ENABLED:
        sd_ckpt = torch.load(HAB_STAGE2_CKPT, map_location="cpu", weights_only=False)
        sd = sd_ckpt["model_state_dict"]
        cv_kappa_l1 = float(sd_ckpt["final_kappas"][1])
        cv_kappa_l2 = float(sd_ckpt["final_kappas"][2])
        cv_delta_l1 = sd["vq_layers.1.delta_kappa.weight"].float()
        cv_delta_l2 = sd["vq_layers.2.delta_kappa.weight"].float()
        curv_module = CurvatureResidualModuleEval(
            kappa_l1=cv_kappa_l1,
            delta_kappa_l1=cv_delta_l1,
            kappa_l2=cv_kappa_l2,
            delta_kappa_l2=cv_delta_l2,
            d_model=128,
            mlp_hidden=CURVATURE_RESIDUAL_MLP_HIDDEN,
            alpha_init=CURVATURE_RESIDUAL_ALPHA_INIT,
        ).to(DEVICE)
        # Try to load 4 αs + f1/f2 weights from Stage3 ckpt (保持 train/eval 一致)
        if os.path.exists(CKPT_PATH):
            _cv_ck = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
            _cv_sd = _cv_ck.get("model_state_dict", _cv_ck)
            _alpha_keys = [
                "curvature_residual_module.alpha_1_enc",
                "curvature_residual_module.alpha_2_enc",
                "curvature_residual_module.alpha_1_dec",
                "curvature_residual_module.alpha_2_dec",
            ]
            _loaded_any = False
            # v23 v1 backward compat: if alpha_1 (not _enc) in ckpt, load as encoder
            if "curvature_residual_module.alpha_1" in _cv_sd and "curvature_residual_module.alpha_1_enc" not in _cv_sd:
                curv_module.alpha_1_enc.data = _cv_sd["curvature_residual_module.alpha_1"].detach().clone()
                _loaded_any = True
            for _ak in _alpha_keys:
                if _ak in _cv_sd:
                    _attr = _ak.split(".")[-1]
                    getattr(curv_module, _attr).data = _cv_sd[_ak].detach().clone()
                    _loaded_any = True
            # Load f1, f2
            try:
                curv_module.f1.load_state_dict({
                    k.split("f1.")[-1]: v for k, v in _cv_sd.items() if k.startswith("curvature_residual_module.f1.")
                })
                _loaded_any = True
            except Exception as _e:
                print(f"[v23 v2] WARN: failed to load f1 from ckpt ({_e}); use init", flush=True)
            try:
                curv_module.f2.load_state_dict({
                    k.split("f2.")[-1]: v for k, v in _cv_sd.items() if k.startswith("curvature_residual_module.f2.")
                })
                _loaded_any = True
            except Exception as _e:
                # f2 might not be in ckpt (v23 v1 had no f2), use init
                pass
            print(f"[v23 v2 curvature residual eval] alphas loaded from ckpt "
                  f"a1_enc={float(curv_module.alpha_1_enc.item()):.4f} "
                  f"a2_enc={float(curv_module.alpha_2_enc.item()):.4f} "
                  f"a1_dec={float(curv_module.alpha_1_dec.item()):.4f} "
                  f"a2_dec={float(curv_module.alpha_2_dec.item()):.4f} "
                  f"loaded_any={_loaded_any}", flush=True)
        else:
            print(f"[v23 v2 curvature residual eval] no ckpt at {CKPT_PATH}, use init", flush=True)
        layer_id_lut_t = torch.from_numpy(SCSB_LAYER_ID_LUT)
        install_curvature_residual_eval(
            model, curv_module, layer_id_lut_t,
            decoder_enabled=CURVATURE_RESIDUAL_DECODER_ENABLED,
        )
        _cv_side = "encoder+decoder (L1+L2)" if CURVATURE_RESIDUAL_DECODER_ENABLED else "encoder-only (L1+L2)"
        print(f"[v23 v2 curvature residual] eval ON: side={_cv_side} "
              f"L1 kappa={cv_kappa_l1:.4f} delta_kappa_std={float(cv_delta_l1.std().item()):.4f} "
              f"L2 kappa={cv_kappa_l2:.4f} delta_kappa_std={float(cv_delta_l2.std().item()):.4f} "
              f"mlp_hidden={CURVATURE_RESIDUAL_MLP_HIDDEN} "
              f"learnable_params={sum(p.numel() for p in curv_module.parameters() if p.requires_grad)}", flush=True)

    # Issue #108 (2026-08-10): HSCSB eval 同步
    hscsb_module = None
    if HSCSB_ENABLED:
        sd_ckpt = torch.load(HSCSB_STAGE2_CKPT, map_location="cpu",
                              weights_only=False)
        sd = sd_ckpt["model_state_dict"]
        codebooks_t = [
            sd["vq_layers.0.embeddings.weight"].float(),
            sd["vq_layers.1.embeddings.weight"].float(),
            sd["vq_layers.2.embeddings.weight"].float(),
        ]
        cs = [float(c) for c in sd_ckpt["final_cs"]]
        kappas = [float(k) for k in sd_ckpt["final_kappas"]]
        hscsb_module = HSCSBModuleEval(
            codebooks_t=codebooks_t,
            cs=cs,
            kappas=kappas,
            layer_id_lut=SCSB_LAYER_ID_LUT,
            alpha_init=HSCSB_ALPHA_INIT,
            beta_l=[HSCSB_BETA_L0, HSCSB_BETA_L1, HSCSB_BETA_L2],
            beta_cross=HSCSB_BETA_CROSS,
            alpha_max=5.0,
            vocab_size=1025,
        ).to(DEVICE)
        install_hscsb_eval(model, hscsb_module, DEVICE)
        print(f"[Issue #108 v31 HSCSB] eval ON: c_per_layer={[f'{c:.4f}' for c in cs]} "
              f"beta_l=[{HSCSB_BETA_L0}, {HSCSB_BETA_L1}, {HSCSB_BETA_L2}] "
              f"beta_cross={HSCSB_BETA_CROSS}", flush=True)

    # Issue #110 (2026-08-10): RDB eval 安装 (与 train 一致)
    if RDB_ENABLED:
        sd_ckpt = torch.load(RDB_STAGE2_CKPT, map_location="cpu",
                              weights_only=False)
        sd = sd_ckpt["model_state_dict"]
        codebooks_t = [
            sd["vq_layers.0.embeddings.weight"].float(),
            sd["vq_layers.1.embeddings.weight"].float(),
            sd["vq_layers.2.embeddings.weight"].float(),
        ]
        cs = [float(c) for c in sd_ckpt["final_cs"]]
        rdb_module = RDBModuleEval(
            codebooks_t=codebooks_t,
            cs=cs,
            layer_id_lut=SCSB_LAYER_ID_LUT,
            alpha_init=RDB_ALPHA_INIT,
            alpha_max=3.0,
            vocab_size=1025,
        ).to(DEVICE)
        install_rdb_eval(model, rdb_module, DEVICE)
        print(f"[Issue #110 v33 RDB] eval ON: c_per_layer={[f'{c:.4f}' for c in cs]} "
              f"alpha_init={RDB_ALPHA_INIT}", flush=True)

    # Issue #111 (2026-08-10): HRes eval 安装 (与 train 一致)
    if HRES_ENABLED:
        sd_ckpt = torch.load(HRES_STAGE2_CKPT, map_location="cpu",
                              weights_only=False)
        cs = [float(c) for c in sd_ckpt["final_cs"]]
        hres_module = HResModuleEval(
            c_per_layer=cs,
            beta_init=HRES_BETA_INIT,
            beta_max=0.5,
        ).to(DEVICE)
        install_hres_eval(model, hres_module, DEVICE)
        print(f"[Issue #111 v34 HRes] eval ON: c_per_layer={[f'{c:.4f}' for c in cs]} "
              f"beta_init={HRES_BETA_INIT}", flush=True)

    # Issue #70: 安装 DecorPromptFormer (candidate bins + alpha gate)
    # 必须在 load_state_dict 之前 add_module(pf_module), 否则 strict load 找不到 pf_module.* keys
    if PROMPT_FORMER_ENABLED:
        from common.decor_prompt_former import DecorPromptFormer
        pf_module = DecorPromptFormer(
            d_model=CONFIG["d_model"],
            vocab_size=CONFIG.get("vocab_size", 1024),
            num_bins=4,
            codes_per_bin=256,
            num_bos_queries=PROMPT_FORMER_NUM_BOS_QUERIES,
            alpha_init=PROMPT_FORMER_ALPHA,
        )
        install_prompt_former_eval(model, pf_module)
        print(f"[Issue #70] prompt_former enabled for eval "
              f"(forward+generate patched, alpha_init={PROMPT_FORMER_ALPHA:.3f}, "
              f"num_bos_queries={PROMPT_FORMER_NUM_BOS_QUERIES}, "
              f"pf_params={sum(p.numel() for p in pf_module.parameters())})",
              flush=True)

    state_dict = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    # Issue #62 ckpt 兼容:
    # - v3 ckpt: key=geo_module.alphas (旧命名)
    # - v4/v5 ckpt: 同时存 geo_module.alphas_raw + geo_module.alphas (alias 冗余)
    # 当前 GeoResidualModuleEval 类只用 alphas_raw; 统一处理:
    if GEO_RESIDUAL:
        # v3 路径: alphas → alphas_raw 重命名
        if "geo_module.alphas" in state_dict and "geo_module.alphas_raw" not in state_dict:
            state_dict["geo_module.alphas_raw"] = state_dict.pop("geo_module.alphas")
            print("[Issue #62] v3 ckpt detected, remapped geo_module.alphas → geo_module.alphas_raw", flush=True)
        # v4/v5 路径: 移除冗余 alphas key, 只保留 alphas_raw
        elif "geo_module.alphas" in state_dict and "geo_module.alphas_raw" in state_dict:
            del state_dict["geo_module.alphas"]
            print("[Issue #62] v4/v5 ckpt detected, dropped redundant geo_module.alphas", flush=True)
    # Issue #63 ckpt 兼容: v2 用 codeword_geo_module.beta (自由漂移), v3 用 codeword_geo_module.beta_raw (smooth clamp)
    if CODEWORD_GEO_ENABLED:
        b_key = "codeword_geo_module.beta"
        br_key = "codeword_geo_module.beta_raw"
        if b_key in state_dict and br_key not in state_dict:
            state_dict[br_key] = state_dict.pop(b_key)
            print("[Issue #63] v2 ckpt detected, remapped codeword_geo_module.beta → codeword_geo_module.beta_raw", flush=True)
        elif b_key in state_dict and br_key in state_dict:
            del state_dict[b_key]
            print("[Issue #63] v2+v3 ckpt detected, dropped redundant codeword_geo_module.beta", flush=True)
    # Issue #141 v85u (2026-08-09) HAB ckpt 加载修复: v77原 ckpt 实际存了 11 HAB keys (U/V/Dbar/lambda_raw/residual_alpha), 之前假设"v77 纯 T5 ckpt"是错的. 必须把 ckpt 的 HAB params 加载到 hab_module, 否则 Stage4 eval 用新 init 值 (lambda_raw=0.1) 跟训练收敛值 (lambda_raw=[-0.057, 0.571, 0.689]) 完全不同 → HAB bias 错误 → 预测全 0 (test_R@10=0).
    hab_keys_in_ckpt = [k for k in state_dict.keys() if k.startswith('hab_module.')]
    if HAB_ENABLED and hab_keys_in_ckpt:
        hab_state = {k[len('hab_module.'):]: state_dict.pop(k) for k in hab_keys_in_ckpt}
        hab_missing, hab_unexpected = hab_module.load_state_dict(hab_state, strict=False)
        print(f"[Stage4/v85u] loaded {len(hab_keys_in_ckpt)} HAB params from ckpt "
              f"(lambda_raw={[round(hab_module.lambda_raw[i].item(), 4) for i in range(3)]}, "
              f"residual_alpha={[round(hab_module.residual_alpha[i].item(), 4) for i in range(3)]})", flush=True)
        if hab_missing:
            print(f"[Stage4/v85u] HAB missing keys: {hab_missing}", flush=True)
        if hab_unexpected:
            print(f"[Stage4/v85u] HAB unexpected keys: {hab_unexpected}", flush=True)
    elif HAB_ENABLED:
        print(f"[Stage4/v85u] no HAB params in ckpt, using init values", flush=True)
    # Issue #224: c_perturb_raw 加载 + 启用 (与 train 端 v85p_repro 配套)
    if HAB_ENABLED and CPERTURB_ENABLED:
        hab_module.c_perturb_enabled = True
        hab_module.c_perturb_scale = float(CPERTURB_SCALE)
        # c_perturb_raw 已通过 hab_state 加载 (在 strict=False 下, 即便 ckpt 中是 trainable tensor 也能加载)
        if hasattr(hab_module, "c_perturb_raw"):
            print(f"[Stage4/Issue #224] c_perturb ENABLED for eval "
                  f"(c_perturb_raw={[round(hab_module.c_perturb_raw[i].item(), 4) for i in range(3)]}, "
                  f"scale={CPERTURB_SCALE})", flush=True)
        else:
            print(f"[Stage4/Issue #224] c_perturb enabled flag set but c_perturb_raw not found in hab_module", flush=True)
    # strict=False: v77 纯 T5 ckpt (134 keys) + HAB forward monkey-patch, state_dict 不存 hab_module.*
    #    keys; strict=True 会 RuntimeError. 不允许 silent fallback (R8) — 缺失 keys 必须 log 出来供审查.
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"[Stage4] load_state_dict missing keys (HAB/geo/codeword 模块未在 ckpt, 接受 — "
              f"{len(missing_keys)} keys): {missing_keys[:6]}...", flush=True)
    if unexpected_keys:
        print(f"[Stage4] load_state_dict unexpected keys (ckpt 含未注册模块, "
              f"{len(unexpected_keys)} keys): {unexpected_keys[:6]}...", flush=True)
    model.to(DEVICE)
    model.eval()

    ds = GenRecDataset(
        dataset_path=EVAL_PARQUET, code_path=SID_NPY, mode="evaluation",
        codebook_size=CODEBOOK_SIZE, max_len=MAX_LEN)
    loader = GenRecDataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    n = len(ds)

    recalls = {f"R@{k}": [] for k in TOP_K}
    ndcgs = {f"NDCG@{k}": [] for k in TOP_K}
    t0 = time.time()
    # Issue #141 ensemble (2026-08-09): 收集 raw top-K predictions per user, 落盘供 ensemble post-process.
    # preds shape: (B, BEAM_SIZE, 4) — 4-token SID code.  R@10 = target 出现在 top-10 of 30.
    all_preds = []
    all_labels = []
    all_history_ids = []
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            input_ids = batch["history"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["target"].to(DEVICE)
            preds = model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=BEAM_SIZE, num_beam_groups=NUM_BEAM_GROUPS if NUM_BEAM_GROUPS > 0 else 1, diversity_penalty=DIVERSITY_PENALTY if NUM_BEAM_GROUPS > 0 else 0.0, length_penalty=LENGTH_PENALTY)
            preds = preds[:, 1:]
            preds = preds.reshape(input_ids.shape[0], BEAM_SIZE, -1)
            # Issue #238 (2026-08-10): Stage4 Poincaré Re-ranking
            # alpha=0 完全等价 v18 baseline (no rerank, 保持 generate 原序)
            # alpha>0 按 R_geo 重排 top-K (鼓励 candidate 与 history 在 Poincaré 球上接近)
            if POINCARE_RERANK:
                from common.poincare_rerank import compute_geo_score_batch
                _codebook_t, _c_t = load_poincare_assets(HAB_STAGE2_CKPT, layer=RERANK_LAYER)
                R_geo = compute_geo_score_batch(input_ids, preds, _codebook_t, _c_t, device=DEVICE)  # (B, K)
                # 策略:
                #   α=0 → 严格保持原 beam 序 (rank_score 主导, R_geo = 0)
                #   α>0 → 按 α*R_geo 重排 (鼓励 candidate 与 history 在 Poincaré 球上接近)
                # rank_score = -beam_idx / K 归一化到 [-1, 0]
                rank_score = -torch.arange(BEAM_SIZE, dtype=torch.float, device=DEVICE).unsqueeze(0).expand(input_ids.shape[0], -1) / BEAM_SIZE
                final_score = rank_score + RERANK_ALPHA * R_geo
                new_order = final_score.argsort(dim=-1, descending=True)  # (B, K)
                preds = torch.gather(preds, 1, new_order.unsqueeze(-1).expand(-1, -1, preds.shape[-1]))
            pos_index = calculate_pos_index(preds, labels, maxk=BEAM_SIZE)
            for k in TOP_K:
                recalls[f"R@{k}"].append(recall_at_k(pos_index, k).mean().item())
                ndcgs[f"NDCG@{k}"].append(ndcg_at_k(pos_index, k).mean().item())
            # Save predictions for ensemble
            all_preds.append(preds.detach().cpu().numpy().tolist())
            all_labels.append(labels.detach().cpu().numpy().tolist())
            all_history_ids.append(input_ids.detach().cpu().numpy().tolist())
            if (bi + 1) % 50 == 0:
                print(f"  {bi+1}/{len(loader)} batches done", flush=True)

    result = {k: sum(v) / len(v) for k, v in recalls.items()}
    result.update({k: sum(v) / len(v) for k, v in ndcgs.items()})
    result["n_eval"] = n
    result["t_eval_s"] = round(time.time() - t0, 1)
    result["ckpt"] = CKPT_PATH
    result["sid_sha256"] = sid_sha
    result["eval_parquet"] = EVAL_PARQUET
    result["tag"] = TAG
    result["done_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    with open(VERDICT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    # Issue #141 ensemble (2026-08-09): 落盘 raw predictions
    preds_arr = np.array([p for batch_preds in all_preds for p in batch_preds], dtype=np.int32)  # (n, BEAM_SIZE, 4)
    labels_arr = np.array([l for batch_labels in all_labels for l in batch_labels], dtype=np.int32)  # (n, 4)
    preds_path = PRODUCT_DIR / "raw_predictions.npz"
    np.savez_compressed(preds_path, preds=preds_arr, labels=labels_arr)
    print(f"  saved raw predictions: {preds_path} shape={preds_arr.shape}", flush=True)
    print(f"=== {TAG} test R@5/10/20 = {result['R@5']:.4f}/{result['R@10']:.4f}/{result['R@20']:.4f} "
          f"NDCG@5/10/20 = {result['NDCG@5']:.4f}/{result['NDCG@10']:.4f}/{result['NDCG@20']:.4f}")
    print(f"=== verdict: {VERDICT_PATH}")


if __name__ == "__main__":
    main()
