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

# Issue #71 Phase A (2026-08-07): eval 时 T_0=T_w=0 → warmup_w=1 立即生效
# HAB_WARMUP_T0_EVAL / HAB_WARMUP_TW_EVAL 在 _args = parse_args() 之后赋值 (line ~99)
# Issue #70: DECOR PromptFormer (candidate bins + alpha gate) eval-time 安装 (与 HAB/GEO 正交)
_argparser.add_argument("--enable_prompt_former", action="store_true", help="Issue #70: 安装 DecorPromptFormer 模块, 加载 pf_module.* 参数, monkey-patch forward+generate")
_argparser.add_argument("--prompt_former_alpha", type=float, default=0.35, help="Issue #70: alpha gate sigmoid 初始值 (Stage3 train 默认 0.35)")
_argparser.add_argument("--prompt_former_num_bos_queries", type=int, default=64, help="Issue #70: learnable bos_queries count (DECOR 默认 64)")
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
    num_layers=6, num_decoder_layers=6, d_model=128, d_ff=1024,  # Issue #141 v85j (2026-08-08): num_decoder_layers 4→6 同步 Stage3 训练
    num_heads=6, d_kv=64, dropout_rate=0.1, vocab_size=1025,
    pad_token_id=0, eos_token_id=0, decoder_start_token_id=0,
    feed_forward_proj="relu",
)
TOP_K = [5, 10, 20]
BEAM_SIZE = 20
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


def calculate_pos_index(preds, labels, maxk=20):
    preds = preds.detach().cpu()
    labels = labels.detach().cpu()
    pos_index = torch.zeros((preds.shape[0], maxk), dtype=torch.bool)
    for i in range(preds.shape[0]):
        cur_label = labels[i].tolist()
        for j in range(maxk):
            if preds[i, j].tolist() == cur_label:
                pos_index[i, j] = True
                break
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
            assert stats["sym_err"] < 1e-6, f"L{stats['layer']} 对称误差 {stats['sym_err']} >= 1e-6"
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
    model.load_state_dict(state_dict)
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
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            input_ids = batch["history"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["target"].to(DEVICE)
            preds = model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=BEAM_SIZE)
            preds = preds[:, 1:]
            preds = preds.reshape(input_ids.shape[0], BEAM_SIZE, -1)
            pos_index = calculate_pos_index(preds, labels, maxk=BEAM_SIZE)
            for k in TOP_K:
                recalls[f"R@{k}"].append(recall_at_k(pos_index, k).mean().item())
                ndcgs[f"NDCG@{k}"].append(ndcg_at_k(pos_index, k).mean().item())
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
    print(f"=== {TAG} test R@5/10/20 = {result['R@5']:.4f}/{result['R@10']:.4f}/{result['R@20']:.4f} "
          f"NDCG@5/10/20 = {result['NDCG@5']:.4f}/{result['NDCG@10']:.4f}/{result['NDCG@20']:.4f}")
    print(f"=== verdict: {VERDICT_PATH}")


if __name__ == "__main__":
    main()
