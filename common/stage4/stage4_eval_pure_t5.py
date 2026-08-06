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
_args = _argparser.parse_args()

CKPT_PATH = _args.ckpt_path
SID_NPY = _args.sid_npy
PRODUCT_DIR = Path(_args.product_dir)
DEVICE = _args.device
TAG = _args.tag
EXPECTED_SID_SHA = _args.expected_sid_sha
GEO_RESIDUAL = _args.geo_residual
# Stage4 = test only (held-out). 硬编码, 不允许覆盖 (valid 由 Stage3 val_trace 覆盖)
EVAL_PARQUET = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/test.parquet"

# 超参 (R30 硬编码 — 变体需 fork 脚本)
BATCH_SIZE = 96
SEED = 42

CODEBOOK_SIZE = [64, 128, 256, 1]
CONFIG = dict(
    num_layers=6, num_decoder_layers=4, d_model=128, d_ff=1024,
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
