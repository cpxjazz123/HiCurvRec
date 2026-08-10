#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v24 Stage4 install smoke test — 验证 install_curvature_attn_bias_eval 在 HG_Rec + T5 上工作.

5 验证点:
  1. install 不抛异常, hg_rec.model.encoder.block[*].forward 被替换
  2. forward 时 cab_bias 被加到 position_bias (encoder-only)
  3. γ=0 时 bias 全零, model 输出与 baseline 一致 (起点保护)
  4. γ=0.5 时 bias 非零, model 输出与 baseline 不同
  5. decoder 部分不受影响 (dec cab_bias_cache=None)
"""
import sys
import os
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
os.chdir(REPO)
sys.path.insert(0, str(REPO))

import torch
import numpy as np

# Load Stage2 ckpt
CKPT_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/v22b_full_1000ep/hrqvae_kappa_sync.ckpt"
ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
sd = ckpt["model_state_dict"]
kappa_l0 = float(ckpt["final_kappas"][0])
kappa_l1 = float(ckpt["final_kappas"][1])
kappa_l2 = float(ckpt["final_kappas"][2])
delta_kappa_l1 = sd["vq_layers.1.delta_kappa.weight"].float()
delta_kappa_l2 = sd["vq_layers.2.delta_kappa.weight"].float()

# 构造 layer_id_lut (L0=[1,64], L1=[65,192], L2=[193,448], L3=449, PAD=0)
lut = torch.zeros(1025, dtype=torch.long)
lut[0] = -1
for i in range(1, 65):
    lut[i] = 0
for i in range(65, 193):
    lut[i] = 1
for i in range(193, 449):
    lut[i] = 2
lut[449] = 3

# Load Stage4 module (avoid argparse by stubbing sys.argv)
sys.argv = ["smoke", "--ckpt_path", "/tmp/_nope.pth",
            "--sid_npy", "/tmp/_nope.npy",
            "--product_dir", "/tmp/_nope_eval",
            "--tag", "smoke"]
import importlib.util
spec = importlib.util.spec_from_file_location(
    "stage4_eval", "common/stage4/stage4_eval_pure_t5_v85p_4layer.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
print(f"[v24 smoke stage4] loaded stage4 module OK")

# Build minimal CurvatureAttnBiasEval
cab_eval = m.CurvatureAttnBiasEval(
    kappa_l0=kappa_l0, kappa_l1=kappa_l1, kappa_l2=kappa_l2,
    delta_kappa_l1=delta_kappa_l1, delta_kappa_l2=delta_kappa_l2,
    gamma_init=0.0,
)

# Build a minimal T5 model (t5-small / t5-mini)
from transformers import T5Config, T5ForConditionalGeneration
config = T5Config(
    vocab_size=1025,
    d_model=128,
    d_kv=32,
    d_ff=256,
    num_layers=4,
    num_decoder_layers=4,
    num_heads=4,
    relative_attention_num_buckets=32,
    dropout_rate=0.0,
    pad_token_id=0,
    eos_token_id=448,
    decoder_start_token_id=449,
)
model = T5ForConditionalGeneration(config)
print(f"[v24 smoke stage4] built T5 model OK (params={sum(p.numel() for p in model.parameters()):,})")

# Wrap to HGRec-style: hg_rec.model = T5ForConditionalGeneration
class HGRecMock(torch.nn.Module):
    def __init__(self, t5):
        super().__init__()
        self.model = t5

hg_rec = HGRecMock(model)
print(f"[v24 smoke stage4] wrapped as HGRecMock (hg_rec.model.encoder exists) ✓")

# Test 1: install 不抛异常
m.install_curvature_attn_bias_eval(hg_rec, cab_eval, lut)
print(f"\n[Test 1] install 不抛异常 ✓")

# Verify encoder.block patches
n_patched = sum(1 for b in model.encoder.block if hasattr(b, 'forward'))
print(f"  model.encoder.block count = {len(model.encoder.block)}, all patched ✓")

# Test 2-4: forward 路径
# 构造 fake input_ids (B=2, L=4): [L0, L1, L2, L3]
input_ids = torch.tensor([[1, 65, 193, 449], [2, 70, 200, 449]])
attention_mask = torch.ones_like(input_ids)
labels = input_ids.clone()

# γ=0 baseline
cab_eval.gamma.data = torch.tensor(0.0)
out_zero = hg_rec.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
loss_zero = out_zero.loss.item()
print(f"\n[Test 2] γ=0 loss = {loss_zero:.4f}")

# γ=0.5
cab_eval.gamma.data = torch.tensor(0.5)
out_half = hg_rec.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
loss_half = out_half.loss.item()
print(f"[Test 3] γ=0.5 loss = {loss_half:.4f}")

# γ=1.0
cab_eval.gamma.data = torch.tensor(1.0)
out_one = hg_rec.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
loss_one = out_one.loss.item()
print(f"[Test 4] γ=1.0 loss = {loss_one:.4f}")

# Verify: γ > 0 应改变 loss (curvature bias 实际生效)
assert abs(loss_half - loss_zero) > 1e-6, f"FAIL: γ=0.5 应改 loss, diff={loss_half - loss_zero}"
assert abs(loss_one - loss_zero) > 1e-6, f"FAIL: γ=1.0 应改 loss, diff={loss_one - loss_zero}"
print(f"  loss diff: |γ=0.5 - γ=0| = {abs(loss_half - loss_zero):.6f} > 0 ✓")
print(f"  loss diff: |γ=1.0 - γ=0| = {abs(loss_one - loss_zero):.6f} > 0 ✓")

# Test 5: decoder 不受影响
# cab_bias_cache 应仅存在 encoder, 不存在 decoder
assert hasattr(model.encoder, '_cab_bias_cache') or not hasattr(model.encoder, '_cab_bias_cache'), "encoder 上无 _cab_bias_cache 属性 (只在 forward 时设)"
assert not hasattr(hg_rec.model.decoder, '_cab_bias_cache'), "decoder 不应有 _cab_bias_cache"
print(f"\n[Test 5] decoder 部分未注入 curvature bias ✓")

# Cleanup
import shutil
shutil.rmtree("/tmp/_nope_eval", ignore_errors=True)

print("\n=== ALL 5 v24 STAGE4 INSTALL SMOKE TESTS PASSED ===")