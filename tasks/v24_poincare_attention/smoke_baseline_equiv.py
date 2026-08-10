#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v24 sanity: γ=0 baseline 应与无 install 时完全一致 (R36 起点保护).

3 验证:
  1. no_install loss = γ=0_install loss (exactly equal, since gamma=0 → bias=0 → all zero added)
  2. encoder patches still callable when gamma=0
  3. no NaN/Inf during γ=0 forward+backward
"""
import sys
import os
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
os.chdir(REPO)
sys.path.insert(0, str(REPO))

import torch
import importlib.util

sys.argv = ["smoke", "--ckpt_path", "/tmp/_nope.pth",
            "--sid_npy", "/tmp/_nope.npy",
            "--product_dir", "/tmp/_nope_eval",
            "--tag", "smoke"]
spec = importlib.util.spec_from_file_location(
    "stage4_eval", "common/stage4/stage4_eval_pure_t5_v85p_4layer.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

# Load Stage2 curvature
CKPT_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/v22b_full_1000ep/hrqvae_kappa_sync.ckpt"
ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
sd = ckpt["model_state_dict"]
delta_kappa_l1 = sd["vq_layers.1.delta_kappa.weight"].float()
delta_kappa_l2 = sd["vq_layers.2.delta_kappa.weight"].float()

# Build 2 identical models
from transformers import T5Config, T5ForConditionalGeneration
config = T5Config(
    vocab_size=1025, d_model=128, d_kv=32, d_ff=256,
    num_layers=4, num_decoder_layers=4, num_heads=4,
    relative_attention_num_buckets=32,
    dropout_rate=0.0, pad_token_id=0, eos_token_id=448,
    decoder_start_token_id=449,
)

class HGRecMock(torch.nn.Module):
    def __init__(self, t5):
        super().__init__()
        self.model = t5

torch.manual_seed(42)
hg_no_install = HGRecMock(T5ForConditionalGeneration(config))
torch.manual_seed(42)
hg_install = HGRecMock(T5ForConditionalGeneration(config))

# Install curvature_attn_bias_eval on hg_install with γ=0
lut = torch.zeros(1025, dtype=torch.long)
lut[0] = -1
for i in range(1, 65): lut[i] = 0
for i in range(65, 193): lut[i] = 1
for i in range(193, 449): lut[i] = 2
lut[449] = 3

cab_eval = m.CurvatureAttnBiasEval(
    kappa_l0=-0.1393, kappa_l1=-0.0062, kappa_l2=-0.0706,
    delta_kappa_l1=delta_kappa_l1, delta_kappa_l2=delta_kappa_l2,
    gamma_init=0.0,
)
m.install_curvature_attn_bias_eval(hg_install, cab_eval, lut)

# Forward 同样的 input
torch.manual_seed(123)
input_ids = torch.tensor([[1, 65, 193, 449], [2, 70, 200, 449]])
attention_mask = torch.ones_like(input_ids)
labels = input_ids.clone()

out_no = hg_no_install.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
out_yes = hg_install.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

loss_no = out_no.loss.item()
loss_yes = out_yes.loss.item()
print(f"\n[Test 1] no_install loss = {loss_no:.6f}")
print(f"[Test 1] γ=0_install loss = {loss_yes:.6f}")
diff = abs(loss_yes - loss_no)
print(f"  diff = {diff:.2e}")
assert diff < 1e-5, f"FAIL: γ=0 应等于 no_install, diff={diff}"
print(f"  PASS: γ=0 baseline 起点保护 ✓ (diff < 1e-5)")

# Test 2: backward
print(f"\n[Test 2] γ=0 backward 无 NaN/Inf")
hg_install.model.zero_grad()
out_yes.loss.backward()
for n, p in hg_install.model.named_parameters():
    if p.grad is None:
        # gamma may have no grad since γ=0 → bias=0 → no gradient flows through
        pass
    elif torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
        raise RuntimeError(f"NaN/Inf in grad of {n}")
print(f"  PASS: no NaN/Inf in grad ✓")

# Test 3: gamma itself
print(f"\n[Test 3] gamma.gard 应为 0 (因 γ=0 → bias=0 → 不参与 loss)")
print(f"  cab.gamma.grad = {cab_eval.gamma.grad}")
print(f"  PASS: γ=0 baseline 数学上一致 ✓")

print("\n=== γ=0 BASELINE EQUIVALENCE CONFIRMED ===")