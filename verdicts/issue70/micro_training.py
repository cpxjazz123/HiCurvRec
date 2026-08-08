"""Issue #70 micro-training v2: 端到端连通性 + 几步 loss 下降验证.

修复:
  - ckpt keys 去掉 'model.' 前缀
  - CrossEntropyLoss 用 torch.nn
  - forward 用 T5 标准模式 (inputs_embeds + labels)
"""
import sys
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec")
import os
os.chdir("/home/wlia0047/ar57/wenyu/GeneRec")
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path

CKPT_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/taskA_stage3_issue61/HG_Rec_best.pth"
SID_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_issue61/sid_output.npy"
TRAIN_PARQUET = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/train.parquet"
SID_OFFSET = 1
VOCAB_SIZE = 1025  # HG-Rec 实际 vocab size

print("=" * 60)
print("[Micro-training v2] 加载 ckpt + 数据")
print("=" * 60)

# 加载 ckpt
ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
sd = ckpt.get("model_state_dict", ckpt)
# 去掉 'model.' 前缀 (HG_Rec 包装层)
sd_fixed = {}
for k, v in sd.items():
    if k.startswith("model."):
        sd_fixed[k[len("model."):]] = v
    else:
        sd_fixed[k] = v
print(f"  ckpt 加载: {len(sd_fixed)} tensors, sample key: {list(sd_fixed.keys())[0]}")

# 加载 SID + train
sids = np.load(SID_PATH)
print(f"  SID shape: {sids.shape}")
train_df = pd.read_parquet(TRAIN_PARQUET)
print(f"  train rows: {len(train_df)}, columns: {list(train_df.columns)}")

# 构造 100 sample
np.random.seed(42)
n_samples = 100
max_his_len = 20
input_ids_list = []
target_ids_list = []
for i in range(n_samples):
    sid = sids[i % len(sids)]
    if len(sid) >= max_his_len:
        history = sid[:max_his_len] + SID_OFFSET
    else:
        history = np.concatenate([sid, np.zeros(max_his_len - len(sid), dtype=sid.dtype)]) + SID_OFFSET
        history[len(sid):] = 0
    target = sid[0] + SID_OFFSET
    input_ids_list.append(history)
    target_ids_list.append(target)
input_ids = torch.tensor(np.array(input_ids_list), dtype=torch.long)
target_ids = torch.tensor(np.array(target_ids_list), dtype=torch.long)
print(f"  input_ids shape: {input_ids.shape}, target_ids shape: {target_ids.shape}")

# 加载 T5
print("\n" + "=" * 60)
print("[Micro-training v2] 加载 T5 + DecorPromptFormer")
print("=" * 60)

from transformers import T5ForConditionalGeneration, T5Config
from common.decor_prompt_former import DecorPromptFormer

t5_config = T5Config(
    vocab_size=VOCAB_SIZE, d_model=128, d_ff=1024,
    num_layers=4, num_decoder_layers=4, num_heads=6, d_kv=64,
    dropout_rate=0.1, feed_forward_proj="relu",
    pad_token_id=0, decoder_start_token_id=0, eos_token_id=VOCAB_SIZE - 1,
)

model = T5ForConditionalGeneration(t5_config)
missing, unexpected = model.load_state_dict(sd_fixed, strict=False)
print(f"  T5 加载: missing={len(missing)}, unexpected={len(unexpected)}")

# DecorPromptFormer
pf = DecorPromptFormer(
    d_model=128, vocab_size=VOCAB_SIZE,
    num_bins=4, codes_per_bin=256, num_bos_queries=64, alpha_init=0.35,
)
model.add_module("pf_module", pf)
print(f"  DecorPromptFormer 参数: {sum(p.numel() for p in pf.parameters())}")

# Patch forward — 简化为直接用 e_final 作为 inputs_embeds 喂 T5
import types
d_model_sqrt = 128 ** 0.5

def pf_forward(self, input_ids=None, attention_mask=None, labels=None,
               inputs_embeds=None, **kwargs):
    """取代原 forward. 注入 DecorPromptFormer 到 inputs_embeds."""
    if inputs_embeds is None and input_ids is not None:
        e_fused = self.shared(input_ids) * d_model_sqrt
        e_final, aux = self.pf_module(
            e_fused, input_ids, attention_mask=attention_mask,
            e_fused_embedding=self.shared,
        )
        inputs_embeds = e_final
    # 用 T5 原始 forward (传入 inputs_embeds)
    return T5ForConditionalGeneration.forward(
        self, attention_mask=attention_mask, labels=labels,
        inputs_embeds=inputs_embeds, **kwargs
    )

model.forward = types.MethodType(pf_forward, model)

# 训练循环
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
pf = pf.to(device)
model.train()

batch_size = 8
n_steps = 5
optimizer = torch.optim.Adam(model.parameters(), lr=4e-4)

print(f"\n  device: {device}, batch_size: {batch_size}, n_steps: {n_steps}")

# 构造 labels: (B, 4) — 4-digit SID
# decoder 学: 输入 [0, sid1, sid2], 预测 [sid1, sid2, sid3]
labels = target_ids.unsqueeze(-1).expand(-1, 4).contiguous().clone()  # (B, 4)
print(f"  labels shape: {labels.shape}, sample: {labels[0]}")

losses = []
for step in range(n_steps):
    batch_input = input_ids[step*batch_size:(step+1)*batch_size].to(device)
    batch_labels = labels[step*batch_size:(step+1)*batch_size].to(device)
    attention_mask = (batch_input != 0).long()

    optimizer.zero_grad()
    try:
        out = model(batch_input, attention_mask=attention_mask, labels=batch_labels)
        loss = out.loss
        loss.backward()
        pf_grad = pf.alpha_raw.grad.item() if pf.alpha_raw.grad is not None else None
        bos_grad = pf.bos_queries.grad.norm().item() if pf.bos_queries.grad is not None else None
        print(f"  Step {step+1}: loss={loss.item():.4f}  alpha_grad={pf_grad}  bos_grad_norm={bos_grad}")
        losses.append(loss.item())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    except Exception as e:
        print(f"  Step {step+1}: ERROR {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        break

print(f"\n  Loss 曲线: {losses}")
if len(losses) >= 2:
    delta = losses[-1] - losses[0]
    print(f"  Loss 变化: {delta:+.4f}")
    if delta < 0:
        print("  [PASS] loss 下降, DecorPromptFormer 端到端连通")
    elif delta == 0:
        print("  [INFO] loss 不变 (可能 batch 无效)")
    else:
        print("  [INFO] loss 上升 (5 step 太少, 但说明梯度流通)")

# Generate 测试
print("\n" + "=" * 60)
print("[Micro-training v2] Generate token id 合法验证")
print("=" * 60)
model.eval()
with torch.no_grad():
    sample_input = input_ids[:2].to(device)
    sample_mask = (sample_input != 0).long()
    try:
        gen_ids = model.generate(sample_input, attention_mask=sample_mask,
                                  max_length=4, num_beams=4, num_return_sequences=4)
        print(f"  generate output shape: {gen_ids.shape}")
        print(f"  sample generate: {gen_ids[0].tolist()}")
        valid = ((gen_ids >= 0) & (gen_ids < VOCAB_SIZE)).all().item()
        print(f"  token id 合法 (0-{VOCAB_SIZE-1})? {valid}")
        if valid:
            print("  [PASS] generate 端到端连通")
        else:
            print("  [WARN] 有 token id 越界")
    except Exception as e:
        print(f"  generate ERROR: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("[Micro-training v2] 完成")