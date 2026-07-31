#!/usr/bin/env python3
"""
Task #386 / Issue #93 [方向C Gate3] metadata warm-up + 同模型 shuffle 修复验证 (R22 + R19, GPU 0)

修复 task383 #90 的 shuffle_diff=0 失败:
- 修复 1: metadata_proj zero-equivalent init (weight=0, bias=0; scale=0 时 metadata 完全不生效)
- 修复 2: metadata_scale warm-up schedule (step 0=0 → step 200=1.0 线性增)
- 修复 3: 训练 500 steps (vs #90 100 steps) + 同模型 on/off/shuffle 三态对比 (排除 init 噪声)
"""
import os, sys, time, json, torch, numpy as np
from pathlib import Path

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = "cuda:0"

from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer

# === 0. config ===
TASK_NAME = "task386_issue93_metadata_warmup_shuffle_fix"
SEED = 42
N_ITEMS = 9922
N_LAYERS = 3
N_METADATA_FEATURES = 4
SEQ_LEN = 10
BATCH = 32
TRAIN_STEPS = 500  # R11.5: 5x vs #90 100 steps 让 metadata_proj 充分学习
WARMUP_STEPS = 200  # metadata_scale 0 → 1.0 线性 warm-up
LR = 1e-3
T5_NAME = "t5-small"
D_MODEL = 512

ISSUE_87_METADATA = "/home/wlia0047/ar57/wenyu/GeneRec/products/task380_issue87_sid_metadata_alignment/sid_metadata.json"
OUT_DIR = Path(f"/home/wlia0047/ar57/wenyu/GeneRec/products/{TASK_NAME}")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_PATH = OUT_DIR / "metadata_warmup_ckpt.pt"
TRACE_FILE = OUT_DIR / "trace_per_step.jsonl"

torch.manual_seed(SEED); np.random.seed(SEED)
print("=" * 70)
print(f"Task #386 / Issue #93 [方向C Gate3] metadata warm-up + shuffle 修复")
print(f"Device: {DEVICE}, T5: {T5_NAME}, train_steps: {TRAIN_STEPS}, warmup_steps: {WARMUP_STEPS}")
print("=" * 70)

# === 1. 加载 metadata ===
print(f"\n[加载 metadata]")
with open(ISSUE_87_METADATA, 'r') as f:
    metadata_list = json.load(f)
metadata_array = np.zeros((N_ITEMS, N_LAYERS, N_METADATA_FEATURES), dtype=np.float32)
for item in metadata_list:
    iid = item['item_id']
    for l_meta in item['layer_metadata']:
        l_id = l_meta['layer_id']
        metadata_array[iid, l_id, 0] = l_meta['kappa_l']
        metadata_array[iid, l_id, 1] = l_meta['scale_l']
        metadata_array[iid, l_id, 2] = l_meta['assignment_confidence']
        metadata_array[iid, l_id, 3] = float(l_meta['mask'])
metadata_t = torch.tensor(metadata_array, dtype=torch.float32).to(DEVICE)
print(f"  metadata shape: {metadata_t.shape}")

# === 2. Toy data ===
print(f"\n[生成 toy data]")
n_seqs = 1000
np.random.seed(SEED)
histories = np.random.randint(0, N_ITEMS, size=(n_seqs, SEQ_LEN - 1)).astype(np.int64)
targets = np.random.randint(0, N_ITEMS, size=(n_seqs,)).astype(np.int64)
histories_t = torch.tensor(histories, dtype=torch.long).to(DEVICE)
targets_t = torch.tensor(targets, dtype=torch.long).to(DEVICE)
def get_batch(batch_size):
    idx = np.random.randint(0, n_seqs, size=batch_size)
    return histories_t[idx], targets_t[idx]

# === 3. T5 + metadata (zero-equivalent init 修复) ===
print(f"\n[加载 T5-small]")
tokenizer = T5Tokenizer.from_pretrained(T5_NAME)
t5_model = T5ForConditionalGeneration.from_pretrained(T5_NAME).to(DEVICE)

class T5WithMetadata(torch.nn.Module):
    def __init__(self, base_t5, n_items, n_layers):
        super().__init__()
        self.t5 = base_t5
        self.n_items = n_items
        self.n_layers = n_layers
        # 修复 1: zero-equivalent init — metadata_proj weight=0, bias=0
        # 这样 scale=0 时 metadata 路径输出全 0, 不影响 T5 forward
        self.metadata_proj = torch.nn.Linear(N_LAYERS * N_METADATA_FEATURES, base_t5.config.d_model)
        torch.nn.init.zeros_(self.metadata_proj.weight)
        torch.nn.init.zeros_(self.metadata_proj.bias)
        # metadata_scale 是 scalar, 由 warm-up schedule 控制
        self.metadata_scale = 0.0
        # item embedding
        self.item_embedding = torch.nn.Embedding(n_items + 100, base_t5.config.d_model)
        with torch.no_grad():
            avg = base_t5.shared.weight.data.mean(dim=0)
            self.item_embedding.weight.data = avg.unsqueeze(0).expand(n_items + 100, -1).clone()

    def forward(self, item_ids, metadata_per_item, metadata_scale=None, shuffle_metadata=False):
        """item_ids: (B, seq_len-1), metadata_per_item: (B, seq_len-1, LAYERS*FEATURES)"""
        item_emb = self.item_embedding(item_ids)  # (B, seq_len-1, d_model)
        if shuffle_metadata:
            # 关键修复: shuffle 后 metadata 必须仍然流过 proj, 但顺序破坏
            B, S = item_ids.shape
            perm = torch.randperm(B * S, device=item_ids.device)
            metadata_per_item = metadata_per_item.view(B * S, -1)[perm].view(B, S, -1)
        metadata_emb = self.metadata_proj(metadata_per_item)  # (B, seq_len-1, d_model)
        scale = metadata_scale if metadata_scale is not None else self.metadata_scale
        # 注意: scale=0 时 metadata_emb * 0 = 0, 不影响 T5
        x = item_emb + scale * metadata_emb  # (B, seq_len-1, d_model)
        # 通过 T5 encoder (简化: 直接用 T5.shared + encoder)
        encoder_outputs = self.t5.encoder(inputs_embeds=x)
        hidden = encoder_outputs.last_hidden_state  # (B, seq_len-1, d_model)
        # LM head over item vocab
        logits = self.t5.lm_head(hidden)  # (B, seq_len-1, vocab_size)
        # 简化: 只看第一个位置的 logits 预测下一个 item
        return logits[:, -1, :self.n_items]  # (B, n_items)

model = T5WithMetadata(t5_model, N_ITEMS, N_LAYERS).to(DEVICE)
print(f"  T5-small + metadata_proj: {sum(p.numel() for p in model.parameters()):,} params")
print(f"  metadata_proj init: weight={model.metadata_proj.weight.abs().sum().item():.4f}, bias={model.metadata_proj.bias.abs().sum().item():.4f} (应≈0)")

# === 4. 训练 + warm-up schedule ===
print(f"\n[训练] {TRAIN_STEPS} steps, warmup {WARMUP_STEPS}")
optim = torch.optim.Adam(model.parameters(), lr=LR)
TRACE_FILE.unlink(missing_ok=True)

def get_metadata_for_items(item_ids_batch):
    """item_ids_batch: (B, S-1) → (B, S-1, LAYERS*FEATURES)"""
    B, S = item_ids_batch.shape
    meta = metadata_t[item_ids_batch.view(-1)]  # (B*(S-1), LAYERS, FEATURES)
    meta = meta.view(B, S, -1)  # (B, S-1, LAYERS*FEATURES)
    return meta

for step in range(1, TRAIN_STEPS + 1):
    # warm-up schedule: scale 0 → 1.0 in WARMUP_STEPS
    model.metadata_scale = min(1.0, step / WARMUP_STEPS)

    hist_b, tgt_b = get_batch(BATCH)
    meta_b = get_metadata_for_items(hist_b)
    optim.zero_grad()
    logits = model(hist_b, meta_b)
    loss = torch.nn.functional.cross_entropy(logits, tgt_b)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
    optim.step()

    if step % 50 == 0 or step == 1 or step == TRAIN_STEPS:
        # per-step trace
        with torch.no_grad():
            meta_proj_weight_norm = model.metadata_proj.weight.norm().item()
            meta_proj_bias_norm = model.metadata_proj.bias.norm().item()
        trace = {
            "step": step,
            "loss": float(loss.item()),
            "metadata_scale": float(model.metadata_scale),
            "grad_norm": float(grad_norm),
            "meta_proj_w_norm": meta_proj_weight_norm,
            "meta_proj_b_norm": meta_proj_bias_norm,
        }
        with open(TRACE_FILE, 'a') as f:
            f.write(json.dumps(trace) + '\n')
        print(f"  step {step}/{TRAIN_STEPS} loss={loss.item():.4f} meta_scale={model.metadata_scale:.3f} grad={grad_norm:.4f} meta_w_norm={meta_proj_weight_norm:.4f} meta_b_norm={meta_proj_bias_norm:.4f}")

# R12: ckpt save
if CKPT_PATH.exists():
    CKPT_PATH.unlink()
torch.save(model.state_dict(), CKPT_PATH)
print(f"\n[R12 ckpt] saved to {CKPT_PATH}")

# === 5. 同模型三态对比 (on/off/shuffle) ===
print(f"\n[同模型三态对比 — 关键 FAIL/PASS 判定]")
model.eval()
test_batch_size = 256
np.random.seed(SEED + 1)
hist_test, tgt_test = get_batch(test_batch_size)
meta_test = get_metadata_for_items(hist_test)

with torch.no_grad():
    # (a) metadata ON: scale=1.0
    logits_on = model(hist_test, meta_test, metadata_scale=1.0, shuffle_metadata=False)
    # (b) metadata OFF: scale=0.0
    logits_off = model(hist_test, meta_test, metadata_scale=0.0, shuffle_metadata=False)
    # (c) metadata SHUFFLE: scale=1.0 但 metadata 顺序破坏
    logits_shuf = model(hist_test, meta_test, metadata_scale=1.0, shuffle_metadata=True)
    # (d) sanity: same input on vs on (数值精度)
    logits_on2 = model(hist_test, meta_test, metadata_scale=1.0, shuffle_metadata=False)

on_off_diff = (logits_on - logits_off).abs().mean().item()
on_on_diff = (logits_on - logits_on2).abs().mean().item()
shuf_diff = (logits_on - logits_shuf).abs().mean().item()
shuf_logits_argmax_match = (logits_on.argmax(dim=-1) == logits_shuf.argmax(dim=-1)).float().mean().item()

print(f"  on vs off diff = {on_off_diff:.6f} (排除 init 噪声: 应 > on_on_diff)")
print(f"  on vs on (sanity) diff = {on_on_diff:.6f} (数值精度噪声基准)")
print(f"  on vs shuffle diff = {shuf_diff:.6f} (目标 > on_on_diff, 排除 0)")
print(f"  on vs shuffle argmax match = {shuf_logits_argmax_match*100:.2f}% (越低说明 metadata 影响越大)")

# === 6. evidence ===
evidence = {
    "task_id": "task386",
    "issue": "#93",
    "fix_applied": "zero-equivalent init + warm-up schedule + same-model on/off/shuffle",
    "train_steps": TRAIN_STEPS,
    "warmup_steps": WARMUP_STEPS,
    "on_off_diff": on_off_diff,
    "on_on_diff": on_on_diff,
    "shuffle_diff": shuf_diff,
    "shuffle_argmax_match": shuf_logits_argmax_match,
    "verdict_path": "verdicts/task386_issue93_metadata_warmup_shuffle_fix_v2.md",
}
with open(OUT_DIR / "evidence_package.json", "w") as f:
    json.dump(evidence, f, indent=2, default=str)
print(f"\n[证据] saved to {OUT_DIR / 'evidence_package.json'}")

# === 7. Issue #93 Gate 3 PASS 判定 ===
# PASS 条件:
#   - shuffle_diff > on_on_diff (排除 0)
#   - on_off_diff > on_on_diff (metadata 路径确实影响 forward)
#   - 方向可重复 (on vs shuffle argmax match < 100%)
#   - ckpt saved, no NaN/Inf
checks = [
    ("shuffle_diff > on_on_diff (metadata 路径有信号)", shuf_diff > on_on_diff * 10),  # 至少 10x 数值噪声
    ("on_off_diff > on_on_diff (on vs off 有真实差异)", on_off_diff > on_on_diff * 10),
    ("shuffle argmax match < 100% (shuffle 真影响 top-1)", shuf_logits_argmax_match < 1.0),
    ("shuffle_diff > 0 (排除 #90 shuffle_diff=0)", shuf_diff > 0.0),
    ("on_off_diff > 0 (排除 init 噪声)", on_off_diff > 0.0),
    ("ckpt_saved", CKPT_PATH.exists()),
    ("no_nan_inf", all(np.isfinite(t['loss']) and np.isfinite(t['grad_norm']) for t in [json.loads(l) for l in open(TRACE_FILE)])),
    ("trace_records >= 10", sum(1 for _ in open(TRACE_FILE)) >= 10),
]
n_pass = sum(1 for _, p in checks if p)
print(f"\n[sanity] PASS: {n_pass}/{len(checks)}")
for name, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

print(f"\n[Issue #93 Gate 3 决策]")
print(f"  Gate 3: {'✅ PASS' if n_pass == len(checks) else '❌ FAIL'} ({n_pass}/{len(checks)})")
print(f"  on_off_diff = {on_off_diff:.6f} vs on_on_diff = {on_on_diff:.6f} (metadata 真信号)")
print(f"  shuffle_diff = {shuf_diff:.6f} (排除 #90 FAIL)")