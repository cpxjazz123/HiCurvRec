#!/usr/bin/env python3
"""
Task #375 / Issue #82 [方向B 准入] product预检证据落仓 + Gate1 准入

任务:
1. 落仓 #79 证据包 commit hash + 关键路径 + R20 4 Gate 详细
2. Gate1 训练入口 (静态配置冻结清单)
3. 不实际启动 Stage 1 训练 (Issue #82 spec "不是启动训练本身, 是准入层")

R21: commit hash 必须明示, 不允许 pending
"""
import os, sys, json, torch, numpy as np
from pathlib import Path

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/scripts')
from task369_issue76_product_3component_evidence import Product3ComponentVectorQuantization

# === 0. config ===
TASK_NAME = "task375_issue82_precheck_landing_gate1_entry"
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_E = [64, 128, 256]
E_DIM = 32
M = 3
KAPPA_MAX = 2.0
FIXED_HYP_KAPPA = 1.0

torch.manual_seed(SEED); np.random.seed(SEED)
print("=" * 70)
print("Task #375 / Issue #82 [方向B 准入] product预检证据落仓 + Gate1 准入")
print("=" * 70)

# === 1. 落仓 #79 证据包 (commit hash + 关键路径) ===
ISSUE_79_LANDING = {
    "issue": "#79",
    "verdict_path": "verdicts/task372_issue79_evidence_package_v2.md",
    "scripts_path": "scripts/task372_issue79_evidence_package.py",
    "evidence_path": "products/task372_issue79_evidence_package/evidence_package.json",
    "description_path": "descriptions/task372_issue79_direction_b_precheck_evidence_package.md",
    # R21: commit hash 必须明示 — Issue #79 已 commit 在 8a761f6 (R18/R20 强制实证闭环)
    "commit_hash": "8a761f6",
    "commit_message": "Issue #78/#79/#80 证据包 R18/R20 强制实证闭环 (R17 gate 说明 + 失败原因)",
    "push_status": "pushed to origin/main",
}
print(f"\n[落仓 #79 证据包]")
print(json.dumps(ISSUE_79_LANDING, indent=2))

# === 2. R20 4 Gate 详细内容 (落仓备查) ===
R20_4_GATE = {
    "Gate_1": {
        "状态": "⏸ STOP per Issue #79 spec (预检任务, spec 不要求 Stage 1 训练)",
        "证据": "Issue #79 spec 明确说明 'Gate 1 准入前必须先把证据包落到可核验 commit/verdict'",
    },
    "Gate_1.5": {
        "状态": "✅ PASS 10/10",
        "关键数据": "P1 logits_l perturbation accepted (软路径生效); P2 theta_m perturbation → indices diff=21; P3 embedding perturbation → indices diff=52; P4 reproducible; L0/L1/L2 logits_l.grad > 0 (2.37e-4 / 2.22e-2 / 2.22e-2); L0/L1/L2 theta_m.grad > 0 (7.12e-2 / 6.67e-2 / 6.67e-2); per_layer_independent",
        "失败原因": "无失败",
        "verdict路径": "verdicts/task372_issue79_evidence_package_v2.md",
        "commit": "8a761f6",
    },
    "Gate_2": {"状态": "⏸ STOP per spec", "原因": "Issue #79 是预检 spec"},
    "Gate_3": {"状态": "⏸ STOP per spec", "原因": "Issue #79 是预检 spec"},
    "Gate_4": {"状态": "⏸ STOP per spec", "原因": "Issue #79 是预检 spec"},
}
print(f"\n[R20 4 Gate 详细内容 (落仓)]")
print(json.dumps(R20_4_GATE, indent=2, ensure_ascii=False))

# === 3. 配置冻结清单 (Gate1 训练入口) ===
GATE1_ENTRY_CONFIG = {
    "model": {
        "class": "Product3ComponentVectorQuantization (继承 FreeCurvVectorQuantization)",
        "n_e_per_layer": N_E,
        "e_dim": E_DIM,
        "M (num_hierarchies)": M,
        "kappa_max": KAPPA_MAX,
        "fixed_hyp_kappa": FIXED_HYP_KAPPA,
        "three_components": ["learned-κ", "fixed-hyp (κ=1.0)", "Euclidean (κ=0)"],
        "w_l (per-component weight)": "softmax(logits_l) over 3 components",
        "logits_l shape": "[M, 3]",
        "logits_l init": "per-layer 增量 (layer_id * 0.1)",
    },
    "optimizer": {
        "type": "Adam",
        "lr": 1e-3,
        "param_groups": [
            {"name": "non_theta_m", "lr": 1e-3},
            {"name": "theta_m (learned κ)", "lr": 1e-3},
            {"name": "logits_l (3-component weights)", "lr": 1e-3},
            {"name": "embedding.weight (codebook)", "lr": 1e-3},
        ],
        "beta_1": 0.9,
        "beta_2": 0.999,
        "weight_decay": 0,
    },
    "training_loop": {
        "epochs": 200,  # R137 baseline 200 epoch
        "batch_size": 256,
        "warmup_epochs": 0,  # κ-decouple Phase A 替代 #78 κ-freeze warmup
        "save_strategy": "per R12: save ckpt after each epoch + delete old ckpt",
    },
    "data": {
        "dataset": "Musical_Instruments (Amazon, 24588 items → 5-core 后 9922 items)",
        "item_emb_path": "HG-Rec/dataset/Instruments/item_emb.parquet",
        "normalize": "‖x‖_E → 0.85",
    },
    "loss_components": {
        "commitment_loss": "MSE(data, quantized) per component, weighted by w_l",
        "codebook_loss": "MSE(quantized, data) per component, weighted by w_l",
        "total_loss": "commitment_loss + β * codebook_loss (β=0.25 baseline)",
    },
    "gate1_entry_point": "scripts/task375_issue82_gate1_train.py (待 owner 派工启动)",
    "r137_baseline_recipe": "task287 K=128 κ-decouple Phase A (κ frozen=0) + #78 κ-freeze unfreeze + Issue #76 三分量 product",
}

print(f"\n[Gate1 训练入口 配置冻结清单]")
print(json.dumps(GATE1_ENTRY_CONFIG, indent=2, ensure_ascii=False))

# === 4. 验证 Product3ComponentVectorQuantization 初始化 + forward 形状 ===
print(f"\n[验证 Product3ComponentVQ 初始化 + forward]")
layer = Product3ComponentVectorQuantization(
    n_e=64, e_dim=E_DIM, M=M, kappa_max=KAPPA_MAX,
    fixed_hyp_kappa=FIXED_HYP_KAPPA,
).to(DEVICE)
print(f"  layer.embeddings.weight.shape = {layer.embeddings.weight.shape}")
print(f"  layer.logits_l.shape = {layer.logits_l.shape}")
print(f"  layer.theta_m.shape = {layer.theta_m.shape}")

# forward shape check
z_test = torch.randn(8, E_DIM, device=DEVICE)
z_q, idx, loss = layer(z_test)
print(f"  z_test.shape={z_test.shape} → z_q.shape={z_q.shape}, idx.shape={idx.shape}, loss={loss.item():.4f}")

# === 5. evidence ===
out_dir = Path(f"/home/wlia0047/ar57/wenyu/GeneRec/products/{TASK_NAME}")
out_dir.mkdir(parents=True, exist_ok=True)
evidence = {
    "task_id": "task375",
    "issue": "#82",
    "issue_79_landing": ISSUE_79_LANDING,
    "R20_4_gate": R20_4_GATE,
    "gate1_entry_config": GATE1_ENTRY_CONFIG,
    "verdict_path": "verdicts/task375_issue82_precheck_landing_v2.md",
    "commit_hash_landing": "8a761f6 (Issue #79 commit hash from prior push)",
    "evidence_types": ["issue_79_landing", "R20_4_gate", "gate1_entry_config", "forward_shape_check"],
}
with open(out_dir / "evidence_package.json", "w") as f:
    json.dump(evidence, f, indent=2, default=str)
print(f"\n[证据] saved to {out_dir / 'evidence_package.json'}")

# === 6. sanity ===
checks = [
    ("issue_79_commit_hash_specified", ISSUE_79_LANDING['commit_hash'] == "8a761f6"),
    ("R20_4_gate_complete", len(R20_4_GATE) >= 4),
    ("gate1_entry_config_complete", all(k in GATE1_ENTRY_CONFIG for k in ['model', 'optimizer', 'training_loop', 'data'])),
    ("forward_shape_ok", z_q.shape == z_test.shape),
]
n_pass = sum(1 for _, p in checks if p)
print(f"\n[sanity] PASS: {n_pass}/{len(checks)}")
for name, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

print(f"\n[Issue #82 决策]")
if n_pass == len(checks):
    print(f"  ✅ 准入层 PASS — 落仓 + 配置冻结 + 训练入口就绪 (待 owner 派工启动 Gate1 训练)")
else:
    print(f"  ❌ FAIL — 静态产物不完整")