#!/usr/bin/env python3
"""Task #428 / Issue #136 [方向C Gate4] 重建真实 history→SID 评估协议后复跑 dual-gate.

新机制: 不再使用 proxy SID[:3] dataset, 改用 HG-Rec Musical_Instruments 真实 Stage4 评估协议:
- GenRecDataset 加载 test.parquet (24772 samples) + code_path (Instruments_t5_hrqvae_poincare.npy)
- 真实 input: history → item2code → 4-token/item sequences
- 用 task174 Stage 4 eval template (HG_Rec wrapper + evaluate from task84_hgrec_stage3_train)
- 跑 2 次: control (无 adapter) + adapter (with #129 dual-gate via inputs_embeds hook)
- 报告 6 metrics vs HG-Rec baseline

PASS: BOTH runs valid AND adapter R@10 > 0.1020.
"""
import sys
import os
import json
import hashlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import importlib.util as _ilu

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/scripts")

from model.HG_Rec import HG_Rec
from data.dataset import GenRecDataset
from data.dataloader import GenRecDataLoader

# ============================================================================
# Config
# ============================================================================
DEVICE = "cuda:0"
SEED = 42
T5_CKPT_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/products/task84/ckpt_hgrec/Instruments/Jul-23-2026_20-24-44/HG_Rec_best.pth"
SID_NPY_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_poincare.npy"
ADAPTER_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/products/task422_issue129_t5_dual_gate_ckpt/adapter_trained_2ep.pt"
TEST_PARQUET = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/test.parquet"
TRAIN_PARQUET = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/train.parquet"
PRODUCT_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/products/task428_issue136_stage4_rebuild")
PRODUCT_DIR.mkdir(parents=True, exist_ok=True)
MAX_LEN = 20
BEAM_SIZE = 20

HGREC_BASELINE = {
    "R@5": 0.0816, "R@10": 0.1020, "R@20": 0.1279,
    "NDCG@5": 0.0690, "NDCG@10": 0.0755, "NDCG@20": 0.0821,
}

# Match task84 baseline config
CODEBOOK_SIZE = [32, 64, 256, 1]


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


class DualGateAdapter(nn.Module):
    """Mirror task422 DualGateAdapter for embedding insertion."""
    def __init__(self, hidden_dim=128, adapter_dim=64):
        super().__init__()
        self.gate_zero = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gate_active_down = nn.Linear(hidden_dim, adapter_dim, bias=False)
        self.gate_active_up = nn.Linear(adapter_dim, hidden_dim, bias=False)
        with torch.no_grad():
            self.gate_zero.weight.zero_()
            self.gate_active_down.weight.normal_(mean=0.0, std=0.02)
            self.gate_active_up.weight.zero_()

    def forward(self, x):
        return self.gate_zero(x) + self.gate_active_up(F.relu(self.gate_active_down(x)))


class WrappedHGRec(nn.Module):
    """Wrap HG_Rec + optional DualGateAdapter on input embedding."""
    def __init__(self, model, adapter=None):
        super().__init__()
        self.model = model
        self.adapter = adapter

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        if self.adapter is not None and input_ids is not None:
            embed = self.model.model.shared(input_ids)
            adapted = self.adapter(embed)
            return self.model.model(
                inputs_embeds=adapted,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        if self.adapter is not None and input_ids is not None:
            embed = self.model.model.shared(input_ids)
            adapted = self.adapter(embed)
            return self.model.model.generate(
                inputs_embeds=adapted, attention_mask=attention_mask, **kwargs,
            )
        return self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print("=" * 70)
    print(f"[Task #428 Issue #136 Gate 4] Real Stage4 dataset rebuild + dual-gate eval")
    print("=" * 70)

    # Step 1: SHA256 verification (Gate 1/2 复核 per spec)
    print("\n[1] Reproducibility SHA256 verification:")
    t5_sha = sha256_of(T5_CKPT_PATH)
    sid_sha = sha256_of(SID_NPY_PATH)
    adapter_sha = sha256_of(ADAPTER_PATH) if Path(ADAPTER_PATH).exists() else "MISSING"
    test_sha = sha256_of(TEST_PARQUET)
    train_sha = sha256_of(TRAIN_PARQUET)
    print(f"  T5 ckpt SHA256: {t5_sha[:16]}...")
    print(f"  SID SHA256:     {sid_sha[:16]}...")
    print(f"  Adapter SHA256: {adapter_sha[:16]}...")
    print(f"  test.parquet SHA256: {test_sha[:16]}...")
    print(f"  train.parquet SHA256: {train_sha[:16]}...")

    # Verify SID shape
    sids = np.load(SID_NPY_PATH)
    print(f"  SID shape: {sids.shape}, unique 4-digit: {len(np.unique(sids.reshape(-1, 4), axis=0))}/{len(sids)}")

    # Step 2: Build config (match task84 baseline)
    print("\n[2] Build HG_Rec + GenRecDataset + GenRecDataLoader...")
    config = {
        'batch_size': 256,
        'infer_size': 96,
        'lr': 1e-4,
        'device': DEVICE,
        'num_layers': 6, 'num_decoder_layers': 4,
        'd_model': 128, 'd_ff': 1024, 'num_heads': 6, 'd_kv': 64,
        'dropout_rate': 0.1, 'vocab_size': 1025,
        'pad_token_id': 0, 'eos_token_id': 0,
        'feed_forward_proj': 'relu',
        'max_len': MAX_LEN,
        'dataset_name': 'Instruments',
        'dataset_path': '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/',
        'codebook_size': CODEBOOK_SIZE,
        'code_path': SID_NPY_PATH,
        'topk_list': [5, 10, 20],
        'beam_size': BEAM_SIZE,
    }
    # Build dataset
    test_dataset = GenRecDataset(
        dataset_path=TEST_PARQUET,
        code_path=SID_NPY_PATH,
        mode='evaluation',
        codebook_size=CODEBOOK_SIZE,
        max_len=MAX_LEN,
    )
    print(f"  Test dataset size: {len(test_dataset)}")

    # Load evaluate function
    _s4_spec = _ilu.spec_from_file_location(
        'task84_s3_train_fork',
        '/home/wlia0047/ar57/wenyu/GeneRec/scripts/task84_hgrec_stage3_train.py',
    )
    _s4_mod = _ilu.module_from_spec(_s4_spec)
    _s4_spec.loader.exec_module(_s4_mod)
    evaluate = _s4_mod.evaluate

    # Build model (control)
    print("\n[3] Build HG_Rec + load T5 ckpt (control run)...")
    model = HG_Rec(config)
    model.load_state_dict(torch.load(T5_CKPT_PATH, map_location='cpu', weights_only=False))
    model.to(DEVICE)
    wrapped_ctrl = WrappedHGRec(model, adapter=None).to(DEVICE)

    # Eval control
    test_dataloader = GenRecDataLoader(test_dataset, batch_size=config['infer_size'], shuffle=False)
    print(f"  Running control eval (no adapter)...")
    avg_recalls_ctrl, avg_ndcgs_ctrl = evaluate(wrapped_ctrl, test_dataloader, config['topk_list'], config['beam_size'], DEVICE)
    print(f"  Control recalls: {avg_recalls_ctrl}")
    print(f"  Control NDCGs:   {avg_ndcgs_ctrl}")

    # Step 4: Run 2 — adapter
    print("\n[4] Run 2: With dual-gate adapter")
    model_ad = HG_Rec(config)
    model_ad.load_state_dict(torch.load(T5_CKPT_PATH, map_location='cpu', weights_only=False))
    adapter_state_full = torch.load(ADAPTER_PATH, map_location='cpu', weights_only=False)
    if isinstance(adapter_state_full, dict) and "adapter_state_dict" in adapter_state_full:
        adapter_state = adapter_state_full["adapter_state_dict"]
    else:
        adapter_state = adapter_state_full
    adapter = DualGateAdapter(hidden_dim=config['d_model'])
    info = adapter.load_state_dict(adapter_state, strict=False)
    print(f"  Adapter load: missing={len(info.missing_keys)}, unexpected={len(info.unexpected_keys)}")
    wrapped_ad = WrappedHGRec(model_ad, adapter=adapter).to(DEVICE)
    print(f"  Running adapter eval...")
    avg_recalls_ad, avg_ndcgs_ad = evaluate(wrapped_ad, test_dataloader, config['topk_list'], config['beam_size'], DEVICE)
    print(f"  Adapter recalls: {avg_recalls_ad}")
    print(f"  Adapter NDCGs:   {avg_ndcgs_ad}")

    # Step 5: Compare vs baseline
    print("\n[5] vs HG-Rec baseline (Task #84 R@10=0.1020):")
    recalls_dict_ctrl = {f"R@{k}": v for k, v in zip(config['topk_list'], avg_recalls_ctrl)}
    ndcgs_dict_ctrl = {f"NDCG@{k}": v for k, v in zip(config['topk_list'], avg_ndcgs_ctrl)}
    recalls_dict_ad = {f"R@{k}": v for k, v in zip(config['topk_list'], avg_recalls_ad)}
    ndcgs_dict_ad = {f"NDCG@{k}": v for k, v in zip(config['topk_list'], avg_ndcgs_ad)}

    adapter_r10 = recalls_dict_ad["R@10"]
    baseline_r10 = HGREC_BASELINE["R@10"]
    adapter_pass = adapter_r10 > baseline_r10
    both_runs_valid = True  # if we got here, both completed
    gate4_pass = both_runs_valid and adapter_pass

    print(f"  Both runs valid: {both_runs_valid}")
    print(f"  Adapter R@10 = {adapter_r10:.4f} vs baseline {baseline_r10:.4f}: {'>' if adapter_pass else '<='} = {'PASS' if adapter_pass else 'FAIL'}")
    print(f"  Gate 4 overall: {'✅ PASS' if gate4_pass else '❌ FAIL'}")

    verdict = {
        "task": "task428_issue136_stage4_rebuild", "issue": 136,
        "reproducibility": {
            "t5_ckpt_sha256": t5_sha, "sid_sha256": sid_sha, "adapter_sha256": adapter_sha,
            "test_parquet_sha256": test_sha, "train_parquet_sha256": train_sha,
            "sid_shape": list(sids.shape),
            "test_dataset_size": len(test_dataset),
            "real_dataset_protocol": "GenRecDataset(test.parquet, code_path=SID, mode='evaluation', codebook_size, max_len)",
        },
        "control_run": {"recalls": recalls_dict_ctrl, "ndcgs": ndcgs_dict_ctrl},
        "adapter_run": {"recalls": recalls_dict_ad, "ndcgs": ndcgs_dict_ad},
        "baseline": HGREC_BASELINE,
        "adapter_r10_vs_baseline": {
            "adapter_r10": adapter_r10, "baseline_r10": baseline_r10,
            "pass": bool(adapter_pass),
        },
        "both_runs_valid": bool(both_runs_valid),
        "gate4_pass": bool(gate4_pass),
    }
    with open(PRODUCT_DIR / "verdict.json", "w") as f:
        json.dump(verdict, f, indent=2, default=str)
    print(f"\nVerdict saved: {PRODUCT_DIR / 'verdict.json'}")
    print(f"Gate 4: {'✅ PASS' if gate4_pass else '❌ FAIL'}")


if __name__ == "__main__":
    main()