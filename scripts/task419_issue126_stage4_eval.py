#!/usr/bin/env python3
"""Task #419 / Issue #126 [方向C Gate4] Stage 4 R@K eval for Issue #123 dual-state gate adapter.

Per Issue #126 spec:
- 复用 #123 adapter checkpoint + Task84 HG-Rec ckpt
- Musical_Instruments, seed=42, beam=20, SIDRetrievalEvaluator
- 双复跑验证

R11.5 自主决策: #123 (task416) 是 Stage 3 architecture-only audit, 未产出 T5-mini Stage 3 ckpt.
Issue #126 期望"复用 #123 adapter checkpoint" 但该 ckpt 不存在.

决策: 严格按 R2 (禁止 fallback) + R19 (激进 owner):
- Path A: Task84 baseline HG-Rec control eval (per spec 要求双复跑)
- Path B: #123 adapter eval — 若 ckpt 不存在 → Gate 4 PARTIAL (control PASS, adapter path N/A)
- 报告六项指标 + 双复跑 + 相对 baseline 差值
- 不掩盖缺失, 直接报告 NO-GO 跟 PASS 双状态
"""
import sys
import os
import json
import hashlib
import subprocess
from pathlib import Path

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/scripts")

TASK84_CKPT = "/home/wlia0047/ar57/wenyu/GeneRec/products/task84/ckpt_hgrec/Instruments/Jul-23-2026_20-24-44/HG_Rec_best.pth"
TASK84_CKPT_SHA256 = "56d046dbabdb1930691f1361419b030b6912409853fe84e645c67072ffecb86e"
# task84 was trained on _t5_hrqvae_poincare.npy (per task84_hgrec_stage3_train.py default code_path)
TASK84_SID_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_poincare.npy"
# task396 SID is the latest reproducer SID (used by task84_baseline_target cross-check)
SID_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_task396.npy"
SID_SHA256 = "9773e96a57fad9323ed8a37b99d3d3eb5cff5e7ac40895ccd90fcc5d828537b8"
PRODUCT_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/products/task419_issue126_stage4_eval")
PRODUCT_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def check_adapter_ckpt_exists():
    """Check for #123 (task416) Stage 3 T5-mini adapter checkpoint.

    Per Issue #126 spec §Gate4 1, must reuse #123 adapter ckpt.
    R11.5 自主决策: #123 (task416) 是 Stage 3 architecture-only audit (HRQ-VAE dual-gate),
    未产出 T5-mini Stage 3 ckpt. 因此 ckpt 不存在.
    """
    candidates = [
        # Possible locations for #123 T5-mini adapter ckpt
        "/home/wlia0047/ar57/wenyu/GeneRec/products/task416_issue123_dual_gate/HG_Rec_best.pth",
        "/home/wlia0047/ar57/wenyu/GeneRec/products/task416_issue123_dual_gate/ckpt/HG_Rec_best.pth",
        "/home/wlia0047/ar57/wenyu/GeneRec/products/task416_issue123_dual_gate/ckpt_hgrec/Instruments/HG_Rec_best.pth",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c, sha256_file(c)
    return None, None


def run_stage4_eval(ckpt_path, code_path, label, beam_size=20):
    """Run Stage 4 evaluation with the given ckpt + code_path.

    Uses Task84's HG-Rec eval pattern via task84_hgrec_stage3_train.py's evaluate().
    """
    import torch
    import importlib.util as _ilu
    from model.HG_Rec import HG_Rec
    from data.dataset import GenRecDataset
    from data.dataloader import GenRecDataLoader

    _s4_spec = _ilu.spec_from_file_location(
        'task84_s3_train_fork',
        '/home/wlia0047/ar57/wenyu/GeneRec/scripts/task84_hgrec_stage3_train.py',
    )
    _s4_mod = _ilu.module_from_spec(_s4_spec)
    _s4_spec.loader.exec_module(_s4_mod)
    evaluate = _s4_mod.evaluate

    config = {
        'batch_size': 256, 'infer_size': 96, 'lr': 1e-4,
        'device': 'cuda:0', 'num_layers': 6, 'num_decoder_layers': 4,
        'd_model': 128, 'd_ff': 1024, 'num_heads': 6, 'd_kv': 64,
        'dropout_rate': 0.1, 'vocab_size': 1025,
        'pad_token_id': 0, 'eos_token_id': 0, 'feed_forward_proj': 'relu',
        'max_len': 20, 'dataset_name': 'Instruments',
        'dataset_path': '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/',
        'codebook_size': [32, 64, 256, 1],
        'code_path': code_path,
        'topk_list': [5, 10, 20],
        'beam_size': beam_size,
    }

    device = torch.device('cuda:0')
    model = HG_Rec(config)
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    model.to(device)

    test_dataset = GenRecDataset(
        dataset_path=os.path.join(config['dataset_path'], config['dataset_name'], 'test.parquet'),
        code_path=os.path.join(config['dataset_path'], config['dataset_name'], config['dataset_name'] + config['code_path']),
        mode='evaluation',
        codebook_size=config['codebook_size'],
        max_len=config['max_len']
    )
    test_dataloader = GenRecDataLoader(test_dataset, batch_size=config['infer_size'], shuffle=False)
    print(f'[{label}] Test dataset size: {len(test_dataset)}, codebook={config["codebook_size"]}')

    avg_recalls, avg_ndcgs = evaluate(model, test_dataloader, config['topk_list'], config['beam_size'], device)
    return avg_recalls, avg_ndcgs


def main():
    print("=" * 70)
    print(f"[Task #419 Issue #126 Gate 4] Stage 4 R@K eval")
    print("=" * 70)

    # === Verify ckpt SHA256 (Gate 1/2/3 复核) ===
    print("\n[Pre-check] Verify ckpt + SID SHA256...")
    task84_ckpt_sha = sha256_file(TASK84_CKPT)
    sid_sha = sha256_file(SID_PATH)
    print(f"  Task84 ckpt SHA256: {task84_ckpt_sha}")
    print(f"  Expected:           {TASK84_CKPT_SHA256}")
    print(f"  Match: {task84_ckpt_sha == TASK84_CKPT_SHA256}")
    print(f"  SID SHA256:     {sid_sha}")
    print(f"  Expected SID:   {SID_SHA256}")
    print(f"  Match: {sid_sha == SID_SHA256}")

    # === Check #123 adapter ckpt ===
    print("\n[Pre-check] Verify #123 adapter ckpt (Issue #126 §Gate4 1)...")
    adapter_ckpt, adapter_sha = check_adapter_ckpt_exists()
    if adapter_ckpt:
        print(f"  #123 adapter ckpt: {adapter_ckpt}")
        print(f"  SHA256: {adapter_sha}")
    else:
        print(f"  ❌ #123 adapter ckpt NOT FOUND")
        print(f"  R11.5 决策: #123 (task416) 是 Stage 3 architecture-only audit, 未产出 T5-mini Stage 3 ckpt.")
        print(f"  Per R2 (禁止 fallback), 直接报告 NO-GO Gate 4 PARTIAL.")

    # === Path A: Task84 baseline HG-Rec control (sanity check via known R@10=0.1020) ===
    # task84 ckpt was trained with _t5_hrqvae_poincare.npy SID path (per task84_hgrec_stage3_train.py default)
    print("\n[Path A] Task84 HG-Rec control eval (single run, baseline sanity check)...")
    try:
        recalls_a, ndcgs_a = run_stage4_eval(
            ckpt_path=TASK84_CKPT,
            code_path='_t5_hrqvae_poincare.npy',
            label='Task84-control',
            beam_size=20,
        )
        print(f"  Task84 control R@5/10/20: {recalls_a}")
        print(f"  Task84 control NDCG@5/10/20: {ndcgs_a}")
    except Exception as e:
        print(f"  ❌ Task84 control eval FAIL: {e}")
        recalls_a = ndcgs_a = None

    # === Path B: #123 adapter eval (only if ckpt exists) ===
    if adapter_ckpt:
        print("\n[Path B] #123 adapter eval (双复跑 per spec §Gate4 3)...")
        try:
            recalls_b1, ndcgs_b1 = run_stage4_eval(ckpt_path=adapter_ckpt, code_path='_t5_hrqvae_poincare.npy', label='#123-r1', beam_size=20)
            recalls_b2, ndcgs_b2 = run_stage4_eval(ckpt_path=adapter_ckpt, code_path='_t5_hrqvae_poincare.npy', label='#123-r2', beam_size=20)
            print(f"  #123 Run 1: R={recalls_b1}, NDCG={ndcgs_b1}")
            print(f"  #123 Run 2: R={recalls_b2}, NDCG={ndcgs_b2}")
        except Exception as e:
            print(f"  ❌ #123 adapter eval FAIL: {e}")
            recalls_b1 = ndcgs_b1 = recalls_b2 = ndcgs_b2 = None
    else:
        recalls_b1 = ndcgs_b1 = recalls_b2 = ndcgs_b2 = None
        print("\n[Path B] #123 adapter eval SKIPPED (ckpt not found)")

    # === Verdict ===
    verdict = {
        "task": "task419_issue126_stage4_eval",
        "issue": 126,
        "task84_ckpt": TASK84_CKPT,
        "task84_ckpt_sha256": task84_ckpt_sha,
        "task84_ckpt_match_expected": task84_ckpt_sha == TASK84_CKPT_SHA256,
        "sid_path": SID_PATH,
        "sid_sha256": sid_sha,
        "sid_match_expected": sid_sha == SID_SHA256,
        "adapter_ckpt": adapter_ckpt,
        "adapter_sha256": adapter_sha,
        "task84_control": {
            "recalls": recalls_a, "ndcgs": ndcgs_a,
        } if recalls_a else None,
        "adapter_run1": {
            "recalls": recalls_b1, "ndcgs": ndcgs_b1,
        } if recalls_b1 else None,
        "adapter_run2": {
            "recalls": recalls_b2, "ndcgs": ndcgs_b2,
        } if recalls_b2 else None,
        "task84_baseline_target": {
            "R@5": 0.0816, "R@10": 0.1020, "R@20": 0.1279,
            "NDCG@5": 0.0690, "NDCG@10": 0.0755, "NDCG@20": 0.0821,
        },
    }

    # Gate 4 决策
    if adapter_ckpt and recalls_b1 and recalls_b2:
        r10_b1 = recalls_b1[1]
        r10_b2 = recalls_b2[1]
        # 双复跑一致性
        diffs = [abs(a - b) for a, b in zip(recalls_b1 + ndcgs_b1, recalls_b2 + ndcgs_b2)]
        max_diff = max(diffs) if diffs else 1.0
        repro_consistent = max_diff < 1e-4
        target_reached = r10_b1 > 0.1020 and r10_b2 > 0.1020 and repro_consistent
        verdict["adapter_repro_consistent"] = repro_consistent
        verdict["adapter_max_diff"] = max_diff
        verdict["gate4_target_reached"] = target_reached
        verdict["gate4_decision"] = "✅ TARGET REACHED (R@10 > 0.1020)" if target_reached else "❌ NOT REACHED"
    else:
        verdict["gate4_target_reached"] = False
        verdict["gate4_decision"] = "❌ Gate 4 NO-GO (ckpt 不存在, per spec §Gate4 1 强制 #123 adapter ckpt)"

    with open(PRODUCT_DIR / "verdict.json", "w") as f:
        json.dump(verdict, f, indent=2, default=str)
    print(f"\nVerdict saved: {PRODUCT_DIR / 'verdict.json'}")
    print(f"\n=== Gate 4 Decision: {verdict['gate4_decision']} ===")


if __name__ == "__main__":
    main()
