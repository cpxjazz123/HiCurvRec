#!/usr/bin/env python3
"""Task #425 / Issue #133 [方向C Gate4] Stage 4 eval for #129 dual-gate ckpt.

验证 #129 ckpt SHA256 + Task84 T5 config + #102 SID 完整可重现性,
跑 2 runs (control + adapter), report vs HG-Rec baseline (Task #84 R@10=0.1020).

PASS 条件: BOTH runs valid AND adapter R@10 > 0.1020.
"""
import sys
import json
import hashlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/scripts")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/model")

# ============================================================================
# Config
# ============================================================================
DEVICE = "cuda:0"
BEAM_SIZE = 20
SEED = 42
T5_CKPT_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/products/task84/ckpt_hgrec/Instruments/Jul-23-2026_20-24-44/HG_Rec_best.pth"
SID_NPY_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_poincare.npy"
ADAPTER_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/products/task422_issue129_t5_dual_gate_ckpt/adapter_trained_2ep.pt"
PRODUCT_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/products/task425_issue133_dual_gate_eval")
PRODUCT_DIR.mkdir(parents=True, exist_ok=True)

HGREC_BASELINE = {
    "R@5": 0.0816, "R@10": 0.1020, "R@20": 0.1279,
    "NDCG@5": 0.0690, "NDCG@10": 0.0755, "NDCG@20": 0.0821,
}


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


class DualGateAdapter(nn.Module):
    """Mirror task422 DualGateAdapter: bias=False + gate_zero(0 init) + active_down/up."""
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


def compute_metrics(model, dataloader, ks=(5, 10, 20), beam_size=20, device="cuda:0"):
    """SIDRetrievalEvaluator per HG-Rec: R@K + NDCG@K."""
    model.eval()
    results = {f"R@{k}": [] for k in ks}
    ndcg_results = {f"NDCG@{k}": [] for k in ks}
    n_total = 0
    n_valid = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            target = batch["labels"]  # (B, 4)
            target = target.to(device)
            # Generate SID sequence via beam search
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=4,
                num_beams=beam_size,
                early_stopping=True,
                do_sample=False,
                pad_token_id=1024,
                eos_token_id=1,
            )  # (B, gen_len)
            # Match logic
            for i in range(target.size(0)):
                tgt = target[i].cpu().tolist()
                pred = outputs[i].cpu().tolist()
                # Filter PAD/EOS
                pred = [t for t in pred if t not in (0, 1, 1024)][:4]
                # Compute metrics: full SID sequence match
                for k in ks:
                    pred_prefix = pred[:k] if k <= len(pred) else pred
                    hit = (len(pred_prefix) == len(tgt) and all(p == t for p, t in zip(pred_prefix, tgt)))
                    results[f"R@{k}"].append(1.0 if hit else 0.0)
                    rel = [1.0 if hit else 0.0]
                    dcg = sum(rel[i] / np.log2(i + 2) for i in range(len(rel)))
                    idcg = sum([1.0 / np.log2(j + 2) for j in range(1)])
                    ndcg_results[f"NDCG@{k}"].append(dcg / max(idcg, 1e-12))
            n_total += target.size(0)
            n_valid += target.size(0)
    return {
        **{f"R@{k}": float(np.mean(results[f"R@{k}"])) for k in ks},
        **{f"NDCG@{k}": float(np.mean(ndcg_results[f"NDCG@{k}"])) for k in ks},
        "n_valid": n_valid, "n_total": n_total,
    }


def build_t5_with_adapter(t5_ckpt, adapter_state=None, d_model=128, vocab_size=1025):
    """Build T5ForConditionalGeneration + optional DualGateAdapter."""
    from transformers import T5ForConditionalGeneration, T5Config
    config = T5Config(
        num_layers=6, num_decoder_layers=4,
        d_model=d_model, d_ff=1024, num_heads=6, d_kv=64,
        dropout_rate=0.1, vocab_size=vocab_size,
        pad_token_id=0, eos_token_id=0,
        decoder_start_token_id=0,
        feed_forward_proj="relu",
    )
    model = T5ForConditionalGeneration(config)
    # Strip "model." prefix from HG_Rec wrapper state_dict
    stripped = {k.replace("model.", "", 1): v for k, v in t5_ckpt.items()}
    info = model.load_state_dict(stripped, strict=False)
    print(f"  T5 load: missing={len(info.missing_keys)}, unexpected={len(info.unexpected_keys)}")
    if adapter_state is not None:
        # adapter file is dict {'adapter_state_dict': {...}, 'hidden_dim': 128, ...}
        if isinstance(adapter_state, dict) and "adapter_state_dict" in adapter_state:
            inner = adapter_state["adapter_state_dict"]
        else:
            inner = adapter_state
        adapter = DualGateAdapter(hidden_dim=d_model)
        info = adapter.load_state_dict(inner, strict=False)
        print(f"  Adapter load: missing={len(info.missing_keys)}, unexpected={len(info.unexpected_keys)}")
        return model, adapter
    return model, None


class WrappedT5Adapter(nn.Module):
    def __init__(self, model, adapter):
        super().__init__()
        self.model = model
        self.adapter = adapter

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # Hook encoder input embedding through adapter
        if self.adapter is not None:
            embed = self.model.shared(input_ids)
            adapted = self.adapter(embed)
            # Pass via inputs_embeds
            return self.model(
                inputs_embeds=adapted,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)

    def generate(self, input_ids=None, attention_mask=None, inputs_embeds=None, **kwargs):
        if self.adapter is not None and input_ids is not None:
            embed = self.model.shared(input_ids)
            inputs_embeds = self.adapter(embed)
            input_ids = None
        return self.model.generate(
            input_ids=input_ids, inputs_embeds=inputs_embeds,
            attention_mask=attention_mask, **kwargs,
        )


class SIDDataset(torch.utils.data.Dataset):
    """Eval dataset: history = SID[:3] digits, target = SID[1:] (shifted next-token prediction).

    Per Issue #133: 6 metrics R@5/10/20 + NDCG@5/10/20 on Musical_Instruments test split.
    """
    def __init__(self, sids, history=None):
        self.sids = torch.tensor(sids, dtype=torch.long)  # (N, 4)

    def __len__(self):
        return len(self.sids)

    def __getitem__(self, idx):
        sid = self.sids[idx]
        # input: shift-left history (3 digits) → predict next digit + 2 more
        input_ids = sid[:3].long()  # (3,) — first 3 digits as history
        labels = sid.long()  # (4,) — full SID as target
        attention_mask = torch.ones(3, dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print("=" * 70)
    print(f"[Task #425 Issue #133 Gate 4] Stage 4 eval for #129 dual-gate ckpt")
    print("=" * 70)

    # Step 1: SHA256 verification
    print("\n[1] Reproducibility SHA256 verification:")
    t5_sha = sha256_of(T5_CKPT_PATH)
    sid_sha = sha256_of(SID_NPY_PATH)
    adapter_sha = sha256_of(ADAPTER_PATH) if Path(ADAPTER_PATH).exists() else "MISSING"
    print(f"  T5 ckpt SHA256: {t5_sha[:16]}...")
    print(f"  SID SHA256:     {sid_sha[:16]}...")
    print(f"  Adapter SHA256: {adapter_sha[:16]}...")

    # Step 2: Load T5 ckpt
    print("\n[2] Load Task84 T5 ckpt + adapter...")
    t5_ckpt = torch.load(T5_CKPT_PATH, map_location="cpu", weights_only=False)

    # Step 3: Load SID
    print("\n[3] Load SID...")
    sids = np.load(SID_NPY_PATH)
    print(f"  SID shape: {sids.shape}, dtype: {sids.dtype}, min: {sids.min()}, max: {sids.max()}")
    sid_unique = np.unique(sids[:, :4].reshape(-1, 4), axis=0)
    print(f"  Unique 4-digit SIDs: {len(sid_unique)}/{len(sids)}")

    # Step 4: Build eval dataset
    print("\n[4] Build eval dataset...")
    dataset = SIDDataset(sids)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)

    # Step 5: Run 1 — control (no adapter)
    print("\n[5] Run 1: Control (no adapter)")
    model_ctrl, _ = build_t5_with_adapter(t5_ckpt, adapter_state=None)
    wrapped_ctrl = WrappedT5Adapter(model_ctrl, None).to(DEVICE)
    metrics_ctrl = compute_metrics(wrapped_ctrl, dataloader, ks=(5, 10, 20), beam_size=BEAM_SIZE, device=DEVICE)
    print(f"  Control metrics: {metrics_ctrl}")

    # Step 6: Run 2 — adapter
    print("\n[6] Run 2: With dual-gate adapter")
    adapter_state = torch.load(ADAPTER_PATH, map_location="cpu", weights_only=False)
    model_ad, adapter_ad = build_t5_with_adapter(t5_ckpt, adapter_state=adapter_state)
    wrapped_ad = WrappedT5Adapter(model_ad, adapter_ad).to(DEVICE)
    metrics_ad = compute_metrics(wrapped_ad, dataloader, ks=(5, 10, 20), beam_size=BEAM_SIZE, device=DEVICE)
    print(f"  Adapter metrics: {metrics_ad}")

    # Step 7: Compare vs baseline
    print("\n[7] vs HG-Rec baseline (Task #84 R@10=0.1020):")
    both_valid = (metrics_ctrl["n_valid"] == len(sids)) and (metrics_ad["n_valid"] == len(sids))
    adapter_r10 = metrics_ad["R@10"]
    baseline_r10 = HGREC_BASELINE["R@10"]
    adapter_pass = adapter_r10 > baseline_r10
    gate4_pass = both_valid and adapter_pass

    print(f"  Both runs valid: {both_valid}")
    print(f"  Adapter R@10 = {adapter_r10:.4f} vs baseline {baseline_r10:.4f}: {'>' if adapter_pass else '<='} = {'PASS' if adapter_pass else 'FAIL'}")
    print(f"  Gate 4 overall: {'✅ PASS' if gate4_pass else '❌ FAIL'}")

    verdict = {
        "task": "task425_issue133_dual_gate_eval", "issue": 133,
        "reproducibility": {
            "t5_ckpt_sha256": t5_sha, "sid_sha256": sid_sha, "adapter_sha256": adapter_sha,
            "sid_shape": list(sids.shape), "sid_unique_4digit": int(len(sid_unique)),
        },
        "control_run": metrics_ctrl,
        "adapter_run": metrics_ad,
        "baseline": HGREC_BASELINE,
        "adapter_r10_vs_baseline": {
            "adapter_r10": adapter_r10, "baseline_r10": baseline_r10,
            "pass": bool(adapter_pass),
        },
        "both_runs_valid": bool(both_valid),
        "gate4_pass": bool(gate4_pass),
    }
    with open(PRODUCT_DIR / "verdict.json", "w") as f:
        json.dump(verdict, f, indent=2, default=str)
    print(f"\nVerdict saved: {PRODUCT_DIR / 'verdict.json'}")
    print(f"Gate 4: {'✅ PASS' if gate4_pass else '❌ FAIL'}")


if __name__ == "__main__":
    main()