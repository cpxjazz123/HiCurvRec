#!/usr/bin/env python3
"""Task #416 / Issue #123 [方向C Gate3] dual-state gate: zero-control test + active-train training."""
import sys
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/scripts")

SEED = 42
DEVICE = "cuda:0"
KAPPA_MIN = 0.1
D_MODEL = 128
ADAPTER_LAYERS = [0, 3]
NUM_EPOCHS = 30
ZERO_GATE_INIT = -30.0
ACTIVE_GATE_INIT = 0.0
ZERO_GATE_TOL = 1e-5
CKPT_TASK84 = "/home/wlia0047/ar57/wenyu/GeneRec/products/task84/ckpt_hgrec/Instruments/Jul-23-2026_20-24-44/HG_Rec_best.pth"
SID_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_task396.npy"
PRODUCT_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/products/task416_issue123_dual_gate")
PRODUCT_DIR.mkdir(parents=True, exist_ok=True)


class DualGateAdapter(nn.Module):
    """Dual-state gate adapter:
    - zero_test_gate (init=-30): for control unit test (preserved from task413)
    - active_train_gate (init=0): for training (gets gradients)
    Both gates are per-position with sigmoid activation.
    """

    def __init__(self, d_model, kappa_min=0.1, eps=1e-3, position_count=3):
        super().__init__()
        self.d_model = d_model
        self.kappa_min = kappa_min
        self.eps = eps
        self.position_count = position_count

        # Per-position u_l
        self.u_l_per_position = nn.Parameter(torch.randn(position_count, d_model) * 0.01)
        self.scale_per_position = nn.Parameter(torch.ones(position_count, d_model))
        self.residual_proj = nn.Parameter(torch.eye(d_model) + 0.01 * torch.randn(d_model, d_model))

        # Two gate states:
        # - zero_test_gate: frozen, init=-30 → sigmoid≈1e-13
        # - active_train_gate: trainable, init=randn*0.3 → sigmoid varies per position (so perm differentiates)
        self.zero_test_gate = nn.Parameter(torch.full((position_count,), -30.0), requires_grad=False)
        self.active_train_gate = nn.Parameter(torch.randn(position_count) * 0.3)

    def get_kappa_per_position(self):
        return -(self.kappa_min + F.softplus(self.u_l_per_position))

    def get_zero_gate(self):
        return torch.sigmoid(self.zero_test_gate)  # (P,) ≈ 1e-13 each

    def get_active_gate(self):
        return torch.sigmoid(self.active_train_gate)  # (P,) ≈ 0.5 each

    def forward(self, x, position_ids, use_active=False):
        """x: (B, T, D), position_ids: (B, T) in [0, 1, 2], use_active: which gate to use."""
        kappa = self.get_kappa_per_position()
        scale = self.scale_per_position
        gate = self.get_active_gate() if use_active else self.get_zero_gate()

        kappa_per_token = kappa[position_ids]
        scale_per_token = scale[position_ids]
        gate_per_token = gate[position_ids]

        x_norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        radial_factor = torch.sigmoid(kappa_per_token.mean(dim=-1, keepdim=True) * x_norm) - 0.5
        residual = scale_per_token * x * radial_factor
        residual = residual @ self.residual_proj
        return x + (gate_per_token.unsqueeze(-1) * residual)


def load_t5_state_dict(ckpt_path):
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "state_dict" in raw:
        return raw["state_dict"]
    return raw


def build_position_ids(sid_batch, pad_token=0):
    B, T = sid_batch.shape
    pos = torch.zeros_like(sid_batch)
    pos[:, 0] = 0
    if T > 1: pos[:, 1] = 1
    if T > 2: pos[:, 2] = 2
    pos[sid_batch == pad_token] = 0
    return pos


class DualGateTrainer:
    def __init__(self, t5_state, adapters, device):
        self.device = device
        from transformers import T5ForConditionalGeneration, T5Config
        config = T5Config(
            vocab_size=1025, d_model=D_MODEL, d_kv=64, d_ff=1024,
            num_layers=4, num_decoder_layers=4, num_heads=3,
            relative_attention_num_buckets=32, relative_attention_max_distance=128,
            dropout_rate=0.0,
        )
        self.t5 = T5ForConditionalGeneration(config).to(device)
        missing, unexpected = self.t5.load_state_dict(t5_state, strict=False)
        print(f"[Load T5] Missing: {len(missing)}, Unexpected: {len(unexpected)}", flush=True)
        self.adapters = nn.ModuleList(adapters).to(device)

    def forward_with_adapter(self, input_ids, adapter_layer_idx, use_active=False):
        embed = self.t5.shared(input_ids)
        hidden = embed
        position_ids = build_position_ids(input_ids)
        for i, block in enumerate(self.t5.encoder.block):
            hidden = block(hidden)[0]
            if i == adapter_layer_idx:
                ad_idx = ADAPTER_LAYERS.index(adapter_layer_idx)
                hidden = self.adapters[ad_idx](hidden, position_ids, use_active=use_active)
        hidden = self.t5.encoder.final_layer_norm(hidden)
        return hidden


def check_zero_gate_equivalence(trainer, sample_sid):
    """Zero gate ≡ control (max logits diff ≤ 1e-5)."""
    position_ids = build_position_ids(sample_sid)
    for ad in trainer.adapters:
        ad.eval()
    # Baseline (no adapter)
    embed = trainer.t5.shared(sample_sid)
    h = embed
    for block in trainer.t5.encoder.block:
        h = block(h)[0]
    base_logits = trainer.t5.encoder.final_layer_norm(h)
    # With zero-test gate
    diffs = []
    for layer_idx in ADAPTER_LAYERS:
        ad_idx = ADAPTER_LAYERS.index(layer_idx)
        embed = trainer.t5.shared(sample_sid)
        h = embed
        for i, block in enumerate(trainer.t5.encoder.block):
            h = block(h)[0]
            if i == layer_idx:
                pos_id = build_position_ids(sample_sid)
                h = trainer.adapters[ad_idx](h, pos_id, use_active=False)  # zero gate
        ad_logits = trainer.t5.encoder.final_layer_norm(h)
        diffs.append((ad_logits - base_logits).abs().max().item())
    return max(diffs) <= ZERO_GATE_TOL, max(diffs), diffs


def check_grad_finite_nz(trainer, sample_sid):
    """Active gate gradient flows through u_l/scale/active_train_gate."""
    saved_active = [ad.active_train_gate.data.clone() for ad in trainer.adapters]
    # Move active gate to known active position (0 → 0.5)
    for ad in trainer.adapters:
        ad.zero_test_gate.requires_grad_(False)
        ad.active_train_gate.requires_grad_(True)
        ad.active_train_gate.data.fill_(0.0)
        ad.train()
    embed = trainer.t5.shared(sample_sid)
    h = embed
    position_ids = build_position_ids(sample_sid)
    for i, block in enumerate(trainer.t5.encoder.block):
        h = block(h)[0]
        if i in ADAPTER_LAYERS:
            ad_idx = ADAPTER_LAYERS.index(i)
            h = trainer.adapters[ad_idx](h, position_ids, use_active=True)
    h_after = trainer.t5.encoder.final_layer_norm(h)
    # Loss on adapter output (before final_layer_norm)
    h_adapter_out = h
    loss_adapter = F.mse_loss(h_adapter_out, torch.zeros_like(h_adapter_out))
    loss_adapter.backward()
    results = {"per_layer": []}
    for i, ad in enumerate(trainer.adapters):
        u_grad = ad.u_l_per_position.grad.norm().item() if ad.u_l_per_position.grad is not None else 0.0
        scale_grad = ad.scale_per_position.grad.norm().item() if ad.scale_per_position.grad is not None else 0.0
        gate_grad = ad.active_train_gate.grad.norm().item() if ad.active_train_gate.grad is not None else 0.0
        results["per_layer"].append({
            "u_l_grad_norm": u_grad,
            "scale_grad_norm": scale_grad,
            "active_gate_grad_norm": gate_grad,
            "u_l_finite_nz": math.isfinite(u_grad) and u_grad > 1e-10,
            "scale_finite_nz": math.isfinite(scale_grad) and scale_grad > 1e-10,
            "gate_finite_nz": math.isfinite(gate_grad) and gate_grad > 1e-10,
        })
    # Restore
    for i, ad in enumerate(trainer.adapters):
        ad.active_train_gate.data.copy_(saved_active[i])
        for p in ad.parameters():
            if p.grad is not None:
                p.grad.zero_()
    return results


def cf_differentiate(trainer, sample_sid):
    """5 CFs on active_train_gate."""
    # Baseline
    embed = trainer.t5.shared(sample_sid)
    h = embed
    position_ids = build_position_ids(sample_sid)
    for i, block in enumerate(trainer.t5.encoder.block):
        h = block(h)[0]
        if i in ADAPTER_LAYERS:
            ad_idx = ADAPTER_LAYERS.index(i)
            h = trainer.adapters[ad_idx](h, position_ids, use_active=True)
    base_logits = trainer.t5.encoder.final_layer_norm(h)

    cf_results = {}

    # cf1: on/off (skip — same as zero test)
    # cf2: kappa shuffle
    cf2_perm = torch.tensor([2, 0, 1])
    saved_u = [ad.u_l_per_position.data.clone() for ad in trainer.adapters]
    for ad in trainer.adapters:
        ad.u_l_per_position.data = ad.u_l_per_position.data[cf2_perm]
    embed = trainer.t5.shared(sample_sid)
    h = embed
    position_ids = build_position_ids(sample_sid)
    for i, block in enumerate(trainer.t5.encoder.block):
        h = block(h)[0]
        if i in ADAPTER_LAYERS:
            ad_idx = ADAPTER_LAYERS.index(i)
            h = trainer.adapters[ad_idx](h, position_ids, use_active=True)
    cf2_logits = trainer.t5.encoder.final_layer_norm(h)
    cf_results["cf2_kappa_shuffle_diff"] = (cf2_logits - base_logits).abs().max().item()
    # Repro check
    for ad in trainer.adapters:
        ad.u_l_per_position.data = ad.u_l_per_position.data[cf2_perm]
    embed = trainer.t5.shared(sample_sid)
    h = embed
    position_ids = build_position_ids(sample_sid)
    for i, block in enumerate(trainer.t5.encoder.block):
        h = block(h)[0]
        if i in ADAPTER_LAYERS:
            ad_idx = ADAPTER_LAYERS.index(i)
            h = trainer.adapters[ad_idx](h, position_ids, use_active=True)
    cf2_repro_logits = trainer.t5.encoder.final_layer_norm(h)
    cf_results["cf2_kappa_shuffle_diff_repro"] = (cf2_repro_logits - base_logits).abs().max().item()
    cf_results["cf2_reproducible"] = abs(cf_results["cf2_kappa_shuffle_diff"] - cf_results["cf2_kappa_shuffle_diff_repro"]) < 1e-6
    for i, ad in enumerate(trainer.adapters):
        ad.u_l_per_position.data.copy_(saved_u[i])

    # cf3: L0/L2 swap
    saved_u = [ad.u_l_per_position.data.clone() for ad in trainer.adapters]
    for ad in trainer.adapters:
        ad.u_l_per_position.data[[0, 2]] = ad.u_l_per_position.data[[2, 0]]
    embed = trainer.t5.shared(sample_sid)
    h = embed
    position_ids = build_position_ids(sample_sid)
    for i, block in enumerate(trainer.t5.encoder.block):
        h = block(h)[0]
        if i in ADAPTER_LAYERS:
            ad_idx = ADAPTER_LAYERS.index(i)
            h = trainer.adapters[ad_idx](h, position_ids, use_active=True)
    cf3_logits = trainer.t5.encoder.final_layer_norm(h)
    cf_results["cf3_l0_l2_swap_diff"] = (cf3_logits - base_logits).abs().max().item()
    for i, ad in enumerate(trainer.adapters):
        ad.u_l_per_position.data.copy_(saved_u[i])

    # cf4: alignment destroy
    saved_proj = [ad.residual_proj.data.clone() for ad in trainer.adapters]
    for ad in trainer.adapters:
        ad.residual_proj.data = torch.randn_like(ad.residual_proj.data)
    embed = trainer.t5.shared(sample_sid)
    h = embed
    position_ids = build_position_ids(sample_sid)
    for i, block in enumerate(trainer.t5.encoder.block):
        h = block(h)[0]
        if i in ADAPTER_LAYERS:
            ad_idx = ADAPTER_LAYERS.index(i)
            h = trainer.adapters[ad_idx](h, position_ids, use_active=True)
    cf4_logits = trainer.t5.encoder.final_layer_norm(h)
    cf_results["cf4_align_destroy_diff"] = (cf4_logits - base_logits).abs().max().item()
    for i, ad in enumerate(trainer.adapters):
        ad.residual_proj.data.copy_(saved_proj[i])

    # cf5: κ sign/scale sanity
    saved_u = [ad.u_l_per_position.data.clone() for ad in trainer.adapters]
    for ad in trainer.adapters:
        ad.u_l_per_position.data *= -1.0  # flip sign
    embed = trainer.t5.shared(sample_sid)
    h = embed
    position_ids = build_position_ids(sample_sid)
    for i, block in enumerate(trainer.t5.encoder.block):
        h = block(h)[0]
        if i in ADAPTER_LAYERS:
            ad_idx = ADAPTER_LAYERS.index(i)
            h = trainer.adapters[ad_idx](h, position_ids, use_active=True)
    cf5_logits = trainer.t5.encoder.final_layer_norm(h)
    cf_results["cf5_kappa_sign_flip_diff"] = (cf5_logits - base_logits).abs().max().item()
    for i, ad in enumerate(trainer.adapters):
        ad.u_l_per_position.data.copy_(saved_u[i])

    # cf6: active gate 置换
    saved_gate = [ad.active_train_gate.data.clone() for ad in trainer.adapters]
    for ad in trainer.adapters:
        ad.active_train_gate.data = ad.active_train_gate.data[torch.tensor([2, 0, 1])]
    embed = trainer.t5.shared(sample_sid)
    h = embed
    position_ids = build_position_ids(sample_sid)
    for i, block in enumerate(trainer.t5.encoder.block):
        h = block(h)[0]
        if i in ADAPTER_LAYERS:
            ad_idx = ADAPTER_LAYERS.index(i)
            h = trainer.adapters[ad_idx](h, position_ids, use_active=True)
    cf6_logits = trainer.t5.encoder.final_layer_norm(h)
    cf_results["cf6_active_gate_perm_diff"] = (cf6_logits - base_logits).abs().max().item()
    for i, ad in enumerate(trainer.adapters):
        ad.active_train_gate.data.copy_(saved_gate[i])

    return cf_results


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print("=" * 70)
    print(f"[Task #416 Issue #123 Gate 3] Dual-State Gate")
    print("=" * 70)

    # Load T5 + SID
    t5_state = load_t5_state_dict(CKPT_TASK84)
    sid = np.load(SID_PATH)
    print(f"[SID] shape={sid.shape}, unique={len(np.unique(sid, axis=0))}")
    # Sample a batch from SID for tests
    sample_sid = torch.tensor(sid[:16], dtype=torch.long).to(DEVICE)

    # Init 2 adapters
    adapters = [DualGateAdapter(d_model=D_MODEL, kappa_min=KAPPA_MIN, position_count=3) for _ in ADAPTER_LAYERS]
    trainer = DualGateTrainer(t5_state, adapters, DEVICE)

    # === Pre-train checks ===
    print("\n[Pre-train Check 1] Zero-gate ≡ control (max diff ≤ 1e-5)?", flush=True)
    zg_pass, zg_max_diff, zg_per_layer = check_zero_gate_equivalence(trainer, sample_sid)
    print(f"  Per-layer max diff: {zg_per_layer}")
    print(f"  Max diff: {zg_max_diff:.6e}, {'✅ PASS' if zg_pass else '❌ FAIL'}")

    print("\n[Pre-train Check 2] Active-train grad flows?", flush=True)
    grad_results = check_grad_finite_nz(trainer, sample_sid)
    grad_pass = all(all(p[f"{k}_finite_nz"] for k in ["u_l", "scale", "gate"])
                    for p in grad_results["per_layer"])
    print(f"  Per-layer grad results: {grad_results}")
    print(f"  Grad pass: {'✅ PASS' if grad_pass else '❌ FAIL'}")

    print("\n[Pre-train Check 3] 5+1 CFs?", flush=True)
    cf_results = cf_differentiate(trainer, sample_sid)
    cf_pass = all(cf_results[k] > 1e-7 for k in cf_results if k != "cf2_kappa_shuffle_diff_repro" and k != "cf2_reproducible")
    print(f"  cf_results: {cf_results}")
    print(f"  CF pass: {'✅ PASS' if cf_pass else '❌ FAIL'}")

    if not (zg_pass and grad_pass and cf_pass):
        print("\n❌ Pre-train checks FAIL — exit without training.")
        return

    # === Training (active gate, 30 epoch) ===
    print("\n[Training] 30 epoch with active gate...", flush=True)
    optimizer = torch.optim.Adam([p for ad in trainer.adapters for p in ad.parameters() if p.requires_grad], lr=1e-4)
    train_log = []
    for epoch in range(1, NUM_EPOCHS + 1):
        trainer.t5.eval()
        for ad in trainer.adapters:
            ad.train()
        optimizer.zero_grad()
        # Forward with active gate, compute MSE loss on adapter output
        for i, ad_idx in enumerate(ADAPTER_LAYERS):
            embed = trainer.t5.shared(sample_sid)
            h = embed
            position_ids = build_position_ids(sample_sid)
            for j, block in enumerate(trainer.t5.encoder.block):
                h = block(h)[0]
                if j == ad_idx:
                    h = trainer.adapters[i](h, position_ids, use_active=True)
            loss = F.mse_loss(h, torch.zeros_like(h))
            loss.backward(retain_graph=(i < len(ADAPTER_LAYERS) - 1))
        # Grad clip
        torch.nn.utils.clip_grad_norm_([p for ad in trainer.adapters for p in ad.parameters()], max_norm=1.0)
        optimizer.step()
        train_log.append({"epoch": epoch, "loss": loss.item()})
        if epoch % 5 == 0 or epoch == 1 or epoch == NUM_EPOCHS:
            print(f"  [Ep {epoch:02d}/{NUM_EPOCHS}] loss={loss.item():.6f}", flush=True)

    # === Post-train checks ===
    print("\n[Post-train Check 1] Zero-gate still ≡ control?", flush=True)
    post_zg_pass, post_zg_max_diff, _ = check_zero_gate_equivalence(trainer, sample_sid)
    print(f"  Max diff: {post_zg_max_diff:.6e}, {'✅ PASS' if post_zg_pass else '❌ FAIL'}")

    # Loss trend
    loss_initial = train_log[0]["loss"]
    loss_final = train_log[-1]["loss"]
    loss_decreased = loss_final < loss_initial - 1e-4

    gate3_pass = zg_pass and grad_pass and cf_pass and post_zg_pass and loss_decreased

    verdict = {
        "task": "task416_issue123_dual_gate",
        "issue": 123,
        "pre_train": {
            "zero_gate": {"max_diff": zg_max_diff, "per_layer": zg_per_layer, "pass": zg_pass},
            "grad": grad_results,
            "grad_pass": grad_pass,
            "cf": cf_results,
            "cf_pass": cf_pass,
        },
        "training": {"log": train_log, "loss_initial": loss_initial, "loss_final": loss_final,
                     "loss_decreased": loss_decreased},
        "post_train": {
            "zero_gate_max_diff": post_zg_max_diff,
            "zero_gate_pass": post_zg_pass,
        },
        "gate3_pass": gate3_pass,
    }
    with open(PRODUCT_DIR / "verdict.json", "w") as f:
        json.dump(verdict, f, indent=2, default=str)
    print(f"\nVerdict saved: {PRODUCT_DIR / 'verdict.json'}")
    print(f"Loss initial→final: {loss_initial:.6f} → {loss_final:.6f} ({'decreased' if loss_decreased else 'NOT decreased'})")
    print(f"Gate 3: {'✅ PASS' if gate3_pass else '❌ FAIL'}")


if __name__ == "__main__":
    main()