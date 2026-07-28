#!/usr/bin/env python3
"""Task #137 R137 fix — θ_init=0.01 probe.

After R137 fix, Run A (θ_init=0) is math fixed point: torch.where selects
eucl branch (no κ dependence). Test θ_init=0.01: tiny positive → torch.where
selects sph branch (κ=0.02) → if grad flows, κ should move. If it does,
then Task #89 retrain can use θ_init=0.01 to escape the fixed point.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
sys.path.insert(0, str(REPO / "HG-Rec"))
from model.hrqvae_free_curv import FreeCurvHRQVAE

EMB_PATH = REPO / "HG-Rec/dataset/Instruments/item_emb.parquet"

df = pd.read_parquet(EMB_PATH)
emb = np.stack(df["emb"].values) if "emb" in df.columns else np.stack(df[df.columns[-1]].values)
emb = torch.tensor(emb, dtype=torch.float32)
print(f"emb: {emb.shape}")

device = torch.device("cuda:0")
torch.manual_seed(42)
np.random.seed(42)

model = FreeCurvHRQVAE(
    in_dim=768, num_emb_list=[64, 128, 256], e_dim=32, M=1,
    kappa_max=2.0, layers=[512, 256, 128], dropout_prob=0.0,
    bn=False, loss_type="mse", quant_loss_weight=1.0, beta=0.25,
    kmeans_init=True, kmeans_iters=5,
    sk_eps=[0.003, 0.003, 0.003], sk_iters=3,
).to(device)

# Probe: θ_init = 0.01 → κ_init = κ_max * tanh(0.01) = 2 * 0.00999967 = 0.02
theta_init = 0.01
with torch.no_grad():
    for vq in model.hrq.vq_layers:
        vq.theta_m.data = torch.tensor([theta_init], dtype=torch.float32, device=device)
        vq.initted = False

print(f"θ_init = {theta_init}")
print(f"κ_init = {2.0 * np.tanh(theta_init):.6f}  (sph branch: k_m > 0 → True)")

model.train()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
dl = DataLoader(TensorDataset(emb), batch_size=256, shuffle=True)

# Step 1 grad check
first = next(iter(dl))[0].to(device)
z = model.encoder(first)
out, rq_loss, _ = model.hrq(z, use_sk=False)
out = model.decoder(out)
loss = F.mse_loss(out, first, reduction="mean") + model.quant_loss_weight * rq_loss
opt.zero_grad()
loss.backward()

print(f"\nStep 1 grad check (θ_init={theta_init}):")
for i, vq in enumerate(model.hrq.vq_layers):
    if vq.theta_m.grad is None:
        print(f"  Layer {i}: grad = None")
    else:
        g = vq.theta_m.grad.norm().item()
        print(f"  Layer {i}: |grad| = {g:.6e}  {'✅' if g > 1e-10 else '❌'}")

# Train 10 epoch
print(f"\n10 epoch trajectory:")
import time
t0 = time.time()
for ep in range(10):
    for batch in dl:
        x = batch[0].to(device)
        out, rq_loss, _ = model(x, use_sk=False)
        loss = F.mse_loss(out, x, reduction="mean") + model.quant_loss_weight * rq_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
    if ep == 0 or (ep + 1) % 2 == 0 or ep == 9:
        theta = [vq.theta_m.item() for vq in model.hrq.vq_layers]
        kappa = [vq.kappa_m().item() for vq in model.hrq.vq_layers]
        print(f"  ep{ep+1:2d}: θ={theta} κ={kappa}  ({time.time()-t0:.1f}s)")

print("\nVerdict:")
final_kappa = [vq.kappa_m().item() for vq in model.hrq.vq_layers]
move = max(abs(vq.theta_m.item() - theta_init) for vq in model.hrq.vq_layers)
print(f"  max |θ_final − θ_init| = {move:.6e}")
if move > 1e-4:
    print("  ✅ THETA MOVES from 0.01 init → R137 fix works for non-zero init")
    print("  → Task #89 retrain should use θ_init=0.01 to escape fixed point")
else:
    print("  ❌ THETA STILL FROZEN at 0.01 init → deeper issue")
    print("  → investigate hyp/sph branch gradient computation")