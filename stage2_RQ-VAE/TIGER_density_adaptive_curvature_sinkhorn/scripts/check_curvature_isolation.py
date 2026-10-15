"""Two models in one process must not be able to change each other's curvature.

The curvature used to live in a module-level pointer, so binding a second model
silently redirected the first model's assignment. These checks pin the isolation the
trainer now relies on, plus the checkpoint path that depends on it.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import adaptive_curvature as ac  # noqa: E402
from model.layers import VQLayer  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        fails.append(name)


def build(curvature: float):
    layers = torch.nn.ModuleList([VQLayer(codebook_size=32, codebook_dim=8) for _ in range(3)])
    rq = torch.nn.Module()
    rq.vq_layers = layers
    buffers = ac.ensure_curvature_buffers(rq)
    for level, layer in enumerate(layers):
        ac.install_adaptive_assignment(layer, level)
    ac.set_curvature(curvature, buffers)
    return rq, buffers, layers


torch.manual_seed(0)
model_a, buffers_a, layers_a = build(0.5)
model_b, buffers_b, layers_b = build(3.5)
probe = torch.randn(16, 8)

check("model A assigns at its own curvature",
      ac.current_curvature(layer=layers_a[0]) == 0.5,
      f"A = {ac.current_curvature(layer=layers_a[0])}")
check("model B assigns at its own curvature",
      ac.current_curvature(layer=layers_b[0]) == 3.5,
      f"B = {ac.current_curvature(layer=layers_b[0])}")

dist_a = layers_a[0]._distances(probe).clone()
dist_b = layers_b[0]._distances(probe).clone()

# now churn the other model, the way a second arm would during training
ac.set_curvature(4.0, buffers_b)
ac.update_curvature_from_tokens(torch.zeros(8, 3, dtype=torch.long), buffers_b, usable_k=32)

check("updating B leaves A's distance matrix untouched",
      torch.allclose(layers_a[0]._distances(probe), dist_a),
      f"max|diff| = {float((layers_a[0]._distances(probe) - dist_a).abs().max()):.2e}")
check("B did move",
      not torch.allclose(layers_b[0]._distances(probe), dist_b),
      f"B changed by {float((layers_b[0]._distances(probe) - dist_b).abs().max()):.4f}")
check("A's curvature is still A's",
      ac.current_curvature(layer=layers_a[0]) == 0.5,
      f"A = {ac.current_curvature(layer=layers_a[0])}")

# re-binding must not reset a trained curvature
ac.ensure_curvature_buffers(model_a)
check("re-binding does not reset an existing curvature",
      ac.current_curvature(layer=layers_a[0]) == 0.5,
      f"A after re-bind = {ac.current_curvature(layer=layers_a[0])}")

# checkpoint round-trip restores the curvature and the distances
saved = model_a.state_dict()
snapshot = layers_a[0]._distances(probe).clone()
curvature_key = "vq_layers.0._adaptive_curvature"
check("the curvature travels in state_dict",
      curvature_key in saved,
      f"value = {float(saved.get(curvature_key, float('nan'))):.4f}")

restored = VQLayer(codebook_size=32, codebook_dim=8)
rq2 = torch.nn.Module()
rq2.vq_layers = torch.nn.ModuleList([restored])
ac.ensure_curvature_buffers(rq2)
ac.install_adaptive_assignment(restored, 0)
rq2.load_state_dict({k: v for k, v in saved.items()
                     if k.startswith('vq_layers.0.')})
check("a reloaded layer reports the restored curvature",
      ac.current_curvature(layer=restored) == 0.5,
      f"restored = {ac.current_curvature(layer=restored)}")
check("a reloaded layer assigns identically",
      torch.allclose(restored._distances(probe), snapshot),
      f"max|diff| = {float((restored._distances(probe) - snapshot).abs().max()):.2e}")

# bounds still hold
ac.set_curvature(99.0, buffers_a)
check("the curvature stays clamped to [c_min, c_max]",
      ac.current_curvature(layer=layers_a[0]) == ac.C_MAX,
      f"clamped to {ac.current_curvature(layer=layers_a[0])}, c_max = {ac.C_MAX}")

print("\n" + ("all curvature-isolation checks passed" if not fails else f"FAILURES: {fails}"))
sys.exit(1 if fails else 0)
