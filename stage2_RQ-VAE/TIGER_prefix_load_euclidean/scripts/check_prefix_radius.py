"""Unit tests for the prefix-keyed radius operator.

Covers the five properties the report names: prefix-key packing, no stale-count
persistence, accounting of the actually-assigned indices, checkpoint reload, and
the metric effect that makes this arm's operator geometric.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.layers import VQLayer  # noqa: E402
from prefix_radius import (  # noqa: E402
    CURVATURE, GEOMETRY, LOAD_STRENGTH, install_prefix_radius, prefix_radii,
)

fails = []


def check(name, ok, detail=""):
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        fails.append(name)


print(f"  geometry = {GEOMETRY}")

# --- prefix-key packing ---------------------------------------------------
n_embed, base = 32, 256
level2 = VQLayer(codebook_size=n_embed, codebook_dim=8, sk_epsilon=0.5, sk_iters=3)
install_prefix_radius(level2, level=2, base=base, geometry=GEOMETRY, c=CURVATURE)
check("level 2 has one counter row per L1 codeword",
      tuple(level2._prefix_counts.shape) == (base, n_embed),
      f"table = {tuple(level2._prefix_counts.shape)}")
level3 = VQLayer(codebook_size=n_embed, codebook_dim=8, sk_epsilon=0.5, sk_iters=3)
install_prefix_radius(level3, level=3, base=base, geometry=GEOMETRY, c=CURVATURE)
check("level 3 has base**2 rows (keys l1*256 + l2)",
      tuple(level3._prefix_counts.shape) == (base * base, n_embed),
      f"table = {tuple(level3._prefix_counts.shape)}")

# the key arithmetic the trainer's trace helper uses
l1 = torch.tensor([0, 5, 300])
l2 = torch.tensor([7, 9, 255])
packed = l1 * base + l2
check("the L3 packing is injective on the tested keys",
      len(set(packed.tolist())) == 3, f"l1={l1.tolist()} -> keys={packed.tolist()}")

# --- the radius rule ------------------------------------------------------
counts = torch.tensor([0.0, 1.0, 100.0])
radii = prefix_radii(counts)
check("a crowded cell is placed further out", radii[2] > radii[1] > radii[0],
      f"radii = {[round(float(r), 4) for r in radii]}")
check("an unused cell is not moved", abs(float(radii[0]) - 1.0) < 1e-6)

# --- the counter: assigned indices, no stale persistence, reload -----------
layer2 = VQLayer(codebook_size=n_embed, codebook_dim=8, sk_epsilon=0.5, sk_iters=3)
install_prefix_radius(layer2, level=2, base=base, geometry=GEOMETRY, c=CURVATURE)
layer2.train()
layer2._prefix_key = torch.zeros(32, dtype=torch.long)
latent = torch.randn(32, 8)
distances = layer2._distances(latent)
chosen = layer2._indices(distances, False)
recorded = torch.nonzero(layer2._prefix_counts.reshape(-1)).reshape(-1) % n_embed
check("the counter records the codes actually assigned",
      torch.equal(torch.sort(torch.unique(recorded)).values,
                  torch.unique(chosen).sort().values),
      f"recorded {recorded.numel()} cells, assigned {chosen.unique().numel()} codes")

sk = layer2._indices(distances, False)
check("Sinkhorn (not argmin) decides on this level",
      not torch.equal(sk, distances.argmin(dim=-1)),
      f"rows differing = {int((sk != distances.argmin(dim=-1)).sum())}/{len(sk)}")

# a cell the batch never visits must still decay
layer2._prefix_counts[7, 3] = 4.0
before = float(layer2._prefix_counts[7, 3].item())
for _ in range(3):
    layer2._indices(layer2._distances(latent), False)
after = float(layer2._prefix_counts[7, 3].item())
check("an unvisited cell decays instead of going stale",
      abs(after - before * 0.9 ** 3) < 1e-5,
      f"cell (7,3): {before:.4f} -> {after:.4f}, expected {before * 0.9 ** 3:.4f}")

saved = layer2.state_dict()
fresh = VQLayer(codebook_size=n_embed, codebook_dim=8, sk_epsilon=0.5, sk_iters=3)
install_prefix_radius(fresh, level=2, base=base, geometry=GEOMETRY, c=CURVATURE)
fresh.load_state_dict(saved)
check("the table round-trips through state_dict",
      torch.allclose(fresh._prefix_counts, layer2._prefix_counts),
      f"max|diff| = {float((fresh._prefix_counts - layer2._prefix_counts).abs().max()):.2e}")

# --- the metric effect (cross-arm) ----------------------------------------
import importlib.util  # noqa: E402

sibling = Path(__file__).resolve().parent.parent.parent / "TIGER_prefix_load_euclidean"
sys.path.insert(0, str(sibling))
spec = importlib.util.spec_from_file_location("prefix_radius_euclid", sibling / "prefix_radius.py")
euclid_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(euclid_module)

torch.manual_seed(2)
probe = torch.randn(64, 8)
base_codes = torch.nn.functional.normalize(torch.randn(n_embed, 8), dim=-1)
keys = torch.randint(0, base, (64,))
g_layer = VQLayer(codebook_size=n_embed, codebook_dim=8)
e_layer = VQLayer(codebook_size=n_embed, codebook_dim=8)
with torch.no_grad():
    g_layer.get_code_embs().copy_(base_codes)
    e_layer.get_code_embs().copy_(base_codes)
install_prefix_radius(g_layer, level=2, base=base, geometry="geodesic", c=CURVATURE)
install_prefix_radius(e_layer, level=2, base=base, geometry="euclidean")
for lay in (g_layer, e_layer):
    lay._prefix_key = keys
    lay._prefix_counts[keys[0], 0] = 60.0        # one crowded (prefix, code) cell
with torch.no_grad():
    g_mean = g_layer._distances(probe).mean(0)
    e_mean = e_layer._distances(probe).mean(0)
check("the two metrics rank the candidates differently",
      not torch.allclose(g_mean, e_mean, atol=1e-4),
      f"max|diff| = {float((g_mean - e_mean).abs().max()):.4f}")

print("\n" + ("all prefix-radius checks passed" if not fails else f"FAILURES: {fails}"))
sys.exit(1 if fails else 0)
