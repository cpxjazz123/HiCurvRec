"""Regression: prefix-geodesic coordinates must use one stable reference RMS.

Run from either M10 tree:
    python scripts/check_reference_rms.py

Checks helper coordinates and full prefix-aware assignment for L2 (L1 keys) and
L3 (packed L1/L2 keys). The Euclidean paired arm runs the same checks.
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.layers import VQLayer  # noqa: E402
from prefix_radius import (  # noqa: E402
    CURVATURE, GEOMETRY, LOAD_STRENGTH, install_prefix_radius, prefix_radii,
)
from scripts.hyp_losses import encode_points, shared_tangent_scale  # noqa: E402

failures = []


def check(name, condition, detail=""):
    print(f"  [{'ok' if condition else 'FAIL'}] {name}" +
          (f" -- {detail}" if detail else ""))
    if not condition:
        failures.append(name)


torch.manual_seed(41)
base, k, dim, n = 16, 16, 8, 14
probe = torch.randn(n, dim) * 0.55
extra = torch.randn(7, dim) * 1.7
subset = torch.tensor([0, 2, 5, 8, 13], dtype=torch.long)

for level in (2, 3):
    layer = VQLayer(codebook_size=k, codebook_dim=dim,
                    sk_epsilon=0.5, sk_iters=3)
    install_prefix_radius(layer, level=level, base=base,
                          geometry=GEOMETRY, c=CURVATURE)
    layer.eval()

    # Two independent prefix buckets; at L3 the key is l1 * base + l2.
    first = 3 if level == 2 else 3 * base + 1
    second = 7 if level == 2 else 7 * base + 2
    keys = torch.tensor([first, second] * (n // 2), dtype=torch.long)
    with torch.no_grad():
        layer._prefix_counts[first, 1] = 11.0
        layer._prefix_counts[second, 5] = 9.0
    book = layer.get_code_embs()
    scale = shared_tangent_scale(book, CURVATURE)
    label = f"{GEOMETRY} L{level}"

    # One fixed codebook RMS for item tensors of different batch sizes.
    image = encode_points(probe, CURVATURE, scale, reference=book)
    image_part = encode_points(probe[subset], CURVATURE, scale, reference=book)
    diff = (image[subset] - image_part).abs().max().item()
    check(f"{label}: same item coordinates in full and subset",
          torch.allclose(image[subset], image_part, atol=1e-6, rtol=1e-6),
          f"max diff {diff:.2e}")

    # Exact same raw vector maps to the same ball point from either side.
    full_book = encode_points(book, CURVATURE, scale, reference=book)
    one_book = encode_points(book[:1], CURVATURE, scale, reference=book)
    check(f"{label}: item/codeword coordinate frame is shared",
          torch.allclose(full_book[:1], one_book, atol=1e-6, rtol=1e-6))

    # Prefix-dependent candidate points are (N,K,d). They must not drift
    # when unrelated rows change the candidate tensor's own RMS.
    directions = F.normalize(book, dim=-1)
    radii = directions.unsqueeze(0) * prefix_radii(
        layer._prefix_counts[keys], LOAD_STRENGTH).unsqueeze(-1)
    mapped = encode_points(radii, CURVATURE, scale, reference=book)
    mapped_part = encode_points(radii[subset], CURVATURE, scale, reference=book)
    check(f"{label}: same prefix candidates in full and subset",
          torch.allclose(mapped[subset], mapped_part, atol=1e-6, rtol=1e-6))

    # Exercise the actual patched VQLayer._distances, not merely a helper.
    layer._prefix_key = keys
    full = layer._distances(probe).detach()
    layer._prefix_key = keys[subset]
    sliced = layer._distances(probe[subset]).detach()
    check(f"{label}: distance matrix is batch-subset invariant",
          torch.allclose(full[subset], sliced, atol=1e-5, rtol=1e-5),
          f"max diff {(full[subset] - sliced).abs().max().item():.2e}")
    check(f"{label}: inference code indices are batch-subset invariant",
          torch.equal(full[subset].argmin(-1), sliced.argmin(-1)))

    layer._prefix_key = torch.cat((keys, torch.full((len(extra),), second)))
    extended = layer._distances(torch.cat((probe, extra), dim=0)).detach()
    check(f"{label}: unrelated rows do not move existing assignments",
          torch.allclose(full, extended[:n], atol=1e-5, rtol=1e-5)
          and torch.equal(full.argmin(-1), extended[:n].argmin(-1)))

    check(f"{label}: all distances and coordinates are finite",
          bool(torch.isfinite(full).all() and torch.isfinite(mapped).all()))
    layer._prefix_key = None

print("\n" + ("all reference-RMS checks passed" if not failures
              else f"FAILED: {failures}"))
sys.exit(1 if failures else 0)
