import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mixed_geometry as G
import native_product as N


def codes_at(scale):
    torch.manual_seed(0)
    return [torch.randn(3, 32, dtype=torch.float64) * scale for _ in range(3)]


def test_small_normals_agree_with_the_tangent_path():
    latent = torch.randn(3, 32, dtype=torch.float64) * 0.02
    codes = codes_at(0.02)
    native = N.reconstruct(codes)
    tangent = N.tangent_reconstruct(latent, codes)
    assert torch.allclose(native, tangent, atol=5e-2)


def test_large_normals_diverge_from_the_tangent_path():
    latent = torch.randn(3, 32, dtype=torch.float64) * 0.9
    codes = codes_at(0.9)
    native = N.reconstruct(codes)
    tangent = N.tangent_reconstruct(latent, codes)
    assert not torch.allclose(native, tangent, atol=1e-2)


def test_residuals_end_at_the_same_state_for_zero_codes():
    latent = torch.randn(3, 32, dtype=torch.float64) * 0.3
    zero = [torch.zeros(3, 32, dtype=torch.float64) for _ in range(3)]
    states = N.residuals(latent, zero)
    assert torch.allclose(states[-1], latent, atol=1e-10)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok   ", name)
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print("FAIL ", name, type(exc).__name__, exc)
    raise SystemExit(1 if failures else 0)
