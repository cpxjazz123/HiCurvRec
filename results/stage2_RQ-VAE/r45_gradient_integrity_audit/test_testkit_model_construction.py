"""Issue #72 — the shared test helpers must actually build a model.

``r5_common.r5_testkit.tiny_model`` called ``T.RQVAE(hidden_channels=..., latent_dim=...,
num_codebooks=..., codebook_size=...)`` while ``RQVAE.__init__(self, config, *, in_dim)`` accepts
none of those keyword names, so it raised ``TypeError`` on every call. Nothing noticed because the
committed per-arm suites only grep trainer source text, assert a module flag, or call an isolated
helper -- they never constructed a model.

These tests exist so that cannot recur silently: each one fails loudly if the helper raises, and
they additionally drive a real forward + backward through the constructed model.

Run: python results/stage2_RQ-VAE/r45_gradient_integrity_audit/test_testkit_model_construction.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
sys.path.insert(0, str(REPO / "results/stage2_RQ-VAE/r5_common"))
import r5_testkit as K  # noqa: E402

ARMS = ["r4_08"] + [f"r5_{i:02d}" for i in range(1, 11)]
MODULES = {arm: K.load_arm(arm) for arm in ARMS}


def test_tiny_model_constructs_a_real_rqvae():
    """The whole point of the helper: it must return a usable model, not raise."""
    for arm in ARMS:
        T = MODULES[arm]
        try:
            model, config = K.tiny_model(T)
        except Exception as exc:  # noqa: BLE001 - the failure IS the finding
            raise AssertionError(
                f"{arm}: tiny_model raised {type(exc).__name__}: {exc}. It must build a real "
                f"RQVAE from the arm's own tokenizer config.") from exc
        assert isinstance(model, torch.nn.Module), f"{arm}: tiny_model did not return a Module"
        assert config is not None, f"{arm}: tiny_model must return the config it built from"
        assert hasattr(model, "encoder") and hasattr(model, "decoder"), (
            f"{arm}: constructed model is missing the encoder/decoder the trainer relies on")
    print(f"  tiny_model built a real RQVAE for all {len(ARMS)} arms", flush=True)


def test_tiny_model_drives_a_real_forward_and_backward():
    """Constructing is not enough: the model must be differentiable end to end."""
    T = MODULES["r5_10"]
    model, _ = K.tiny_model(T, seed=7, in_dim=16, codebook_dim=4, hidden=8, size=8)
    emb = torch.randn(8, 16, generator=torch.Generator().manual_seed(11))
    reconstructed, quant_loss, unused, tokens = model.forward(emb)
    assert reconstructed.shape == emb.shape, (reconstructed.shape, emb.shape)
    assert tokens.shape[0] == emb.shape[0], tokens.shape
    total, recon = model.compute_loss(emb, reconstructed, quant_loss)
    assert torch.isfinite(total) and torch.isfinite(recon), (float(total), float(recon))
    model.zero_grad(set_to_none=False)
    total.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "backward produced no gradients at all"
    assert any(float(g.abs().sum()) > 0.0 for g in grads), "every gradient was exactly zero"
    print(f"  forward/backward through the constructed model: "
          f"{len(grads)} parameter tensors with gradients", flush=True)


def test_tiny_model_is_deterministic_for_a_given_seed():
    T = MODULES["r5_01"]
    first, _ = K.tiny_model(T, seed=99)
    second, _ = K.tiny_model(T, seed=99)
    for a, b in zip(first.parameters(), second.parameters()):
        assert torch.equal(a.detach(), b.detach()), "same seed must give identical weights"
    print("  identical weights for the same seed", flush=True)


def main() -> int:
    tests = [
        test_tiny_model_constructs_a_real_rqvae,
        test_tiny_model_drives_a_real_forward_and_backward,
        test_tiny_model_is_deterministic_for_a_given_seed,
    ]
    failed = 0
    for test in tests:
        try:
            test()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {test.__name__}\n     {exc}", flush=True)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"ERROR {test.__name__}\n     {type(exc).__name__}: {exc}", flush=True)
        else:
            print(f"pass {test.__name__}", flush=True)
    print(f"\n{len(tests) - failed}/{len(tests)} passed", flush=True)
    if failed:
        print("TESTKIT_TESTS_RED", flush=True)
        return 1
    print("ALL TESTS PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
