"""Fail-closed transfer of the locked iter8 Stage2 warm-start."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

LOCKED_WARMSTART_PATH = Path(
    "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/"
    "curvature_RQ-VAE_iter8/out/rqvae/instruments/rqvae_best.pth"
)
LOCKED_WARMSTART_SHA256 = (
    "189e0affb156b9cdbd11f38a6fdb1127804e3f3095edd23c55f84825beb56c3b"
)
EXPECTED_FIXED_BUFFER_KEYS = frozenset(
    f"layers.{index}._fixed_c" for index in range(3)
)
LEGACY_CURVATURE_KEYS = frozenset(
    key
    for index in range(3)
    for key in (f"layers.{index}.c_layer_scale", f"layers.{index}._fixed_c")
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_warm_start(model: nn.Module, expected_curvatures: Sequence[float]) -> dict[str, Any]:
    """Validate the locked bytes and transfer every compatible non-curvature tensor."""
    if len(expected_curvatures) != 3:
        raise ValueError("expected fixed curvature must contain exactly three layers")
    path = LOCKED_WARMSTART_PATH
    if not path.is_absolute() or not path.is_file():
        raise FileNotFoundError(f"locked warm-start is missing: {path}")
    resolved_path = path.resolve(strict=True)
    expected_path = LOCKED_WARMSTART_PATH.resolve(strict=True)
    if resolved_path != expected_path:
        raise RuntimeError(
            f"warm-start path resolved unexpectedly: {resolved_path} != {expected_path}"
        )
    digest = sha256_file(resolved_path)
    if digest != LOCKED_WARMSTART_SHA256:
        raise RuntimeError(
            f"warm-start SHA256 mismatch: {digest} != {LOCKED_WARMSTART_SHA256}"
        )

    device = next(model.parameters()).device
    checkpoint = torch.load(resolved_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise RuntimeError("locked warm-start must contain a mapping-valued model state_dict")
    source = checkpoint["model"]
    target = model.state_dict()
    if not target:
        raise RuntimeError("target model state_dict is empty")

    for name, value in source.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise RuntimeError(
                f"warm-start model entry is not a string/tensor pair: {name!r}"
            )
    source_keys = set(source)
    unexpected_source = source_keys - set(target) - LEGACY_CURVATURE_KEYS
    if unexpected_source:
        raise RuntimeError(
            f"warm-start contains unexplained source keys: {sorted(unexpected_source)}"
        )
    missing_nonfixed = (set(target) - EXPECTED_FIXED_BUFFER_KEYS) - source_keys
    if missing_nonfixed:
        raise RuntimeError(
            f"warm-start is missing non-curvature tensors: {sorted(missing_nonfixed)}"
        )

    transfer: dict[str, torch.Tensor] = {}
    for name, value in source.items():
        if name in LEGACY_CURVATURE_KEYS:
            continue
        expected = target[name]
        if value.shape != expected.shape:
            raise RuntimeError(
                f"warm-start shape mismatch for {name}: {tuple(value.shape)} "
                f"!= {tuple(expected.shape)}"
            )
        if value.dtype != expected.dtype:
            raise RuntimeError(
                f"warm-start dtype mismatch for {name}: {value.dtype} != {expected.dtype}"
            )
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all().item():
            raise RuntimeError(f"warm-start tensor is non-finite: {name}")
        transfer[name] = value

    expected_transfer = set(target) - EXPECTED_FIXED_BUFFER_KEYS
    if set(transfer) != expected_transfer or not transfer:
        raise RuntimeError(
            "warm-start compatible transfer set is incomplete: "
            f"transferred={len(transfer)} expected={len(expected_transfer)}"
        )

    missing, unexpected = model.load_state_dict(transfer, strict=False)
    if set(missing) != EXPECTED_FIXED_BUFFER_KEYS or unexpected:
        raise RuntimeError(
            f"warm-start state transfer mismatch: missing={sorted(missing)} "
            f"unexpected={sorted(unexpected)}"
        )
    loaded = model.state_dict()
    for name, value in transfer.items():
        if not torch.equal(loaded[name], value):
            raise RuntimeError(f"warm-start transfer verification failed for {name}")

    parameter_names = {name for name, _ in model.named_parameters()}
    fixed_values = []
    for index, expected_value in enumerate(expected_curvatures):
        key = f"layers.{index}._fixed_c"
        layer = model.layers[index]
        buffer = layer._fixed_c
        if key in parameter_names or buffer.requires_grad:
            raise RuntimeError(f"fixed curvature is trainable or parameterized: {key}")
        value = float(buffer.detach().cpu().item())
        if abs(value - float(expected_value)) > 1e-6:
            raise RuntimeError(
                f"fixed curvature changed during warm-start: {key}={value} "
                f"expected={expected_value}"
            )
        fixed_values.append(value)

    skipped = sorted(set(source) & LEGACY_CURVATURE_KEYS)
    return {
        "path": str(resolved_path),
        "sha256": digest,
        "transferred_tensor_count": len(transfer),
        "skipped_legacy_curvature_keys": skipped,
        "expected_missing_fixed_buffer_keys": sorted(EXPECTED_FIXED_BUFFER_KEYS),
        "fixed_curvatures": fixed_values,
    }
