"""Shared helpers for the R5 arm test suites (Issues #61-#70).

Parent for RED is the pinned R4-08 tree ``TIGER_r4_08``. GREEN loads each arm's own tree.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
sys.path.insert(0, str(REPO_ROOT / "results/stage2_RQ-VAE/r5_common"))

FORBIDDEN_IN_TRAINER = ("valid.parquet", "test.parquet")


def load_arm(arm: str):
    """Import one arm's train_rqvae. ``R5_TEST_TREE`` overrides the tree path (RED phase)."""
    override = os.environ.get("R5_TEST_TREE") or os.environ.get("R4_TEST_TREE")
    tree = Path(override) if override else REPO_ROOT / "stage2_RQ-VAE" / f"TIGER_{arm}"
    # Drop previously imported trainer so RED/GREEN can switch trees in-process.
    sys.modules.pop("train_rqvae", None)
    sys.path.insert(0, str(tree))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import train_rqvae  # noqa: PLC0415
    return train_rqvae


def arm_dir(arm: str) -> Path:
    return REPO_ROOT / "results/stage2_RQ-VAE" / arm


def tiny_model(T, seed: int = 0):
    torch.manual_seed(seed)
    model = T.RQVAE(
        hidden_channels=[12, 10, 8],
        latent_dim=8,
        num_codebooks=3,
        codebook_size=[8, 8, 8],
    )
    return model, None


def check_train_only(T) -> None:
    text = Path(T.__file__).read_text(encoding="utf-8")
    for forbidden in FORBIDDEN_IN_TRAINER:
        assert forbidden not in text, f"trainer references forbidden split file {forbidden}"


def check_sid_contract(T) -> None:
    assert getattr(T, "CODEBOOK_NUM", 3) == 3
    assert list(getattr(T, "CODEBOOK_SIZE", [256, 256, 256]))[:3] == [256, 256, 256]


def test_stage3_gate_fail_closed_and_mock_launcher() -> None:
    import json
    import tempfile
    import r5_gate as G

    calls: list[str] = []

    def mock_launcher() -> None:
        calls.append("launched")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        sid_path = tmp_dir / "item_sids.json"
        ckpt_path = tmp_dir / "rqvae_best.pth"
        metrics_path = tmp_dir / "training_metrics.jsonl"
        metrics, gate = G.evaluate(
            arm="r5_tmp",
            issue=0,
            item_sids_path=sid_path,
            metrics_path=metrics_path,
            tests_passed=True,
        )
        assert gate["stage3_allowed"] is False
        assert G.maybe_launch_stage3(gate, mock_launcher) is False
        assert calls == []


def run(name: str, tests: list[Any]) -> None:
    all_tests = list(tests)
    if all(
        getattr(t, "__name__", "") != "test_stage3_gate_fail_closed_and_mock_launcher"
        for t in all_tests
    ):
        all_tests.append(test_stage3_gate_fail_closed_and_mock_launcher)
    failures = 0
    for test in all_tests:
        label = getattr(test, "__name__", repr(test))
        try:
            test()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL: {label}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR: {label}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS: {label}")
    if failures:
        raise SystemExit(f"{failures} test(s) failed")
    print("ALL TESTS PASSED")
