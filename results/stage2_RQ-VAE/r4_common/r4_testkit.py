"""Shared helpers for the R4 arm test suites (issues #51-#60).

Every ``test_r4_NN.py`` imports this module so the contract-level assertions
(train-only data audit, 3-token SID contract, integrated mini training step) are
written once and behave identically across arms.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
FORBIDDEN_IN_TRAINER = ("valid.parquet", "test.parquet", "behavior_pairs_valid_s42.npz")
FORBIDDEN_IN_TRAINER_CSV = ("valid.parquet", "test.parquet")


def load_arm(arm: str):
    """Import one R4 arm's ``train_rqvae`` with its tree on ``sys.path``.

    Set ``R4_TEST_TREE`` to the absolute path of a tree directory to load that tree
    instead. The RED phase points it at the canonical Resample tree, so every mechanism
    test is shown failing on the unchanged parent rather than on a half-written arm.

    The module only defines constants and functions at import time, so this is
    side-effect free, but the torch import is slow on this filesystem: callers
    should give the process a generous timeout.
    """
    override = os.environ.get("R4_TEST_TREE")
    tree = Path(override) if override else REPO_ROOT / "stage2_RQ-VAE" / f"TIGER_{arm}"
    sys.path.insert(0, str(tree))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import train_rqvae  # noqa: PLC0415 - deliberately late
    return train_rqvae


def arm_dir(arm: str) -> Path:
    return REPO_ROOT / "results/stage2_RQ-VAE" / arm


def test_stage3_gate_fail_closed_and_mock_launcher() -> None:
    """Verify the fail-closed Stage 3 gate across all boundary and failure cases."""
    import json
    import tempfile
    import r4_gate

    calls: list[str] = []

    def mock_launcher() -> None:
        calls.append("launched")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        sid_path = tmp_dir / "item_sids.json"
        ckpt_path = tmp_dir / "rqvae_best.pth"
        metrics_path = tmp_dir / "training_metrics.jsonl"

        # 1. Absent SID / checkpoint / metrics -> False
        _, gate_missing = r4_gate.evaluate(
            arm="r4_test",
            issue=0,
            item_sids_path=sid_path,
            metrics_path=metrics_path,
            tests_passed=True,
        )
        assert gate_missing["stage3_allowed"] is False
        assert r4_gate.maybe_launch_stage3(gate_missing, mock_launcher) is False
        assert len(calls) == 0

        # Prepare valid files using canonical SID as fixture
        sid_path.write_bytes(r4_gate.CANONICAL_SID_PATH.read_bytes())
        # Issue #72: the strict gate requires completion to be PROVABLE, so the positive fixture
        # must supply the same evidence a real run does: a readable checkpoint carrying its own
        # epoch metadata, and a linked ckpt_saved record. The previous fixture wrote the byte
        # string b"checkpoint-bytes" and a lone train_end row, so it satisfied the old gate only
        # because that gate defaulted a missing epoch to 3000 and never opened the checkpoint.
        import torch  # noqa: PLC0415

        torch.save({"state_dict": {}, "epoch": 3000, "selection": "final_epoch"}, ckpt_path)
        metrics_path.write_text(
            json.dumps({"event": "ckpt_saved", "epoch": 3000,
                        "path": str(ckpt_path), "reason": "final_epoch"}) + "\n"
            + json.dumps({"event": "train_end", "epoch": 3000}) + "\n",
            encoding="utf-8")

        # 2. Equal to baseline (canonical Resample SID has ER_L1 == 16.38953942232631) -> False
        m_eq, gate_eq = r4_gate.evaluate(
            arm="r4_test",
            issue=0,
            item_sids_path=sid_path,
            metrics_path=metrics_path,
            tests_passed=True,
        )
        assert abs(m_eq["ER_L1"] - r4_gate.BASELINE_ER_L1) < 1e-12
        assert gate_eq["stage3_allowed"] is False
        assert r4_gate.maybe_launch_stage3(gate_eq, mock_launcher) is False

        # Mock hard_l1_er for synthetic ER cases (below, above, NaN, Inf, zero neg_share)
        real_er = r4_gate.hard_l1_er
        try:
            def _case(er_val: float, neg_val: float = 0.005, n_pos: int = 52855, n_neg: int = 10):
                r4_gate.hard_l1_er = lambda _mat: {
                    "n_positive_rows": n_pos,
                    "n_negatives_per_row": n_neg,
                    "pos_share": er_val * neg_val if neg_val > 0 else 0.08,
                    "neg_share": neg_val,
                    "ER_L1": er_val,
                }
                return r4_gate.evaluate(
                    arm="r4_test",
                    issue=0,
                    item_sids_path=sid_path,
                    metrics_path=metrics_path,
                    tests_passed=True,
                )[1]

            # 3. Below baseline -> False
            g_below = _case(r4_gate.BASELINE_ER_L1 - 1e-6)
            assert g_below["stage3_allowed"] is False
            assert r4_gate.maybe_launch_stage3(g_below, mock_launcher) is False

            # 4. NaN -> False
            g_nan = _case(float("nan"))
            assert g_nan["stage3_allowed"] is False
            assert r4_gate.maybe_launch_stage3(g_nan, mock_launcher) is False

            # 5. Inf -> False
            g_inf = _case(float("inf"))
            assert g_inf["stage3_allowed"] is False
            assert r4_gate.maybe_launch_stage3(g_inf, mock_launcher) is False

            # 6. Zero neg_share -> False
            g_zero_neg = _case(20.0, neg_val=0.0)
            assert g_zero_neg["stage3_allowed"] is False
            assert r4_gate.maybe_launch_stage3(g_zero_neg, mock_launcher) is False

            # 7. Wrong support -> False
            g_bad_sup = _case(20.0, n_pos=100)
            assert g_bad_sup["stage3_allowed"] is False
            assert r4_gate.maybe_launch_stage3(g_bad_sup, mock_launcher) is False

            # 8. Strictly above baseline with all checks valid -> True
            g_above = _case(r4_gate.BASELINE_ER_L1 + 1e-6)
            assert g_above["stage3_allowed"] is True

            # 9. Incomplete training (epoch 2999) even when ER > baseline -> False
            metrics_path.write_text(json.dumps({"event": "train_end", "epoch": 2999}) + "\n", encoding="utf-8")
            g_incomplete = _case(r4_gate.BASELINE_ER_L1 + 1.0)
            assert g_incomplete["stage3_allowed"] is False
            assert r4_gate.maybe_launch_stage3(g_incomplete, mock_launcher) is False
            metrics_path.write_text(json.dumps({"event": "train_end", "epoch": 3000}) + "\n", encoding="utf-8")

            # 10. Wrong hash even when ER > baseline -> False
            _, g_bad_hash = r4_gate.evaluate(
                arm="r4_test",
                issue=0,
                item_sids_path=sid_path,
                metrics_path=metrics_path,
                tests_passed=True,
                canonical_sid_sha="0" * 64,
            )
            assert g_bad_hash["stage3_allowed"] is False
            assert r4_gate.maybe_launch_stage3(g_bad_hash, mock_launcher) is False

            # 11. Failed tests even when ER > baseline -> False
            _, g_failed_tests = r4_gate.evaluate(
                arm="r4_test",
                issue=0,
                item_sids_path=sid_path,
                metrics_path=metrics_path,
                tests_passed=False,
            )
            assert g_failed_tests["stage3_allowed"] is False
            assert r4_gate.maybe_launch_stage3(g_failed_tests, mock_launcher) is False

            # Confirm mock launcher was never called on any false gate, and runs on g_above
            assert len(calls) == 0
            assert r4_gate.maybe_launch_stage3(g_above, mock_launcher) is True
            assert len(calls) == 1
        finally:
            r4_gate.hard_l1_er = real_er


def run(name: str, tests: list[Any]) -> None:
    """Run every test in order; print one PASS line each; exit non-zero on failure."""
    all_tests = list(tests)
    if all(getattr(t, "__name__", "") != "test_stage3_gate_fail_closed_and_mock_launcher" for t in all_tests):
        all_tests.append(test_stage3_gate_fail_closed_and_mock_launcher)
    failures = 0
    for test in all_tests:
        label = getattr(test, "__name__", repr(test))
        try:
            test()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL: {label}: {exc}")
        except Exception as exc:  # noqa: BLE001 - surface any error as a failure
            failures += 1
            print(f"ERROR: {label}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS: {label}")
    if failures:
        raise SystemExit(f"{failures} test(s) failed")
    print("ALL TESTS PASSED")


def check_train_only(trainer) -> None:
    """The trainer may read train parquet + Stage1 embeddings, nothing else.

    The quantizer itself lives in the ``model/`` package, so the baseline
    quantizer paths are audited there rather than in the trainer text.
    """
    tree = Path(trainer.__file__).parent
    source = (Path(trainer.__file__)).read_text(encoding="utf-8")
    for needle in FORBIDDEN_IN_TRAINER:
        assert needle not in source, f"trainer references frozen evaluation data: {needle}"
    assert "train.parquet" in source, "trainer must still read the original train parquet"
    assert "sentence_t5.npy" in source, "trainer must still read the Stage1 embeddings"
    model_source = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted((tree / "model").rglob("*.py"))
    )
    for needle in ("class VQLayer", "init_codebook", "sinkhorn"):
        assert needle in model_source, f"quantizer package lost its baseline {needle} path"


def check_sid_contract(trainer) -> None:
    """The 4th exported token is collision extension only; the first three survive."""
    tokens = np.array(
        [
            [5, 7, 9],
            [5, 7, 9],
            [5, 7, 9],
            [5, 8, 9],
        ],
        dtype=np.int64,
    )
    sizes = [16, 16, 16]
    extended = trainer._extend_collisions(tokens, sizes)
    assert extended.shape == (4, 4), extended.shape
    assert np.array_equal(extended[:, :3], tokens), "collision extension changed the first 3 tokens"
    assert len(set(map(tuple, extended.tolist()))) == 4, "collision extension did not make rows unique"
    # Rows sharing a prefix get distinct 4th tokens; distinct prefixes start over.
    assert extended[0, 3] != extended[1, 3] and extended[1, 3] != extended[2, 3], extended[:, 3]
    assert extended[0, 3] == extended[3, 3], extended[:, 3]
    # The extra level numbers the occurrences inside each prefix group, so its values
    # start where the three real levels end and span one slot per distinct prefix.
    from collections import Counter
    group_sizes = Counter(tuple(row) for row in tokens.tolist())
    assert int(extended[:, 3].min()) == sum(sizes), extended[:, 3]
    assert int(extended[:, 3].max()) < sum(sizes) + max(group_sizes.values()), extended[:, 3]


def tiny_model(trainer, in_dim: int = 12, codebook_dim: int = 4, hidden: int = 8, size: int = 5):
    """A real ``RQVAE`` built from the arm's own tokenizer config, on CPU."""
    from types import SimpleNamespace  # noqa: PLC0415

    cfg = trainer._tokenizer_config()
    cfg = SimpleNamespace(
        hidden_sizes=[hidden, hidden],
        codebook_num=3,
        codebook_size=(size, size, size),
        codebook_dim=codebook_dim,
        dropout=0.0,
        beta=cfg.beta,
        vq_type=cfg.vq_type,
        ema_decay=cfg.ema_decay,
        fix_code_embs=False,
        sk_epsilon=cfg.sk_epsilon,
        sk_iters=cfg.sk_iters,
    )
    from model import RQVAE  # noqa: PLC0415 - resolved against the arm's tree

    torch.manual_seed(1234)
    model = RQVAE(cfg, in_dim=in_dim)
    model.eval()
    return model, cfg


def mini_step(trainer, seed: int = 1234) -> dict[str, Any]:
    """One real encoder -> 3-level quantization -> decoder -> backward -> step.

    Returns the loss, the encoder grad norm and a copy of the encoder parameters,
    so callers can compare a mechanism-on run against a mechanism-off run.
    """
    model, _ = tiny_model(trainer)
    torch.manual_seed(seed)
    batch = torch.randn(7, 12)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    optimizer.zero_grad(set_to_none=True)
    reconstructed, quant_loss, _, _ = model(batch)
    loss, recon = model.compute_loss(batch, reconstructed, quant_loss)
    loss.backward()
    encoder_params = [p for p in model.encoder.parameters()]
    grad_norm = float(
        torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in encoder_params if p.grad is not None))
    )
    for p in encoder_params:
        assert p.grad is not None, "encoder received no gradient in the integrated step"
        assert torch.isfinite(p.grad).all(), "encoder gradient is not finite"
    optimizer.step()
    params = [p.detach().clone() for p in encoder_params]
    return {
        "loss": float(loss.detach()),
        "recon": float(recon.detach()),
        "encoder_grad_norm": grad_norm,
        "params": params,
        "finite": bool(torch.isfinite(loss.detach())),
    }


def behaviour_inputs(trainer, seed: int = 7, pairs: int = 4, codebook_dim: int = 4):
    """Realistic inputs for ``behavior_contrastive_loss``: residuals + ids."""
    torch.manual_seed(seed)
    residuals = torch.randn(3, 2 * pairs, codebook_dim)
    source_ids = torch.arange(pairs, dtype=torch.long)
    target_ids = torch.arange(pairs, 2 * pairs, dtype=torch.long)
    return residuals, source_ids, target_ids, None
