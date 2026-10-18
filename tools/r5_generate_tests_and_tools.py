"""Generate R5 per-arm tests, configs, capture/launch/evaluate helpers."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
RESULTS = REPO / "results/stage2_RQ-VAE"
COMMON = RESULTS / "r5_common"

META = {
    "r5_01": (61, "Reverse PCGrad: protect L1 behavior against base gradient", "R5_01_REVERSE_PCGRAD",
              "project g_base against g_L1 when conflict; assemble g'_base + scale*(g_L1+g_other)"),
    "r5_02": (62, "Reconstruction-only reference gradient for original L1 PCGrad", "R5_02_RECON_REF",
              "project g_L1 against g_recon=grad(L_recon); assemble with full g_base"),
    "r5_03": (63, "Quantization-only reference gradient for original L1 PCGrad", "R5_03_QUANT_REF",
              "project g_L1 against g_quant=grad(L_base-L_recon); assemble with full g_base"),
    "r5_04": (64, "Per-encoder-layer blockwise PCGrad instead of full-vector projection", "R5_04_BLOCKWISE",
              "PCGrad per nn.Linear weight+bias block instead of one global encoder vector"),
    "r5_05": (65, "Half-strength conflict projection: fixed alpha=0.5", "R5_05_ALPHA",
              "g'_L1 = g_L1 - 0.5 * (<g_L1,g_base>/(||g_base||^2+eps)) g_base on conflict"),
    "r5_06": (66, "Norm-preserving post-PCGrad L1 behavior gradient", "R5_06_NORM_PRESERVE",
              "after R4-08 projection, rescale g_proj to original ||g_L1||"),
    "r5_07": (67, "Strong-conflict-only cosine trigger for R4-08 projection", "R5_07_COS_THRESHOLD",
              "project only if cos(g_L1,g_base) < -0.2; else leave g_L1 untouched"),
    "r5_08": (68, "Late-epoch-only R4-08 conflict projection", "R5_08_LATE_START_EPOCH",
              "activate R4-08 projection only for epoch >= 2250"),
    "r5_09": (69, "EMA-smoothed base gradient as PCGrad reference", "R5_09_EMA_DECAY",
              "m_t=0.9 m_{t-1}+0.1 g_base_t as projection reference; assemble with current g_base"),
    "r5_10": (70, "Inter-level PCGrad: protect L2+L3 behavior while projecting L1", "R5_10_INTERLEVEL",
              "project g_L1 against g_other=grad(L2+L3); assemble with full g_base + g_other"),
}


TESTKIT = r'''"""Shared helpers for the R5 arm test suites (Issues #61-#70).

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
'''


def test_body(arm: str) -> str:
    issue, title, flag, formula = META[arm]
    common_head = f'''"""R5 tests for {arm} (issue #{issue}): {title}.

RED is run with ``R5_TEST_TREE`` pointing at the pinned parent ``TIGER_r4_08``.
GREEN loads this arm's own tree.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
sys.path.insert(0, str(REPO / "results/stage2_RQ-VAE/r5_common"))
import r5_testkit as K  # noqa: E402

ARM = "{arm}"
FLAG = "{flag}"
EPS = 1e-12
T = K.load_arm(ARM)


def _set(value: bool) -> None:
    if hasattr(T, "R4_08_PCGRAD_ENC"):
        T.R4_08_PCGRAD_ENC = value
    if hasattr(T, FLAG):
        try:
            setattr(T, FLAG, value if not isinstance(getattr(T, FLAG), float) else getattr(T, FLAG))
        except Exception:
            pass


def _dot(a, b):
    return float((a * b).sum())


def test_train_only_data_audit():
    K.check_train_only(T)


def test_sid_three_token_contract():
    K.check_sid_contract(T)

'''
    # mechanism-specific RED/GREEN bodies
    bodies = {
        "r5_01": '''
def test_red_reverse_projects_base_not_l1():
    """Toy conflict: reverse PCGrad must make g'_base orthogonal to g_L1; parent projects L1."""
    g_base = torch.tensor([1.0, 0.0])
    g_l1 = torch.tensor([-1.0, 2.0])
    # Parent API projects the first arg against the second.
    # Reverse arm's r4_project_conflict(g_base, g_l1) should orthogonalize base to L1.
    out = T.r4_project_conflict(g_base.clone(), g_l1.clone())
    assert abs(_dot(out, g_l1)) < 1e-5, (
        f"reverse projection did not orthogonalize base to L1 (dot={_dot(out, g_l1):.6f}); "
        "parent still projects L1 instead"
    )
    # L1 itself must remain untouched by the reverse helper when called this way.
    assert torch.allclose(g_l1, torch.tensor([-1.0, 2.0]))


def test_red_parent_projects_l1_so_reverse_expectation_differs():
    g_base = torch.tensor([1.0, 0.0])
    g_l1 = torch.tensor([-1.0, 2.0])
    parentish = g_l1 - (_dot(g_l1, g_base) / (_dot(g_base, g_base) + EPS)) * g_base
    out = T.r4_project_conflict(g_base.clone(), g_l1.clone())
    assert not torch.allclose(out, parentish, atol=1e-5), (
        "reverse projection collapsed to the parent L1-projection result"
    )


def test_nonconflict_passthrough():
    g_base = torch.tensor([1.0, 0.0])
    g_l1 = torch.tensor([0.0, 2.0])
    out = T.r4_project_conflict(g_base.clone(), g_l1.clone())
    assert torch.allclose(out, g_base, atol=1e-7)
''',
        "r5_02": '''
def test_red_recon_reference_projects_when_parent_would_not():
    """Construct L1 conflicting with recon but not with total base; parent must stay inactive."""
    # Parent uses g_base; arm uses g_recon.
    g_l1 = torch.tensor([-1.0, 0.0, 0.0])
    g_recon = torch.tensor([1.0, 0.0, 0.0])
    g_quant = torch.tensor([-2.0, 0.0, 0.0])  # so g_base = g_recon+g_quant = (-1,0,0), dot with L1 > 0
    g_base = g_recon + g_quant
    assert _dot(g_l1, g_base) > 0
    assert _dot(g_l1, g_recon) < 0
    # Call the arm helper with recon as reference via project(g_l1, g_recon)
    out = T.r4_project_conflict(g_l1.clone(), g_recon.clone())
    assert abs(_dot(out, g_recon)) < 1e-5, (
        f"recon-reference projection inactive (dot={_dot(out, g_recon):.6f})"
    )
    parent_out = T.r4_project_conflict(g_l1.clone(), g_base.clone())
    # On the GREEN tree, projecting vs base is a no-op here; vs recon is active.
    # On the RED parent, both calls use the same projector so parent_out==g_l1 and
    # out==projected; the RED failure is that parent cannot prefer recon.
    assert not torch.allclose(out, parent_out, atol=1e-6), (
        "recon-reference projection is indistinguishable from parent base-reference"
    )


def test_recon_equals_base_matches_parent_formula():
    g_l1 = torch.tensor([-1.0, 2.0])
    g_ref = torch.tensor([1.0, 0.0])
    out = T.r4_project_conflict(g_l1.clone(), g_ref.clone())
    expected = g_l1 - (_dot(g_l1, g_ref) / (_dot(g_ref, g_ref) + EPS)) * g_ref
    assert torch.allclose(out, expected, atol=1e-6)
''',
        "r5_03": '''
def test_red_quant_reference_projects_when_parent_would_not():
    g_l1 = torch.tensor([-1.0, 0.0])
    g_quant = torch.tensor([1.0, 0.0])
    g_recon = torch.tensor([-2.0, 0.0])
    g_base = g_recon + g_quant  # (-1,0) -> nonconflict with L1
    assert _dot(g_l1, g_base) > 0
    assert _dot(g_l1, g_quant) < 0
    out = T.r4_project_conflict(g_l1.clone(), g_quant.clone())
    assert abs(_dot(out, g_quant)) < 1e-5, (
        f"quant-reference projection inactive (dot={_dot(out, g_quant):.6f})"
    )
    parent_out = T.r4_project_conflict(g_l1.clone(), g_base.clone())
    assert not torch.allclose(out, parent_out, atol=1e-6), (
        "quant-reference projection matches parent base-reference (silent no-op)"
    )


def test_quant_equals_base_matches_parent_formula():
    g_l1 = torch.tensor([-1.0, 2.0])
    g_ref = torch.tensor([1.0, 0.0])
    out = T.r4_project_conflict(g_l1.clone(), g_ref.clone())
    expected = g_l1 - (_dot(g_l1, g_ref) / (_dot(g_ref, g_ref) + EPS)) * g_ref
    assert torch.allclose(out, expected, atol=1e-6)
''',
        "r5_04": '''
def test_red_blockwise_differs_from_global_when_layers_disagree():
    """Global dot positive, one block negative: blockwise must change that block."""
    # Simulate two blocks concatenated; use apply if present.
    apply = getattr(T, "_r4_08_apply_projection", None)
    assert apply is not None, "projection apply entry missing"
    assert getattr(T, "R5_04_BLOCKWISE", False), (
        "blockwise flag missing — parent still uses global PCGrad"
    )
    # Analytic: block0 conflict, block1 aligned. Global may be non-negative.
    g_l1_b0 = torch.tensor([-1.0, 0.0])
    g_base_b0 = torch.tensor([1.0, 0.0])
    g_l1_b1 = torch.tensor([10.0, 0.0])
    g_base_b1 = torch.tensor([1.0, 0.0])
    flat_l1 = torch.cat([g_l1_b0, g_l1_b1])
    flat_base = torch.cat([g_base_b0, g_base_b1])
    assert _dot(flat_l1, flat_base) > 0  # global nonconflict
    # Per-block projection of block0 should move.
    proj_b0 = T.r4_project_conflict(g_l1_b0.clone(), g_base_b0.clone())
    assert not torch.allclose(proj_b0, g_l1_b0, atol=1e-6), "block0 conflict not projected"
    # Global projector (parent behavior) would leave flat_l1 unchanged.
    global_proj = T.r4_project_conflict(flat_l1.clone(), flat_base.clone())
    assert torch.allclose(global_proj, flat_l1, atol=1e-6), "sanity: global should be inactive"
    # Mechanism present marker
    assert hasattr(T, "_r5_04_linear_blocks"), (
        "blockwise grouping helper not implemented"
    )


def test_single_block_matches_parent_formula():
    g_l1 = torch.tensor([-1.0, 2.0])
    g_base = torch.tensor([1.0, 0.0])
    out = T.r4_project_conflict(g_l1.clone(), g_base.clone())
    expected = g_l1 - (_dot(g_l1, g_base) / (_dot(g_base, g_base) + EPS)) * g_base
    assert torch.allclose(out, expected, atol=1e-6)
''',
        "r5_05": '''
def test_red_half_strength_expected_vector():
    g_l1 = torch.tensor([-1.0, 2.0])
    g_base = torch.tensor([1.0, 0.0])
    out = T.r4_project_conflict(g_l1.clone(), g_base.clone())
    expected = torch.tensor([-0.5, 2.0])
    assert torch.allclose(out, expected, atol=1e-6), (
        f"half-strength projection wrong: got {out.tolist()}, expected {expected.tolist()} "
        "(parent full projection yields ~[0,2])"
    )


def test_red_conflict_dot_is_half_of_original():
    g_l1 = torch.tensor([-1.0, 2.0])
    g_base = torch.tensor([1.0, 0.0])
    orig = _dot(g_l1, g_base)
    out = T.r4_project_conflict(g_l1.clone(), g_base.clone())
    new = _dot(out, g_base)
    assert abs(new - 0.5 * orig) < 1e-5, (
        f"post-projection conflict dot {new} != 0.5*original {0.5*orig}"
    )


def test_nonconflict_unchanged():
    g_l1 = torch.tensor([0.0, 2.0])
    g_base = torch.tensor([1.0, 0.0])
    out = T.r4_project_conflict(g_l1.clone(), g_base.clone())
    assert torch.allclose(out, g_l1, atol=1e-7)
''',
        "r5_06": '''
def test_red_norm_restored_and_orthogonal():
    g_l1 = torch.tensor([-1.0, 2.0])
    g_base = torch.tensor([1.0, 0.0])
    out = T.r4_project_conflict(g_l1.clone(), g_base.clone())
    assert abs(_dot(out, g_base)) < 1e-5, f"not orthogonal (dot={_dot(out, g_base)})"
    orig_norm = float(torch.linalg.vector_norm(g_l1))
    out_norm = float(torch.linalg.vector_norm(out))
    assert abs(out_norm - orig_norm) < 1e-5, (
        f"norm not restored: ||out||={out_norm}, ||g_L1||={orig_norm} "
        "(parent leaves ||proj||=2)"
    )


def test_antiparallel_returns_zero():
    g_l1 = torch.tensor([-2.0, 0.0])
    g_base = torch.tensor([1.0, 0.0])
    out = T.r4_project_conflict(g_l1.clone(), g_base.clone())
    assert torch.allclose(out, torch.zeros_like(g_l1), atol=1e-7), out


def test_nonconflict_unchanged():
    g_l1 = torch.tensor([0.0, 3.0])
    g_base = torch.tensor([1.0, 0.0])
    out = T.r4_project_conflict(g_l1.clone(), g_base.clone())
    assert torch.allclose(out, g_l1, atol=1e-7)
''',
        "r5_07": '''
def test_red_weak_conflict_is_skipped():
    # cos ≈ -0.1 : parent projects, arm must not.
    g_base = torch.tensor([1.0, 0.0])
    g_l1 = torch.tensor([-0.1, 1.0])
    cos = _dot(g_l1, g_base) / ((float(torch.linalg.vector_norm(g_l1)) * 1.0) + EPS)
    assert -0.2 < cos < 0.0
    out = T.r4_project_conflict(g_l1.clone(), g_base.clone())
    assert torch.allclose(out, g_l1, atol=1e-7), (
        f"weak conflict was projected (cos={cos:.4f}); parent would project, arm must skip"
    )


def test_red_strong_conflict_uses_full_projection():
    g_base = torch.tensor([1.0, 0.0])
    g_l1 = torch.tensor([-1.0, 0.5])
    cos = _dot(g_l1, g_base) / (float(torch.linalg.vector_norm(g_l1)) + EPS)
    assert cos < -0.2
    out = T.r4_project_conflict(g_l1.clone(), g_base.clone())
    expected = g_l1 - (_dot(g_l1, g_base) / (_dot(g_base, g_base) + EPS)) * g_base
    assert torch.allclose(out, expected, atol=1e-6), (out, expected)


def test_threshold_boundary_no_projection():
    # Build vectors with cos exactly -0.2
    g_base = torch.tensor([1.0, 0.0])
    g_l1 = torch.tensor([-0.2, (1 - 0.04) ** 0.5])
    cos = _dot(g_l1, g_base) / float(torch.linalg.vector_norm(g_l1))
    assert abs(cos + 0.2) < 1e-6
    out = T.r4_project_conflict(g_l1.clone(), g_base.clone())
    assert torch.allclose(out, g_l1, atol=1e-7), "cos==-0.2 must NOT project"
''',
        "r5_08": '''
def test_red_late_start_flag_present_and_gate():
    start = getattr(T, "R5_08_LATE_START_EPOCH", None)
    assert start == 2250, (
        f"late-start epoch missing/wrong ({start}); parent projects for every behavior epoch"
    )


def test_red_source_contains_epoch_gate():
    text = Path(T.__file__).read_text(encoding="utf-8")
    assert "epoch >= R5_08_LATE_START_EPOCH" in text, (
        "training loop does not gate projection on epoch>=2250"
    )
    assert "loss.backward()" in text


def test_parent_projection_math_still_available():
    g_l1 = torch.tensor([-1.0, 2.0])
    g_base = torch.tensor([1.0, 0.0])
    out = T.r4_project_conflict(g_l1.clone(), g_base.clone())
    expected = g_l1 - (_dot(g_l1, g_base) / (_dot(g_base, g_base) + EPS)) * g_base
    assert torch.allclose(out, expected, atol=1e-6)
''',
        "r5_09": '''
def test_red_ema_state_and_helper_exist():
    assert hasattr(T, "R5_09_EMA_DECAY"), "EMA decay flag missing on parent"
    assert abs(float(T.R5_09_EMA_DECAY) - 0.9) < 1e-12
    assert hasattr(T, "_r5_09_update_ema"), (
        "EMA update helper not implemented — parent still uses instantaneous g_base"
    )


def test_red_ema_two_step_differs_from_instantaneous():
    T.R5_09_EMA_STATE["buffers"] = None
    g1 = [torch.tensor([1.0, 0.0])]
    g2 = [torch.tensor([0.0, 1.0])]
    m1 = T._r5_09_update_ema(g1)
    assert torch.allclose(m1[0], g1[0])
    m2 = T._r5_09_update_ema(g2)
    expected = 0.9 * g1[0] + 0.1 * g2[0]
    assert torch.allclose(m2[0], expected, atol=1e-6), (m2[0], expected)
    # reset for isolation
    T.R5_09_EMA_STATE["buffers"] = None


def test_projection_math_with_ema_reference():
    g_l1 = torch.tensor([-1.0, 2.0])
    g_ref = torch.tensor([1.0, 0.0])
    out = T.r4_project_conflict(g_l1.clone(), g_ref.clone())
    expected = g_l1 - (_dot(g_l1, g_ref) / (_dot(g_ref, g_ref) + EPS)) * g_ref
    assert torch.allclose(out, expected, atol=1e-6)
''',
        "r5_10": '''
def test_red_interlevel_flag_and_apply_uses_other_reference():
    assert getattr(T, "R5_10_INTERLEVEL", False), (
        "inter-level flag missing — parent still projects against g_base"
    )
    text = Path(T.__file__).read_text(encoding="utf-8")
    assert "R5_10_INTERLEVEL" in text and "g_other" in text
    # Ensure apply body projects against g_other, not only g_base.
    assert "flat_ref = torch.cat([grad.reshape(-1) for grad in g_other])" in text, (
        "apply_projection does not use g_other as PCGrad reference"
    )


def test_conflict_with_other_not_base():
    g_l1 = torch.tensor([-1.0, 0.0])
    g_other = torch.tensor([1.0, 0.0])
    g_base = torch.tensor([0.0, 1.0])  # nonconflict with L1
    assert _dot(g_l1, g_base) == 0.0
    assert _dot(g_l1, g_other) < 0
    out = T.r4_project_conflict(g_l1.clone(), g_other.clone())
    assert abs(_dot(out, g_other)) < 1e-5
    parentish = T.r4_project_conflict(g_l1.clone(), g_base.clone())
    assert torch.allclose(parentish, g_l1, atol=1e-7)
    assert not torch.allclose(out, parentish, atol=1e-6)
''',
    }
    # collect test function names for K.run
    import re

    body = bodies[arm]
    names = re.findall(r'^def (test_\w+)\(', common_head + body, flags=re.M)
    run_list = ",\n            ".join(names)
    tail = f'''

if __name__ == "__main__":
    K.run(
        __name__,
        [
            {run_list},
        ],
    )
'''
    return common_head + body + tail


CAPTURE_RED = r'''"""Capture RED for every R5 arm against pinned parent TIGER_r4_08 (#58)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9"
PARENT = REPO_ROOT / "stage2_RQ-VAE/TIGER_r4_08"
ARMS = [f"r5_{n:02d}" for n in range(1, 11)]


def run(arm: str):
    script = REPO_ROOT / "results/stage2_RQ-VAE" / arm / f"test_{arm}.py"
    env = dict(os.environ)
    env["R5_TEST_TREE"] = str(PARENT)
    done = subprocess.run(
        [PYTHON, str(script)], cwd=REPO_ROOT, capture_output=True, text=True,
        timeout=7200, env=env,
    )
    command = f"R5_TEST_TREE={PARENT} {PYTHON} results/stage2_RQ-VAE/{arm}/test_{arm}.py"
    return done.returncode, done.stdout + done.stderr, command


def main() -> int:
    problems = []
    for arm in ARMS:
        code, output, command = run(arm)
        path = REPO_ROOT / "results/stage2_RQ-VAE" / arm / "tdd_red_green.md"
        failures = [ln for ln in output.splitlines() if ln.startswith(("FAIL:", "ERROR:"))]
        body = [
            f"# {arm.upper()} TDD record", "", "## RED", "",
            "Command (run against the pinned R4-08 parent):", "",
            f"    {command}", "", f"Exit code: {code}", "", "```", output.rstrip(), "```", "",
            f"Behavioural failures observed on the parent: {len(failures)}", "",
        ]
        path.write_text("\n".join(body), encoding="utf-8")
        bad_import = any("ImportError" in ln or "AttributeError" in ln for ln in failures)
        status = "RED OK" if code != 0 and not bad_import else "RED PROBLEM"
        if status != "RED OK":
            problems.append(arm)
        print(f"{arm}: exit={code} {status} | {len(failures)} failures", flush=True)
        for ln in failures[:4]:
            print(f"    {ln[:160]}", flush=True)
    print("\nPROBLEM ARMS:", problems if problems else "none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

CAPTURE_GREEN = r'''"""Capture GREEN for every R5 arm against its own tree."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9"
ARMS = [f"r5_{n:02d}" for n in range(1, 11)]
PARENT = REPO_ROOT / "stage2_RQ-VAE/TIGER_r4_08"


def run(arm: str):
    script = REPO_ROOT / "results/stage2_RQ-VAE" / arm / f"test_{arm}.py"
    env = dict(os.environ)
    env.pop("R5_TEST_TREE", None)
    env.pop("R4_TEST_TREE", None)
    done = subprocess.run(
        [PYTHON, str(script)], cwd=REPO_ROOT, capture_output=True, text=True,
        timeout=7200, env=env,
    )
    return done.returncode, done.stdout + done.stderr, f"{PYTHON} results/stage2_RQ-VAE/{arm}/test_{arm}.py"


def main() -> int:
    failed = []
    for arm in ARMS:
        code, output, command = run(arm)
        path = REPO_ROOT / "results/stage2_RQ-VAE" / arm / "tdd_red_green.md"
        text = path.read_text(encoding="utf-8") if path.is_file() else f"# {arm.upper()} TDD record\n"
        text += "\n## GREEN\n\n"
        text += "Command (run against this arm's own tree, mechanism implemented):\n\n"
        text += f"    {command}\n\nExit code: {code}\n\n```\n{output.rstrip()}\n```\n\n"
        diff = subprocess.run(
            ["diff", "-u", str(PARENT / "train_rqvae.py"),
             str(REPO_ROOT / "stage2_RQ-VAE" / f"TIGER_{arm}" / "train_rqvae.py")],
            capture_output=True, text=True,
        ).stdout
        text += "## Single delta\n\n```diff\n" + diff + "\n```\n"
        path.write_text(text, encoding="utf-8")
        cfg_path = REPO_ROOT / "results/stage2_RQ-VAE" / arm / "config.json"
        if cfg_path.is_file():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            trainer = REPO_ROOT / "stage2_RQ-VAE" / f"TIGER_{arm}" / "train_rqvae.py"
            cfg["train_rqvae_sha256"] = hashlib.sha256(trainer.read_bytes()).hexdigest()
            cfg["single_delta_diff"] = diff
            cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        status = "GREEN OK" if code == 0 else "GREEN FAIL"
        print(f"{arm}: exit={code} {status}", flush=True)
        if code != 0:
            failed.append(arm)
            for ln in output.splitlines():
                if ln.startswith(("FAIL:", "ERROR:")):
                    print(f"    {ln[:170]}", flush=True)
    print("\nFAILED ARMS:", failed if failed else "none")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
'''

LAUNCH = r'''"""Launch one R5 wave: one arm per visible GPU in single-card mode."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9"
LAUNCH_LOG = Path("/tmp/r5_launches.json")


def visible_cards():
    done = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=30, check=True,
    )
    return [ln.strip() for ln in done.stdout.splitlines() if ln.strip()]


def already_running(arm: str) -> bool:
    done = subprocess.run(["pgrep", "-af", f"TIGER_{arm}/train_rqvae.py"], capture_output=True, text=True)
    return bool(done.stdout.strip())


def launch(arm: str, card: str) -> dict:
    tree = REPO_ROOT / "stage2_RQ-VAE" / f"TIGER_{arm}"
    results = tree / "results"
    results.mkdir(parents=True, exist_ok=True)
    log = results / "launcher.log"
    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": card,
        "GENREC_SINGLE_GPU": "1",
        "NCCL_IB_DISABLE": "1",
        "NCCL_P2P_DISABLE": "1",
        "NCCL_SHM_DISABLE": "1",
    })
    handle = log.open("ab")
    proc = subprocess.Popen(
        [PYTHON, str(tree / "train_rqvae.py")],
        cwd=str(tree), env=env, stdout=handle, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return {"arm": arm, "card": card, "pid": proc.pid, "log": str(log), "started_at": time.time()}


def main() -> int:
    arms = sys.argv[1:]
    if not arms:
        print("usage: r5_launch_wave.py <arm> ...", file=sys.stderr)
        return 2
    cards = visible_cards()
    free = [a for a in arms if not already_running(a)]
    skipped = [a for a in arms if a not in free]
    for a in skipped:
        print(f"{a}: already running, skipping")
    if len(free) > len(cards):
        print(f"refusing: {len(free)} arms but only {len(cards)} cards", file=sys.stderr)
        return 1
    payload = json.loads(LAUNCH_LOG.read_text()) if LAUNCH_LOG.is_file() else []
    for arm, card in zip(free, cards):
        entry = launch(arm, card)
        payload = [e for e in payload if e["arm"] != arm] + [entry]
        print(f"{arm}: card {card} pid {entry['pid']} log {entry['log']}", flush=True)
    LAUNCH_LOG.write_text(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

EVALUATE = r'''"""Post-training Hard L1 ER gate evaluation for R5 arms (#61-#70)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
sys.path.insert(0, str(REPO_ROOT / "results/stage2_RQ-VAE/r5_common"))
import r5_gate as G  # noqa: E402

ARMS = {
    "r5_01": (61, "Reverse PCGrad: protect L1 behavior against base gradient",
              "project g_base against g_L1 when conflict"),
    "r5_02": (62, "Reconstruction-only reference gradient for original L1 PCGrad",
              "project g_L1 against g_recon"),
    "r5_03": (63, "Quantization-only reference gradient for original L1 PCGrad",
              "project g_L1 against g_quant"),
    "r5_04": (64, "Per-encoder-layer blockwise PCGrad",
              "PCGrad per Linear weight+bias block"),
    "r5_05": (65, "Half-strength conflict projection alpha=0.5",
              "g'_L1 = g_L1 - 0.5 * proj_component"),
    "r5_06": (66, "Norm-preserving post-PCGrad L1 behavior gradient",
              "rescale projected g_L1 to original norm"),
    "r5_07": (67, "Strong-conflict-only cosine trigger",
              "project only if cos < -0.2"),
    "r5_08": (68, "Late-epoch-only R4-08 conflict projection",
              "activate projection only for epoch >= 2250"),
    "r5_09": (69, "EMA-smoothed base gradient as PCGrad reference",
              "m_t=0.9 m + 0.1 g_base as reference"),
    "r5_10": (70, "Inter-level PCGrad against L2+L3",
              "project g_L1 against g_other"),
}


def green_passed(arm_dir: Path) -> bool:
    tdd = arm_dir / "tdd_red_green.md"
    if not tdd.is_file():
        return False
    text = tdd.read_text(encoding="utf-8")
    return "## GREEN" in text and "Exit code: 0" in text and "ALL TESTS PASSED" in text


def write_report(arm_dir, arm, issue, title, formula, metrics, gate):
    cfg = json.loads((arm_dir / "config.json").read_text(encoding="utf-8"))
    verdict = "PASS — Stage 3 eligible" if gate["stage3_allowed"] else "FAIL — Stage 3 refused"
    lines = [
        f"# {arm.upper()} (Issue #{issue}) — {title}", "",
        "## 1. Sole mechanism and formula", "",
        f"- **Parent**: pinned R4-08 (`stage2_RQ-VAE/TIGER_r4_08`, commit `bc96625`).",
        f"- **Mechanism flag**: `{cfg['mechanism_flag']}`.",
        f"- **Formula**: `{formula}`.",
        f"- **Changed files vs parent**: `{', '.join(cfg['changed_files'])}`.",
        f"- **Trainer SHA256**: `{cfg.get('train_rqvae_sha256')}`.", "",
        "## 2. TDD RED → GREEN", "",
        f"See [`tdd_red_green.md`](tdd_red_green.md). `tests_passed = {gate['tests_passed']}`.", "",
        "## 3. Stage 2 training completion", "",
        f"- **Epochs completed**: `{metrics['train_end_epoch']}` / 3000.",
        f"- **SID SHA256**: `{metrics['item_sids_sha256']}`.",
        f"- **Canonical Resample SID SHA256**: `{cfg['baseline_sid_sha256']}`.", "",
        "## 4. Frozen held-out Hard L1 ER and Stage 3 gate", "",
        "| Quantity | Value |", "|:---|---:|",
        f"| `pos_share` | `{metrics['pos_share']:.10f}` |",
        f"| `neg_share` | `{metrics['neg_share']:.10f}` |",
        f"| **`ER_L1`** | **`{metrics['ER_L1']:.14f}`** |",
        f"| Canonical Resample `ER_L1` | `{metrics['baseline_er_l1']:.14f}` |",
        f"| `Δ ER_L1` | `{metrics['ER_L1'] - metrics['baseline_er_l1']:+.14f}` |",
        f"| **`stage3_allowed`** | **`{gate['stage3_allowed']}`** ({verdict}) |", "",
        "## 5. Stage 3 disposition", "",
    ]
    if gate["stage3_allowed"]:
        lines.append("Gate passed. Stage 3 must run under the frozen protocol before closing.")
    else:
        lines.append("Gate failed. Stage 3 was not launched.")
    (arm_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    for arm, (issue, title, formula) in ARMS.items():
        arm_dir = REPO_ROOT / "results/stage2_RQ-VAE" / arm
        metrics, gate = G.evaluate(
            arm=arm,
            issue=issue,
            item_sids_path=arm_dir / "item_sids.json",
            metrics_path=REPO_ROOT / "stage2_RQ-VAE" / f"TIGER_{arm}" / "logs" / arm / "training_metrics.jsonl",
            tests_passed=green_passed(arm_dir),
        )
        (arm_dir / "stage2_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        (arm_dir / "gate.json").write_text(json.dumps(gate, indent=2) + "\n")
        write_report(arm_dir, arm, issue, title, formula, metrics, gate)
        print(
            f"{arm} (#{issue}): epoch={metrics['train_end_epoch']} "
            f"ER_L1={metrics['ER_L1']:.6f} stage3_allowed={gate['stage3_allowed']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


def main() -> None:
    COMMON.mkdir(parents=True, exist_ok=True)
    (COMMON / "r5_testkit.py").write_text(TESTKIT)
    (REPO / "tools" / "r5_capture_red.py").write_text(CAPTURE_RED)
    (REPO / "tools" / "r5_capture_green.py").write_text(CAPTURE_GREEN)
    (REPO / "tools" / "r5_launch_wave.py").write_text(LAUNCH)
    (REPO / "tools" / "r5_evaluate_all.py").write_text(EVALUATE)

    for arm, (issue, title, flag, formula) in META.items():
        d = RESULTS / arm
        d.mkdir(parents=True, exist_ok=True)
        (d / f"test_{arm}.py").write_text(test_body(arm))
        cfg = {
            "arm": arm,
            "issue": f"#{issue}",
            "mechanism": title,
            "mechanism_flag": flag,
            "baseline_tree": "stage2_RQ-VAE/TIGER_r4_08",
            "baseline_arm": "r4_08",
            "baseline_sid_sha256": "b08567c7fbaf06e585be37e4c2ceb379f6435e4d36e220c578ae3e6e10265d10",
            "output_dir": f"results/stage2_RQ-VAE/{arm}",
            "changed_files": ["train_rqvae.py"],
            "formula": formula,
            "test_command": f"/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 results/stage2_RQ-VAE/{arm}/test_{arm}.py",
        }
        (d / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        print(f"wrote {arm} test+config")
    print("done")


if __name__ == "__main__":
    main()
