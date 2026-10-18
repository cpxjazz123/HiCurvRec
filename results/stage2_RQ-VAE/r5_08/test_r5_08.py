"""R5 tests for r5_08 (issue #68). RED uses R5_TEST_TREE=TIGER_r4_08."""
from __future__ import annotations
import sys
from pathlib import Path
import torch
REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
sys.path.insert(0, str(REPO / "results/stage2_RQ-VAE/r5_common"))
import r5_testkit as K
ARM = "r5_08"
T = K.load_arm(ARM)
EPS = 1e-12

def _dot(a,b):
    return float((a*b).sum())

def test_train_only_data_audit():
    K.check_train_only(T)

def test_sid_three_token_contract():
    K.check_sid_contract(T)

def test_red_late_start_flag():
    assert getattr(T, "R5_08_LATE_START_EPOCH", None) == 2250, (
        "late-start flag missing/wrong on parent"
    )

def test_red_loop_gates_projection():
    src = Path(T.__file__).read_text(encoding="utf-8")
    assert "epoch >= R5_08_LATE_START_EPOCH" in src

def test_projection_math_unchanged():
    g_l1 = torch.tensor([-1.0, 2.0]); g_base = torch.tensor([1.0, 0.0])
    out = T.r4_project_conflict(g_l1.clone(), g_base.clone())
    expected = g_l1 - (_dot(g_l1,g_base)/(_dot(g_base,g_base)+EPS))*g_base
    assert torch.allclose(out, expected, atol=1e-6)

if __name__ == "__main__":
    K.run(__name__, [test_red_late_start_flag, test_red_loop_gates_projection, test_projection_math_unchanged,
        test_train_only_data_audit, test_sid_three_token_contract])
