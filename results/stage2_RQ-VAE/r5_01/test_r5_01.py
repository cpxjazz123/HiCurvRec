"""R5 tests for r5_01 (issue #61). RED uses R5_TEST_TREE=TIGER_r4_08."""
from __future__ import annotations
import sys
from pathlib import Path
import torch
REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
sys.path.insert(0, str(REPO / "results/stage2_RQ-VAE/r5_common"))
import r5_testkit as K
ARM = "r5_01"
T = K.load_arm(ARM)
EPS = 1e-12

def _dot(a,b):
    return float((a*b).sum())

def test_train_only_data_audit():
    K.check_train_only(T)

def test_sid_three_token_contract():
    K.check_sid_contract(T)

def test_red_reverse_flag_active():
    assert getattr(T, "R5_01_REVERSE_PCGRAD", False), (
        "reverse PCGrad flag absent — parent still projects L1 against base"
    )

def test_red_apply_projects_base_operand():
    src = Path(T.__file__).read_text(encoding="utf-8")
    assert "r4_project_conflict(flat_base, flat_l1)" in src, (
        "apply_projection does not project g_base against g_L1"
    )
    assert "g_base_proj" in src

def test_helper_reverse_call_orthogonalizes_base():
    g_base = torch.tensor([1.0, 0.0])
    g_l1 = torch.tensor([-1.0, 2.0])
    out = T.r4_project_conflict(g_base.clone(), g_l1.clone())
    assert abs(_dot(out, g_l1)) < 1e-5, out

if __name__ == "__main__":
    K.run(__name__, [test_red_reverse_flag_active, test_red_apply_projects_base_operand,
        test_helper_reverse_call_orthogonalizes_base, test_train_only_data_audit, test_sid_three_token_contract])
