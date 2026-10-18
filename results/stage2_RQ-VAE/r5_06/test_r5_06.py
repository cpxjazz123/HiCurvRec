"""R5 tests for r5_06 (issue #66). RED uses R5_TEST_TREE=TIGER_r4_08."""
from __future__ import annotations
import sys
from pathlib import Path
import torch
REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
sys.path.insert(0, str(REPO / "results/stage2_RQ-VAE/r5_common"))
import r5_testkit as K
ARM = "r5_06"
T = K.load_arm(ARM)
EPS = 1e-12

def _dot(a,b):
    return float((a*b).sum())

def test_train_only_data_audit():
    K.check_train_only(T)

def test_sid_three_token_contract():
    K.check_sid_contract(T)

def test_red_norm_restored():
    g_l1 = torch.tensor([-1.0, 2.0]); g_base = torch.tensor([1.0, 0.0])
    out = T.r4_project_conflict(g_l1.clone(), g_base.clone())
    assert abs(_dot(out, g_base)) < 1e-5
    assert abs(float(torch.linalg.vector_norm(out)) - float(torch.linalg.vector_norm(g_l1))) < 1e-5, (
        f"norm not restored; parent leaves ||proj||=2, got {float(torch.linalg.vector_norm(out))}"
    )

def test_antiparallel_zero():
    out = T.r4_project_conflict(torch.tensor([-2.0, 0.0]), torch.tensor([1.0, 0.0]))
    assert torch.allclose(out, torch.zeros(2), atol=1e-7)

def test_flag():
    assert getattr(T, "R5_06_NORM_PRESERVE", False)

if __name__ == "__main__":
    K.run(__name__, [test_red_norm_restored, test_antiparallel_zero, test_flag,
        test_train_only_data_audit, test_sid_three_token_contract])
