"""R5 tests for r5_05 (issue #65). RED uses R5_TEST_TREE=TIGER_r4_08."""
from __future__ import annotations
import sys
from pathlib import Path
import torch
REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
sys.path.insert(0, str(REPO / "results/stage2_RQ-VAE/r5_common"))
import r5_testkit as K
ARM = "r5_05"
T = K.load_arm(ARM)
EPS = 1e-12

def _dot(a,b):
    return float((a*b).sum())

def test_train_only_data_audit():
    K.check_train_only(T)

def test_sid_three_token_contract():
    K.check_sid_contract(T)

def test_red_half_strength_vector():
    g_l1 = torch.tensor([-1.0, 2.0]); g_base = torch.tensor([1.0, 0.0])
    out = T.r4_project_conflict(g_l1.clone(), g_base.clone())
    expected = torch.tensor([-0.5, 2.0])
    assert torch.allclose(out, expected, atol=1e-6), (
        f"got {out.tolist()} expected {expected.tolist()} (parent full proj ~[0,2])"
    )

def test_red_alpha_flag():
    assert abs(float(getattr(T, "R5_05_ALPHA", 1.0)) - 0.5) < 1e-12

def test_nonconflict_unchanged():
    g_l1 = torch.tensor([0.0, 2.0]); g_base = torch.tensor([1.0, 0.0])
    assert torch.allclose(T.r4_project_conflict(g_l1.clone(), g_base.clone()), g_l1, atol=1e-7)

if __name__ == "__main__":
    K.run(__name__, [test_red_half_strength_vector, test_red_alpha_flag, test_nonconflict_unchanged,
        test_train_only_data_audit, test_sid_three_token_contract])
