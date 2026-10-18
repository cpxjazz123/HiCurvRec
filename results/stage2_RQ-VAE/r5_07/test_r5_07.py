"""R5 tests for r5_07 (issue #67). RED uses R5_TEST_TREE=TIGER_r4_08."""
from __future__ import annotations
import sys
from pathlib import Path
import torch
REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
sys.path.insert(0, str(REPO / "results/stage2_RQ-VAE/r5_common"))
import r5_testkit as K
ARM = "r5_07"
T = K.load_arm(ARM)
EPS = 1e-12

def _dot(a,b):
    return float((a*b).sum())

def test_train_only_data_audit():
    K.check_train_only(T)

def test_sid_three_token_contract():
    K.check_sid_contract(T)

def test_red_weak_conflict_skipped():
    g_base = torch.tensor([1.0, 0.0]); g_l1 = torch.tensor([-0.1, 1.0])
    out = T.r4_project_conflict(g_l1.clone(), g_base.clone())
    assert torch.allclose(out, g_l1, atol=1e-7), "weak conflict projected (parent would project)"

def test_red_strong_conflict_projected():
    g_base = torch.tensor([1.0, 0.0]); g_l1 = torch.tensor([-1.0, 0.5])
    out = T.r4_project_conflict(g_l1.clone(), g_base.clone())
    expected = g_l1 - (_dot(g_l1,g_base)/(_dot(g_base,g_base)+EPS))*g_base
    assert torch.allclose(out, expected, atol=1e-6)

def test_threshold_flag():
    assert abs(float(getattr(T, "R5_07_COS_THRESHOLD", 0.0)) + 0.2) < 1e-12

if __name__ == "__main__":
    K.run(__name__, [test_red_weak_conflict_skipped, test_red_strong_conflict_projected, test_threshold_flag,
        test_train_only_data_audit, test_sid_three_token_contract])
