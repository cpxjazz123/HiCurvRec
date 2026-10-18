"""R5 tests for r5_10 (issue #70). RED uses R5_TEST_TREE=TIGER_r4_08."""
from __future__ import annotations
import sys
from pathlib import Path
import torch
REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
sys.path.insert(0, str(REPO / "results/stage2_RQ-VAE/r5_common"))
import r5_testkit as K
ARM = "r5_10"
T = K.load_arm(ARM)
EPS = 1e-12

def _dot(a,b):
    return float((a*b).sum())

def test_train_only_data_audit():
    K.check_train_only(T)

def test_sid_three_token_contract():
    K.check_sid_contract(T)

def test_red_interlevel_flag():
    assert getattr(T, "R5_10_INTERLEVEL", False), "inter-level flag absent on parent"

def test_red_apply_projects_against_other():
    src = Path(T.__file__).read_text(encoding="utf-8")
    assert "flat_ref = torch.cat([grad.reshape(-1) for grad in g_other])" in src

def test_project_against_other_math():
    g_l1 = torch.tensor([-1.0, 0.0]); g_other = torch.tensor([1.0, 0.0])
    out = T.r4_project_conflict(g_l1.clone(), g_other.clone())
    assert abs(_dot(out, g_other)) < 1e-5

if __name__ == "__main__":
    K.run(__name__, [test_red_interlevel_flag, test_red_apply_projects_against_other,
        test_project_against_other_math, test_train_only_data_audit, test_sid_three_token_contract])
