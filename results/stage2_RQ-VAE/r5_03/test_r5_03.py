"""R5 tests for r5_03 (issue #63). RED uses R5_TEST_TREE=TIGER_r4_08."""
from __future__ import annotations
import sys
from pathlib import Path
import torch
REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
sys.path.insert(0, str(REPO / "results/stage2_RQ-VAE/r5_common"))
import r5_testkit as K
ARM = "r5_03"
T = K.load_arm(ARM)
EPS = 1e-12

def _dot(a,b):
    return float((a*b).sum())

def test_train_only_data_audit():
    K.check_train_only(T)

def test_sid_three_token_contract():
    K.check_sid_contract(T)

def test_red_quant_flag_active():
    assert getattr(T, "R5_03_QUANT_REF", False), "quant-reference flag absent on parent"

def test_red_apply_builds_quant_reference():
    src = Path(T.__file__).read_text(encoding="utf-8")
    assert "quant_loss = base_loss - recon_loss" in src
    assert "recon_loss=recon_loss" in src

def test_project_formula():
    g_l1 = torch.tensor([-1.0, 2.0]); g_ref = torch.tensor([1.0, 0.0])
    out = T.r4_project_conflict(g_l1.clone(), g_ref.clone())
    expected = g_l1 - (_dot(g_l1,g_ref)/(_dot(g_ref,g_ref)+EPS))*g_ref
    assert torch.allclose(out, expected, atol=1e-6)

if __name__ == "__main__":
    K.run(__name__, [test_red_quant_flag_active, test_red_apply_builds_quant_reference,
        test_project_formula, test_train_only_data_audit, test_sid_three_token_contract])
