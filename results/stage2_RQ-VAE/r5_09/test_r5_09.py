"""R5 tests for r5_09 (issue #69). RED uses R5_TEST_TREE=TIGER_r4_08."""
from __future__ import annotations
import sys
from pathlib import Path
import torch
REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
sys.path.insert(0, str(REPO / "results/stage2_RQ-VAE/r5_common"))
import r5_testkit as K
ARM = "r5_09"
T = K.load_arm(ARM)
EPS = 1e-12

def _dot(a,b):
    return float((a*b).sum())

def test_train_only_data_audit():
    K.check_train_only(T)

def test_sid_three_token_contract():
    K.check_sid_contract(T)

def test_red_ema_flag_and_helper():
    assert abs(float(getattr(T, "R5_09_EMA_DECAY", 0)) - 0.9) < 1e-12, "EMA decay missing on parent"
    assert hasattr(T, "_r5_09_update_ema"), "EMA helper missing on parent"

def test_ema_two_step():
    assert hasattr(T, "R5_09_EMA_STATE") and hasattr(T, "_r5_09_update_ema"), (
        "EMA state/helper missing on parent"
    )
    T.R5_09_EMA_STATE["buffers"] = None
    m1 = T._r5_09_update_ema([torch.tensor([1.0, 0.0])])
    m2 = T._r5_09_update_ema([torch.tensor([0.0, 1.0])])
    assert torch.allclose(m2[0], torch.tensor([0.9, 0.1]), atol=1e-6)
    T.R5_09_EMA_STATE["buffers"] = None

def test_apply_uses_ema_reference():
    src = Path(T.__file__).read_text(encoding="utf-8")
    assert "g_ref = _r5_09_update_ema(g_base)" in src

if __name__ == "__main__":
    K.run(__name__, [test_red_ema_flag_and_helper, test_ema_two_step, test_apply_uses_ema_reference,
        test_train_only_data_audit, test_sid_three_token_contract])
