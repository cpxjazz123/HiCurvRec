"""The declared domain is enforced: out-of-domain codewords and degenerate
scales are rejected, not silently evaluated."""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.research.stage2.reference import build_reference

import structure as S

D = torch.float64
CONTRACT = json.loads((Path(__file__).resolve().parents[1] / "reference_contract.json").read_text())
CALL = {c["name"]: c for c in CONTRACT["calls"]}["prefix_surface_1d"]


def test_a_codeword_outside_the_h_domain_is_rejected():
    h = torch.ones(1, 3, dtype=D)
    s = torch.ones(1, 1, dtype=D)
    codes = [torch.zeros(1, 3, 32, dtype=D) for _ in range(3)]
    codes[0][0, 0, 16] = 2.0
    try:
        S.reference_prefix_native(codes, h, s)
    except ValueError:
        pass
    else:
        raise AssertionError("a codeword with H norm 2 was accepted")


def test_degenerate_h_and_s_are_rejected_by_the_reference():
    ref, _ = build_reference(CALL)
    codes = (torch.rand(1, 18, dtype=D) - 0.5) * 0.1
    s = torch.ones(1, 1, dtype=D)
    bad_h = torch.ones(1, 3, dtype=D)
    bad_h[0, 1] = 0.0
    for args in (dict(codes=codes, h=bad_h, s=s),
                 dict(codes=codes, h=torch.ones(1, 3, dtype=D), s=torch.zeros(1, 1, dtype=D))):
        try:
            ref(**args)
        except ValueError:
            pass
        else:
            raise AssertionError("a degenerate h/s was accepted")


if __name__ == "__main__":
    test_a_codeword_outside_the_h_domain_is_rejected()
    test_degenerate_h_and_s_are_rejected_by_the_reference()
    print("ok")
