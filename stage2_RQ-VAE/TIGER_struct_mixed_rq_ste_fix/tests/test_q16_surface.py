"""The DRAFT surface must match the executable adapter, and the prefix draft must
keep the symmetric stress rather than a one-sided hinge."""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import structure as S

SPEC = json.loads((Path(__file__).resolve().parents[1] / "math_spec.json").read_text())


def test_the_args_match_the_adapter_widths_and_the_callable_is_importable():
    surf = SPEC["continuous_surface"]
    widths = [a["width"] for a in surf["args"]]
    assert widths == [6, 3, 1]
    assert sum(widths) == len(surf["symbols"]) == 10
    module, name = surf["callable"].split(":")
    assert hasattr(__import__(module), name)


def test_coincident_points_give_one_not_zero():
    points = torch.zeros(1, 6, dtype=torch.float64)
    u = torch.zeros(1, 3, dtype=torch.float64)
    s = torch.ones(1, dtype=torch.float64)
    loss = S.reference_objective_surface(points, u, s)
    assert abs(float(loss) - 1.0) < 1e-9


def test_the_prefix_draft_keeps_one_scale_and_no_next_item_target():
    prefix = SPEC["continuous_surface"]["prefix_surface"]
    assert prefix["callable"] == "NOT_IMPLEMENTED"
    text = " ".join(prefix["statements"])
    assert "next_item" not in text and "s1" not in text
    assert "L_prefix = (L1 + L2 + L3) / 3" in text


if __name__ == "__main__":
    test_the_args_match_the_adapter_widths_and_the_callable_is_importable()
    test_coincident_points_give_one_not_zero()
    test_the_prefix_draft_keeps_one_scale_and_no_next_item_target()
    print("ok")
