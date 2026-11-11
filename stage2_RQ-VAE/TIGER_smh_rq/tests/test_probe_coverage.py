"""The probe must label coverage honestly: no substitution, no silent fallback."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import importlib.util
spec = importlib.util.spec_from_file_location("probe", str(Path(__file__).resolve().parent / "probe_collision_gradients.py"))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_shared_and_different_pairs_are_labelled_by_full_sid():
    tokens = torch.tensor([[1, 2, 3], [1, 2, 3], [4, 5, 6]])
    pairs = probe.classify_pairs(tokens)
    assert pairs["shared_sid"][0:2] == (0, 1) and pairs["shared_sid"][2] == "shared_sid"
    assert pairs["different_sid"][2] == "different_sid"
    assert pairs["not_covered"] == {}


def test_no_shared_sid_is_not_covered_not_substituted():
    tokens = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    pairs = probe.classify_pairs(tokens)
    assert pairs["shared_sid"] is None
    assert "shared_sid" in pairs["not_covered"]


def test_two_items_with_one_shared_sid_cover_shared_only():
    # two different items, same full SID: a real collision, so shared_sid is COVERED
    tokens = torch.tensor([[1, 2, 3], [1, 2, 3]])
    pairs = probe.classify_pairs(tokens)
    assert pairs["shared_sid"][0:2] == (0, 1)
    assert "shared_sid" not in pairs["not_covered"]
    assert pairs["different_sid"] is None
    assert "different_sid" in pairs["not_covered"]


def test_fewer_than_two_items_leaves_both_not_covered():
    tokens = torch.tensor([[1, 2, 3]])
    pairs = probe.classify_pairs(tokens)
    assert pairs["shared_sid"] is None and pairs["different_sid"] is None
    assert set(pairs["not_covered"]) == {"shared_sid", "different_sid"}


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok   ", name)
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print("FAIL ", name, type(exc).__name__, exc)
    raise SystemExit(1 if failures else 0)
