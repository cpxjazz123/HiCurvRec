"""The authoritative surface, the first contract call and the distance subface
must be the same text, because the compiler compares calls[0] to the main face."""
import json
from pathlib import Path

SPEC = json.loads((Path(__file__).resolve().parents[1] / "math_spec.json").read_text())
CONTRACT = json.loads((Path(__file__).resolve().parents[1] / "reference_contract.json").read_text())


def test_the_first_call_is_the_main_structure_face():
    surf = SPEC["continuous_surface"]
    assert CONTRACT["calls"][0]["name"] == "structure_objective"
    assert CONTRACT["calls"][0]["statements"] == surf["statements"]


def test_the_distance_subface_matches_the_distance_call():
    calls = {c["name"]: c for c in CONTRACT["calls"]}
    assert SPEC["continuous_surface"]["distance_surface"]["statements"] == \
        calls["mixed_distance"]["statements"]


if __name__ == "__main__":
    test_the_first_call_is_the_main_structure_face()
    test_the_distance_subface_matches_the_distance_call()
    print("ok")
