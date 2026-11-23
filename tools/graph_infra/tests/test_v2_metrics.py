"""v2 metric acceptance tests (issue #246 C24-C27)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import reference_hand as ref  # noqa: E402
from tools.graph_infra import metrics  # noqa: E402


def test_subgraph_summary_distortion_known_answer():
    report = metrics.subgraph_summary([np.array([0.3, 0.4, 0.8])], threshold=0.5,
                                      direction="minimize")
    assert abs(report["subgraph_pass_fraction"] - 2 / 3) < 1e-12
    assert report["center_all_pass_fraction"] == 0.0
    assert report["center_any_pass_fraction"] == 1.0
    assert abs(report["worst"] - 0.8) < 1e-12
    assert abs(report["p90"] - 0.72) < 1e-12
    assert report["centers_without_valid_subgraph"] == 0


def test_subgraph_summary_matches_the_reference_on_random_inputs():
    rng = np.random.default_rng(7)
    values = [rng.random(4), rng.random(3), rng.random(1)]
    for direction, threshold in (("minimize", 0.5), ("maximize", 0.5)):
        produced = metrics.subgraph_summary(values, threshold=threshold, direction=direction)
        expected = ref.subgraph_summary(values, threshold, direction)
        for key in ("subgraph_pass_fraction", "center_all_pass_fraction",
                    "center_any_pass_fraction", "worst", "p90"):
            assert abs(produced[key] - expected[key]) < 1e-12, (key, produced[key], expected[key])


def test_subgraph_summary_counts_centres_without_valid_subgraphs():
    report = metrics.subgraph_summary([np.array([]), np.array([0.9])], threshold=0.5,
                                      direction="minimize")
    assert report["centers_without_valid_subgraph"] == 1
    assert report["subgraphs"] == 1
    assert abs(report["subgraph_pass_fraction"] - 0.0) < 1e-12


def test_prefix_at_k_requires_both_lengths():
    pairs = [((1,), (1, 2, 3)), ((1, 2, 3), (1, 9, 3))]
    assert metrics.prefix_at_k(pairs, 2) == 0.0, "a code shorter than k cannot share a 2-prefix"
    pairs = [((1, 2), (1, 2, 3)), ((1, 2, 3), (1, 9, 3))]
    assert metrics.prefix_at_k(pairs, 2) == 0.5


def test_collision_and_duplicate_prefix_vs_full():
    codes = [[1, 2, 3, 0], [1, 2, 3, 1], [1, 2, 4, 0]]
    report = metrics.sid_structure(codes)
    expected = ref.collision_duplicate(codes)
    assert report["distinct_prefix3"] == 2
    assert report["distinct_full"] == 3, "the disambiguation token separates the first two"
    assert abs(report["collision_prefix3"] - 1 / 3) < 1e-12
    assert abs(report["duplicate_full"] - 0.0) < 1e-12
    for key in ("collision_prefix3", "duplicate_prefix3", "collision_full", "duplicate_full"):
        assert abs(report[key] - expected[key]) < 1e-12, key


def test_sid_audit_reports_every_defect():
    table = {0: [1, 2, 3, 0], 1: [1, 2, 3, 0], 2: [1, 2], 3: [1, 2, 3, 999]}
    report = metrics.sid_audit(table, item_ids=[0, 1, 2, 3, 4], width=4, max_token=255)
    assert report["rows"] == 4
    assert report["missing_items"] == [4]
    assert report["bad_length"] == [2]
    assert report["out_of_range"] == [3]
    assert report["duplicate_rows"] == [1]
    assert report["ok"] is False


def test_sid_audit_accepts_a_valid_table():
    table = {0: [1, 2, 3, 0], 1: [4, 5, 6, 1]}
    report = metrics.sid_audit(table, item_ids=[0, 1], width=4, max_token=255)
    assert report["ok"] is True and report["missing_items"] == [] and report["duplicate_rows"] == []


def test_entropy_states_its_unit():
    report = metrics.codebook_stats(np.array([[0, 0], [0, 0], [1, 0]]), unit="bits")
    assert report[0]["entropy_unit"] == "bits"
    assert abs(report[0]["entropy"] - 0.9182958340544896) < 1e-12


def main() -> int:
    tests = [(name, value) for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    failures = []
    for name, function in tests:
        try:
            function()
            print(f"PASS {name}")
        except Exception as error:  # noqa: BLE001
            failures.append((name, str(error)))
            print(f"FAIL {name}: {type(error).__name__}: {error}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
