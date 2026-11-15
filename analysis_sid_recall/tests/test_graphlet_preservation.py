"""The same-node preservation test must react to a missing edge and to an extra one.

Constructed graphs only: the behaviour set defines the instance's structure, so a quantised graph that
drops an edge, or that adds an edge among the same items, has to fail the exact test.
"""

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "graphlet_preservation",
    HERE.parents[1] / "results" / "research" / "issue215" / "graphlet_preservation.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

TRIANGLE = {(0, 1), (1, 2), (0, 2)}
INSTANCE = [(0, 1, 2)]


def test_identical_graph_preserves():
    exact, share = module.preserved(TRIANGLE, TRIANGLE, INSTANCE)
    assert exact == 1.0 and share == 1.0


def test_missing_edge_breaks_exact_but_is_counted_in_share():
    quantised = TRIANGLE - {(0, 2)}
    exact, share = module.preserved(quantised, TRIANGLE, INSTANCE)
    assert exact == 0.0
    assert abs(share - 2 / 3) < 1e-12


def test_edge_outside_the_instance_is_ignored():
    quantised = set(TRIANGLE) | {(0, 3)}   # 3 is outside the instance, so it cannot matter
    exact, share = module.preserved(quantised, TRIANGLE, INSTANCE)
    assert exact == 1.0 and share == 1.0


def test_extra_edge_inside_the_instance_breaks_exact():
    behaviour = {(0, 1), (1, 2), (2, 3)}      # a path on four items
    quantised = behaviour | {(0, 2)}          # the quantised graph closes a triangle inside them
    exact, share = module.preserved(quantised, behaviour, [(0, 1, 2, 3)])
    assert exact == 0.0
    assert share == 1.0


def test_making_an_edge_among_instance_items_breaks_exact():
    behaviour = {(0, 1)}          # the instance is a single edge
    quantised = {(0, 1), (1, 2)}  # the quantised graph closes a triangle on the same items
    exact, share = module.preserved(quantised, behaviour, [(0, 1, 2)])
    assert exact == 0.0
    assert share == 1.0


if __name__ == "__main__":
    test_identical_graph_preserves()
    test_missing_edge_breaks_exact_but_is_counted_in_share()
    test_edge_outside_the_instance_is_ignored()
    test_extra_edge_inside_the_instance_breaks_exact()
    test_making_an_edge_among_instance_items_breaks_exact()
    print("graphlet-preservation tests pass")
