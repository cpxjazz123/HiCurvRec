"""The routing table: its identity, its four modes, and the errors it must raise.

    python3 -m pytest stage2_RQ-VAE/TIGER_role_geom_rq/tests/test_role_routing.py
"""
from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path

import torch

TREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TREE))

from model.role_geometry import (ROLE_SPACES, SPACES, RoleTableError, build_weights,
                                 load_role_table, sha256_of)


def write_table(directory: Path, rows: list[tuple[int, int]]) -> Path:
    path = directory / "roles.csv"
    path.write_text("item_id,role\n" + "".join(f"{item},{role}\n" for item, role in rows), encoding="utf-8")
    return path


def test_the_table_is_accepted_only_with_its_frozen_hash() -> None:
    with tempfile.TemporaryDirectory(dir=str(TREE / "tests")) as tmp:
        path = write_table(Path(tmp), [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)])
        digest = sha256_of(path)
        roles = load_role_table(path, digest)
        assert roles == {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
        try:
            load_role_table(path, hashlib.sha256(b"other").hexdigest())
        except RoleTableError as exc:
            assert "sha256" in str(exc)
        else:
            raise AssertionError("a wrong hash must be refused")


def test_a_missing_table_and_an_unknown_role_are_errors() -> None:
    with tempfile.TemporaryDirectory(dir=str(TREE / "tests")) as tmp:
        directory = Path(tmp)
        try:
            load_role_table(directory / "absent.csv", "0" * 64)
        except RoleTableError as exc:
            assert "missing" in str(exc)
        else:
            raise AssertionError("a missing table must be refused")
        path = write_table(directory, [(0, 0), (1, 9)])
        try:
            load_role_table(path, sha256_of(path))
        except RoleTableError as exc:
            assert "unknown role" in str(exc)
        else:
            raise AssertionError("an unknown role must be refused")


def test_off_routes_everything_to_the_hyperbolic_branch() -> None:
    roles = {item: item % 6 for item in range(40)}
    weights = build_weights("OFF", 40, roles, list(range(40)))[0]
    assert weights.shape == (40, 3)
    assert torch.allclose(weights[:, SPACES.index("hyperbolic")], torch.ones(40))
    assert float(weights[:, SPACES.index("euclidean")].sum()) == 0.0
    assert float(weights[:, SPACES.index("spherical")].sum()) == 0.0


def test_role_routing_follows_the_frozen_mapping_and_sums_to_one() -> None:
    roles = {item: item % 6 for item in range(60)}
    weights = build_weights("ROLE", 60, roles, list(range(60)))[0]
    for role, space in ROLE_SPACES.items():
        index = SPACES.index(space)
        rows = weights[torch.arange(60) % 6 == role]
        assert torch.allclose(rows[:, index], torch.ones(len(rows)))
        assert float(rows.sum()) == float(len(rows))
    assert float(weights.min()) >= 0.0


def test_the_global_arm_uses_the_population_mean_over_the_training_items() -> None:
    roles = {item: item % 6 for item in range(60)}
    train_ids = list(range(0, 60, 2))          # only the even items are training items
    weights = build_weights("GLOBAL", 60, roles, train_ids)[0]
    assert torch.allclose(weights[0], weights[59])          # every row carries the same vector
    expected = torch.zeros(3, dtype=torch.float64)
    for item in train_ids:
        space = ROLE_SPACES[roles[item]]
        expected[SPACES.index(space)] += 1.0
    expected /= len(train_ids)
    assert torch.allclose(weights[0].double(), expected, atol=1e-6)
    assert abs(float(weights[0].sum()) - 1.0) < 1e-6


def test_the_shuffled_arm_keeps_the_histogram_and_moves_the_labels() -> None:
    roles = {item: item % 6 for item in range(60)}
    role_weights = build_weights("ROLE", 60, roles, list(range(60)))[0]
    shuffled = build_weights("SHUFFLED", 60, roles, list(range(60)))[0]
    # the routing histogram is the role arm's
    for index, space in enumerate(SPACES):
        assert abs(float(role_weights[:, index].sum()) - float(shuffled[:, index].sum())) < 1e-6
    # but the items are routed elsewhere
    assert float((role_weights == shuffled).all(dim=1).float().mean()) < 0.5
    # and the permutation is reproducible
    again = build_weights("SHUFFLED", 60, roles, list(range(60)))[0]
    assert torch.equal(shuffled, again)


def test_an_item_without_a_role_takes_the_documented_fallback() -> None:
    roles = {item: item % 6 for item in range(10)}
    weights, fallbacks = build_weights("ROLE", 20, roles, list(range(20)), fallback_role=0)
    assert fallbacks == 10                       # items 10..19 have no role
    index = SPACES.index(ROLE_SPACES[0])
    assert torch.allclose(weights[10:, index], torch.ones(10))
    assert torch.allclose(weights[:10].sum(dim=1), torch.ones(10))
