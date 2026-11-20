"""The Teacher is only usable if a reload proves it is the same frozen function.

Issue #222 A: a `state_dict` without its architecture, vocabulary and SHA cannot
be checked by a loader, and a teacher whose parameters still receive gradient is
not frozen.  These tests pin the contract:

  * round-trip logits identical (the same batch, same device),
  * every parameter `requires_grad_(False)` and `grad` None after a backward,
  * subtype logits permutation invariant, orbit logits permutation equivariant,
  * a payload whose vocabulary disagrees with its class counts is refused,
  * a truncated payload is refused instead of silently yielding a random head.
"""
from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from teacher_model import (TeacherFormatError, build_teacher,  # noqa: E402
                           load_frozen_teacher, save_teacher)

#: Scratch lives inside the repository (CLAUDE.md 86): no /tmp artefacts.
TMP = HERE / "_scratch"
TMP.mkdir(parents=True, exist_ok=True)


def _payload(n_subtypes: int = 5, n_orbits: int = 7):
    torch.manual_seed(3)
    teacher = build_teacher({"dim": 8, "hidden": 16, "layers": 2,
                             "n_subtypes": n_subtypes, "n_orbits": n_orbits})
    return teacher, {"dim": 8, "hidden": 16, "layers": 2,
                     "n_subtypes": n_subtypes, "n_orbits": n_orbits}


def _batch(seed: int = 11, count: int = 4, width: int = 3):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(count, width, 8, generator=generator)
    adjacency = (torch.rand(count, width, width, generator=generator) < 0.4).float()
    adjacency[:, torch.arange(width), torch.arange(width)] = 0.0
    return x, adjacency


def test_round_trip_logits_are_identical(tmp_path=None):
    teacher, config = _payload()
    target = TMP / "roundtrip.pt"
    record = save_teacher(target, teacher, config=config, vocab={"mode": "directed"})
    assert record["sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    x, adjacency = _batch()
    with torch.no_grad():
        before_type, before_orbit, _ = teacher(x, adjacency)
    reloaded, loaded = load_frozen_teacher(target)
    assert loaded["config"] == config
    with torch.no_grad():
        after_type, after_orbit, _ = reloaded(x, adjacency)
    assert torch.equal(before_type, after_type), "subtype logits changed on reload"
    assert torch.equal(before_orbit, after_orbit), "orbit logits changed on reload"
    assert record["probe_identical"] is True


def test_teacher_is_frozen_and_receives_no_gradient():
    teacher, config = _payload()
    target = TMP / "frozen.pt"
    save_teacher(target, teacher, config=config, vocab={"mode": "directed"})
    reloaded, _ = load_frozen_teacher(target)
    assert not reloaded.training
    assert all(not parameter.requires_grad for parameter in reloaded.parameters())
    x, adjacency = _batch()
    x.requires_grad_(True)
    type_logits, orbit_logits, hidden = reloaded(x, adjacency)
    (type_logits.sum() + orbit_logits.sum() + hidden.sum()).backward()
    assert x.grad is not None and x.grad.abs().sum() > 0, "the input path was cut"
    for name, parameter in reloaded.named_parameters():
        assert parameter.grad is None, f"{name} received gradient while frozen"


def test_subtype_is_permutation_invariant_and_orbit_equivariant():
    teacher = build_teacher({"dim": 8, "hidden": 16, "layers": 2,
                             "n_subtypes": 5, "n_orbits": 7})
    teacher.eval()
    x, adjacency = _batch(seed=5)
    permutation = torch.tensor([2, 0, 1])
    x_perm = x[:, permutation]
    adjacency_perm = adjacency[:, permutation][:, :, permutation]
    with torch.no_grad():
        type_a, orbit_a, hidden_a = teacher(x, adjacency)
        type_b, orbit_b, hidden_b = teacher(x_perm, adjacency_perm)
    assert torch.allclose(type_a, type_b, atol=1e-6), "subtype readout is not invariant"
    assert torch.allclose(orbit_a[:, permutation], orbit_b, atol=1e-6), \
        "orbit readout is not equivariant"
    assert torch.allclose(hidden_a[:, permutation], hidden_b, atol=1e-6), \
        "node hidden states are not equivariant"


def test_vocabulary_disagreement_is_refused():
    teacher, config = _payload(n_subtypes=5, n_orbits=7)
    target = TMP / "mismatch.pt"
    save_teacher(target, teacher, config=config, vocab={"mode": "directed"})
    with pytest.raises(TeacherFormatError):
        load_frozen_teacher(target, expect_vocab={"mode": "directed",
                                                  "n_subtypes": 6, "n_orbits": 7})
    with pytest.raises(TeacherFormatError):
        load_frozen_teacher(target, expect_vocab={"mode": "undirected",
                                                  "n_subtypes": 5, "n_orbits": 7})


def test_truncated_payload_is_refused():
    teacher, config = _payload()
    target = TMP / "truncated.pt"
    save_teacher(target, teacher, config=config, vocab={"mode": "directed"})
    truncated = TMP / "truncated_broken.pt"
    truncated.write_bytes(target.read_bytes()[:target.stat().st_size // 3])
    with pytest.raises(TeacherFormatError):
        load_frozen_teacher(truncated)


def test_reloading_does_not_modify_the_checkpoint():
    teacher, config = _payload()
    target = TMP / "stable.pt"
    save_teacher(target, teacher, config=config, vocab={"mode": "directed"})
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    load_frozen_teacher(target)
    load_frozen_teacher(target)
    assert hashlib.sha256(target.read_bytes()).hexdigest() == digest


def teardown_module(module):
    shutil.rmtree(TMP, ignore_errors=True)
