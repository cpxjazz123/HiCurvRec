"""The Teacher artefact: save -> freeze -> reload -> identical predictions.

The arm checkpoint carries only the encoder and quantiser, so without this
contract the jointly trained head died with the process.  These tests pin the
three properties the Teacher is for: it is frozen after loading, it predicts
bitwise the same as before saving, and a mismatched vocabulary is refused.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import teacher  # noqa: E402
from directed_role import DirectedTypeRoleHead  # noqa: E402

VOCAB = {"mode": "directed", "n_subtypes": 7, "n_orbits": 11}


def make_head(seed: int = 7) -> DirectedTypeRoleHead:
    torch.manual_seed(seed)
    return DirectedTypeRoleHead(dim=8, hidden=16, n_subtypes=VOCAB["n_subtypes"],
                                n_orbits=VOCAB["n_orbits"])


def batch(seed: int = 3, instances: int = 5, width: int = 3):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(instances, width, 8, generator=generator)


def test_round_trip_predictions_are_identical(tmp_path):
    head = make_head()
    torch.save({"probe": head.state_dict()}, tmp_path / "unused.pt")   # keep the seed honest
    teacher.save_teacher(tmp_path / "teacher.pt", head, vocab=VOCAB)
    reloaded, payload = teacher.load_teacher(tmp_path / "teacher.pt")
    assert payload["config"] == teacher.teacher_config(head)

    x = batch()
    with torch.no_grad():
        type_before, role_before = head(x)
        type_after, role_after = reloaded(x)
    assert torch.equal(type_before, type_after)
    assert torch.equal(role_before, role_after)


def test_loaded_teacher_is_frozen(tmp_path):
    head = make_head()
    teacher.save_teacher(tmp_path / "teacher.pt", head, vocab=VOCAB)
    reloaded, _ = teacher.load_teacher(tmp_path / "teacher.pt")
    assert not reloaded.training
    assert all(not parameter.requires_grad for parameter in reloaded.parameters())

    before = {name: tensor.clone() for name, tensor in reloaded.state_dict().items()}
    optimizer = torch.optim.SGD(reloaded.parameters(), lr=0.1)
    x = batch()
    type_logits, role_logits = reloaded(x)
    # A frozen Teacher produces no autograd graph at all: that is the property,
    # not merely "gradients happen to be zero".
    assert not type_logits.requires_grad
    assert not role_logits.requires_grad
    with pytest.raises(RuntimeError):
        (type_logits.sum() + role_logits.sum()).backward()
    optimizer.step()                      # nothing to step: parameters must not move
    after = reloaded.state_dict()
    for name, tensor in before.items():
        assert torch.equal(tensor, after[name]), f"{name} moved after a frozen step"


def test_vocabulary_mismatch_is_refused(tmp_path):
    head = make_head()
    teacher.save_teacher(tmp_path / "teacher.pt", head, vocab=VOCAB)
    with pytest.raises(ValueError, match="vocabulary mismatch"):
        teacher.load_teacher(tmp_path / "teacher.pt",
                             expect_vocab={"mode": "undirected", "n_subtypes": 5, "n_orbits": 4})


def test_bare_state_dict_is_refused(tmp_path):
    head = make_head()
    torch.save(head.state_dict(), tmp_path / "bare.pt")
    with pytest.raises(ValueError, match="not a teacher payload"):
        teacher.load_teacher(tmp_path / "bare.pt")


def test_class_count_disagreement_is_refused_at_save(tmp_path):
    head = make_head()
    with pytest.raises(ValueError, match="disagree on n_subtypes"):
        teacher.save_teacher(tmp_path / "teacher.pt", head,
                             vocab={"mode": "directed", "n_subtypes": 5, "n_orbits": 11})


def test_legacy_pretrained_head_migrates(tmp_path):
    """The shipped artefact is a bare state_dict; wrapping must not change it."""
    legacy = Path(__file__).resolve().parents[3] / "results" / "research" / \
        "directed_graphlet_role" / "head_pretrained_directed.pt"
    if not legacy.is_file():
        pytest.skip("legacy pretrained head not present")
    state = torch.load(legacy, map_location="cpu")
    config = {"dim": 32, "hidden": 96, "n_subtypes": 47, "n_orbits": 376, "interaction": True}
    legacy_head = teacher.build_head(config)
    legacy_head.load_state_dict(state, strict=True)
    payload = teacher.wrap_state_dict(state, config=config,
                                      vocab={"mode": "directed", "n_subtypes": 47,
                                             "n_orbits": 376})
    torch.save(payload, tmp_path / "legacy_teacher.pt")
    reloaded, _ = teacher.load_teacher(tmp_path / "legacy_teacher.pt")

    x = torch.randn(4, 4, 32, generator=torch.Generator().manual_seed(11))
    with torch.no_grad():
        before = legacy_head(x)
        after = reloaded(x)
    assert torch.equal(before[0], after[0])
    assert torch.equal(before[1], after[1])


def test_trainer_helper_writes_a_loadable_frozen_teacher(tmp_path):
    """The exact helper the trainer calls must produce a reloadable Teacher."""
    head = make_head()
    path, _payload = teacher.save_arm_teacher(
        tmp_path, head, vocab=VOCAB, arm="D1", label_mode="directed",
        heldout={"subtype": {"macro_f1": 0.5}, "role": {"macro_over_families": 0.25}})
    assert path == tmp_path / "teacher.pt"
    reloaded, payload = teacher.load_teacher(path, expect_vocab=VOCAB)
    assert payload["extra"]["arm"] == "D1"
    assert payload["extra"]["heldout_orbit_macro_over_families"] == 0.25
    x = batch()
    with torch.no_grad():
        assert torch.equal(head(x)[0], reloaded(x)[0])
    assert all(not parameter.requires_grad for parameter in reloaded.parameters())
