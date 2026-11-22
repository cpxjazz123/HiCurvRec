"""The directed Teacher contract: save, freeze, reload, identical predictions.

The Stage 2 arms train a `DirectedTypeRoleHead` jointly, but the arm checkpoint
holds only the encoder and quantiser (19 keys), so the trained head used to be
lost with the process.  This module owns the one artefact that must survive:

    save_teacher(path, head, vocab=...)   -> a payload with format, config,
                                             vocabulary and state_dict
    load_teacher(path)                    -> eval() + requires_grad_(False)

`load_teacher` refuses anything that is not a teacher payload and refuses a
vocabulary whose class counts disagree with the saved head, so a mismatched head
cannot be scored silently.  Prediction identity is asserted by the tests, not
assumed: the same batch must produce bitwise-equal logits before and after a
save/load round trip.
"""
from __future__ import annotations

from pathlib import Path

import torch

from directed_role import DirectedTypeRoleHead

TEACHER_FORMAT = 1


def teacher_config(head: DirectedTypeRoleHead) -> dict:
    """Dimensions read off the module, so the payload cannot drift from it."""
    return {
        "dim": int(head.phi[0].in_features),
        "hidden": int(head.phi[0].out_features),
        "n_subtypes": int(head.type_head[-1].out_features),
        "n_orbits": int(head.role_head[-1].out_features),
        "interaction": bool(head.interaction),
    }


def freeze(head: DirectedTypeRoleHead) -> DirectedTypeRoleHead:
    """Inference mode and no gradient: a frozen Teacher must not move."""
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    return head


def build_head(config: dict) -> DirectedTypeRoleHead:
    return DirectedTypeRoleHead(dim=int(config["dim"]), hidden=int(config["hidden"]),
                                n_subtypes=int(config["n_subtypes"]),
                                n_orbits=int(config["n_orbits"]),
                                interaction=bool(config["interaction"]))


def save_teacher(path, head: DirectedTypeRoleHead, *, vocab: dict, extra: dict | None = None) -> dict:
    config = teacher_config(head)
    recorded = {"mode": vocab.get("mode"),
                "n_subtypes": int(vocab["n_subtypes"]),
                "n_orbits": int(vocab["n_orbits"])}
    for key in ("n_subtypes", "n_orbits"):
        if recorded[key] != config[key]:
            raise ValueError(f"head and vocabulary disagree on {key}: "
                             f"{config[key]} != {recorded[key]}")
    payload = {
        "format": TEACHER_FORMAT,
        "config": config,
        "vocab": recorded,
        "state_dict": {name: tensor.detach().cpu().clone()
                       for name, tensor in head.state_dict().items()},
        "extra": dict(extra or {}),
    }
    torch.save(payload, path)
    return payload


def load_teacher(path, *, device: str = "cpu", expect_vocab: dict | None = None):
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "config" not in payload or "state_dict" not in payload:
        raise ValueError(
            f"{path} is not a teacher payload (got {type(payload).__name__}); "
            "a bare state_dict must be wrapped with save_teacher first")
    if int(payload.get("format", 0)) != TEACHER_FORMAT:
        raise ValueError(f"teacher format {payload.get('format')} is not {TEACHER_FORMAT}")
    vocabulary = payload.get("vocab") or {}
    if expect_vocab:
        for key in ("mode", "n_subtypes", "n_orbits"):
            if key in expect_vocab and vocabulary.get(key) != expect_vocab[key]:
                raise ValueError(f"teacher vocabulary mismatch on {key}: "
                                 f"saved {vocabulary.get(key)!r} != expected {expect_vocab[key]!r}")
    head = build_head(payload["config"])
    head.load_state_dict(payload["state_dict"], strict=True)
    freeze(head)
    return head.to(device), payload


def save_arm_teacher(log_dir, head, *, vocab: dict, arm: str, label_mode: str,
                     heldout: dict):
    """The call the trainer makes: write the head under the log dir, with its scores."""
    path = Path(log_dir) / "teacher.pt"
    payload = save_teacher(path, head, vocab=vocab, extra={
        "arm": arm,
        "label_mode": label_mode,
        "heldout_subtype_macro_f1": heldout["subtype"]["macro_f1"],
        "heldout_orbit_macro_over_families": heldout["role"]["macro_over_families"],
    })
    return path, payload


def wrap_state_dict(state_dict, *, config: dict, vocab: dict, extra: dict | None = None) -> dict:
    """Turn a legacy bare state_dict into a teacher payload (no file written)."""
    head = build_head(config)
    head.load_state_dict(state_dict, strict=True)
    return {"format": TEACHER_FORMAT, "config": teacher_config(head),
            "vocab": {"mode": vocab.get("mode"), "n_subtypes": int(vocab["n_subtypes"]),
                      "n_orbits": int(vocab["n_orbits"])},
            "state_dict": {name: tensor.detach().cpu().clone()
                           for name, tensor in head.state_dict().items()},
            "extra": dict(extra or {})}
