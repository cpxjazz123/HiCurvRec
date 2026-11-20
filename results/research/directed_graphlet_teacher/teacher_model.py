"""Issue #222 A: the directed subgraph Teacher as one reloadable, frozen object.

`results/research/directed_graphlet_role/subgraph_predictor.py` trains the same
network but keeps nothing: encoder and readout die with the process and the
class counts live only in a JSON file beside a bare state dict.  A Teacher that
cannot be reloaded cannot generate soft targets, so the artefact is the unit:

    build_teacher(config)                  the architecture, from the config alone
    save_teacher(path, teacher, ...)       payload + sha256 + a reload proof
    load_frozen_teacher(path, expect_vocab) strict load, eval(), no gradient

The forward pass is

    forward(x[B,k,768], adjacency[B,k,k])
        -> type_logits[B,n_subtypes], orbit_logits[B,k,n_orbits], node_hidden[B,k,hidden]

with in/out separated message passing (2 layers, hidden 128) exactly as the
predictor that produced 0.7121 / 0.6101 on held-out instances.  Subtype is
permutation INVARIANT over the k nodes, orbit is permutation EQUIVARIANT: a
readout that moved when the nodes were reordered would be reading position, not
structure.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent

TEACHER_FORMAT = 2
REQUIRED_CONFIG = ("dim", "hidden", "layers", "n_subtypes", "n_orbits")


class TeacherFormatError(RuntimeError):
    """The file is not a teacher payload, or it disagrees with what was asked."""


class SubgraphEncoder(nn.Module):
    """In/out separated message passing on the instance's induced subgraph."""

    def __init__(self, dim: int, hidden: int = 128, layers: int = 2):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(dim, hidden), nn.GELU())
        self.self_proj = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.in_proj = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.out_proj = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        h = self.input(x)
        for layer in range(len(self.self_proj)):
            degree_in = adjacency.sum(1).clamp(min=1e-6)
            degree_out = adjacency.sum(2).clamp(min=1e-6)
            in_message = torch.einsum("bji,bjh->bih", adjacency, h) / degree_in.unsqueeze(-1)
            out_message = torch.einsum("bij,bjh->bih", adjacency, h) / degree_out.unsqueeze(-1)
            h = F.gelu(self.self_proj[layer](h) + self.in_proj[layer](in_message)
                       + self.out_proj[layer](out_message))
        return h


class Readout(nn.Module):
    """Pooled subtype head + node-wise, context-aware orbit head."""

    def __init__(self, dim: int, n_subtypes: int, n_orbits: int, hidden: int = 128):
        super().__init__()
        self.subtype = nn.Sequential(nn.Linear(2 * dim, hidden), nn.GELU(),
                                     nn.Linear(hidden, n_subtypes))
        self.context = nn.Linear(2 * dim, dim)
        self.orbit = nn.Sequential(nn.Linear(3 * dim, hidden), nn.GELU(),
                                   nn.Linear(hidden, n_orbits))

    def forward(self, h: torch.Tensor):
        pooled = torch.cat([h.mean(1), h.amax(1)], -1)
        context = self.context(pooled).unsqueeze(1).expand(-1, h.shape[1], -1)
        return self.subtype(pooled), self.orbit(torch.cat([h, context, h * context], -1))


class DirectedSubgraphTeacher(nn.Module):
    """Encoder + readout; one object that can be checkpointed as a whole."""

    def __init__(self, config: dict):
        super().__init__()
        missing = [key for key in REQUIRED_CONFIG if key not in config]
        if missing:
            raise TeacherFormatError("config is missing %s" % ", ".join(missing))
        self.config = {key: int(config[key]) for key in REQUIRED_CONFIG}
        if self.config["layers"] < 1 or self.config["hidden"] < 1 or self.config["dim"] < 1:
            raise TeacherFormatError("non-positive width in %r" % (self.config,))
        self.encoder = SubgraphEncoder(self.config["dim"], self.config["hidden"],
                                       self.config["layers"])
        self.readout = Readout(self.config["hidden"], self.config["n_subtypes"],
                               self.config["n_orbits"], self.config["hidden"])

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor):
        hidden = self.encoder(x, adjacency)
        type_logits, orbit_logits = self.readout(hidden)
        return type_logits, orbit_logits, hidden


def build_teacher(config: dict) -> DirectedSubgraphTeacher:
    return DirectedSubgraphTeacher(config)


def teacher_config(teacher: DirectedSubgraphTeacher) -> dict:
    """Read the widths back off the module so a payload cannot drift from it."""
    return {
        "dim": int(teacher.encoder.input[0].in_features),
        "hidden": int(teacher.encoder.input[0].out_features),
        "layers": len(teacher.encoder.self_proj),
        "n_subtypes": int(teacher.readout.subtype[-1].out_features),
        "n_orbits": int(teacher.readout.orbit[-1].out_features),
    }


def freeze(teacher: DirectedSubgraphTeacher) -> DirectedSubgraphTeacher:
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def _probe_batch(config: dict, seed: int = 222):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(4, 3, config["dim"], generator=generator)
    adjacency = (torch.rand(4, 3, 3, generator=generator) < 0.4).float()
    adjacency[:, torch.arange(3), torch.arange(3)] = 0.0
    return x, adjacency


def save_teacher(path, teacher: DirectedSubgraphTeacher, *, config: dict | None = None,
                 vocab: dict, extra: dict | None = None) -> dict:
    """Write the payload, then prove a reload reproduces the logits exactly."""
    import hashlib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    config = dict(config or teacher_config(teacher))
    recorded_vocab = {"mode": str(vocab.get("mode", "directed")),
                      "n_subtypes": int(teacher_config(teacher)["n_subtypes"]),
                      "n_orbits": int(teacher_config(teacher)["n_orbits"])}
    payload = {"format": TEACHER_FORMAT, "config": config, "vocab": recorded_vocab,
               "state_dict": {key: value.detach().cpu().clone()
                              for key, value in teacher.state_dict().items()},
               "extra": dict(extra or {})}
    torch.save(payload, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    device = next(teacher.parameters()).device
    x, adjacency = (tensor.to(device) for tensor in _probe_batch(config))
    original = freeze(teacher)
    with torch.no_grad():
        before_type, before_orbit, before_hidden = original(x, adjacency)
    reloaded, _ = load_frozen_teacher(path, expect_vocab=recorded_vocab)
    reloaded.to(device)
    with torch.no_grad():
        after_type, after_orbit, after_hidden = reloaded(x, adjacency)
    identical = (torch.equal(before_type, after_type)
                 and torch.equal(before_orbit, after_orbit)
                 and torch.equal(before_hidden, after_hidden))
    if not identical:
        raise TeacherFormatError("reloaded teacher disagrees with the saved one")
    record = {"path": str(path), "sha256": digest, "config": config,
              "vocab": recorded_vocab, "probe_identical": True,
              "probe": {"seed": 222, "batch": 4, "nodes": 3},
              "extra": payload["extra"]}
    return record


def load_frozen_teacher(path, expect_vocab: dict | None = None):
    """Strict load; eval(); no gradient.  Returns (teacher, record)."""
    path = Path(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:  # truncated or foreign file
        raise TeacherFormatError("cannot read %s: %s" % (path, error)) from error
    if not isinstance(payload, dict) or payload.get("format") != TEACHER_FORMAT:
        raise TeacherFormatError("%s is not a teacher payload (format %r)"
                                 % (path, payload.get("format") if isinstance(payload, dict) else None))
    config = payload.get("config")
    vocab = payload.get("vocab")
    if not isinstance(config, dict) or not isinstance(vocab, dict):
        raise TeacherFormatError("payload without config/vocab")
    teacher = build_teacher(config)
    try:
        teacher.load_state_dict(payload["state_dict"], strict=True)
    except Exception as error:
        raise TeacherFormatError("state dict does not match the config: %s" % error) from error

    recorded = {"mode": vocab.get("mode"),
                "n_subtypes": int(vocab["n_subtypes"]),
                "n_orbits": int(vocab["n_orbits"])}
    if expect_vocab is not None:
        for key, value in recorded.items():
            if key in expect_vocab and expect_vocab[key] != value:
                raise TeacherFormatError("vocabulary %s is %r, the caller expects %r"
                                         % (key, value, expect_vocab[key]))
    freeze(teacher)
    record = {"path": str(path), "format": payload["format"], "config": dict(config),
              "vocab": recorded, "extra": payload.get("extra", {})}
    return teacher, record
