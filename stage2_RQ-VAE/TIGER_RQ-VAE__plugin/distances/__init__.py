"""The distance registry: which vectors a codeword is compared against, and how.

Every distance is a module beside this one with its own `NAME`, `DESCRIPTION` and
function `distances(latent, codebook) -> (B, K)`. The training entry point imports
the built object and hands it to the model; the model hands it to the quantiser,
which calls it and never holds a formula of its own.

    from distances import build_distance
    DISTANCE = build_distance()
    model = RQVAE(config, in_dim=dim, distance=DISTANCE)

Adding a metric means adding a module and one entry below -- and that is a mechanism
change, not a configuration knob: a candidate that swaps the metric is a different
experiment, so the choice is made in source.
"""

from __future__ import annotations

from . import distance

__all__ = ["DISTANCES", "DistanceError", "build_distance", "distance"]

#: name -> what the metric is, and the callable that computes it.
DISTANCES = {
    "euclidean": {
        "description": "squared Euclidean distance |x|^2 - 2 x.c + |c|^2",
        "function": distance.distances,
    },
}

#: The metric the frozen baseline trains with.
DEFAULT = "euclidean"


class DistanceError(RuntimeError):
    """The requested metric does not exist; training must not start."""


def _register_from_mechanism() -> None:
    """Let a tree-local ``mechanism.py`` add its distance without editing this registry.

    The mechanism module is optional: the frozen baseline has no ``mechanism.py`` and
    behaves exactly as before. When the file exists it may export

        DISTANCE = {"name": "...", "function": callable, "description": "..."}

    and that entry is registered on first lookup. Importing the mechanism lazily (inside
    build_distance) keeps the import direction one-way: mechanism -> distances.
    """
    try:
        import mechanism as _mechanism
    except ImportError:
        return
    entry = getattr(_mechanism, "DISTANCE", None)
    if not isinstance(entry, dict):
        return
    name = entry.get("name")
    function = entry.get("function")
    if isinstance(name, str) and callable(function):
        DISTANCES.setdefault(name, {
            "description": entry.get("description", "registered by mechanism.py"),
            "function": function,
        })


def build_distance(config=None, name: str | None = None):
    """Return the distance callable named by `name`, then `config`, then the default."""
    if name is None and config is not None:
        for key in ("distance", "DISTANCE"):
            if isinstance(config, dict):
                name = config.get(key) or name
            elif hasattr(config, key):
                name = getattr(config, key) or name
    name = name or DEFAULT
    entry = DISTANCES.get(name)
    if entry is None:
        _register_from_mechanism()
        entry = DISTANCES.get(name)
    if entry is None:
        raise DistanceError("unknown distance %r; the registry holds %s"
                            % (name, ", ".join(sorted(DISTANCES))))
    return entry["function"]
