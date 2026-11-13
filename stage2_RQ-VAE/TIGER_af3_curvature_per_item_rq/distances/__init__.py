"""The distance registry: which vectors a codeword is compared against, and how.

Every distance is a module beside this one with its own `NAME`, `DESCRIPTION` and
a callable. The training entry point imports the built object and hands it to the
model; the model hands it to the quantiser, which calls it and never holds a formula
of its own.

    from distances import build_distance
    DISTANCE = build_distance(name="per_item_curvature", curvature_table=TABLE)
    model = RQVAE(config, in_dim=dim, distance=DISTANCE)

Fork note (arm `TIGER_af3_curvature_per_item_rq`): the frozen baseline holds a single
`euclidean` entry and a `build_distance(config=None, name=None)` that takes no keyword
arguments. Two kinds of entry are allowed here, the same convention the sibling arm
`TIGER_af3_curvature_drive_rq` already established:

* `function` -- a plain `(latent, codebook) -> (B, K)` callable that needs nothing
  else (`euclidean`, kept pointing at `distance.distances`).
* `factory` -- a class whose instances need runtime data the registry cannot know
  (`per_item_curvature` needs the frozen curvature table). `build_distance` forwards
  its keyword arguments to the factory. Such an object may expose
  `needs_rows = True` so the quantiser passes the batch's item rows.
"""

from __future__ import annotations

from . import distance, per_item_curvature

__all__ = ["DISTANCES", "DistanceError", "build_distance", "distance", "per_item_curvature"]

#: name -> what the metric is, and the callable (or factory) that computes it.
DISTANCES = {
    "euclidean": {
        "description": "squared Euclidean distance |x|^2 - 2 x.c + |c|^2",
        "function": distance.distances,
    },
    "per_item_curvature": {
        "description": per_item_curvature.DESCRIPTION,
        "factory": per_item_curvature.PerItemCurvature,
    },
}

#: The metric the frozen baseline trains with.
DEFAULT = "euclidean"


class DistanceError(RuntimeError):
    """The requested metric does not exist; training must not start."""


def build_distance(config=None, name: str | None = None, **kwargs):
    """Return the metric named by `name`, then `config`, then the default.

    A factory entry receives `kwargs`; a plain function entry is returned as is, so
    the baseline call `build_distance(CONFIG)` keeps returning `euclidean`.
    """
    if name is None and config is not None:
        for key in ("distance", "DISTANCE"):
            if isinstance(config, dict):
                name = config.get(key) or name
            elif hasattr(config, key):
                name = getattr(config, key) or name
    name = name or DEFAULT
    entry = DISTANCES.get(name)
    if entry is None:
        raise DistanceError("unknown distance %r; the registry holds %s"
                            % (name, ", ".join(sorted(DISTANCES))))
    if "factory" in entry:
        return entry["factory"](**kwargs)
    if kwargs:
        raise DistanceError("distance %r takes no runtime arguments, got %s"
                            % (name, ", ".join(sorted(kwargs))))
    return entry["function"]
