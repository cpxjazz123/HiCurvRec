"""Re-export of the repository's ONE implementation of `HYPERBOLIC_GEOMETRY_CONTRACT.md`.

The contract's own docstring says a candidate composes the operators, it never
re-derives them. So this file keeps no copy of the geometry: it loads
`tools/research/geometry/poincare_contract.py` (the single implementation) and
re-exports its public API. A copy inside the arm would be a second source of truth
that could silently drift from the frozen contract.

The repository root is found by the `stage2_RQ-VAE` marker (no absolute path and no
parent counting), exactly like `train_rqvae.py`'s `_repository_root`.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

CANONICAL_RELATIVE_PATH = ("tools", "research", "geometry", "poincare_contract.py")


def repository_root() -> Path:
    """The repository root, found by walking up to the `stage2_RQ-VAE` marker."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "stage2_RQ-VAE").is_dir():
            return parent
    raise RuntimeError("cannot locate the repository root above " + str(here))


def canonical_path() -> Path:
    """The frozen contract implementation this module re-exports."""
    path = repository_root()
    for part in CANONICAL_RELATIVE_PATH:
        path = path / part
    if not path.is_file():
        raise RuntimeError(
            "the geometry contract is missing: expected "
            + "/".join(("tools", "research", "geometry", "poincare_contract.py"))
            + " below the repository root, found nothing at "
            + str(path)
        )
    return path


def _load_canonical():
    path = canonical_path()
    spec = importlib.util.spec_from_file_location(__name__ + "._canonical", path)
    if spec is None or spec.loader is None:  # pragma: no cover - import machinery guard
        raise RuntimeError("cannot load the geometry contract from " + str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_canonical = _load_canonical()

#: The whole public API of the contract, re-exported by name.
__all__ = list(getattr(_canonical, "__all__", ()))

for _name in __all__:
    globals()[_name] = getattr(_canonical, _name)

# The two constants the caller compares against (math.json constants BALL_EPS /
# ARTANH_EPS) are part of `__all__` already; keep explicit aliases for readability.
BALL_EPS = _canonical.BALL_EPS
ARTANH_EPS = _canonical.ARTANH_EPS
