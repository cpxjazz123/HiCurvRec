"""Test bootstrap for the candidate tree `TIGER_af3_curvature_factor_mass_rq`.

The tree directory is not a package (its name contains a dash), so the tests put the
tree root on `sys.path` exactly the way the trainer does when it runs from there:
`import factor_share_geometry`, `from distances import ...`, `from model import ...`.

The repository root is found by the `stage2_RQ-VAE` marker, never by counting parents.
"""

from __future__ import annotations

import sys
from pathlib import Path

ARM = "TIGER_af3_curvature_factor_mass_rq"


def repository_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "stage2_RQ-VAE").is_dir():
            return parent
    raise RuntimeError("cannot locate the repository root above " + str(here))


TREE = repository_root() / "stage2_RQ-VAE" / ARM
if not TREE.is_dir():
    raise RuntimeError("the candidate tree is missing: " + str(TREE))
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))
