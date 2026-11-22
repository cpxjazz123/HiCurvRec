"""The floor installer must be able to protect the frozen baseline itself.

Every candidate starts as a byte copy of ``stage2_RQ-VAE/baseline/TIGER_RQ-VAE``, and the
launcher installs the anti-collapse floor into that copy.  The installer used to know only
two revision of ``model/layers.py`` -- the hoisted-``cost`` form and the EMA form -- so the
frozen baseline's inline ``self._distances(latent)`` forward matched no anchor and every
launch died with ``UnknownTree`` before the first epoch.  This test is the reproduction:
it copies the baseline, installs the floor, and requires a fully guarded tree.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from tools.research.stage2.anticollapse import patch_tree  # noqa: E402

BASELINE = ROOT / "stage2_RQ-VAE" / "baseline" / "TIGER_RQ-VAE"


def test_floor_installs_into_the_frozen_baseline_copy():
    assert BASELINE.is_dir(), "the frozen baseline must exist"
    workspace = Path(tempfile.mkdtemp(prefix="floor_install_", dir=ROOT / "results" / "research"))
    tree = workspace / "TIGER_floor_probe"
    try:
        # `dataset/` holds a dangling absolute symlink to a RecBole-era parquet that the
        # trainer never reads (only a docstring mentions it); copying it breaks copytree.
        shutil.copytree(BASELINE, tree,
                        ignore=shutil.ignore_patterns("__pycache__", "logs", "dataset"))
        result = patch_tree.apply_floor(tree)
        assert result["changed"] is True, result
        # the floor module is delivered next to the quantiser
        assert (tree / patch_tree.TARGET_NAME).is_file()
        text = (tree / "model" / "layers.py").read_text(encoding="utf-8")
        for marker in ("constrained_assignment(", "verify_assignment(", "verify_codebook("):
            assert marker in text, marker
        # the scene that killed the launch: every assignment must be guarded now
        assert patch_tree.unguarded_sites(tree) == []
        # a second install is a no-op rather than a second guard on the same line
        again = patch_tree.apply_floor(tree)
        assert again["changed"] is False, again
        assert (tree / "model" / "layers.py").read_text(encoding="utf-8").count(
            "verify_assignment(") == text.count("verify_assignment(")
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
