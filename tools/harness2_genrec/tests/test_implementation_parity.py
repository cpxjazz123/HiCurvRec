"""The proposal-implementation parity check's tests (issue #112).

The check exists because a JOIN once passed for a mechanism that was a reduced copy of the
approved proposal: the tree had lost the proposal's distinctive component and the contract had
been narrowed to match, so every gate agreed and the narrowing was invisible.

Needs torch, so run it under the project interpreter:

    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 -m pytest \
        tools/harness2_genrec/tests/test_implementation_parity.py -q
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "tools" / "harness2_genrec") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools" / "harness2_genrec"))

import implementation_parity as parity  # noqa: E402

torch = pytest.importorskip("torch")

ARM = REPO_ROOT / "stage2_RQ-VAE/TIGER_hyp_product_l1_quantiser"


def test_the_current_tree_carries_every_declared_component():
    result = parity.check(ARM, "hyp_product_l1_quantiser")
    assert result["ok"], result["verdict"]
    assert result["missing"] == []
    ids = {entry["id"] for entry in result["components"]}
    assert ids == {"product_assignment", "role_supervision", "role_separation"}


def test_a_tree_without_the_role_supervision_is_refused(tmp_path):
    """The reduced version -- the assignment but no supervision -- must not pass."""
    tree = tmp_path / "arm"
    shutil.copytree(ARM, tree)
    module = tree / "scripts" / "hyp_losses.py"
    text = module.read_text()
    # the REDUCED version: the assignment, but the arm's active mechanism is not the
    # role-separated term any more
    text = text.replace("        ARM_CONFIGS[_name][\"curvature_b\"] = (\n            ARM_CONFIGS[_name][\"curvature\"] * PRODUCT_CURVATURE_RATIO)",
                        "        ARM_CONFIGS[_name][\"curvature_b\"] = (\n            ARM_CONFIGS[_name][\"curvature\"] * PRODUCT_CURVATURE_RATIO)\n        ARM_CONFIGS[_name][\"mechanism\"] = \"product\"")
    module.write_text(text)
    result = parity.check(tree, "hyp_product_l1_quantiser")
    assert not result["ok"]
    assert "role_supervision" in result["missing"]
    assert "reduced mechanism" in result["verdict"]


def test_an_unknown_mechanism_declares_nothing_and_is_refused():
    result = parity.check(ARM, "a_mechanism_that_does_not_exist")
    assert not result["ok"]
    assert result["components"] == []
