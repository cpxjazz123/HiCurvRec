"""The forensic import's tests (issue #110, WP0).

The import has to be read-only, it has to count what it found, and it has to keep the
three kinds of material apart: an imported record, a new scientific arm and a synthetic
math test. Counting them together is how a campaign reports progress it did not make.

    /home/wlia0047/ar57_scratch/wenyu/harness2/.venv/bin/python -m pytest \
        tools/genrec_agents/tests/test_archives.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.genrec_agents import archives as A  # noqa: E402


def test_the_import_counts_what_it_found():
    document = A.import_archives(REPO_ROOT)
    assert document["counts"]["archives_missing"] == 0
    assert document["counts"]["old_archive_imports"] == len(document["imports"])
    assert document["counts"]["new_scientific_arms"] == 0
    assert document["counts"]["synthetic_math_tests"] == 0
    by_name = {entry["name"]: entry for entry in document["imports"]}
    assert by_name["historical_inventory"]["rows"] == 38
    assert by_name["campaign_store"]["manifests"] > 30
    assert by_name["campaign_proposals"]["proposals"] > 50
    assert by_name["campaign_audit"]["rows"] > 100


def test_the_frozen_assets_are_hashed():
    """A later comparison has to be able to say what it was compared against."""
    document = A.import_archives(REPO_ROOT)
    by_name = {entry["name"]: entry for entry in document["imports"]}
    for name in ("frozen_control_export", "frozen_pair_archive", "train_parquet"):
        assert by_name[name]["sha256"], name
        assert by_name[name]["bytes"] > 0


def test_the_import_writes_only_into_the_new_namespace(tmp_path):
    document = A.import_archives(REPO_ROOT)
    path = A.write_import(document, tmp_path / "archive_import.json")
    assert path.is_file()
    assert json.loads(path.read_text())["counts"]["archives_missing"] == 0
    # the earlier campaigns are untouched: the import only reads them
    assert A.NAMESPACE.name == "hyperbolic_research_runs"


def test_the_r8_hyperbolic_trees_are_imported_as_studied_source():
    document = A.import_archives(REPO_ROOT)
    names = {entry["name"] for entry in document["imports"]}
    for tree in A.R8_HYPERBOLIC_TREES:
        assert f"r8_tree::{tree}" in names
