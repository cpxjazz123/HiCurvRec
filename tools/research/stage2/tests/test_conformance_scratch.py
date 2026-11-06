"""The conformance scratch directory is keyed on the tree, not on the copy.

The fault this refuses: ``check_entry`` runs the PROMOTED copy of a tree (a tree
under ``results/`` resolves a different repository root than the tree that
trains, so the admission copies it into ``stage2_RQ-VAE/``), while
``check_identifier`` reads the scratch directory of the ORIGINAL tree.  The
promoted copy's name carries a digest suffix, ``_admission_<name>_<digest>``, so
stripping only the ``_admission_`` prefix left the digest on the directory name:
the entry check wrote ``item_sids.json`` into
``_conformance/<name>_<digest>/`` and the identifier check looked in
``_conformance/<name>/`` and reported NOT_RUN on a tree that had exported a
valid identifier.  Measured on the curvature-stratum-sinkhorn-v1 candidate:

    PASS | entry      | one epoch runs and exports the identifier
    NOT_RUN | identifier | the entry check did not produce an identifier

The two checks must resolve the same directory for the same tree.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from tools.research.stage2 import checks, contract  # noqa: E402


def test_the_promoted_copy_writes_where_the_identifier_check_reads(tmp_path):
    tree = tmp_path / "curvature-stratum-sinkhorn-v1"
    (tree / "model").mkdir(parents=True)
    (tree / "train_rqvae.py").write_text("EPOCHS = 1\n", encoding="utf-8")
    original = checks._scratch_dir(tree)
    promoted = checks._admission_tree(tree)
    try:
        # The copy's own name carries the digest; its scratch directory must not.
        assert promoted.name.startswith("_admission_")
        assert promoted.name != "_admission_" + tree.name
        assert checks._scratch_dir(promoted) == original
    finally:
        checks._discard_admission_tree(tree)


def test_two_trees_of_the_same_name_keep_their_own_copies(tmp_path):
    """The digest still does its job: the copies do not collide."""
    first = tmp_path / "a" / "candidate"
    second = tmp_path / "b" / "candidate"
    for tree in (first, second):
        (tree / "model").mkdir(parents=True)
        (tree / "train_rqvae.py").write_text("EPOCHS = 1\n", encoding="utf-8")
    left = checks._admission_tree(first)
    right = checks._admission_tree(second)
    try:
        assert left != right
        # ...while both write into the tree's own scratch directory: the name is
        # the identity the review hashes, and the entry check runs one at a time.
        assert checks._scratch_dir(left) == checks._scratch_dir(right)
    finally:
        checks._discard_admission_tree(first)
        checks._discard_admission_tree(second)


def test_the_scratch_directory_is_outside_the_tree(tmp_path):
    tree = tmp_path / "candidate"
    (tree / "model").mkdir(parents=True)
    (tree / "train_rqvae.py").write_text("EPOCHS = 1\n", encoding="utf-8")
    scratch = checks._scratch_dir(tree)
    assert scratch.is_dir()
    assert contract.resolve_root() in scratch.parents
    assert tree not in scratch.parents
