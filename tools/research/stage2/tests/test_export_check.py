"""The export checker end to end, on a tree small enough to read in one screen.

The property under test is the one the 2026-11-18 audit found broken on 12 of 14
arms: the table the trainer stored must equal the table the saved checkpoint
produces.  Here a fake tree makes the two disagree on purpose, so the checker has
to report it, and in rewrite mode replace the table while keeping the trainer's
copy beside it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from tools.research.stage2.export_check import check, write_manifest
from tools.research.stage2.export_policy import manifest_path

TRAINER = '''
from types import SimpleNamespace

CURVATURE = 1.0
ALPHA_MODE = ""


def _tokenizer_config():
    return SimpleNamespace(codebook_num=3, codebook_size=(4, 4, 4), codebook_dim=32)


def _extend_collisions(raw, sizes):
    return raw
'''

MODEL = '''
import torch
from torch import nn


class RQVAE(nn.Module):
    def __init__(self, config, *, in_dim):
        super().__init__()
        self.bump = nn.Parameter(torch.zeros(1))

    def get_indices(self, embeddings, *, infer_use_sk=False):
        value = int(round(float(self.bump.detach())))
        return torch.full((embeddings.shape[0], 3), value, dtype=torch.long)
'''


def build_tree(tmp_path: Path) -> Path:
    tree = tmp_path / "TIGER_fake"
    (tree / "model").mkdir(parents=True)
    (tree / "train_rqvae.py").write_text(TRAINER, encoding="utf-8")
    (tree / "model" / "__init__.py").write_text(MODEL, encoding="utf-8")
    return tree


def build_assets(tmp_path: Path, value: int) -> tuple[Path, Path, Path]:
    embeddings = tmp_path / "embeddings.npy"
    np.save(embeddings, np.zeros((5, 768), dtype=np.float32))
    sid = tmp_path / "item_sids.json"
    sid.write_text(json.dumps({str(i): [0, 0, 0] for i in range(5)}), encoding="utf-8")
    checkpoint = tmp_path / "rqvae_best.pth"
    torch.save({"state_dict": {"bump": torch.tensor([float(value)])}}, checkpoint)
    return embeddings, sid, checkpoint


def test_matching_export_is_verified(tmp_path: Path) -> None:
    tree = build_tree(tmp_path)
    embeddings, sid, checkpoint = build_assets(tmp_path, value=0)
    manifest = check(tree, checkpoint, sid, embeddings)
    assert manifest["tokens_match"] is True
    assert manifest["differing_items"] == 0
    assert manifest["same_l3"] == 1.0
    assert manifest["rewritten_from_checkpoint"] is False


def test_a_different_table_is_reported_not_hidden(tmp_path: Path) -> None:
    tree = build_tree(tmp_path)
    embeddings, sid, checkpoint = build_assets(tmp_path, value=1)
    manifest = check(tree, checkpoint, sid, embeddings)
    assert manifest["tokens_match"] is False
    assert manifest["differing_items"] == 5
    # Read-only by default: the trainer's table is untouched.
    assert json.loads(sid.read_text())["0"] == [0, 0, 0]


def test_rewrite_makes_the_table_the_checkpoints_and_keeps_the_original(tmp_path: Path) -> None:
    tree = build_tree(tmp_path)
    embeddings, sid, checkpoint = build_assets(tmp_path, value=1)
    before = sid.read_text(encoding="utf-8")
    manifest = check(tree, checkpoint, sid, embeddings, rewrite=True)
    assert manifest["rewritten_from_checkpoint"] is True
    assert manifest["original_differing_items"] == 5
    assert manifest["tokens_match"] is True
    assert json.loads(sid.read_text())["0"] == [1, 1, 1]
    assert (tmp_path / "item_sids.trainer.json").read_text(encoding="utf-8") == before
    # The manifest names the new table, so the gate can trust it.
    assert manifest["sid_sha256"] == __import__("hashlib").sha256(sid.read_bytes()).hexdigest()


def test_manifest_is_written_atomically(tmp_path: Path) -> None:
    path = tmp_path / "export_manifest.json"
    write_manifest({"schema": "x"}, path)
    assert json.loads(path.read_text())["schema"] == "x"
    assert not path.with_suffix(".json.partial").exists()
    assert manifest_path(tmp_path / "item_sids.json") == path


def test_the_checker_refuses_a_checkpoint_that_does_not_fit_the_tree(tmp_path: Path) -> None:
    tree = build_tree(tmp_path)
    embeddings, sid, _ = build_assets(tmp_path, value=0)
    checkpoint = tmp_path / "bad.pth"
    torch.save({"state_dict": {"other": torch.zeros(3)}}, checkpoint)
    with pytest.raises(SystemExit, match="does not fit"):
        check(tree, checkpoint, sid, embeddings)
