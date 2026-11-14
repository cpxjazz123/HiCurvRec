"""The plugin host: no mechanism == the frozen baseline; one mechanism == one file.

These tests never train. They check the seams the host offers (dataset shape, context
hand-off, prepare/readouts/switches, lazy distance registration) and the default path
that must stay identical to the baseline.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import TensorDataset

sys.path.insert(0, __file__.rsplit("/tests/", 1)[0])

import train_rqvae  # noqa: E402


def test_without_mechanism_the_host_behaves_like_the_baseline(monkeypatch):
    monkeypatch.setattr(train_rqvae, "MECHANISM", None)
    assert train_rqvae._mechanism_needs_item_ids() is False
    embeddings = torch.zeros(4, 8)
    dataset = train_rqvae._mechanism_dataset(embeddings, None)
    assert isinstance(dataset, TensorDataset)
    (row,) = dataset[0]
    assert row.shape == (8,)
    assert train_rqvae._mechanism_prepare([0, 1], [0, 1, 2]) == {}
    assert train_rqvae._mechanism_readouts() == {}
    assert train_rqvae._mechanism_switches() == {}
    train_rqvae._mechanism_context(torch.tensor([0]), True)  # no-op, must not raise


def test_with_mechanism_the_dataset_carries_item_ids(monkeypatch):
    seen = {}

    def prepare(train_ids, all_ids):
        seen["prepare"] = (list(train_ids), list(all_ids))
        return {"items": len(all_ids)}

    def set_context(item_ids, training):
        seen.setdefault("context", []).append((list(item_ids), training))

    fake = SimpleNamespace(
        needs_item_ids=True,
        prepare=prepare,
        set_context=set_context,
        readouts=lambda: {"ratio": 0.5},
        switches=lambda: {"behaviour_off": False},
        DISTANCE={"name": "fake_cost"},
    )
    monkeypatch.setattr(train_rqvae, "MECHANISM", fake)

    embeddings = torch.zeros(4, 8)
    item_ids = torch.tensor([7, 8, 9, 10])
    dataset = train_rqvae._mechanism_dataset(embeddings, item_ids)
    embedding_row, id_row = dataset[0]
    assert embedding_row.shape == (8,) and int(id_row) == 7
    assert train_rqvae._mechanism_prepare([7, 8], [7, 8, 9, 10]) == {"items": 4}
    assert seen["prepare"] == ([7, 8], [7, 8, 9, 10])
    train_rqvae._mechanism_context(item_ids, True)
    assert seen["context"] == [([7, 8, 9, 10], True)]
    assert train_rqvae._mechanism_readouts() == {"ratio": 0.5}
    assert train_rqvae._mechanism_switches() == {"behaviour_off": False}


def test_mechanism_distance_is_registered_lazily(tmp_path, monkeypatch):
    import importlib
    import distances

    module = tmp_path / "mechanism.py"
    module.write_text(
        "def _cost(latent, codebook):\n"
        "    return latent @ codebook.t()\n"
        "DISTANCE = {'name': 'plugin_cost', 'function': _cost, 'description': 'from a test'}\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("mechanism", None)
    try:
        with pytest.raises(distances.DistanceError):
            distances.build_distance(name="missing_metric")
        fn = distances.build_distance(name="plugin_cost")
        assert callable(fn)
        assert fn(torch.ones(1, 3), torch.ones(2, 3)).shape == (1, 2)
    finally:
        sys.modules.pop("mechanism", None)
        # 注册表是进程级的：把测试加进去的那条清掉，避免影响别的用例。
        distances.DISTANCES.pop("plugin_cost", None)
