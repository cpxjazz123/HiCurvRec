#!/usr/bin/env python3
"""The level-1 codebook is widened to 1024 and every consumer follows that width.

The mechanism is unchanged from ``TIGER_graphlet_align_rq``: the chain/fork/merge
hinges and the Sinkhorn assignment stay as they are. Only the amount of level-1
capacity changes, because the measured concentration-alignment curve
(``SUMMARY_PER_GRAPHLET_RULES.md`` §33) says structural alignment is bought with
level-1 concentration, and 256 codes run out of budget before the gain is visible.

These tests fail on the copied tree (256 codes everywhere) and pass only when the
width is really 1024 on the config, on the quantizer, and on the export offsets.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

TREE = Path(__file__).resolve().parent.parent


def _trainer():
    sys.path.insert(0, str(TREE))
    import train_rqvae  # noqa: PLC0415

    return train_rqvae


def test_config_declares_the_wide_level_one():
    trainer = _trainer()
    sizes = list(trainer.CODEBOOK_SIZE)
    assert sizes == [1024, 256, 256], sizes
    assert int(trainer.CODEBOOK_NUM) == len(sizes)
    assert list(trainer._tokenizer_config().codebook_size) == [1024, 256, 256]


def test_quantizer_builds_one_width_per_level():
    from model.layers import RQLayer  # noqa: PLC0415

    layer = RQLayer(_trainer()._tokenizer_config())
    assert [int(size) for size in layer.codebook_sizes] == [1024, 256, 256]
    assert tuple(layer.vq_layers[0].get_code_embs().shape)[0] == 1024


def test_distance_matrix_is_wide_at_level_one_only():
    from model.layers import RQLayer  # noqa: PLC0415

    config = _trainer()._tokenizer_config()
    layer = RQLayer(config)
    latent = torch.randn(8, int(config.codebook_dim))
    assert tuple(layer.vq_layers[0]._distances(latent).shape) == (8, 1024)
    for level in layer.vq_layers[1:]:
        assert tuple(level._distances(latent).shape) == (8, 256)


def test_export_offsets_follow_the_declared_widths():
    train_rqvae = _trainer()

    sizes = [1024, 256, 256]
    tokens = np.array([[0, 0, 0], [1023, 255, 255]], dtype=np.int64)
    extended = train_rqvae._extend_collisions(tokens, sizes)
    assert extended.shape == (2, 4)
    # the collision counter lives above the three code blocks, so its base is the sum
    assert int(extended[:, 3].min()) == sum(sizes)
