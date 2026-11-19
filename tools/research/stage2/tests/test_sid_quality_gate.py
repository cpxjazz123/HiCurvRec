"""The Stage 3 entry gate: three-level collision below 5% and quantisation error in band.

Both numbers are measured on the arm's own SID and checkpoint, and the gate is
proved here on synthetic arms whose two quantities are set by construction:

* collision 0.04 with quantisation error 0.40  -> allowed;
* collision 0.10                              -> refused (items share an SID);
* quantisation error 0.02 (codebook memorises) -> refused;
* quantisation error 10.0 (vectors pressed)    -> refused;
* no checkpoint                               -> refused, never assumed fine.

The bands are the ones measured over 74 arms: every arm above recall 0.06 stayed
below 4.5% collision, and every arm outside the 0.20-0.60 quantisation band was
a memorising (<=0.10, `dul_rq` 0.026 recall) or a pressed (>7) table.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
SCRATCH = REPO / "results" / "research" / "_selftest_sid_gate"


def _arm(name: str, *, scale: float, distinct: int, with_checkpoint: bool = True) -> Path:
    folder = SCRATCH / name
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    direction = rng.normal(size=(100, 4)).astype("float32")
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    embeddings = (direction * scale).astype("float32")
    np.save(folder / "embeddings.npy", embeddings)

    payload = {}
    for item in range(100):
        triple = [item % 5, (item // 5) % 5, (item // 25) % 5] if item < distinct else [0, 0, 0]
        payload[str(item)] = triple + [768]
    (folder / "item_sids.json").write_text(json.dumps(payload))

    if with_checkpoint:
        state = {
            "encoder.mlp.0.weight": torch.eye(4),
            "rq.vq_layers.0.embed.weight": torch.zeros(6, 4),
            "rq.vq_layers.1.embed.weight": torch.zeros(6, 4),
            "rq.vq_layers.2.embed.weight": torch.zeros(6, 4),
        }
        torch.save({"state_dict": state}, folder / "rqvae_best.pth")
    return folder / "item_sids.json"


def test_inside_the_band_is_allowed():
    folder = _arm("inside", scale=0.40, distinct=96)
    from tools.research.stage2.sid_quality import evaluate
    verdict = evaluate(folder, embeddings_path=folder.parent / "embeddings.npy")
    assert verdict["allow"] is True, verdict
    assert verdict["collision"] == pytest.approx(0.04)
    assert verdict["quant_error"] == pytest.approx(0.40, abs=1e-3)


def test_collision_above_five_percent_is_refused():
    folder = _arm("collided", scale=0.40, distinct=90)
    from tools.research.stage2.sid_quality import evaluate
    verdict = evaluate(folder, embeddings_path=folder.parent / "embeddings.npy")
    assert verdict["allow"] is False
    assert "collision" in verdict["reason"]


def test_memorising_codebook_is_refused():
    folder = _arm("memorised", scale=0.02, distinct=96)
    from tools.research.stage2.sid_quality import evaluate
    verdict = evaluate(folder, embeddings_path=folder.parent / "embeddings.npy")
    assert verdict["allow"] is False
    assert "quantisation" in verdict["reason"]


def test_pressed_codebook_is_refused():
    folder = _arm("pressed", scale=10.0, distinct=96)
    from tools.research.stage2.sid_quality import evaluate
    verdict = evaluate(folder, embeddings_path=folder.parent / "embeddings.npy")
    assert verdict["allow"] is False
    assert "quantisation" in verdict["reason"]


def test_missing_checkpoint_fails_closed():
    folder = _arm("nockpt", scale=0.40, distinct=96, with_checkpoint=False)
    from tools.research.stage2.sid_quality import evaluate
    verdict = evaluate(folder, embeddings_path=folder.parent / "embeddings.npy")
    assert verdict["allow"] is False
    assert "checkpoint" in verdict["reason"]

import numpy as np
import pytest
import torch
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
SCRATCH = REPO / "results" / "research" / "_selftest_sid_gate"


def _arm(name: str, *, scale: float, distinct: int, with_checkpoint: bool = True,
         dim: int = 4) -> Path:
    folder = SCRATCH / name
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    direction = rng.normal(size=(100, dim)).astype("float32")
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    embeddings = (direction * scale).astype("float32")
    np.save(folder / "embeddings.npy", embeddings)

    payload = {}
    for item in range(100):
        triple = [item % 5, (item // 5) % 5, (item // 25) % 5] if item < distinct else [0, 0, 0]
        payload[str(item)] = triple + [768]
    (folder / "item_sids.json").write_text(json.dumps(payload))

    if with_checkpoint:
        state = {
            "encoder.mlp.0.weight": torch.eye(dim),
            "rq.vq_layers.0.embed.weight": torch.zeros(6, dim),
            "rq.vq_layers.1.embed.weight": torch.zeros(6, dim),
            "rq.vq_layers.2.embed.weight": torch.zeros(6, dim),
        }
        torch.save({"state_dict": state}, folder / "rqvae_best.pth")
    return folder / "item_sids.json"


def test_inside_the_band_is_allowed():
    folder = _arm("inside", scale=0.40, distinct=96)
    from tools.research.stage2.sid_quality import evaluate
    verdict = evaluate(folder, embeddings_path=folder.parent / "embeddings.npy")
    assert verdict["allow"] is True, verdict
    assert verdict["collision"] == pytest.approx(0.04)
    assert verdict["quant_error"] == pytest.approx(0.40, abs=1e-3)


def test_collision_above_ten_percent_is_refused():
    folder = _arm("collided", scale=0.40, distinct=85)
    from tools.research.stage2.sid_quality import evaluate
    verdict = evaluate(folder, embeddings_path=folder.parent / "embeddings.npy")
    assert verdict["allow"] is False
    assert "collision" in verdict["reason"]


def test_memorising_codebook_is_refused():
    folder = _arm("memorised", scale=0.02, distinct=96)
    from tools.research.stage2.sid_quality import evaluate
    verdict = evaluate(folder, embeddings_path=folder.parent / "embeddings.npy")
    assert verdict["allow"] is False
    assert "quantisation" in verdict["reason"]


def test_pressed_codebook_is_refused():
    folder = _arm("pressed", scale=10.0, distinct=96)
    from tools.research.stage2.sid_quality import evaluate
    verdict = evaluate(folder, embeddings_path=folder.parent / "embeddings.npy")
    assert verdict["allow"] is False
    assert "quantisation" in verdict["reason"]


def test_unorganised_output_is_refused():
    folder = _arm("unorganised", scale=0.40, distinct=96, dim=20)
    from tools.research.stage2.sid_quality import evaluate
    verdict = evaluate(folder, embeddings_path=folder.parent / "embeddings.npy")
    assert verdict["allow"] is False
    assert "effective rank" in verdict["reason"]
    assert verdict["effective_rank"] > 12


def test_missing_checkpoint_fails_closed():
    folder = _arm("nockpt", scale=0.40, distinct=96, with_checkpoint=False)
    from tools.research.stage2.sid_quality import evaluate
    verdict = evaluate(folder, embeddings_path=folder.parent / "embeddings.npy")
    assert verdict["allow"] is False
    assert "checkpoint" in verdict["reason"]
