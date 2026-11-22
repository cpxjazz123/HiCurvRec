"""The Stage 3 entry gate: three-level collision below 40% and quantisation error in band.

Both numbers are measured on the arm's own SID and checkpoint, and the gate is
proved here on synthetic arms whose two quantities are set by construction:

* collision 0.04 with quantisation error 0.40  -> allowed;
* collision 0.35                              -> allowed (owner raised the limit to 0.4);
* collision 0.45                              -> refused (items share an SID);
* quantisation error 0.02 (codebook memorises) -> allowed, reported as a warning;
* quantisation error 10.0 (vectors pressed)    -> allowed, reported as a warning;
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
    folder = _arm("inside", scale=0.07, distinct=96)
    from tools.research.stage2.sid_quality import evaluate
    verdict = evaluate(folder, embeddings_path=folder.parent / "embeddings.npy")
    assert verdict["allow"] is True, verdict
    assert verdict["collision"] == pytest.approx(0.04)
    assert verdict["quant_error"] == pytest.approx(0.07, abs=1e-3)


def test_gate_is_forty_percent():
    from tools.research.stage2.sid_quality import COLLISION_LIMIT
    assert COLLISION_LIMIT == 0.4


def test_collision_below_the_forty_percent_limit_is_allowed():
    folder = _arm("collided_35", scale=0.07, distinct=65)
    from tools.research.stage2.sid_quality import evaluate
    verdict = evaluate(folder, embeddings_path=folder.parent / "embeddings.npy")
    assert verdict["collision"] == pytest.approx(0.35)
    assert verdict["allow"] is True, verdict


def test_collision_above_the_forty_percent_limit_is_refused():
    folder = _arm("collided_45", scale=0.07, distinct=55)
    from tools.research.stage2.sid_quality import evaluate
    verdict = evaluate(folder, embeddings_path=folder.parent / "embeddings.npy")
    assert verdict["collision"] == pytest.approx(0.45)
    assert verdict["allow"] is False
    assert "collision" in verdict["reason"]


def test_memorising_codebook_is_reported():
    folder = _arm("memorised", scale=0.01, distinct=96)
    from tools.research.stage2.sid_quality import evaluate
    verdict = evaluate(folder, embeddings_path=folder.parent / "embeddings.npy")
    assert verdict["allow"] is True
    assert any("quantisation" in warning for warning in verdict["warnings"])


def test_pressed_codebook_is_reported():
    folder = _arm("pressed", scale=10.0, distinct=96)
    from tools.research.stage2.sid_quality import evaluate
    verdict = evaluate(folder, embeddings_path=folder.parent / "embeddings.npy")
    assert verdict["allow"] is True
    assert any("quantisation" in warning for warning in verdict["warnings"])


def test_unorganised_output_is_reported():
    folder = _arm("unorganised", scale=0.07, distinct=96, dim=20)
    from tools.research.stage2.sid_quality import evaluate
    verdict = evaluate(folder, embeddings_path=folder.parent / "embeddings.npy")
    assert verdict["allow"] is True
    assert any("effective rank" in warning for warning in verdict["warnings"])
    assert verdict["effective_rank"] > 8


def test_missing_checkpoint_fails_closed():
    folder = _arm("nockpt", scale=0.40, distinct=96, with_checkpoint=False)
    from tools.research.stage2.sid_quality import evaluate
    verdict = evaluate(folder, embeddings_path=folder.parent / "embeddings.npy")
    assert verdict["allow"] is False
    assert "checkpoint" in verdict["reason"]


def _write_arm(name: str, state: dict, tokens: list, embeddings) -> Path:
    folder = SCRATCH / name
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "embeddings.npy", np.asarray(embeddings, dtype="float32"))
    payload = {str(index): list(triple) + [768] for index, triple in enumerate(tokens)}
    (folder / "item_sids.json").write_text(json.dumps(payload))
    torch.save({"state_dict": state}, folder / "rqvae_best.pth")
    return folder / "item_sids.json"


def test_encoder_bias_is_part_of_the_forward():
    rng = np.random.default_rng(3)
    direction = rng.normal(size=(100, 4)).astype("float32")
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    embeddings = direction * 0.4
    bias = torch.full((4,), 0.1)
    tokens = [[item % 5, (item // 5) % 5, (item // 25) % 5] for item in range(100)]
    state = {
        "encoder.mlp.1.weight": torch.eye(4),
        "encoder.mlp.1.bias": bias,
        "rq.vq_layers.0.embed.weight": torch.zeros(6, 4),
        "rq.vq_layers.1.embed.weight": torch.zeros(6, 4),
        "rq.vq_layers.2.embed.weight": torch.zeros(6, 4),
    }
    sid = _write_arm("biased", state, tokens, embeddings)
    from tools.research.stage2.sid_quality import evaluate
    verdict = evaluate(sid, embeddings_path=sid.parent / "embeddings.npy")
    expected = float(np.linalg.norm(embeddings + 0.1, axis=1).mean())
    assert verdict["quant_error"] == pytest.approx(expected, abs=1e-4), verdict


def test_simvq_codes_are_read_through_the_projection():
    rng = np.random.default_rng(4)
    direction = rng.normal(size=(100, 4)).astype("float32")
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    embeddings = direction * 0.4
    tokens = [[item % 5, (item // 5) % 5, (item // 25) % 5] for item in range(100)]
    state = {
        "encoder.mlp.1.weight": torch.eye(4),
        "rq.vq_layers.0.embed.weight": torch.ones(6, 4),
        "rq.vq_layers.0.embed_proj.weight": 0.5 * torch.eye(4),
        "rq.vq_layers.1.embed.weight": torch.ones(6, 4),
        "rq.vq_layers.1.embed_proj.weight": 0.5 * torch.eye(4),
        "rq.vq_layers.2.embed.weight": torch.ones(6, 4),
        "rq.vq_layers.2.embed_proj.weight": 0.5 * torch.eye(4),
    }
    sid = _write_arm("simvq", state, tokens, embeddings)
    from tools.research.stage2.sid_quality import evaluate
    verdict = evaluate(sid, embeddings_path=sid.parent / "embeddings.npy")
    projected = float(np.linalg.norm(embeddings - 1.5, axis=1).mean())
    raw = float(np.linalg.norm(embeddings - 3.0, axis=1).mean())
    assert verdict["quant_error"] == pytest.approx(projected, abs=1e-4), verdict
    assert abs(verdict["quant_error"] - raw) > 0.5


def test_ema_codebook_key_without_weight_suffix_is_read():
    rng = np.random.default_rng(5)
    direction = rng.normal(size=(100, 4)).astype("float32")
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    embeddings = direction * 0.4
    tokens = [[item % 5, (item // 5) % 5, (item // 25) % 5] for item in range(100)]
    state = {
        "encoder.mlp.1.weight": torch.eye(4),
        "rq.vq_layers.0.embed": torch.zeros(6, 4),
        "rq.vq_layers.1.embed": torch.zeros(6, 4),
        "rq.vq_layers.2.embed": torch.zeros(6, 4),
    }
    sid = _write_arm("ema", state, tokens, embeddings)
    from tools.research.stage2.sid_quality import evaluate
    verdict = evaluate(sid, embeddings_path=sid.parent / "embeddings.npy")
    assert verdict["quant_error"] == pytest.approx(0.4, abs=1e-3), verdict


def test_artifact_gate_matches_the_launch_gate():
    from tools.research.stage2.anticollapse import artifact
    from tools.research.stage2 import sid_quality
    assert artifact.COLLISION_GATE == sid_quality.COLLISION_LIMIT
