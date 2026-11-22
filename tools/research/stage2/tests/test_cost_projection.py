"""The frozen-run cost projection must use the launch layout's steps per epoch.

The projection is what the owner reads while a Stage 2 job runs, and it is built
from the one-epoch smoke. The number of steps per epoch is not a constant of the
trainer: ``ddp4`` sees ``ceil(24556 / 4096) = 6`` steps, the single-card branch
sees ``ceil(24556 / 1024) = 24``. Reading the single-card figure under ``ddp4``
scaled the projection by four, and the smoke's own start-up time scaled it by
another ~8.7: the first #150 smoke projected 10.5 h for a run that took 1090 s.

These tests pin the layout half of that, and that the result is labelled as the
upper bound it is.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SERVER = Path(__file__).resolve().parents[3] / "genrec-mcp" / "server.py"


def load_server():
    spec = importlib.util.spec_from_file_location("genrec_server_for_projection", SERVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SERVER_MODULE = load_server()


def test_the_frozen_recipe_has_six_steps_per_epoch_under_ddp4():
    """24556 TRAIN items over a global batch of 4096 is six steps, not 24."""
    assert SERVER_MODULE.STAGE2_FROZEN_STEPS_PER_EPOCH_DDP4 == 6
    assert SERVER_MODULE.STAGE2_FROZEN_STEPS_PER_EPOCH_SINGLE == 24
    derived = -(-SERVER_MODULE.STAGE2_TRAIN_ITEMS
                // (SERVER_MODULE.STAGE2_BATCH_PER_RANK * SERVER_MODULE.STAGE2_FROZEN_WORLD_SIZE))
    assert derived == SERVER_MODULE.STAGE2_FROZEN_STEPS_PER_EPOCH_DDP4


def test_a_ddp_smoke_projects_with_the_ddp_step_count():
    """The #150 smoke: 12.599 s over 24 steps must project ~2.6 h, not ~10.5 h."""
    projection = SERVER_MODULE.project_frozen_run_cost(
        {"elapsed_s": 12.599, "steps": 24, "launch_mode": "ddp"})
    assert projection["steps_per_epoch"] == 6
    expected = (12.599 / 24) * 3000 * 6 / 3600.0
    assert projection["projected_hours"] == pytest.approx(expected)
    assert projection["projected_hours"] == pytest.approx(2.62, abs=0.02)
    assert projection["projected_hours"] < 10.5 / 3.0


def test_a_single_card_smoke_keeps_the_single_card_step_count():
    projection = SERVER_MODULE.project_frozen_run_cost(
        {"elapsed_s": 12.0, "steps": 24, "launch_mode": "single"})
    assert projection["steps_per_epoch"] == 24
    assert projection["projected_hours"] == pytest.approx((12.0 / 24) * 3000 * 24 / 3600.0)


def test_an_unlabelled_smoke_follows_the_configured_strategy():
    projection = SERVER_MODULE.project_frozen_run_cost({"elapsed_s": 6.0, "steps": 24})
    if SERVER_MODULE.STAGE2_LAUNCH_STRATEGY == "ddp4":
        assert projection["steps_per_epoch"] == 6
    else:
        assert projection["steps_per_epoch"] == 24


def test_the_projection_says_it_is_an_upper_bound():
    projection = SERVER_MODULE.project_frozen_run_cost(
        {"elapsed_s": 12.599, "steps": 24, "launch_mode": "ddp"})
    assert projection["upper_bound"] is True
    assert "start-up" in projection["note"]
    assert projection["budget_hours"] == SERVER_MODULE.STAGE2_COST_BUDGET_HOURS


def test_a_smoke_without_timing_refuses_instead_of_guessing():
    for summary in (None, {}, {"elapsed_s": 0, "steps": 24}, {"elapsed_s": 12.0}):
        projection = SERVER_MODULE.project_frozen_run_cost(summary)
        assert projection["projected_hours"] is None
        assert projection["reason"]
