"""Stage3 entry refuses a SID whose three-level collision is not below 40%.

The limit lives in `tools/research/stage2/sid_quality.py` and the MCP reaches it through
`server.stage3_sid_quality`, so this file proves the MCP-side path on synthetic arms whose
collision is set by construction.  It never lets a candidate through to the trainer: the
passing case stops at the quality verdict, not at `tool_stage3_start`.
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools" / "genrec-mcp"))

import server  # noqa: E402

BASE = ROOT / "results" / "stage2_RQ-VAE"
ITEMS = 24587  # the gate compares the SID against stage1's own embedding table
WIDTH = 768
CODES = 256


def _arm(distinct: int) -> Path:
    """A SID whose collision is `1 - distinct / ITEMS` by construction.

    Codes stay below `CODES` so the vector measurement can index its codebooks.
    """
    folder = Path(tempfile.mkdtemp(prefix="collision_gate_", dir=BASE))
    triples = [[index // 65536, (index // 256) % 256, index % 256]
               for index in range(distinct)]
    payload = {str(item): list(triples[item % distinct]) + [768] for item in range(ITEMS)}
    (folder / "item_sids.json").write_text(json.dumps(payload))
    state = {
        "encoder.mlp.0.weight": torch.eye(WIDTH),
        **{f"rq.vq_layers.{level}.embed.weight": torch.zeros(CODES, WIDTH)
           for level in range(3)},
    }
    torch.save({"state_dict": state}, folder / "rqvae_best.pth")
    return folder / "item_sids.json"


def test_a_sid_above_the_forty_percent_limit_is_refused():
    sid = _arm(distinct=1)
    try:
        with mock.patch.object(server, "stage3_export_gate",
                               return_value={"allow": True, "state": "STUB"}):
            result = server.tool_stage3_start(str(sid))
    finally:
        shutil.rmtree(sid.parent)
    assert result["started"] is False
    assert "0.40" in result["refused"], result["refused"]
    assert result["collision"] == pytest.approx(1.0 - 1.0 / ITEMS, abs=1e-6)


def test_a_sid_below_the_forty_percent_limit_passes_the_quality_gate():
    sid = _arm(distinct=15982)  # 1 - 15982 / 24587 = 0.34998
    try:
        verdict = server.stage3_sid_quality(sid)
    finally:
        shutil.rmtree(sid.parent)
    assert verdict["collision"] == pytest.approx(0.35, abs=1e-4)
    assert verdict["allow"] is True, verdict


def test_the_waiver_is_off_unless_the_caller_asks_for_it():
    """`evaluate(..., waive_collision=False)` is the default and must stay a refusal."""
    sys.path.insert(0, str(ROOT))
    from tools.research.stage2 import sid_quality

    sid = _arm(distinct=1)
    try:
        verdict = sid_quality.evaluate(sid)
    finally:
        shutil.rmtree(sid.parent)
    assert verdict["allow"] is False
    assert verdict["collision"] == pytest.approx(1.0 - 1.0 / ITEMS, abs=1e-6)
    assert "waived" not in verdict


def test_waiving_the_collision_gate_keeps_the_measured_collision_on_record():
    """A waived run stays a refusal-flavoured record: the number is reported, not hidden.

    The waiver exists so a deliberate out-of-gate run can be measured; it must never
    look like a pass, so the verdict carries the real collision and an explicit
    ``collision_gate_waived`` flag for the caller to record.
    """
    sys.path.insert(0, str(ROOT))
    from tools.research.stage2 import sid_quality

    sid = _arm(distinct=1)
    try:
        verdict = sid_quality.evaluate(sid, waive_collision=True)
    finally:
        shutil.rmtree(sid.parent)
    assert verdict["allow"] is True, verdict
    assert verdict["collision_gate_waived"] is True
    assert verdict["collision"] == pytest.approx(1.0 - 1.0 / ITEMS, abs=1e-6)
    assert "waived" in verdict["reason"].lower()


def test_the_mcp_gate_forwards_the_waiver_and_records_it():
    """The MCP path must pass the flag through and surface it in its own verdict."""
    sid = _arm(distinct=1)
    try:
        with mock.patch.object(server, "stage3_export_gate",
                               return_value={"allow": True, "state": "STUB"}):
            result = server.tool_stage3_start(
                str(sid), waive_collision=True)
    finally:
        shutil.rmtree(sid.parent)
    # It may still refuse later (no embeddings, no live GPU), but the collision
    # gate must no longer be the reason and the waiver must be reported.
    assert result.get("collision_gate_waived") is True, result
    assert "0.40" not in str(result.get("refused", "")), result.get("refused")


def test_the_launch_state_carries_the_waiver_and_the_measured_collision():
    """The persisted launch state must carry the waiver, not just the return value.

    A6 entered Stage 3 outside the gate; the only durable trace is
    ``.cache/genrec-mcp/stage3__<variant>.json``.  If the waiver is missing there the
    run cannot be told apart from a normal pass after the fact, so
    ``tool_stage3_start`` must copy it into the state it saves.
    """
    source = Path(server.__file__).read_text(encoding="utf-8")
    start = source.index("    state = {")
    end = source.index("\n    }", start)
    block = source[start:end]
    # The saved state must name both the waiver and the collision it waives; a bare
    # pid/variant record cannot distinguish an out-of-gate run from a normal pass.
    assert "collision_gate_waived" in block, block
    assert '"collision": collision' in block or '"collision": collision,' in block, block


if __name__ == "__main__":
    test_a_sid_above_the_forty_percent_limit_is_refused()
    test_a_sid_below_the_forty_percent_limit_passes_the_quality_gate()
    test_the_waiver_is_off_unless_the_caller_asks_for_it()
    test_waiving_the_collision_gate_keeps_the_measured_collision_on_record()
    test_the_mcp_gate_forwards_the_waiver_and_records_it()
    print("ok")
