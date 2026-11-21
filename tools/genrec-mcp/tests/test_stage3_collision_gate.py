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


if __name__ == "__main__":
    test_a_sid_above_the_forty_percent_limit_is_refused()
    test_a_sid_below_the_forty_percent_limit_passes_the_quality_gate()
    print("ok")
