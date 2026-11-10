"""Stage3 entry refuses a SID whose three-level collision is not below 40%."""
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools" / "genrec-mcp"))

import server  # noqa: E402

BASE = ROOT / "results" / "stage2_RQ-VAE"


def _sid_dir(rows):
    path = Path(tempfile.mkdtemp(prefix="collision_gate_", dir=BASE))
    (path / "item_sids.json").write_text(json.dumps(rows))
    return path


def test_a_high_collision_sid_is_refused():
    path = _sid_dir({str(i): [1, 1, 1] for i in range(100)})
    try:
        result = server.tool_stage3_start(str(path / "item_sids.json"))
    finally:
        shutil.rmtree(path)
    assert result["started"] is False
    assert "0.40" in result["refused"]
    assert result["collision"] == 0.99


def test_a_low_collision_sid_passes_the_collision_gate():
    path = _sid_dir({str(i): [i, i + 1, i + 2] for i in range(100)})
    try:
        result = server.tool_stage3_start(str(path / "item_sids.json"))
    finally:
        shutil.rmtree(path)
    assert "collision" not in result.get("refused", "")


if __name__ == "__main__":
    test_a_high_collision_sid_is_refused()
    test_a_low_collision_sid_passes_the_collision_gate()
    print("ok")
